#include "tpu_zero_model.h"

#include <string>
#include <glog/logging.h>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/env.h"

namespace tf = tensorflow;

const std::string input_tensor_name = "inputs_";
const std::string policy_tensor_name = "policy_";
const std::string value_tensor_name = "value_";

// A GraphDef containing the ops required to initialize and shutdown a TPU.
// This proto was generated from the script oneoffs/generate_tpu_graph_def.py.
constexpr auto kTpuOpsGraphDef = R"(
node {
  name: "ConfigureDistributedTPU"
  op: "ConfigureDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
  attr {
    key: "embedding_config"
    value {
      s: ""
    }
  }
  attr {
    key: "is_global_init"
    value {
      b: false
    }
  }
  attr {
    key: "tpu_embedding_config"
    value {
      s: ""
    }
  }
}
node {
  name: "ShutdownDistributedTPU"
  op: "ShutdownDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
}
library {
}
)";

TpuZeroModel::TpuZeroModel(int tpu) : m_main_session(nullptr), m_session(nullptr), m_tpu(tpu) {}

TpuZeroModel::~TpuZeroModel() {
  if (m_session) {
    LOG(INFO) << "Closing worker session";
    tf::Status status = m_session->Close();
    if (!status.ok()) {
      LOG(ERROR) << "Error closing worker session: " << status.ToString();
    }
  }
  if (m_main_session != nullptr) {
    LOG(INFO) << "Shutting down TPU " << m_tpu_name;
    tf::Status status = m_main_session->Run({}, {}, {"ShutdownDistributedTPU"}, nullptr);
    if (!status.ok()) {
      LOG(ERROR) << "Error closing TPU session: " << status.ToString();
    }

    LOG(INFO) << "Closing main session";
    status = m_main_session->Close();
    if (!status.ok()) {
      LOG(ERROR) << "Error closing TensorFlow session: " << status.ToString();
    }
  }
}

std::unique_ptr<tf::Session> TpuZeroModel::CreateSession(const tf::GraphDef& graph_def,
                                                         const std::string& tpu_name) {
  tf::SessionOptions options;
  options.target = tpu_name;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);
  std::unique_ptr<tf::Session> session(tf::NewSession(options));
  if (session == nullptr) {
    LOG(ERROR) << "Could not create TensorFlow session.";
  }
  if (m_main_session == nullptr && session != nullptr) {
    LOG(INFO) << "Create main session succ";
  } else {
    LOG(INFO) << "Create worker session succ";
  }

  tf::Status status = session->Create(graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "Error creating graph: " << status.ToString();
  }
  LOG(INFO) << "Create graph succ";
  return session;
}

int TpuZeroModel::Init(const ModelConfig &model_config) {
  m_tpu_name = model_config.tpu_name();
  const std::string model_path = model_config.tpu_model_path();

  // Make sure tpu_name looks like a valid name.
  CHECK(absl::StartsWith(m_tpu_name, "grpc://"));

  tf::GraphDef graph_def;
  ::tensorflow::protobuf::TextFormat::ParseFromString(kTpuOpsGraphDef, &graph_def);

  m_main_session = CreateSession(graph_def, m_tpu_name);

  LOG(INFO) << "Initializing TPU " << m_tpu_name;
  tf::Status status = m_main_session->Run({}, {}, {"ConfigureDistributedTPU"}, nullptr);

  auto* env = tf::Env::Default();
  status = tf::ReadBinaryProto(env, model_path, &graph_def);
    if (!status.ok()) {
    LOG(ERROR) << "Error loading graph from " << model_path << ": " << status.ToString();
    return ERR_RESTORE_VAR;
  }
  LOG(INFO) << "Load checkpoint succ";

  // Check that we're actually loading a TPU model.
  bool found_tpu_op = false;
  for (const auto& node : graph_def.node()) {
    if (absl::StartsWithIgnoreCase(node.name(), "tpu")) {
      found_tpu_op = true;
      break;
    }
  }
  CHECK(found_tpu_op) << "didn't find any ops starting with \"tpu\" this "
                         "model looks like it wasn't compiled for TPU";

  // Count the number of times the model is replicated. There should be eight,
  // one replica for each TPU core.
  m_num_replicas = 0;
  for (const auto& node : graph_def.node()) {
    absl::string_view name = node.name();
    if (absl::ConsumePrefix(&name, input_tensor_name)) {
      int replica;
      CHECK(absl::SimpleAtoi(name, &replica));
      m_num_replicas = std::max(m_num_replicas, replica + 1);
    }
  }
  LOG(INFO) << "Found " << m_num_replicas << " model replicas in graph " << model_path;
  CHECK(m_num_replicas > 0);

  m_session = CreateSession(graph_def, m_tpu_name);

  // Prepare inputs and outputs
  m_per_core_batch_size = 1;
  m_inputs.clear();
  for (int i = 0; i < m_num_replicas; ++i) {
    m_inputs.emplace_back(
        absl::StrCat(input_tensor_name, i),
        tf::Tensor(tf::DT_FLOAT, tf::TensorShape({m_per_core_batch_size, INPUT_DIM})));
  }

  m_output_names.clear();
  for (int i = 0; i < m_num_replicas; ++i) {
    m_output_names.push_back(absl::StrCat(policy_tensor_name, i));
    m_output_names.push_back(absl::StrCat(value_tensor_name, i));
  }

  // Warm up the engine
  std::vector<std::vector<bool>> inputs(8, std::vector<bool>(INPUT_DIM, false));
  std::vector<std::vector<float>> policy;
  std::vector<float> value;
  Forward(inputs, policy, value);

  return 0;
}

int TpuZeroModel::Forward(const std::vector<std::vector<bool>> &inputs,
                          std::vector<std::vector<float>> &policy,
                          std::vector<float> &value) {
  const int batch_size = inputs.size();
  if (batch_size == 0) {
    LOG(ERROR) << "Error batch size can not be 0.";
    return ERR_INVALID_INPUT;
  }

  std::vector<std::vector<float>> inputsT(batch_size, std::vector<float> (INPUT_DIM, 0.0f));
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < INPUT_DIM; ++j) {
      inputsT[i][j] = inputs[i][j];
    }
  }

  int per_core_batch_size = (batch_size + m_num_replicas - 1) / m_num_replicas;
  if (per_core_batch_size > m_per_core_batch_size) {
    LOG(INFO) << "Change core batch size from " << m_per_core_batch_size << " to " << per_core_batch_size;
    m_per_core_batch_size = per_core_batch_size;
    m_inputs.clear();
    for (int i = 0; i < m_num_replicas; ++i) {
      m_inputs.emplace_back(
          absl::StrCat(input_tensor_name, i),
          tf::Tensor(tf::DT_FLOAT, tf::TensorShape({m_per_core_batch_size, INPUT_DIM})));
    }
  }

  // Split the input features across all replicas.
  for (int replica = 0; replica < m_num_replicas; ++replica) {
    size_t begin = replica * per_core_batch_size;
    size_t end = std::min(batch_size, (replica + 1) * per_core_batch_size);
    auto* data = m_inputs[replica].second.flat<float>().data();
    for (size_t i = begin; i < end; ++i) {
      if (inputsT[i].size() != INPUT_DIM) {
        LOG(ERROR) << "Error input dim not match, need " << INPUT_DIM << ", got " << inputsT[i].size();
        return ERR_INVALID_INPUT;
      }
      data = std::copy(inputsT[i].begin(), inputsT[i].end(), data);
    }
  }

  // Run the model.
  std::vector<tf::Tensor> network_outputs;
  tf::Status status = m_session->Run(m_inputs, m_output_names, {}, &network_outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Error session run: " << status.ToString();
    return ERR_SESSION_RUN;
  }

  // Copy the policy and value out of the output tensors.
  policy.resize(batch_size);
  value.resize(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    size_t replica = i / per_core_batch_size;
    size_t j = i % per_core_batch_size;

    const auto& policy_tensor = network_outputs[replica * 2].flat<float>();
    const auto& value_tensor = network_outputs[replica * 2 + 1].flat<float>();

    policy[i].resize(OUTPUT_DIM);
    std::memcpy(policy[i].data(), policy_tensor.data() + j * OUTPUT_DIM, sizeof(float) * OUTPUT_DIM);

    value[i] = -value_tensor.data()[j];
  }

  return 0;
}

int TpuZeroModel::GetGlobalStep(int &global_step) {
  std::vector<tf::Tensor> network_outputs;
  tf::Status status = m_session->Run({}, {"global_step"}, {}, &network_outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Error session run: " << status.ToString();
    return ERR_SESSION_RUN;
  }

  global_step = network_outputs[0].scalar<int64_t>()();
  return 0;
}
