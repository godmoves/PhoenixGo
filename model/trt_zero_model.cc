#include "trt_zero_model.h"

#if USE_TENSORRT

#include <fstream>
#include <string>

#include <glog/logging.h>

#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"
#include "tensorrt/include/NvUffParser.h"

const std::string input_tensor_name = "inputs";
const std::string policy_tensor_name = "policy";
const std::string value_tensor_name = "value";

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR: LOG(ERROR) << msg; break;
      case Severity::kERROR: LOG(ERROR) << msg; break;
      case Severity::kWARNING: LOG(WARNING) << msg; break;
      case Severity::kINFO: LOG(INFO) << msg; break;
    }
  }
} g_logger;

TrtZeroModel::TrtZeroModel(int gpu)
    : m_builder(nullptr),
      m_network(nullptr),
      m_parser(nullptr),
      m_engine(nullptr),
      m_runtime(nullptr),
      m_context(nullptr),
      m_gpu(gpu),
      m_global_step(0) {}

TrtZeroModel::~TrtZeroModel() {
  if (m_context) {
    m_context->destroy();
  }
  if (m_engine) {
    m_engine->destroy();
  }
  if (m_runtime) {
    m_runtime->destroy();
  }
  if (m_parser) {
    m_parser->destroy();
  }
  if (m_network) {
    m_network->destroy();
  }
  if (m_builder) {
    m_builder->destroy();
  }
  for (auto buf : m_cuda_buf) {
    int ret = cudaFree(buf);
    if (ret != 0) {
      LOG(ERROR) << "cuda free err " << ret;
    }
  }
}

int TrtZeroModel::InitPlanModel(const ModelConfig &model_config, fs::path &model_path) {
  fs::path model_dir = model_config.model_dir();

  model_path = model_config.trt_plan_model_path();
  if (model_path.is_relative()) {
    model_path = model_dir / model_path;
  }

  std::ostringstream model_ss(std::ios::binary);
  if (!(model_ss << std::ifstream(model_path.string(), std::ios::binary).rdbuf())) {
    PLOG(ERROR) << "read tensorrt plan model '" << model_path << "' error";
    return ERR_READ_TRT_MODEL;
  }
  std::string model_str = model_ss.str();

  m_runtime = nvinfer1::createInferRuntime(g_logger);
  m_engine = m_runtime->deserializeCudaEngine(model_str.c_str(), model_str.size(), nullptr);
  if (m_engine == nullptr) {
    PLOG(ERROR) << "load cuda engine error";
    return ERR_LOAD_TRT_ENGINE;
  } 

  return 0;
}

int TrtZeroModel::InitUffModel(const ModelConfig &model_config, fs::path &model_path) {
  fs::path model_dir = model_config.model_dir();

  model_path = model_config.trt_uff_model_path();
  if (model_path.is_relative()) {
    model_path = model_dir / model_path;
  }

  m_builder = nvinfer1::createInferBuilder(g_logger);
  m_network = m_builder->createNetwork();
  m_parser = nvuffparser::createUffParser();

  m_parser->registerInput(input_tensor_name.c_str(),
                          nvinfer1::DimsCHW(FEATURE_COUNT, BOARD_SIZE, BOARD_SIZE),
                          nvuffparser::UffInputOrder::kNCHW);
  m_parser->registerOutput(policy_tensor_name.c_str());
  m_parser->registerOutput(value_tensor_name.c_str());

  if (!m_parser->parse(model_path.c_str(), *m_network, nvinfer1::DataType::kFLOAT)) {
    PLOG(ERROR) << "parse tensorrt uff model '" << model_path << "' error";
    m_builder->destroy();
    m_network->destroy();
    m_parser->destroy();
    return ERR_READ_TRT_MODEL;
  }

  if (model_config.enable_fp16()) {
    if (!m_builder->platformHasFastFp16()) {
      LOG(WARNING) << "fast fp16 is not supported by the platform";
    }
    m_builder->setFp16Mode(true);
    // Force the engine use FP16 precision
    // m_builder->setStrictTypeConstraints(true);
  }

  // Batch size larger than 16 is not recommended.
  m_builder->setMaxBatchSize(16);
  m_builder->setMaxWorkspaceSize(1 << 30);
  m_engine = m_builder->buildCudaEngine(*m_network);
  if (m_engine == nullptr) {
    PLOG(ERROR) << "load cuda engine error";
    return ERR_LOAD_TRT_ENGINE;
  }

  return 0;
}

int TrtZeroModel::Init(const ModelConfig &model_config) {
  cudaSetDevice(m_gpu);

  if (model_config.trt_uff_model_path() != "" && model_config.trt_plan_model_path() != "") {
    LOG(WARNING) << "both tensorrt uff and plan model path are set, use uff model by default";
  }

  if (model_config.trt_uff_model_path() == "" && model_config.enable_fp16()) {
      LOG(WARNING) << "enable_fp16 option is not supported by plan model";
  }

  fs::path model_path;
  if (model_config.trt_uff_model_path() != "") {
    InitUffModel(model_config, model_path);
  } else {
    InitPlanModel(model_config, model_path);
  }

  m_context = m_engine->createExecutionContext();

  int batch_size = m_engine->getMaxBatchSize();
  LOG(INFO) << "tensorrt max batch size: " << batch_size;
  for (int i = 0; i < m_engine->getNbBindings(); ++i) {
    auto dim = m_engine->getBindingDimensions(i);
    std::string dim_str = "(";
    int size = 1;
    for (int i = 0; i < dim.nbDims; ++i) {
      if (i)
        dim_str += ", ";
      dim_str += std::to_string(dim.d[i]);
      size *= dim.d[i];
    }
    dim_str += ")";
    LOG(INFO) << "tensorrt binding: " << m_engine->getBindingName(i) << " " << dim_str;

    void *buf;
    int ret;
    ret = cudaMalloc(&buf, batch_size * size * sizeof(float));
    if (ret != 0) {
      LOG(ERROR) << "cuda malloc err " << ret;
      return ERR_CUDA_MALLOC;
    }
    m_cuda_buf.push_back(buf);
  }

  if (!(std::ifstream(model_path.string() + ".step") >> m_global_step)) {
    LOG(WARNING) << "read global step from " << model_path.string() << ".step failed";
  }

  return 0;
}

int TrtZeroModel::Forward(const std::vector<std::vector<bool>> &inputs,
                          std::vector<std::vector<float>> &policy,
                          std::vector<float> &value) {
  int batch_size = inputs.size();
  if (batch_size == 0) {
    LOG(ERROR) << "Error batch size can not be 0.";
    return ERR_INVALID_INPUT;
  }

  std::vector<float> inputs_flat(batch_size * INPUT_DIM);
  for (int i = 0; i < batch_size; ++i) {
    if (inputs[i].size() != INPUT_DIM) {
      LOG(ERROR) << "Error input dim not match, need " << INPUT_DIM
                 << ", got " << inputs[i].size();
      return ERR_INVALID_INPUT;
    }
    for (int j = 0; j < INPUT_DIM; ++j) {
      inputs_flat[i * INPUT_DIM + j] = inputs[i][j];
    }
  }

  int ret = cudaMemcpy(m_cuda_buf[0], inputs_flat.data(),
                       inputs_flat.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (ret != 0) {
    LOG(ERROR) << "cuda memcpy err " << ret;
    return ERR_CUDA_MEMCPY;
  }

  m_context->execute(batch_size, m_cuda_buf.data());

  std::vector<float> internal_value(batch_size);
  ret = cudaMemcpy(internal_value.data(), m_cuda_buf[1],
                   internal_value.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (ret != 0) {
    LOG(ERROR) << "cuda memcpy err " << ret;
    return ERR_CUDA_MEMCPY;
  }
  value.resize(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    value[i] = -internal_value[i];
  }

  std::vector<float> policy_flat(batch_size * OUTPUT_DIM);
  ret = cudaMemcpy(policy_flat.data(), m_cuda_buf[2],
                   policy_flat.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (ret != 0) {
    LOG(ERROR) << "cuda memcpy err " << ret;
    return ERR_CUDA_MEMCPY;
  }
  policy.resize(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    policy[i].resize(OUTPUT_DIM);
    for (int j = 0; j < OUTPUT_DIM; ++j) {
      policy[i][j] = policy_flat[i * OUTPUT_DIM + j];
    }
  }

  return 0;
}

int TrtZeroModel::GetGlobalStep(int &global_step) {
  global_step = m_global_step;
  return 0;
}

#else // USE_TENSORRT

#include <glog/logging.h>

TrtZeroModel::TrtZeroModel(int gpu) {
  LOG(FATAL) << "TensorRT is not enable!";
}

TrtZeroModel::~TrtZeroModel() {
  LOG(FATAL) << "TensorRT is not enable!";
}

int TrtZeroModel::Init(const ModelConfig &model_config) {
  LOG(FATAL) << "TensorRT is not enable!";
  return 0;
}

int TrtZeroModel::Forward(const std::vector<std::vector<bool>> &inputs,
                          std::vector<std::vector<float>> &policy,
                          std::vector<float> &value) {
  LOG(FATAL) << "TensorRT is not enable!";
  return 0;
}

int TrtZeroModel::GetGlobalStep(int &global_step) {
  LOG(FATAL) << "TensorRT is not enable!";
  return 0;
}

#endif // USE_TENSORRT
