#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model/model_config.pb.h"
#include "model/zero_model_base.h"

namespace tensorflow {
class Session;
class Tensor;
class GraphDef;
}

class TpuZeroModel final : public ZeroModelBase {
public:
  TpuZeroModel(int tpu);
  ~TpuZeroModel();

  int Init(const ModelConfig &model_config) override;

  // input  [batch, board_size * board_size * features]
  // policy [batch, board_size * board_size + 1]
  int Forward(const std::vector<std::vector<bool>> &inputs,
              std::vector<std::vector<float>> &policy,
              std::vector<float> &value) override;

  int GetGlobalStep(int &global_step) override;

private:
  std::unique_ptr<tensorflow::Session> CreateSession(
    const tensorflow::GraphDef& graph_def, const std::string& tpu_name);

private:
  std::unique_ptr<tensorflow::Session> m_main_session;
  std::unique_ptr<tensorflow::Session> m_session;
  std::string m_tpu_name;
  std::vector<std::pair<std::string, tensorflow::Tensor>> m_inputs;
  std::vector<std::string> m_output_names;
  int m_per_core_batch_size;
  int m_tpu;
  int m_num_replicas;
};
