#pragma once

#include <memory>

#include "model/model_config.pb.h"
#include "model/zero_model_base.h"

namespace tensorflow {
class Session;
}

class TfZeroModel final : public ZeroModelBase {
public:
  TfZeroModel(int gpu);
  ~TfZeroModel();

  int Init(const ModelConfig &model_config) override;

  // input  [batch, board_size * board_size * features]
  // policy [batch, board_size * board_size + 1]
  int Forward(const std::vector<std::vector<bool>> &inputs,
              std::vector<std::vector<float>> &policy,
              std::vector<float> &value) override;

  int GetGlobalStep(int &global_step) override;

  static void SetMKLEnv(const ModelConfig &model_config);

private:
  std::unique_ptr<tensorflow::Session> m_session;
  int m_gpu;
};
