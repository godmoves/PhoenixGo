#pragma once

#include <memory>

#include "model/zero_model_base.h"

#include "dist/dist_config.pb.h"
#include "dist/dist_zero_model.grpc.pb.h"
#include "dist/leaky_bucket.h"

class DistZeroModelClient final : public ZeroModelBase {
public:
  DistZeroModelClient(const std::string &server_adress,
                      const DistConfig &dist_config);

  int Init(const ModelConfig &model_config) override;

  int Forward(const std::vector<std::vector<bool>> &inputs,
              std::vector<std::vector<float>> &policy,
              std::vector<float> &value) override;

  int GetGlobalStep(int &global_step) override;

  void Wait() override;

private:
  DistConfig m_config;
  std::string m_server_address;
  std::unique_ptr<DistZeroModel::Stub> m_stub;
  LeakyBucket m_leaky_bucket;
};
