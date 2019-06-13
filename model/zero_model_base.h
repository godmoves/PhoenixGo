#pragma once

#include <functional>
#include <vector>

#include <gflags/gflags.h>

#include "common/errordef.h"
#include "common/go_comm.h"

#include "model/model_config.pb.h"


class ZeroModelBase {
public:
  enum {
    BOARD_SIZE = GoComm::BOARD_SIZE,
    FEATURE_COUNT = GoFeature::FEATURE_COUNT,
    INPUT_DIM = BOARD_SIZE * BOARD_SIZE * FEATURE_COUNT,
    OUTPUT_DIM = BOARD_SIZE * BOARD_SIZE + 1,
  };

  typedef std::function<void(int, std::vector<std::vector<float>>, std::vector<float>)> callback_t;

  virtual ~ZeroModelBase() {}

  virtual int Init(const ModelConfig &model_config) = 0;

  virtual int Forward(const std::vector<std::vector<bool>> &inputs,
                      std::vector<std::vector<float>> &policy,
                      std::vector<float> &value) = 0;

  virtual void Forward(const std::vector<std::vector<bool>> &inputs,
                       callback_t callback) {
    std::vector<std::vector<float>> policy;
    std::vector<float> value;
    int ret = Forward(inputs, policy, value);
    callback(ret, std::move(policy), std::move(value));
  }

  virtual int GetGlobalStep(int &global_step) = 0;

  virtual int RpcQueueSize() { return 0; }

  virtual void Wait() {}
};
