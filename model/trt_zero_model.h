#pragma once

#include <memory>

#include <boost/filesystem.hpp>

#include "model/model_config.pb.h"
#include "model/zero_model_base.h"

namespace fs = boost::filesystem;

namespace nvinfer1 {
class IBuilder;
class INetworkDefinition;
class ICudaEngine;
class IRuntime;
class IExecutionContext;
}

namespace nvuffparser {
class IUffParser;
}

class TrtZeroModel final : public ZeroModelBase {
public:
  TrtZeroModel(int gpu);
  ~TrtZeroModel();

  int Init(const ModelConfig &model_config) override;

  // input  [batch, board_size * board_size * features]
  // policy [batch, board_size * board_size + 1]
  int Forward(const std::vector<std::vector<bool>> &inputs,
              std::vector<std::vector<float>> &policy,
              std::vector<float> &value) override;

  int GetGlobalStep(int &global_step) override;

private:
  int InitPlanModel(const ModelConfig &model_config, fs::path &model_path);
  int InitUffModel(const ModelConfig &model_config, fs::path &model_path);

private:
  nvinfer1::IBuilder *m_builder;
  nvinfer1::INetworkDefinition *m_network;
  nvuffparser::IUffParser *m_parser;
  nvinfer1::ICudaEngine *m_engine;
  nvinfer1::IRuntime *m_runtime;
  nvinfer1::IExecutionContext *m_context;
  std::vector<void *> m_cuda_buf;
  int m_gpu;
  int m_global_step;
};
