#include "mcts_config.h"

#include <fstream>

#include <glog/logging.h>
#include <google/protobuf/text_format.h>

std::unique_ptr<MCTSConfig> LoadConfig(const char *config_path) {
  auto config = std::unique_ptr<MCTSConfig>(new MCTSConfig);
  std::ostringstream conf_ss;
  if (!(conf_ss << std::ifstream(config_path).rdbuf())) {
    PLOG(ERROR) << "read config file " << config_path << " error";
    return nullptr;
  }
  if (!google::protobuf::TextFormat::ParseFromString(conf_ss.str(), config.get())) {
    LOG(ERROR) << "parse config file " << config_path << " error! buf=" << conf_ss.str();
    return nullptr;
  }
  return config;
}

std::unique_ptr<MCTSConfig> LoadConfig(const std::string &config_path) {
  return LoadConfig(config_path.c_str());
}
