#pragma once

#include <memory>
#include <string>

#include "mcts/mcts_config.pb.h"

std::unique_ptr<MCTSConfig> LoadConfig(const char *config_path);
std::unique_ptr<MCTSConfig> LoadConfig(const std::string &config_path);
