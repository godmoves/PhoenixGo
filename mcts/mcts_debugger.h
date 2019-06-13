#pragma once

#include <string>

#include "common/go_comm.h"

class MCTSEngine;
class MCTSDebugger {
public:
  MCTSDebugger(MCTSEngine *engine);

  void Debug(); // call before move

  std::string GetDebugStr();
  std::string GetLastMoveDebugStr(); // call after move
  std::string GetMoveIndexStr(int ith);
  void UpdateLastMoveDebugStr();

  std::string GetMainMovePath(int rank = 0);
  void PrintTree(int depth, int topk, const std::string &prefix = "");

private:
  MCTSEngine *m_engine;
  std::string m_last_move_debug_str;
};
