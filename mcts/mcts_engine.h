/*
 * Tencent is pleased to support the open source community by making PhoenixGo
 * available.
 *
 * Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <atomic>
#include <future>
#include <thread>
#include <vector>

#include "common/go_comm.h"
#include "common/go_state.h"
#include "common/task_queue.h"
#include "common/thread_conductor.h"
#include "common/timer.h"
#include "common/wait_group.h"
#include "model/zero_model_base.h"

#include "byo_yomi_timer.h"
#include "mcts_config.h"
#include "mcts_debugger.h"
#include "mcts_monitor.h"

const int64_t k_action_value_base = 1 << 16;
const int k_unexpanded = 0;
const int k_expanding = 1;
const int k_expanded = 2;

typedef std::function<void(int, std::vector<float>, float)> EvalCallback;

struct EvalTask {
  std::vector<bool> features;
  EvalCallback callback;
};

class OutputAnalysisData {
public:
  OutputAnalysisData(const std::string& move, int visits, float winrate,
                     float policy, std::string pv)
      : m_move(move),
        m_visits(visits),
        m_winrate(winrate),
        m_policy(policy), 
        m_pv(pv) {};

  std::string get_info_string(int order) const {
    auto tmp = "info move " + m_move +
               " visits " + std::to_string(m_visits) +
               " winrate " + std::to_string(m_winrate) +
               " network " + std::to_string(m_policy);
    if (order >= 0) {
      tmp += " order " + std::to_string(order);
    }
    tmp += " pv " + m_pv;
    return tmp;
  }

  friend bool operator<(const OutputAnalysisData& a, const OutputAnalysisData& b) {
    if (a.m_visits == b.m_visits) {
      return a.m_winrate < b.m_winrate;
    }
    return a.m_visits < b.m_visits;
  }

private:
  std::string m_move;
  int m_visits;
  float m_winrate;
  float m_policy;
  std::string m_pv;
};

class MCTSEngine {
public:
  MCTSEngine(const MCTSConfig &config);
  ~MCTSEngine();

  void Reset();
  std::string Undo();
  void Move(GoCoordId x, GoCoordId y);
  void GenMove(GoCoordId &x, GoCoordId &y);
  void GenMove(GoCoordId &x, GoCoordId &y, std::vector<int> &visit_count, float &v_resign);
  const GoState &GetBoard();
  MCTSConfig &GetConfig();
  void SetPendingConfig(std::unique_ptr<MCTSConfig> config);
  MCTSDebugger &GetDebugger();
  int GetModelGlobalStep();
  ByoYomiTimer &GetByoYomiTimer();

private:
  TreeNode *InitNode(TreeNode *node, TreeNode *fa, int move, float prior_prob);
  TreeNode *FindChild(TreeNode *node, int move);

  void Eval(const GoState &board, EvalCallback callback);
  void EvalRoutine(std::unique_ptr<ZeroModelBase> model);

  TreeNode *Select(GoState &board);
  TreeNode *SelectChild(TreeNode *node);
  int Expand(TreeNode *node, GoState &board, const std::vector<float> &policy);
  void Backup(TreeNode *node, float value, int ch_len);
  void UndoVirtualLoss(TreeNode *node);

  bool CheckEarlyStop(int64_t timeout_us);
  bool CheckUnstable();
  bool CheckBehind();

  int64_t GetSearchTimeoutUs();
  int64_t GetSearchOvertimeUs(int64_t timeout_us);

  void Search();
  void SearchWait(int64_t timeout_us, bool is_overtime);
  void SearchResume();
  void SearchPause();
  void SearchRoutine();

  void ChangeRoot(TreeNode *node);
  void InitRoot();

  void DeleteRoutine();
  int DeleteTree(TreeNode *node);

  int GetBestMove(float &v_resign);
  int GetSamplingMove(float temperature);
  std::vector<int> GetVisitCount(TreeNode *node);

  template <class T>
  void TransformFeatures(T &features, int mode, bool reverse = false);
  void TransformCoord(GoCoordId &x, GoCoordId &y, int mode, bool reverse = false);

  void ApplyTemperature(std::vector<float> &probs, float temperature);

  void TTableUpdate(uint64_t hash, int64_t value);
  void TTableSync(TreeNode *node);
  void TTableClear();

  void EvalCacheInsert(uint64_t hash, const std::vector<float> policy, float value);
  bool EvalCacheFind(uint64_t hash, std::vector<float> &policy, float &value);

  bool IsPassDisable();
  void OutputAnalysis(TreeNode *parent);

private:
  MCTSConfig m_config;
  std::unique_ptr<MCTSConfig> m_pending_config;

  TreeNode *m_root;
  GoState m_board;

  std::vector<std::thread> m_eval_threads;
  TaskQueue<EvalTask> m_eval_task_queue;
  WaitGroup m_eval_threads_init_wg;
  WaitGroup m_eval_tasks_wg;
  std::atomic<int> m_model_global_step;

  std::vector<std::thread> m_search_threads;
  ThreadConductor m_search_threads_conductor;
  bool m_is_searching;

  std::thread m_delete_thread;
  TaskQueue<TreeNode *> m_delete_queue;

  std::atomic<int> m_simulation_counter;
  Timer m_search_timer;

  int m_num_moves;
  std::string m_moves_str;

  int m_gen_passes;

  ByoYomiTimer m_byo_yomi_timer;

  MCTSMonitor m_monitor;
  MCTSDebugger m_debugger;

  friend class MCTSMonitor;
  friend class MCTSDebugger;

  std::vector<GoCoordId> m_move_history;
};
