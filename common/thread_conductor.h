#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "wait_group.h"

class ThreadConductor {
public:
  ThreadConductor();
  ~ThreadConductor();

  void Pause();
  void Resume(int num_threads);
  void Wait();
  void AckPause();
  bool Join(int64_t timeout_us = -1);
  void Sleep(int64_t duration_us);
  bool IsRunning();
  void Terminate();
  bool IsTerminate();

private:
  std::atomic<int> m_state;
  std::mutex m_mutex;
  std::condition_variable m_cond;

  WaitGroup m_resume_wg;
  WaitGroup m_pause_wg;

private:
  static const int k_pause = 0;
  static const int k_running = 1;
  static const int k_terminate = 2;
};
