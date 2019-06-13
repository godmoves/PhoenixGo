#include "wait_group.h"

void WaitGroup::Add(int v) {
  bool notify;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_counter += v;
    if (m_counter < 0) {
      throw std::runtime_error("WaitGroup::Add(): m_counter < 0");
    }
    notify = (m_counter == 0);
  }
  if (notify) {
    m_cond.notify_all();
  }
}

void WaitGroup::Done() {
  Add(-1);
}

bool WaitGroup::Wait(int64_t timeout_us) {
  std::unique_lock<std::mutex> lock(m_mutex);
  if (timeout_us < 0) {
    m_cond.wait(lock, [this] { return m_counter == 0; });
    return true;
  } else {
    return m_cond.wait_for(lock, std::chrono::microseconds(timeout_us),
                           [this] { return m_counter == 0; });
  }
}
