#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

template <class T> class TaskQueue {
public:
  TaskQueue(int capacity = 0)
      : m_capacity(capacity),
        m_size(0),
        m_is_close(false) {}

  template <class U> void Push(U &&elem) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_capacity > 0) {
      m_push_cond.wait(lock, [this] { return m_queue.size() < m_capacity; });
    }
    m_queue.push_back(std::forward<U>(elem));
    m_size = m_queue.size();
    lock.unlock();
    m_pop_cond.notify_one();
  }

  template <class U> void PushFront(U &&elem) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_queue.push_front(std::forward<U>(elem));
    m_size = m_queue.size();
    lock.unlock();
    m_pop_cond.notify_one();
  }

  bool Pop(T &elem, int64_t timeout_us = -1) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (timeout_us < 0) {
      m_pop_cond.wait(lock, [this] { return !m_queue.empty() || m_is_close; });
    } else {
      if (!m_pop_cond.wait_for(
              lock, std::chrono::microseconds(timeout_us),
              [this] { return !m_queue.empty() || m_is_close; })) {
        return false;
      }
    }
    if (m_queue.empty()) {
      return false;
    }
    elem = std::move(m_queue.front());
    m_queue.pop_front();
    m_size = m_queue.size();
    lock.unlock();
    if (m_capacity > 0) {
      m_push_cond.notify_one();
    }
    return true;
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_is_close = true;
    }
    m_push_cond.notify_all();
    m_pop_cond.notify_all();
  }

  bool IsClose() const {
    return m_is_close;
  }

  int Size() const {
    return m_size;
  }

private:
  std::deque<T> m_queue;
  int m_capacity;
  std::atomic<int> m_size;
  std::mutex m_mutex;
  std::condition_variable m_push_cond;
  std::condition_variable m_pop_cond;
  std::atomic<bool> m_is_close;
};
