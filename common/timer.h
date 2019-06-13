#pragma once

#include <chrono>

class Timer {
public:
  Timer();
  void Reset();
  int64_t sec() const;
  int64_t ms() const;
  int64_t us() const;
  float fsec() const;
  float fms() const;
  float fus() const;

private:
  typedef std::chrono::system_clock clock;
  clock::time_point m_start;
};
