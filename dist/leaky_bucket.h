#pragma once

#include <chrono>

class LeakyBucket {
public:
  LeakyBucket(int bucket_size, int refill_period_ms);
  void ConsumeToken();
  bool Empty();
  void WaitRefill();

private:
  int m_bucket_size;
  int m_tokens;
  typedef std::chrono::system_clock clock;
  clock::duration m_refill_period;
  clock::time_point m_last_refill;
};
