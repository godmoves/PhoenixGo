#include <thread>

#include "leaky_bucket.h"

LeakyBucket::LeakyBucket(int bucket_size, int refill_period_ms)
    : m_bucket_size(bucket_size),
      m_tokens(bucket_size),
      m_refill_period(std::chrono::milliseconds(refill_period_ms)),
      m_last_refill(clock::now()) {}

void LeakyBucket::ConsumeToken() {
  auto now = clock::now();
  if (now - m_last_refill > m_refill_period) {
    m_last_refill = now;
    m_tokens = m_bucket_size;
  }
  --m_tokens;
}

bool LeakyBucket::Empty() {
  return m_tokens <= 0;
}

void LeakyBucket::WaitRefill() {
  if (m_tokens <= 0) {
    std::this_thread::sleep_until(m_last_refill + m_refill_period);
    m_last_refill = clock::now();
    m_tokens = m_bucket_size;
  }
}
