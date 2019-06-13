#include "timer.h"

using namespace std::chrono;

Timer::Timer() : m_start(clock::now()) {}

void Timer::Reset() { m_start = clock::now(); }

int64_t Timer::sec() const {
  return duration_cast<seconds>(clock::now() - m_start).count();
}

int64_t Timer::ms() const {
  return duration_cast<milliseconds>(clock::now() - m_start).count();
}

int64_t Timer::us() const {
  return duration_cast<microseconds>(clock::now() - m_start).count();
}

float Timer::fsec() const {
  return std::chrono::duration<float>(clock::now() - m_start).count();
}

float Timer::fms() const {
  return std::chrono::duration<float, std::milli>(clock::now() - m_start).count();
}

float Timer::fus() const {
  return std::chrono::duration<float, std::micro>(clock::now() - m_start).count();
}
