#pragma once

#include "common/go_comm.h"
#include "common/timer.h"

class ByoYomiTimer {
public:
  ByoYomiTimer();
  void Set(float main_time, float byo_yomi_time);
  void Reset();
  bool IsEnable();
  void HandOff();
  void SetRemainTime(GoStoneColor color, float time);
  float GetRemainTime(GoStoneColor color);
  float GetByoYomiTime();

private:
  bool m_enable;
  float m_remain_time[2];
  float m_byo_yomi_time;
  GoStoneColor m_curr_player;
  Timer m_timer;
};
