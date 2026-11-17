#include "robot_envs.h"
#include <cassert>
#include <cmath>
#include <vector>

int main() {
  Rng rng(123);
  RobotCfg tb_cfg = RobotCfg::parse("rnoise=0.05,rdelay=2,rlag=1.5", 300);
  TurtleBotNav turtle(tb_cfg);
  assert(turtle.n_state() == 32 && turtle.act_dim() == 2);
  turtle.reset(rng);
  std::vector<float> obs((size_t)turtle.n_state());
  turtle.state_feat(obs.data());
  for (float x : obs) assert(std::isfinite(x));
  bool done = false; float action[2] = {0.1f, -0.1f};
  turtle.step_cont(action, done, rng);

  RobotCfg arm_cfg = RobotCfg::parse("rnoise=0.05,rdelay=2,rlag=1.5,feat=3,fk=0.25", 100);
  Arm2Link arm(arm_cfg);
  assert(arm.n_state() == 15 && arm.act_dim() == 2);
  arm.reset(rng); obs.assign((size_t)arm.n_state(), 0.f); arm.state_feat(obs.data());
  for (float x : obs) assert(std::isfinite(x));
  done = false; arm.step_cont(action, done, rng);
  arm.state_feat(obs.data());
  for (float x : obs) assert(std::isfinite(x));
  return 0;
}
