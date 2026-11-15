#include "envs.h"
#include "concept_env.h"
#include <cassert>
#include <cmath>

int main() {
  Rng rng(123);
  int legal_actions[128];
  float features[128];
  bool done = false;

  ConnectFour connect_four;
  connect_four.reset(rng);
  assert(connect_four.n_state() == 84);
  assert(connect_four.legal(legal_actions) >= 1);
  connect_four.state_feat(features);
  for (int i = 0; i < connect_four.n_state(); i++) assert(features[i] == 0.f || features[i] == 1.f);
  ConnectFour win_board;
  for (int col = 0; col < 4; col++) win_board.board[5][col] = 1;
  assert(ConnectFour::wins(win_board.board, 1));
  for (int col = 0; col < 4; col++) win_board.board[5][col] = 0;
  for (int row = 2; row < 6; row++) win_board.board[row][0] = 1;
  assert(ConnectFour::wins(win_board.board, 1));

  Nim nim;
  nim.remaining = 1;
  assert(nim.legal(legal_actions) == 1 && legal_actions[0] == 0);
  assert(nim.step(0, done, rng) == 1.f && done);
  nim.remaining = 2;
  done = false;
  assert(nim.step(1, done, rng) == 1.f && done);

  LineWorld line;
  line.position = 0;
  line.steps = 0;
  assert(line.legal(legal_actions) == 1 && legal_actions[0] == 1);
  for (int i = 0; i < LineWorld::kGoal - 1; i++) {
    done = false;
    assert(line.step(1, done, rng) == -0.01f);
    assert(!done);
  }
  done = false;
  assert(line.step(1, done, rng) == 1.f && done);
  assert(line.make()->n_state() == line.n_state());

  ConceptEnv wrapped(std::make_unique<LineWorld>(), {"state_magnitude"});
  assert(wrapped.n_action_feat() == 2);
  wrapped.reset(rng);
  float left_features[2], right_features[2];
  wrapped.action_feat(0, left_features);
  wrapped.action_feat(1, right_features);
  assert(left_features[0] == right_features[0]);
  assert(std::fabs(left_features[1] - std::tanh(-1.f)) < 1e-6f);
  assert(std::fabs(right_features[1] - std::tanh(1.f)) < 1e-6f);
  TicTacToe structural_ttt;
  structural_ttt.b[0] = 1;
  structural_ttt.b[1] = 1;
  std::vector<float> ttt_state(18);
  structural_ttt.state_feat(ttt_state.data());
  float structural_value = 0.f;
  assert(structural_ttt.additional_primitive("board_own_immediate_wins", ttt_state, structural_value));
  assert(std::fabs(structural_value - 1.f / 9.f) < 1e-6f);
  assert(structural_ttt.additional_primitive("board_occupancy", ttt_state, structural_value));
  assert(std::fabs(structural_value - 2.f / 9.f) < 1e-6f);

  ConnectFour structural_c4;
  structural_c4.board[5][0] = 1;
  structural_c4.board[5][1] = 1;
  structural_c4.board[5][2] = 1;
  std::vector<float> c4_state(84);
  structural_c4.state_feat(c4_state.data());
  assert(structural_c4.additional_primitive("board_own_immediate_wins", c4_state, structural_value));
  assert(std::fabs(structural_value - 1.f / 42.f) < 1e-6f);

  ConceptEnv temporal(std::make_unique<LineWorld>(), {"transition_delta_mean"});
  temporal.reset(rng);
  float temporal_features[2];
  temporal.action_feat(1, temporal_features);
  assert(std::fabs(temporal_features[0]) < 1e-7f); // no previous transition after reset
  done = false;
  temporal.step(1, done, rng);
  temporal.action_feat(1, temporal_features);
  const float expected_delta_mean = (0.1f - 1.f / 30.f) / 2.f;
  assert(std::fabs(temporal_features[0] - std::tanh(expected_delta_mean)) < 1e-6f);
  temporal.reset(rng);
  temporal.action_feat(1, temporal_features);
  assert(std::fabs(temporal_features[0]) < 1e-7f); // reset clears history

  ConceptEnv identity_only(std::make_unique<LineWorld>(), {"action_norm"});
  assert(identity_only.n_action_feat() == 1);
}
