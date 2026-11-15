#include "envs.h"
#include "concept_env.h"
#include <iostream>
int main(){
  std::vector<float> p={0.125f,0.25f,0.0625f,0.5f,-1.f,0.75f,0.3f,0.25f,0.5f,-0.5f,0.75f,1.25f,0.5f,0.25f,0.125f,0.1f,0.4f,0.2f,0.8f,0.15f,0.65f,0.35f,0.5f,0.125f,0.25f,0.5f,0.6f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f};
  std::vector<std::string> e={"div(action_norm,action_norm)","div(min(add(state_var,state_abs_mean),mul(state_max_abs,state_abs_mean)),action_norm)","div(sub(max(legal_norm,action_norm),add(legal_norm,state_mean)),action_norm)","state_magnitude","state_dispersion","state_boundary_margin","abs(neg(state_min))","sqrt(neg(state_min))","log1p(state_max)","tanh(square(state_mean))","state_sparsity","state_median","state_iqr","state_skewness","state_entropy","state_position_moment","state_position_spread","state_adjacent_difference","state_sign_change_fraction","transition_delta_mean","transition_delta_abs_mean","transition_delta_max_abs","transition_state_alignment","board_own_open_lines","board_opponent_open_lines","board_own_immediate_wins","board_opponent_immediate_wins","board_center_control","board_occupancy"};
  for(auto&s:e) std::cout.precision(9),std::cout<<ConceptExpr{s}.eval(p)<<"\n";
  CartPole cp; TicTacToe ttt; float margin=0.f;
  if(std::string(kConceptPrimitiveContract)!="generic_state_action_v8") return 1;
  if(!cp.additional_primitive("state_boundary_margin",{0.6f,0.f,-0.8f,0.f},margin) || std::fabs(margin-0.2f)>1e-6f) return 1;
  if(ttt.additional_primitive("state_boundary_margin",{0.f},margin)) return 1;
  if(std::fabs(ConceptExpr{"abs(neg(state_min))"}.eval(p)-std::tanh(0.5f))>1e-6f) return 1;
  if(std::fabs(ConceptExpr{"sqrt(neg(state_min))"}.eval(p)-std::tanh(std::sqrt(0.5f)))>1e-6f) return 1;
  if(std::fabs(ConceptExpr{"log1p(state_max)"}.eval(p)-std::tanh(std::log1p(0.75f)))>1e-6f) return 1;
}
