#pragma once
#include "env.h"
#include <string>
#include <vector>
#include <cmath>
#include <cctype>
#include <algorithm>
#include <stdexcept>
#include <limits>

constexpr const char* kConceptPrimitiveContract = "generic_state_action_v8";

// Small expression evaluator for persisted GP concepts.
// Unary functions take one expression; binary functions take two.
struct ConceptExpr {
  std::string s;
  static float prim(const std::string& n, const std::vector<float>& p) {
    static const char* names[] = {
      "state_mean","state_abs_mean","state_var","state_max_abs","action_norm",
      "legal_norm","state_magnitude","state_dispersion","state_boundary_margin",
      "state_min","state_max","state_range","state_positive_fraction",
      "state_negative_fraction","state_sparsity","state_median","state_iqr",
      "state_skewness","state_entropy","state_position_moment","state_position_spread",
      "state_adjacent_difference","state_sign_change_fraction",
      "transition_delta_mean","transition_delta_abs_mean",
      "transition_delta_max_abs","transition_state_alignment",
      "board_own_open_lines","board_opponent_open_lines",
      "board_own_immediate_wins","board_opponent_immediate_wins",
      "board_center_control","board_occupancy",
      "action_wins_now","action_blocks_opponent"
    };
    for (int i=0;i<35;i++) if (n==names[i]) {
      if ((size_t)i>=p.size()) throw std::runtime_error("missing GP primitive value: "+n);
      if (!std::isfinite(p[i])) throw std::runtime_error("unavailable GP primitive: "+n);
      return p[i];
    }
    throw std::runtime_error("unknown GP primitive: "+n);
  }
  static bool unary(const std::string& op) {
    return op=="abs" || op=="neg" || op=="square" || op=="sqrt" ||
           op=="log1p" || op=="tanh";
  }
  static float apply_unary(const std::string& op, float a) {
    if(op=="abs") return std::fabs(a);
    if(op=="neg") return -a;
    if(op=="square") return a*a;
    if(op=="sqrt") return std::sqrt(std::fabs(a));
    if(op=="log1p") return std::log1p(std::fabs(a));
    if(op=="tanh") return std::tanh(a);
    throw std::runtime_error("unknown GP unary op: "+op);
  }
  static float parse(const std::string& x, size_t& i, const std::vector<float>& p) {
    while (i<x.size() && std::isspace((unsigned char)x[i])) ++i;
    size_t b=i; while (i<x.size() && (std::isalnum((unsigned char)x[i]) || x[i]=='_')) ++i;
    if (b==i) throw std::runtime_error("bad GP expression");
    std::string op=x.substr(b,i-b);
    while (i<x.size() && std::isspace((unsigned char)x[i])) ++i;
    if (i>=x.size() || x[i]!='(') return prim(op,p);
    ++i; float a=parse(x,i,p);
    if(unary(op)) {
      while (i<x.size() && std::isspace((unsigned char)x[i])) ++i;
      if (i>=x.size() || x[i]!=')') throw std::runtime_error("expected close");
      ++i;
      return apply_unary(op,a);
    }
    while (i<x.size() && std::isspace((unsigned char)x[i])) ++i;
    if (i>=x.size() || x[i]!=',') throw std::runtime_error("expected comma"); ++i;
    float bval=parse(x,i,p); while (i<x.size() && std::isspace((unsigned char)x[i])) ++i;
    if (i>=x.size() || x[i]!=')') throw std::runtime_error("expected close"); ++i;
    if(op=="add") return a+bval; if(op=="sub") return a-bval; if(op=="mul") return a*bval;
    if(op=="div") return a/(std::fabs(bval)<1e-4f?(bval<0?-1e-4f:1e-4f):bval);
    if(op=="min") return std::min(a,bval); if(op=="max") return std::max(a,bval);
    throw std::runtime_error("unknown GP op: "+op);
  }
  float eval(const std::vector<float>& p) const {
    size_t i=0; float v=parse(s,i,p);
    while (i<s.size() && std::isspace((unsigned char)s[i])) ++i;
    if(i!=s.size()) throw std::runtime_error("trailing data in GP expression");
    if(!std::isfinite(v)) return 0.f;
    return std::tanh(std::max(-10.f,std::min(10.f,v)));
  }
};

// Wraps either fixed-action or per-action games into a common per-action scorer.
// All transferred concepts use the same typed, normalized primitive interface.
class ConceptEnv : public Env {
  std::unique_ptr<Env> base;
  std::vector<ConceptExpr> concepts;
  mutable std::vector<float> state;
  std::vector<float> prims;
  std::vector<float> last_delta;
  bool has_transition=false;
  bool act_onehot = false;   // append one-hot action identity to the per-action features
  int state_onehot_k = 0;    // >0: for 1-D integer-valued states s=v/(k-1), feed the net one-hot(v) (k+1 inputs); primitives still use the raw state
public:
  ConceptEnv(std::unique_ptr<Env> e, const std::vector<std::string>& ex, bool act_onehot_ = false, int state_onehot_k_ = 0)
    : base(std::move(e)), state(base->n_state()), prims(35), last_delta(base->n_state(),0.f), act_onehot(act_onehot_), state_onehot_k(state_onehot_k_) {
    if (state_onehot_k > 0 && base->n_state() != 1) throw std::runtime_error("state_onehot requires a 1-D state");
    for(auto& x:ex) concepts.push_back({x});
    if(std::find(ex.begin(),ex.end(),"action_norm")==ex.end()) concepts.push_back({"action_norm"});
  }
  bool per_action() const override { return true; }
  int n_state() const override { return state_onehot_k > 0 ? state_onehot_k + 1 : base->n_state(); }
  int n_action_feat() const override { return (int)concepts.size() + (act_onehot ? base->n_actions() : 0); }
  int n_actions() const override { return base->n_actions(); }
  void reset(Rng& r) override { base->reset(r); std::fill(last_delta.begin(),last_delta.end(),0.f); has_transition=false; }
  int legal(int* out) override { return base->legal(out); }
  void state_feat(float* out) override {
    if (state_onehot_k > 0) { float v; base->state_feat(&v); int k = (int)std::lround(v * (state_onehot_k - 1)); for (int i = 0; i <= state_onehot_k; i++) out[i] = i == k ? 1.f : 0.f; }
    else base->state_feat(out);
  }
  void action_feat(int a, float* out) override {
    base->state_feat(state.data());
    float mean=0, ab=0, mx=0; for(float x:state){mean+=x;ab+=std::fabs(x);mx=std::max(mx,std::fabs(x));}
    mean/=std::max(1,(int)state.size()); ab/=std::max(1,(int)state.size());
    float var=0; for(float x:state)var+=(x-mean)*(x-mean); var/=std::max(1,(int)state.size());
    int lg[128]; int n=base->legal(lg);
    prims[0]=mean; prims[1]=ab; prims[2]=var; prims[3]=mx;
    prims[4]=base->n_actions()>1 ? (2.f*a/(base->n_actions()-1)-1.f) : 0.f;
    prims[5]=base->n_actions()>1 ? (float)n/base->n_actions() : 1.f;
    float magnitude_sq=0.f;
    for(float x:state)magnitude_sq+=x*x;
    prims[6]=std::sqrt(magnitude_sq/std::max(1,(int)state.size()));
    prims[7]=std::sqrt(var);
    prims[8]=std::numeric_limits<float>::quiet_NaN();
    float boundary_margin=0.f;
    if(base->additional_primitive("state_boundary_margin",state,boundary_margin))
      prims[8]=boundary_margin;
    auto bounds=std::minmax_element(state.begin(),state.end());
    prims[9]=state.empty()?0.f:*bounds.first;
    prims[10]=state.empty()?0.f:*bounds.second;
    prims[11]=prims[10]-prims[9];
    int positive=0,negative=0,sparse=0;
    for(float value:state) {
      positive+=value>0.f;
      negative+=value<0.f;
      sparse+=std::fabs(value)<=1e-6f;
    }
    float count=std::max(1,(int)state.size());
    prims[12]=(float)positive/count;
    prims[13]=(float)negative/count;
    prims[14]=(float)sparse/count;
    std::vector<float> sorted=state;
    std::sort(sorted.begin(),sorted.end());
    auto quantile=[&](float q) {
      if(sorted.empty()) return 0.f;
      float pos=q*(sorted.size()-1); size_t lo=(size_t)pos, hi=std::min(lo+1,sorted.size()-1);
      return sorted[lo]+(pos-lo)*(sorted[hi]-sorted[lo]);
    };
    prims[15]=quantile(0.5f);
    prims[16]=quantile(0.75f)-quantile(0.25f);
    float third=0.f; for(float value:state) { float d=value-mean; third+=d*d*d; }
    prims[17]=var>1e-12f ? (third/count)/std::pow(var,1.5f) : 0.f;
    float mass=0.f; for(float value:state) mass+=std::fabs(value);
    float entropy=0.f;
    if(mass>1e-12f && state.size()>1) {
      for(float value:state) if(std::fabs(value)>0.f) { float probability=std::fabs(value)/mass; entropy-=probability*std::log(probability); }
      entropy/=std::log((float)state.size());
    }
    prims[18]=entropy;
    float position_moment=0.f, position_spread=0.f, abs_mass=0.f;
    float adjacent_difference=0.f; int sign_changes=0;
    for(size_t i=0;i<state.size();i++) {
      float position=state.size()>1 ? (2.f*(float)i/(state.size()-1)-1.f) : 0.f;
      position_moment+=state[i]*position;
      position_spread+=std::fabs(state[i])*position*position;
      abs_mass+=std::fabs(state[i]);
      if(i>0) {
        adjacent_difference+=std::fabs(state[i]-state[i-1]);
        sign_changes+=(state[i]*state[i-1]<0.f);
      }
    }
    prims[19]=state.empty()?0.f:position_moment/state.size();
    prims[20]=abs_mass>1e-12f?position_spread/abs_mass:0.f;
    prims[21]=state.size()>1?adjacent_difference/(state.size()-1):0.f;
    prims[22]=state.size()>1?(float)sign_changes/(state.size()-1):0.f;
    float delta_mean=0.f,delta_abs=0.f,delta_max=0.f,dot=0.f,state_norm=0.f,delta_norm=0.f;
    for(size_t i=0;i<state.size();i++) {
      float d=has_transition?last_delta[i]:0.f;
      delta_mean+=d; delta_abs+=std::fabs(d); delta_max=std::max(delta_max,std::fabs(d));
      dot+=state[i]*d; state_norm+=state[i]*state[i]; delta_norm+=d*d;
    }
    float count_state=std::max(1,(int)state.size());
    prims[23]=delta_mean/count_state;
    prims[24]=delta_abs/count_state;
    prims[25]=delta_max;
    prims[26]=(state_norm>1e-12f && delta_norm>1e-12f)?dot/std::sqrt(state_norm*delta_norm):0.f;
    static const char* board_names[] = {
      "board_own_open_lines","board_opponent_open_lines",
      "board_own_immediate_wins","board_opponent_immediate_wins",
      "board_center_control","board_occupancy"
    };
    for(int i=0;i<6;i++) {
      float value=std::numeric_limits<float>::quiet_NaN();
      if(base->additional_primitive(board_names[i],state,value)) prims[27+i]=value;
      else prims[27+i]=std::numeric_limits<float>::quiet_NaN();
    }
    { float tac[2]={std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN()};
      if(base->n_action_tactics()==2) base->action_tactics(a,tac);
      prims[33]=tac[0]; prims[34]=tac[1]; }
    for(size_t i=0;i<concepts.size();i++) out[i]=concepts[i].eval(prims);
    if(act_onehot) for(int i=0;i<base->n_actions();i++) out[concepts.size()+i]=(i==a)?1.f:0.f;
  }
  float step(int a,bool& d,Rng& r) override {
    std::vector<float> before(base->n_state()),after(base->n_state());
    base->state_feat(before.data());
    float reward=base->step(a,d,r);
    base->state_feat(after.data());
    for(size_t i=0;i<last_delta.size();i++) last_delta[i]=after[i]-before[i];
    has_transition=true;
    return reward;
  }
  int episode_result() const override { return base->episode_result(); }
  std::unique_ptr<Env> make() const override { return std::make_unique<ConceptEnv>(base->make(), raw(), act_onehot, state_onehot_k); }
  std::vector<std::string> raw() const { std::vector<std::string> x; for(auto& c:concepts)x.push_back(c.s); return x; }
};
