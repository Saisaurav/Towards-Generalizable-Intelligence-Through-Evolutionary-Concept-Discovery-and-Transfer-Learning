#include "envs.h"
#include "concept_env.h"
#include <cstdio>
int main(){
  for(int which=0;which<2;which++){
    std::unique_ptr<Env> b; if(which==0){auto e=std::make_unique<TicTacToe>(); e->nf=0; b=std::move(e);} else b=std::make_unique<ConnectFour>();
    ConceptEnv env(std::move(b), {"action_wins_now","action_blocks_opponent"}); Rng rng(9); int lg[64]; int nf=env.n_action_feat(); std::vector<float> f(nf), st(env.n_state());
    int printed=0;
    for(int ep=0;ep<400&&printed<400;ep++){ env.reset(rng); bool d=false; while(!d&&printed<400){ int nl=env.legal(lg); env.state_feat(st.data());
        for(int k=0;k<nl;k++){ env.action_feat(lg[k],f.data()); std::printf("%s %d",which?"c4":"ttt",lg[k]); for(float v:st) std::printf(" %g",v); std::printf(" | %g %g\n",f[0],f[1]); printed++; }
        bool dd=false; env.step(lg[rng.below(nl)],dd,rng); d=dd; } }
  }
}
