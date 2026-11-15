#!/bin/bash
# TTT -> Connect Four with the v8 vocabulary. Standard learner; target trained from scratch every run; only expression strings carry over.
cd "$(dirname "$0")/.."
F=results_v8/features; O=results_v8/transfer; STD="act_onehot=1 greedy_eval=1 pop=3 ppo_iters=4 budget=150000 ckpt_every=3"; g=connect_four
run(){ tag=$1; shift; python3 experiments/batch.py $O $tag "$@"; }
run source_oracle  "game=$g repr=concepts concepts=$F/source_ttt_to_c4_oracle.txt $STD"
run source_rollout "game=$g repr=concepts concepts=$F/source_ttt_to_c4_rollout.txt $STD"
run random         "game=$g repr=concepts concepts=$F/random_${g}_s{seed}.txt $STD"
run target_trained "game=$g repr=concepts concepts=$F/target_trained_$g.txt $STD"
run scratch        "game=$g repr=aid $STD"
run sxa            "game=$g repr=sxa $STD"
echo V8_MATRIX_DONE
