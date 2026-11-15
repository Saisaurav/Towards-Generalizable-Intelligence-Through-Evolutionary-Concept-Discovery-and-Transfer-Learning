#!/bin/bash
# v9 concepts + audited standard learner. Target learner trained from scratch in every run; only expression strings carry over.
cd "$(dirname "$0")/.."
F=results_v9/features; O=results_v9/transfer
STD="act_onehot=1 greedy_eval=1 pop=3 ppo_iters=4 budget=150000 ckpt_every=3"
run(){ tag=$1; shift; python3 experiments/batch.py $O $tag "$@"; }
for g in nim cartpole connect_four; do
  X="$STD"; [ $g = nim ] && X="$STD state_onehot=22"
  run scratch "game=$g repr=aid $X"
  run random "game=$g repr=concepts concepts=$F/random_${g}_s{seed}.txt $X"
  run sxa "game=$g repr=sxa $X"
  run target_trained "game=$g repr=concepts concepts=$F/target_trained_$g.txt $X"
done
run source_rollout "game=nim repr=concepts concepts=$F/source_c4_to_nim_rollout.txt $STD state_onehot=22"
for t in rollout oracle; do
  run source_$t "game=cartpole repr=concepts concepts=$F/source_ttt_to_cartpole_$t.txt $STD"
  run source_$t "game=connect_four repr=concepts concepts=$F/source_ttt_to_c4_$t.txt $STD"
done
echo V9_MATRIX_DONE
