#!/bin/bash
cd "$(dirname "$0")/.."
for it in 8 32; do for g in nim ttt cartpole connect_four; do python3 experiments/batch.py results_v2/standalone onehot_it$it "game=$g repr=onehot ppo_iters=$it budget=150000"; done; done
for it in 8 32; do for g in ttt cartpole; do python3 experiments/batch.py results_v2/standalone fixed_it$it "game=$g repr=fixed ppo_iters=$it budget=150000"; done; done
echo SWEEP2_DONE
