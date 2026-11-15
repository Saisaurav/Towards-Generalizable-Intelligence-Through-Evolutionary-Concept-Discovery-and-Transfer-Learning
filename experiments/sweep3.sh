#!/bin/bash
cd "$(dirname "$0")/.."
for pop in 6 48; do
 for g in nim ttt cartpole connect_four; do python3 experiments/batch.py results_v2/standalone onehot_pop$pop "game=$g repr=onehot pop=$pop budget=150000"; done
 for g in ttt cartpole; do python3 experiments/batch.py results_v2/standalone fixed_pop$pop "game=$g repr=fixed pop=$pop budget=150000"; done
done
echo SWEEP3_DONE
