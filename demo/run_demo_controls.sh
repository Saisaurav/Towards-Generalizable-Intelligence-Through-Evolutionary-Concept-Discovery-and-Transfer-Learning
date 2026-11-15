#!/bin/bash
cd "$(dirname "$0")/.."
for g in demo_1_open_field demo_2_wall_maze demo_3_hazard_corridor; do
  python3 experiments/batch.py results_v2/demo ${g}_random "game=grid grid=demo/$g.txt repr=concepts concepts=results_v2/features/random_grid_s{seed}.txt pop=6 budget=100000 ckpt_every=2"
  python3 experiments/batch.py results_v2/demo ${g}_sxa "game=grid grid=demo/$g.txt repr=sxa pop=6 budget=100000 ckpt_every=2"
done
echo CONTROLS_DONE
