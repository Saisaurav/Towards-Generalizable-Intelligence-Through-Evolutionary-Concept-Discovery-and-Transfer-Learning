#!/bin/bash
# Trains each cached demo game from scratch, with and without bank concepts (10 seeds), and stores raw JSON under results_v2/demo/.
cd "$(dirname "$0")/.."
for g in demo_1_open_field demo_2_wall_maze demo_3_hazard_corridor; do
  python3 experiments/batch.py results_v2/demo ${g}_without "game=grid grid=demo/$g.txt repr=aid pop=6 budget=100000 ckpt_every=2"
  python3 experiments/batch.py results_v2/demo ${g}_with "game=grid grid=demo/$g.txt repr=concepts concepts=demo/grid_concepts.txt pop=6 budget=100000 ckpt_every=2"
done
echo DEMOS_DONE
