#!/bin/bash
cd "$(dirname "$0")/.."
F=results_v2/features; O=results_v2/transfer; C="pop=6 budget=150000 ckpt_every=3"
run(){ tag=$1; shift; python3 experiments/batch.py $O $tag "$@"; }
for pair in "cartpole ttt_to_cartpole" "nim c4_to_nim"; do set -- $pair; g=$1; p=$2
  run scratch "game=$g repr=aid $C"
  run source "game=$g repr=concepts concepts=$F/source_$p.txt $C"
  run random "game=$g repr=concepts concepts=$F/random_${g}_s{seed}.txt $C"
  run sxa "game=$g repr=sxa $C"
  run target_trained "game=$g repr=concepts concepts=$F/target_trained_$g.txt $C"
done
echo MATRIX_DONE
