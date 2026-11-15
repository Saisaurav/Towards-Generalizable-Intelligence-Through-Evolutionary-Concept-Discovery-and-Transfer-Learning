#!/bin/bash
# v9 concept discovery (user's improved generator) for the evaluation pairs. Same GP budget everywhere: pop 100, 30 generations, seed 2026.
cd "$(dirname "$0")/.."
O=results_v9/gp
run(){ name=$1; shift; [ -f $O/$name/downstream_shortlist.txt ] || python3 gp_transfer.py "$@" --seed 2026 --pop 100 --gens 30 --out $O/$name > $O/$name.log 2>&1; echo "$name rc=$?"; }
run ttt_to_cartpole_rollout  --source ttt --target cartpole --teacher-policy rollout
run ttt_to_cartpole_oracle   --source ttt --target cartpole --teacher-policy oracle
run ttt_to_c4_rollout        --source ttt --target connect_four --teacher-policy rollout
run ttt_to_c4_oracle         --source ttt --target connect_four --teacher-policy oracle
run c4_to_nim_rollout        --source connect_four --target nim --teacher-policy rollout
run tt_cartpole_rollout      --source cartpole --target cartpole --same-game-control --teacher-policy rollout
run tt_connect_four_rollout  --source connect_four --target connect_four --same-game-control --teacher-policy rollout
run tt_nim_rollout           --source nim --target nim --same-game-control --teacher-policy rollout
run tt_nim_oracle            --source nim --target nim --same-game-control --teacher-policy oracle
echo DISCOVERY_DONE
