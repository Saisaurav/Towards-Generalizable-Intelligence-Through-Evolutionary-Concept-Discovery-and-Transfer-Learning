#!/bin/bash
# Discovery with the v8 vocabulary (adds action_wins_now / action_blocks_opponent). Same GP budget and seed as v9 runs.
cd "$(dirname "$0")/.."
O=results_v8/gp
run(){ name=$1; shift; [ -f $O/$name/downstream_shortlist.txt ] || python3 gp_transfer.py "$@" --seed 2026 --pop 100 --gens 30 --out $O/$name > $O/$name.log 2>&1; echo "$name rc=$?"; }
run ttt_to_c4_rollout   --source ttt --target connect_four --teacher-policy rollout
run ttt_to_c4_oracle    --source ttt --target connect_four --teacher-policy oracle
run tt_connect_four_rollout --source connect_four --target connect_four --same-game-control --teacher-policy rollout
echo V8_DISCOVERY_DONE
