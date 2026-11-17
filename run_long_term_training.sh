#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 install_builtin_envs.py
make -C src brain
python3 train_long_term_suite.py --games all --trials 3 --minutes 45 --resume
python3 deploy_trained_policies.py --games all
