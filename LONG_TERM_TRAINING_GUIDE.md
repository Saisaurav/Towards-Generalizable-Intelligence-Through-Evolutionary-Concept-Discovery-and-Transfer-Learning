# Long-term training: chess endgames + four games

This suite trains the existing NEAT+PPO brain on six built-in environments:

- Chess KQK — king and queen versus king
- Chess KRK — king and rook versus king
- Pong
- Tic-Tac-Toe
- Snake
- Connect 4

## 1. Build and validate

The launcher reinstalls the six built-ins and rebuilds `src/brain` from the current C++ sources. Python dependencies used by the existing platform must already be installed in the active environment.

Linux/macOS/WSL:

```bash
./run_long_term_training.sh
```

Windows (with Python and MinGW-w64 `g++` on PATH):

```bat
run_long_term_training.bat
```

Or run the steps individually:

```bash
python install_builtin_envs.py
# Linux/macOS/WSL:
make -C src brain
# Windows PowerShell (from src):
# g++ -std=c++17 -O2 -pthread main.cpp -o brain.exe
python train_long_term_suite.py --games all --trials 3 --minutes 45
python deploy_trained_policies.py --games all
```

## 2. Default training schedule

The default schedule is **3 independent trials × up to 45 minutes for each of 6 games** (up to 13.5 hours of learner time, plus validation and evaluation). The script uses the existing `EnvTrainer`, the repository concept bank, per-action training for turn-based games, and fixed action heads for real-time games. It uses the environment's scripted-action hook for imitation where supported; Pong and Snake use a small imitation weight to expose that expert signal to their fixed heads.

Each trial is a fresh training run, not a continuation of the previous trial's neural weights. The current trainer exports policies at the end of a run but does not expose a reliable mid-run optimizer/genome resume API. `--resume` therefore skips **completed** trials; it cannot recover a trial interrupted before its final policy was written. Genome artifacts and generation logs are retained per trial for analysis.

## 3. Resume / smaller tests

Resume completed trials after interruption:

```bash
python train_long_term_suite.py --games all --trials 3 --minutes 45 --resume
```

Quick smoke run for the two chess endgames:

```bash
python train_long_term_suite.py --games chess_kqk,chess_krk --trials 1 --minutes 1 --eval-episodes 10 --final-eval-episodes 10 --expert-eval-episodes 10 --calibration-episodes 10
```

Train one game for longer:

```bash
python train_long_term_suite.py --games connect4 --trials 4 --minutes 50
```

## 4. Outputs

```text
training_runs/long_term_suite/
  suite_status.json
  chess_kqk/
    summary.json
    trials/trial_01/{progress.jsonl,result.json,genomes/}
    policies/{trial_01.json,champion.json}
  ...
```

Each completed policy is evaluated against random play and, where available, the environment's rule-based/scripted opponent. The champion is selected by win + half-draw rate against random play for multiplayer games, or mean return for Snake. Rule-based results are reported separately and are not hidden by the champion score.

## 5. Deploy to the website

After training:

```bash
python deploy_trained_policies.py --games all
```

This copies each champion to the matching saved environment's `generated_envs/<id>/policies/brain.json`, which is the path the existing `EnvService` loads. Restart the web backend after deploying so it reloads the policy. Keep the `training_runs` directory as the experiment record.

## 6. Important interpretation

A successful training run means the brain produced a valid policy and the evaluation completed. It does **not** automatically mean the policy is expert-level. Use the saved evaluation metrics, especially the separate rule-based-opponent results, to judge actual performance. The chess endgames have an expert/scripted-action hook available for imitation and benchmarking; training still has to demonstrate that the learned policy can convert positions reliably.
