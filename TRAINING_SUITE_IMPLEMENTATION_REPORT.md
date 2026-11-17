# Long-term training suite implementation report

## What was added

- `train_long_term_suite.py`: long-running orchestration over the existing `EnvTrainer` for KQK, KRK, Pong, Tic-Tac-Toe, Snake, and Connect 4.
- `deploy_trained_policies.py`: installs each selected champion at the exact policy path used by the existing website `EnvService`.
- `run_long_term_training.sh` and `run_long_term_training.bat`: build/install/train/deploy launchers for Linux/macOS/WSL and Windows.
- `LONG_TERM_TRAINING_GUIDE.md`: usage, schedule, outputs, resume behavior, and interpretation.

## Verified in this workspace

- Python compilation passed for both new Python scripts.
- The builtin installer revalidated and installed all six environments without warnings.
- The C++ `brain` executable compiled successfully from `src/main.cpp` (compiler emitted existing style/initializer warnings, no build errors).
- A one-generation end-to-end training smoke run successfully produced and evaluated a policy for each of the six environments. The smoke run used tiny populations and very small evaluation counts, so it verifies the integration path only; its scores are not meaningful evidence of learning quality.
- The deployment script was exercised against five smoke policies and wrote them to the expected EnvService paths. Those temporary smoke policies were removed afterwards; no untrained/weak smoke policy is left deployed.
- `python -m pytest -q test_chess_endgames.py` passed: `1 passed`.

## Training design and limits

- Chess uses the existing 322-value observation and 64 discrete actions. The brain learns the two-phase piece-selection/destination-action protocol already implemented by the environments. Both chess environments expose a scripted-action hook; the per-action learner's existing imitation path uses it.
- Turn-based games use the existing per-action head. Real-time Pong and Snake use the fixed action head and a small scripted-action imitation weight.
- Each game defaults to three independent 45-minute runs. The best policy is selected using a held-out evaluation against random play; rule-based opponent evaluation is also recorded for games with that benchmark.
- These are independent trials, not sequential fine-tuning. The current `EnvTrainer` API does not provide a supported way to resume the optimizer/genome state from a partially completed run. `--resume` skips completed trials and retains their results; an interrupted in-progress trial must be rerun. Generation genome artifacts are retained for analysis.

The smoke scores are deliberately not included as evidence that the policies are strong. Run the full schedule and inspect `result.json` and `summary.json` for actual performance.
