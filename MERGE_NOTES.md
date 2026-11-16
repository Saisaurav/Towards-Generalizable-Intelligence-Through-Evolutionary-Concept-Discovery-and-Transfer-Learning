# Merge notes: frontend/concept-refresh branch + step3 real-time learner

Base `0107d0a`. Left side: `d5305da` + uncommitted work (concept visualizer frontend, generation-20 GP concept
refresh, expert/imitation hooks). Right side: uncommitted step3 work (fixed-head real-time learner, learnability gate,
Snake/Pong fixtures, `baseline/`, `robotics/`).

## What was kept from where
- **C++ core** (`src/`): old side (functional superset: continuous head, `expert_action`/`--imitation`,
  `--max-episode-steps`, validation) plus step3's `StopLimits`/`--min-generations`. `robotics_env.h` is the same
  PointMass/StackEnv in both; old version kept.
- **Frontend**: old rewrite. Added the step3 "success rate" wording (`training_quality_mode == "return"`).
- **Python trainer** (`env_trainer.py`): step3's calibration/quality mode, `--min/--max-generations`, head selection;
  old's concept refresh (armed for the per-action head only). Removed the old title-regex Snake hack and the 256-step
  real-time cap (replaced by runtime `max_frames`/`stall_frames`).
- **Prompts**: step3's "MAKE THE GAME LEARNABLE" section + frame-cap text; the old Snake paragraph was trimmed so it no
  longer asks for the raw body in the observation.

## Behaviour decisions to review
1. `--head fixed` (real-time) now defaults to the config measured in `baseline/results`
   (hidden 16, iters 16, train-fitness, persist-adam, carry 0.5, ent 0, **imitation off**) and population 8.
   The step3 trainer did not pass these, so its default path was the weaker one.
2. Imitation of `scripted_action` stays on for the per-action head (old behaviour), off for the fixed head, and is
   opt-in there via `--imitation` / `GAME_ENV_IMITATION`. `scripted_action` is only sent over the relay when needed.
3. `src/brain.exe` was removed from the tree (stale for the merged source). Rebuild: `cd src && make`.

## Verification
- Merged `brain` on Snake fixture, step3 explicit flags: identical to pristine step3 (gen 30: 97/100, return 19.27).
- Merged Python `EnvTrainer` defaults on the Snake fixture: target reached in ~25 s (gen 14, 93% success);
  Python `NeatPolicy` greedy return +17.7 vs random -0.98.
- 130 Python tests + C++ regression/integration tests pass.
- Not verified: an LLM-generated Snake end-to-end (needs an API key), Windows build.
