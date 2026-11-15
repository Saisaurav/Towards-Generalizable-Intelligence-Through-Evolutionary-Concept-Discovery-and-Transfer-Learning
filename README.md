# Cross-game concept transfer prototype

This package contains an experimental NEAT+PPO learner and GP concept-transfer pipeline. It currently includes Tic-Tac-Toe, CartPole, Connect Four, Nim, and LineWorld.

## Current status (audited and re-run)

See `RESEARCH_REPORT.md` for the full audit, standalone-learner results, transfer matrix and final assessment. Summary:

- A real NEAT bug was fixed (initial MLP genomes now share innovation history); regression tests verify the hidden layer, gradients, Lamarckian write-back and inheritance.
- Standalone learners: CartPole is solved with the native 2-logit head under a greedy policy; Tic-Tac-Toe reaches 0.88 win-or-draw without tactical features; Connect Four beats random only; Nim is not learned with the scalar encoding.
- **Cross-game transfer was not demonstrated.** Source-discovered GP concepts did not improve CartPole learning over scratch or random features, and raw state x action features beat them; the Connect Four to Nim pair is uninformative.
- v9 concept generator (richer primitives, oracle teacher) was merged and evaluated with the audited learner (`RESEARCH_REPORT.md` section J): still no measurable source-to-target transfer; only target-game concepts helped, and only on Nim.
- With action-conditioned primitives (`generic_state_action_v8`: supplied win-now / block-opponent lookahead) TTT-discovered concepts improve Connect Four from 0.79 to 0.97 (`RESEARCH_REPORT.md` section K), but a primitives-only control matches them, so the gain comes from the primitives, not from GP discovery.
- Supported environments: Tic-Tac-Toe, CartPole, Connect Four, Nim, LineWorld, and LLM-parameterised grid worlds (width/height/start/goal/walls/hazards only; not arbitrary game generation).
- Known limitations: single-core experiments, 10 seeds, pilots only for the grid demos, concept primitives are state aggregates, transfer recorded at pair level.

Reproduce (from the repo root):

```bash
cd src && make runner regression_test env_smoke_test lamarckian_hidden_test && ./regression_test && cd ..
python3 -m unittest test_concept_bank test_llm_game_platform
python3 experiments/batch.py results_v2/standalone my_run "game=cartpole repr=fixed budget=150000"   # 10 seeds, JSON per run
python3 gp_transfer.py --source ttt --target cartpole --seed 2026 --pop 100 --gens 30 --out results_v2/gp/ttt_to_cartpole
python3 experiments/make_features.py && experiments/transfer_matrix.sh && python3 experiments/analyze_transfer.py
PYTHONPATH=. python3 experiments/build_bank.py && python3 experiments/build_frontend_data.py
python3 -m llm_game_platform.server     # then open http://127.0.0.1:8000/research (cached data, works offline)
```

## Project goal and research question

The longer-term project is a platform where a user describes a task, the system
creates an environment, an agent learns to solve it, and knowledge discovered
through earlier tasks can help with later ones. Games are the initial controlled
testbed; the broader goal is to investigate whether reusable computational
concepts can make learning unfamiliar tasks more efficient.

This is both a learning-system research prototype and a planned user-facing
product. It now generates and trains on LLM-written Python environments (see
`GENERATED_ENVIRONMENTS.md`), but that removes interface limits only; it is
not evidence of general-purpose intelligence. The existing cross-game results are preliminary;
see `EXPERIMENT_REPORT.md` and `RESTART_EXPERIMENT_REPORT.md` for methodology,
results, and limitations.

## Learning architecture and transfer boundary

The research core combines:

- **NEAT** to evolve neural-network topologies and populations.
- **PPO** to optimize task-specific policies.
- **Genetic Programming (GP)** to discover mathematical state/action features
  that may be useful on other tasks.

Each target task starts with a fresh learner. The intended transfer is compatible
GP-discovered concepts as input features—not neural-network weights or topology,
PPO optimizer state, or training trajectories. The target agent must learn anew
how to use those features. Concept compatibility and source-game fitness alone
are not evidence that a concept helps a target; the downstream admission
experiment compares fresh learners under paired seeds and a fixed budget.

The current C++ learner accepts a fixed set of built-in environments through
the `Env` interface. The Python GP discovery and experiment pipeline works with
the supported environments listed below. Neither component currently accepts
arbitrary LLM-generated game rules.

## Environments

| Environment | Rules and observation | Actions |
|---|---|---|
| `ttt` | 3x3 Tic-Tac-Toe against a random opponent; the opponent sometimes opens | Place in an empty cell |
| `cartpole` | Classic four-value CartPole simulation with a 500-step horizon | Push left or right |
| `connect_four` | 6x7 Connect Four against a random opponent; the opponent sometimes opens | Drop in a non-full column |
| `nim` | Single heap, 10–21 objects at reset; take 1–3, then a random opponent takes 1–3; taking the last object wins | Take 1, 2, or 3 (illegal takes are masked) |
| `lineworld` | Deterministic 11-position line; start at positions 0–4, reach position 10 within 30 steps | Move left or right (moves off the line are masked) |

These are controlled experimental environments, not claims of broad game coverage. Connect Four and Nim use random opponents; their results measure learning against those opponents, not optimal play.

## Exact oracle teacher datasets (v9)

For controlled comparisons, GP discovery also supports `--teacher-policy oracle` on TicTacToe, Nim, and LineWorld. These environments have exact action-value evaluators: minimax for TicTacToe, optimal normal-play recursion for the one-heap Nim rules, and deterministic shortest-path continuation for LineWorld. The resulting labels are action values centered by the mean value of legal actions, matching the existing advantage-style dataset interface.

Example: `python gp_transfer.py --source ttt --target connect_four --teacher-policy oracle`. The default remains `--teacher-policy rollout`. Oracle mode is deliberately rejected for CartPole and Connect Four because this project does not yet implement exact evaluators for those environments. The target feature dataset uses the selected oracle only when the target environment supports it; otherwise it falls back to the rollout dataset builder. Summary metadata records the teacher policy and dataset protocol.

Oracle labels are not automatically transferable concepts. They are a cleaner supervision source in the supported games, and downstream target-learner tests remain necessary.

## Counterfactual rollout data protocol (v8)

The GP dataset builders now use an environment-calibrated action-comparison seed protocol (`game_calibrated_rollout_seeds_v3`). For environments where pilot measurements showed lower variance in action-value differences, legal actions are evaluated with common random numbers (the same rollout seed list per action). For CartPole and Nim, the pilot showed higher variance under common random numbers, so each action receives independent rollout seeds instead. This choice affects only source dataset label generation; the primitive contract and C++ concept runtime remain `generic_state_action_v7`.

The pilot compared the variance of an estimated two-action value difference over repeated experiments on one representative fixed state per environment. Ratios of paired-seed variance to independent-seed variance were: TicTacToe 0.376, Connect Four 0.765, CartPole 1.364, Nim 1.469, and LineWorld 0.381. These are pilot results, not universal guarantees across all states or policies. Recalibrate with broader state samples before making general claims.

## Build

```bash
cd src
make
make test-envs
```

## Run a target learner

```bash
./brain <ttt|cartpole|connect_four|nim|lineworld> <concept-expression-file> <env-step-budget> <seed> [population]
```

The target learner is always created from scratch. The expression file supplies only the action-conditioned concept features.
The runtime automatically includes `action_norm` unless the file already contains that exact primitive expression, so otherwise state-only concepts do not make all legal actions indistinguishable.

## GP discovery and admission

From the package root:

```bash
python gp_transfer.py --out results --source ttt --seed 2026 --pop 60 --gens 16 --downstream-budget 150000
python gp_transfer.py --out results_reverse --source cartpole --seed 2027 --pop 60 --gens 16 --downstream-budget 150000
```

The GP uses NSGA-II over five objectives: worst-of-train/validation signed correlation, validation signed correlation, train-validation consistency, expression simplicity, and behavioral novelty. Signed correlation means an inverted feature is not treated as equally predictive; the grammar can discover explicit negation if useful. A separate source test split is reported and is not used during evolution. Constant or duplicate candidates and primitives unavailable in the target are filtered before the downstream shortlist is formed. Select any supported source and a different target with `--source` and `--target`; the default source-target pairs are TTT→CartPole, CartPole→TTT, ConnectFour→Nim, Nim→LineWorld, and LineWorld→ConnectFour. Dataset sizes can be overridden with `--nstates N` and `--rollouts N` (0 uses environment defaults); these apply independently to the train, validation, test, and target feature datasets.

For example:

```bash
python gp_transfer.py --out results_c4_to_nim --source connect_four --target nim --seed 2028 --pop 60 --gens 16
python gp_transfer.py --out results_nim_to_line --source nim --target lineworld --seed 2029 --pop 60 --gens 16
python gp_transfer.py --out results_line_to_c4 --source lineworld --target connect_four --seed 2030 --pop 60 --gens 16
```

The shared primitive contract is now `generic_state_action_v7`. It retains all v4 primitives and adds four order-sensitive descriptors of the normalized observation vector:

| Primitive | Meaning | Availability |
|---|---|---|
| `state_min` / `state_max` | Minimum / maximum normalized state value | All five games |
| `state_range` | Difference between maximum and minimum state values | All five games |
| `state_positive_fraction` / `state_negative_fraction` | Fraction of state values above / below zero | All five games |
| `state_sparsity` | Fraction of state values whose absolute value is at most `1e-6` | All five games |
| `state_median` | Median normalized state value | All five games |
| `state_iqr` | 75th minus 25th percentile | All five games |
| `state_skewness` | Standardized third central moment | All five games |
| `state_entropy` | Normalized Shannon entropy of absolute state-value mass | All five games |

The v4 distribution primitives are permutation-invariant summaries. The v5 ordered-vector descriptors are deliberately order-sensitive and can capture coarse positional patterns in a flattened observation, but they are **not** guaranteed to have the same meaning across games. They are not board-topology-aware and are not temporal features. In particular, a one-dimensional Nim observation makes all four v5 ordered-vector descriptors degenerate or uninformative. Treat them as experimental inputs, not as established transferable abstractions.

| Primitive | Meaning | Availability |
|---|---|---|
| `state_position_moment` | Signed first moment over normalized feature indices | All five games; order-sensitive |
| `state_position_spread` | Absolute-mass-weighted spread over normalized feature indices | All five games; order-sensitive |
| `state_adjacent_difference` | Mean absolute difference between adjacent state features | All five games; order-sensitive |
| `state_sign_change_fraction` | Fraction of adjacent feature pairs with opposite nonzero signs | All five games; order-sensitive |

The GP fitness uses signed correlations and penalizes train/validation disagreement. Candidate screening now requires at least one state-dependent primitive and a minimum source robust correlation (`--min-source-robust-corr`, default `0.02`), preventing action-only expressions from being presented as discovered state concepts. Downstream admission is stricter: by default the mean and median paired improvement must be positive and at least 60% of paired seeds must improve (`--downstream-min-win-fraction`, default `0.6`). This is still a screening heuristic, not a statistical significance test; use more seeds and confidence intervals for research claims.

The GP grammar also includes protected unary functions `abs`, `neg`, `square`,
`sqrt` (of the absolute value), `log1p` (of the absolute value), and `tanh`,
alongside its binary arithmetic, `min`, and `max` operations. Expressions are
bounded and evaluated consistently by the Python discovery/policy code and the
C++ learner. Availability and definitions are recorded in `gp_transfer.py`;
the C++ environment reports support for environment-specific primitives.

`--downstream-budget` is the fixed environment-step budget for each fresh target learner. Candidates are tested one at a time against the action-identity baseline with the same five feature slots, paired seeds, population size, and budget. The default paired seeds are `11,22,33,44,55,66,77,88,99,111`; override with `--downstream-seeds`. A concept is admitted only when its mean paired held-out return improvement exceeds `--downstream-min-improvement` (default `0`), its median paired improvement is positive, and its win fraction meets `--downstream-min-win-fraction` (default `0.6`). The executable defaults to `src/brain`; build it first with `cd src && make`.

Without `--downstream-budget`, the run stops at a shortlist: `downstream_shortlist.txt` and `concept_candidates.json` contain pending candidates, while `concept_bank.json`, `transferred_concepts.txt`, and `compatible_target_concepts.txt` contain only downstream-admitted concepts and therefore remain empty. This avoids treating compatibility or source-game fitness as evidence of target utility. Downstream evaluation uses target rewards by design; the earlier compatibility screen uses target features only.

## Novelty monitor

`gp_transfer.py` records structural and behavioral novelty each generation. After 8 consecutive generations without a globally new behavior, it now restarts the population: the Pareto elite archive is retained, fresh random trees are injected, and archived trees seed stronger mutation for the configured number of generations. A restart is successful only if it yields a globally new behavior during its observation window. By default, evolution stops after 3 consecutive unsuccessful restarts.

Configure restart behavior with `--stagnation-window`, `--max-unsuccessful-restarts`, `--restart-random-fraction`, and `--restart-mutation-generations`. `novelty_history.json` records injected random/mutated individuals and each restart outcome. Run the focused GP tests with:

```bash
python -m unittest -v test_novelty_monitor test_gp_selection
```

## Prompt-to-playable game prototype

`llm_game_platform/` adds a prompt-to-game service for bounded JSON grid games.
One LLM request produces a game contract, which a trusted Python engine
validates. The existing C++ brain then trains a fresh policy on a generic grid
adapter using compatible concepts from `results/concept_bank.json`. In parallel,
a second LLM request produces declarative presentation settings; the bundled
trusted browser renderer uses those settings to display the game and run the
trained policy. Game-contract generation runs asynchronously with visible browser
status and errors; both LLM outputs have bounded validation-and-repair loops.
Generated source code is not executed.

Configure an OpenAI-compatible chat endpoint, then start the server from the
package root:

```powershell
$env:LLM_API_KEY = "your-api-key"
$env:LLM_MODEL = "your-model"
# Optional for compatible providers:
$env:LLM_BASE_URL = "https://api.openai.com/v1"
# Optional; default is 45 seconds per model request:
$env:LLM_REQUEST_TIMEOUT_SECONDS = "120"
python -m pip install -r requirements.txt
cd src
make
cd ..
python -m llm_game_platform.server
```

Open `http://127.0.0.1:8000`, describe a grid game, play with the arrow
buttons, or wait for training and choose **Let brain play**. The build requires
the repository's C++17 compiler and make tool. Training can be configured with
`GAME_TRAINING_BUDGET` (default `10000` environment steps),
`GAME_TRAINING_POPULATION` (default `24`), `GAME_TRAINING_TIMEOUT` (default
`900` seconds), `GAME_CONCEPT_BANK` (default `results/concept_bank.json`), and
`GAME_BRAIN_BIN` (to select a non-default brain executable). Slow model endpoints
can use `LLM_REQUEST_TIMEOUT_SECONDS` (default `45` seconds per request).

Run focused tests with:

```bash
python -m unittest -v test_llm_game_platform
```

## Integrated platform: generated environments + research brain

This tree merges the prompt-to-game platform (web UI, LLM environment generation, sandboxed
runtime) with the research brain (NEAT+PPO learner, GP concept generator, transfer experiments).
Start with `docs/ARCHITECTURE.md`; HTTP endpoints are in `docs/API.md`; what was replaced, retained,
modified and added, the compatibility issues found, and exactly which tests were run (and which were not)
are in `docs/INTEGRATION_REPORT.md`.

Quick start (from this directory):

```bash
python3 -m pip install -r requirements.txt
cd src && make brain runner && cd ..          # builds the learner (now with `ext` mode) and the experiment runner
export LLM_API_KEY=...  LLM_MODEL=...         # any OpenAI-compatible endpoint (see llm_game_platform/llm.py)
python3 -m llm_game_platform.server           # http://127.0.0.1:8000  (default tab: "Any game (generated Python)")
                                              # http://127.0.0.1:8000/research  -> cached research results viewer
python3 generate_env.py --acceptance ttt --train   # CLI: generate + validate + train (needs a live LLM)
python3 play_cli.py generated_env_fixtures/ttt.py  # terminal play of a hand-written fixture, no LLM needed
```

Tests: `python3 -m unittest test_generated_envs test_llm_game_platform test_concept_bank` (platform; slow),
`python3 -m unittest test_novelty_monitor test_gp_selection test_action_primitives` (research),
`cd src && make regression_test env_smoke_test && ./regression_test && ./env_smoke_test` (C++).
Note: `make test-envs` references `lamarckian_hidden_test.cpp`, which is absent from the V8 archive this was
built from, so that aggregate target fails exactly as it did in V8; use the explicit targets above.

Concepts reaching generated games come from `results/concept_bank.json` through V8's
`screen_concepts` filter. Per the bank's own note none of these concepts was admitted as a demonstrated
transfer win, so training a generated game with them is not evidence of transfer.

## Product architecture and implementation status

The intended end-to-end workflow is:

1. A user describes a game or task.
2. The game-spec LLM produces a contract; validation failures are sent back for
   bounded repair.
3. After the contract validates, fresh brain training and a second LLM request
   for declarative runtime settings run concurrently.
4. The generic grid adapter supplies observations, legal actions, rewards, and
   terminal conditions to the existing NEAT+PPO training path with compatible
   concept-bank features.
5. A validated exported policy is connected to the same trusted game engine and
   browser renderer used for human play.
6. Invalid runtime settings are sent back to the runtime LLM with validation
   errors for bounded repair; no generated source code is executed.

Status in this repository:

| Component | Status |
|---|---|
| NEAT/PPO learning core and five experimental environments | Implemented; see build and learner instructions above |
| GP concept discovery, compatibility screening, novelty monitoring, and downstream admission | Implemented as an experimental pipeline |
| Prompt-to-game flow | Implemented for validated bounded JSON grid-game contracts |
| Generated-game adapter to the C++ `Env` contract | Implemented for the grid-game contract |
| Brain training and policy export for generated grid games | Implemented; starts a fresh learner and uses compatible generic concept expressions |
| Runtime generation | Implemented as declarative UI settings generated independently of the game contract |
| Runtime self-healing | Implemented as bounded validation-error feedback for both generated JSON outputs; no LLM source code is executed |
| Browser play by the trained brain | Implemented using the exported policy and the same trusted Python game environment |
| Learning-curve view, concept browser, cached demos (`/research`) | Implemented; reads cached result files only |
| LLM-generated Python environments (any game/simulation, multi-agent, opponents, generic 2D renderer) | Implemented; see `GENERATED_ENVIRONMENTS.md` (sandboxed, validated; C++ learner unchanged) |
| Per-generation fitness/species chart in the browser | Implemented for generated environments; no full population visualizer |

This integration is intentionally limited to the validated grid contract. The
runtime LLM can choose presentation data but cannot alter rules, emit executable
code, or bypass the trusted engine. Concept entries that require primitives
unavailable to the generic grid adapter (such as CartPole-only features) are
excluded. (Arbitrary generated game logic is handled by the separate generated-
environment pipeline below; there is still no full live population visualizer.)

## Report

See `EXPERIMENT_REPORT.md` for the previous experiment's methodology, results, caveats, and the updated next-experiment protocol. `results/` and `results_reverse/` contain historical concept banks, novelty histories, seed logs, and summary CSVs.


## Concept Generator v6 — transition-aware primitives

The shared contract is now `generic_state_action_v6`. It retains all v5 primitives and adds four transition descriptors, available only in CartPole and LineWorld:

| Primitive | Definition | Availability |
|---|---|---|
| `transition_delta_mean` | Mean normalized observation change from the previous environment step | CartPole, LineWorld |
| `transition_delta_abs_mean` | Mean absolute normalized observation change | CartPole, LineWorld |
| `transition_delta_max_abs` | Maximum absolute normalized observation change | CartPole, LineWorld |
| `transition_state_alignment` | Cosine alignment between current observation and its previous-step change | CartPole, LineWorld |

The C++ `ConceptEnv` stores the most recent normalized state transition, exposes these values to concept expressions, and clears transition history on reset. Python dataset generation uses the last observed transition for CartPole samples and a one-step predecessor transition for LineWorld samples. Other games mark these primitive columns unavailable, and the GP generator excludes them through environment-specific primitive metadata.

These are first-order transition features, not full trajectory models: no multi-step history, acceleration, learned dynamics, graph topology, or hierarchical concept composition is claimed in v6. The Python LLM grid policy accepts v6 banks but deliberately excludes transition-only primitives from grid-game loading.


## Concept Generator v7 — topology-aware board concepts

v7 retains all v6 primitives and adds six typed board descriptors available only in TicTacToe and Connect Four:

| Primitive | Definition |
|---|---|
| `board_own_open_lines` | Fraction of winning lines containing own marks and no opponent marks |
| `board_opponent_open_lines` | Fraction of winning lines containing opponent marks and no own marks |
| `board_own_immediate_wins` | Fraction of distinct empty cells completing an own winning line in one move |
| `board_opponent_immediate_wins` | Fraction of distinct empty cells completing an opponent winning line in one move |
| `board_center_control` | Normalized own-minus-opponent center occupancy |
| `board_occupancy` | Fraction of occupied cells |

TicTacToe uses three-cell winning lines; Connect Four uses four-cell winning lines. These are derived from the typed board layouts rather than generic adjacent-vector arithmetic. The Python discovery code and C++ concept runtime share the same definitions. Grid-game policy loading accepts v7 banks but filters these game-specific primitives from generic generated-grid policies.
