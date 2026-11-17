# Frontend integration notes

- Replaced `llm_game_platform/web/index.html` with the supplied Frameforge HTML visual design.
- Existing environment generation, editable source validation, training logic, saved-environment loading, runtime actions, brain controls, concept-bank endpoint, and research viewer remain intact. The trainer received an additive optional genome recorder and the backend received read-only genome retrieval endpoints.
- The Game Builder includes the new layout plus the existing live environment workflow and source editor.
- The Training Visualizer uses live generation history for the chart and loads actual per-generation population snapshots for network inspection.
- The former placeholder Gazebo page is now a Robot Lab with a dummy custom-file upload area and two working built-in task cards: TurtleBot navigation and a 2-link arm. The upload does not parse custom robot files. The browser preview is a JavaScript surrogate, not a live Gazebo renderer.
- The Concept Bank is populated from the existing `/api/concepts` endpoint and keeps the existing search, compatibility status, active-concept status, concept graph, and metadata rendering.


## Per-generation neural genome recording (added)

- The NEAT population is snapshotted after fitness evaluation and speciation, immediately before reproduction replaces the current population.
- Snapshots include every individual, fitness/previous fitness, species, node genes, connection genes (including disabled state and optimizer moments), continuous-action log-standard-deviation parameters, and generation number.
- Recording uses a bounded background writer. It performs no extra environment rollouts, fitness evaluations, PPO updates, or mutation/crossover operations. There is a small in-memory copy and serialization/I/O cost; the writer only blocks if its short bounded queue falls behind.
- Generated snapshots are stored under `generated_envs/<environment-id>/policies/genomes/`. Self-play runs and concept-refresh restarts use separate subdirectories.
- API: `GET /api/envs/<id>/genomes` lists available generations; `GET /api/envs/<id>/genomes/<run>/<generation>` returns the actual population snapshot.
- The Training Visualizer loads these snapshots and displays every individual in a selected generation, with a topology thumbnail and inspectable fitness, species, nodes, and connection weights.
- The source changes require rebuilding the bundled C++ brain executable once (`cd src && make brain`). The previous prebuilt Windows executable is retained but is detected as stale rather than being run with an unsupported command-line option.


## Robotics workspace integration (added)

- Merged the separate robotics workspace's `src/robot_envs.h`, domain-randomized PointMass / `StackEnv` / `FilterEnv` research updates, and robotics-aware policy export into the main C++ trainer. Existing game training options and genome recording remain in place. The existing `genome.h` acyclic-crossover fix is retained.
- Added `llm_game_platform/robotics_service.py` and `/api/robotics/{status,train}` plus `/api/robotics/policy/<kind>` endpoints. Training is performed by the same C++ NEAT + PPO brain, with a dedicated build target so the old prebuilt Windows brain is not mistaken for the robotics-capable binary.
- Added `web/robotics_sim.js`, which evaluates the exported NEAT topology and mirrors the two C++ surrogate environments in the browser.
- Included trained seed-3 policy checkpoints and one final-population genome snapshot per environment under `robotics_sim/`. Full per-generation snapshots are recorded for new frontend training runs under `robotics_sim/runs/`.
- Merged the expanded PointMass reliability harness and result set (`robotics/run_reliability.py`, `robotics/results/`, `ROBOTICS_RESULTS.md`). Added environment specifications, ROS 2 adapter examples and tests. The ROS 2 nodes are adapters for compatible Gazebo models/controllers; they do not spawn a world, parse URDF/SDF uploads, or certify sim-to-real transfer.
- The included C++ surrogate held-out results (300 episodes) are TurtleBot 86.3% success and arm 89.0% success. These are single-seed surrogate results, not Gazebo or hardware validation.
