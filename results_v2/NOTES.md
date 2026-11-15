# Running notes (Stage 3/4)
- Nim: learner plateaus at greedy return ~+0.25 across hidden=32, lr, ent, ppo_iters, 4x budget. Fixed policy "always take 3" scores +0.248 vs random opponent
  (always-1: -0.228, always-2: +0.083), so the plateau is a constant policy, NOT learned game structure. Optimal vs random: +0.991.
- sweep3 (pop 48) stopped manually; connect_four onehot_pop48 and fixed_pop48 incomplete.
- Evaluation: greedy and stochastic both recorded; stochastic is far lower because policies stay high-entropy.
