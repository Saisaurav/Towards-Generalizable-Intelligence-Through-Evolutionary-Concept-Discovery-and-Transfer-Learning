# Standalone learner results (generated; every configuration that was run, none omitted)

Metric: held-out greedy policy, 1000 episodes, vs the training-time random opponent (CartPole: episode length). 10 seeds each, 150,000 PPO rollout steps unless the tag says otherwise. Seeds 11,22,33,44,55,66,77,88,99,111. "stochastic" = sampling from the policy.

Exact-optimum references (expectimax, not used in training): TTT vs random: return 0.960, win-or-draw 0.996; Nim vs random: return 0.991. Nim constant policy "always take 3": +0.248.


## cartpole

| config tag | ok | failed | pop | PPO iters | mean | median | sd | worst | best | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| aid_actoh | 10 | 0 | 6 | 2 | 349.127 | 297.173 | 115.202 | 216.962 | 495.548 | 500-rate greedy 0.36 (min 0.00); stochastic length 110 |
| base_aid | 10 | 0 | 24 | 2 | 107.293 | 53.802 | 142.640 | 9.331 | 500.000 | 500-rate greedy 0.10 (min 0.00); stochastic length 24 |
| confirm_std | 10 | 0 | 3 | 4 | 499.570 | 500.000 | 1.060 | 496.462 | 500.000 | 500-rate greedy 0.99 (min 0.97); stochastic length 166 |
| cp_actoh_h32 | 10 | 0 | 6 | 2 | 353.162 | 329.723 | 121.649 | 196.358 | 500.000 | 500-rate greedy 0.37 (min 0.00); stochastic length 186 |
| cp_actoh_h32_pop24 | 10 | 0 | 24 | 2 | 130.848 | 89.722 | 125.700 | 46.407 | 493.713 | 500-rate greedy 0.08 (min 0.00); stochastic length 30 |
| cp_actoh_pop1 | 10 | 0 | 1 | 2 | 313.286 | 499.100 | 229.856 | 9.347 | 500.000 | 500-rate greedy 0.60 (min 0.00); stochastic length 166 |
| cp_actoh_pop2 | 10 | 0 | 2 | 2 | 399.573 | 498.515 | 123.558 | 207.044 | 500.000 | 500-rate greedy 0.62 (min 0.00); stochastic length 237 |
| cp_actoh_pop24 | 10 | 0 | 24 | 2 | 86.025 | 67.838 | 67.684 | 9.363 | 204.304 | 500-rate greedy 0.00 (min 0.00); stochastic length 26 |
| cp_actoh_pop2_it4 | 10 | 0 | 2 | 4 | 427.230 | 500.000 | 150.435 | 51.044 | 500.000 | 500-rate greedy 0.80 (min 0.00); stochastic length 290 |
| cp_actoh_pop3 | 10 | 0 | 3 | 2 | 407.671 | 497.536 | 131.296 | 155.335 | 500.000 | 500-rate greedy 0.61 (min 0.00); stochastic length 201 |
| cp_actoh_pop3_it1 | 10 | 0 | 3 | 1 | 332.592 | 297.120 | 117.373 | 204.432 | 500.000 | 500-rate greedy 0.30 (min 0.00); stochastic length 194 |
| cp_actoh_pop3_it4 | 10 | 0 | 3 | 4 | 438.576 | 498.592 | 92.274 | 268.290 | 500.000 | 500-rate greedy 0.71 (min 0.00); stochastic length 219 |
| cp_actoh_pop3_it8 | 10 | 0 | 3 | 8 | 426.536 | 500.000 | 102.805 | 197.422 | 500.000 | 500-rate greedy 0.61 (min 0.00); stochastic length 237 |
| cp_actoh_pop3_lr1e3 | 10 | 0 | 3 | 2 | 341.237 | 335.159 | 107.498 | 194.148 | 499.455 | 500-rate greedy 0.29 (min 0.00); stochastic length 104 |
| cp_actoh_pop6_it4 | 10 | 0 | 6 | 4 | 395.514 | 453.488 | 112.066 | 179.162 | 499.905 | 500-rate greedy 0.53 (min 0.00); stochastic length 110 |
| cp_pop3_it4_greedyfit | 10 | 0 | 3 | 4 | 479.419 | 500.000 | 59.851 | 299.941 | 500.000 | 500-rate greedy 0.89 (min 0.00); stochastic length 207 |
| fixed_it32 | 10 | 0 | 24 | 32 | 274.130 | 225.177 | 103.875 | 154.285 | 500.000 | 500-rate greedy 0.15 (min 0.00); stochastic length 79 |
| fixed_it8 | 10 | 0 | 24 | 8 | 377.730 | 409.367 | 125.744 | 215.624 | 500.000 | 500-rate greedy 0.51 (min 0.00); stochastic length 125 |
| fixed_pop6 | 10 | 0 | 6 | 2 | 298.097 | 301.433 | 110.384 | 141.442 | 495.719 | 500-rate greedy 0.16 (min 0.00); stochastic length 146 |
| onehot_it32 | 10 | 0 | 24 | 32 | 196.337 | 182.457 | 57.612 | 139.709 | 340.358 | 500-rate greedy 0.03 (min 0.00); stochastic length 36 |
| onehot_it8 | 10 | 0 | 24 | 8 | 145.878 | 150.547 | 47.203 | 57.608 | 215.671 | 500-rate greedy 0.00 (min 0.00); stochastic length 34 |
| onehot_pop48 | 10 | 0 | 48 | 2 | 27.787 | 9.789 | 26.673 | 9.317 | 92.979 | 500-rate greedy 0.00 (min 0.00); stochastic length 24 |
| onehot_pop6 | 10 | 0 | 6 | 2 | 366.633 | 366.970 | 110.830 | 233.188 | 488.233 | 500-rate greedy 0.43 (min 0.00); stochastic length 134 |
| repr_fixed | 10 | 0 | 24 | 2 | 476.170 | 500.000 | 65.386 | 280.755 | 500.000 | 500-rate greedy 0.89 (min 0.00); stochastic length 219 |
| repr_onehot | 10 | 0 | 24 | 2 | 140.321 | 124.244 | 135.680 | 9.407 | 495.535 | 500-rate greedy 0.09 (min 0.00); stochastic length 25 |

## connect_four

| config tag | ok | failed | pop | PPO iters | mean | median | sd | worst | best | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| aid_actoh | 10 | 0 | 6 | 2 | 0.777 | 0.780 | 0.028 | 0.712 | 0.806 | win-or-draw 0.89 (min 0.86); stochastic +0.708; vs strategic opp -0.869 |
| aid_actoh_greedyfit | 10 | 0 | 6 | 2 | 0.790 | 0.796 | 0.027 | 0.730 | 0.826 | win-or-draw 0.89 (min 0.86); stochastic +0.708; vs strategic opp -0.865 |
| base_aid | 10 | 0 | 24 | 2 | 0.609 | 0.616 | 0.025 | 0.544 | 0.630 | win-or-draw 0.80 (min 0.77); stochastic +0.013; vs strategic opp -0.998 |
| confirm_std | 10 | 0 | 3 | 4 | 0.771 | 0.788 | 0.035 | 0.714 | 0.812 | win-or-draw 0.89 (min 0.86); stochastic +0.728; vs strategic opp -0.861 |
| onehot_it32 | 10 | 0 | 24 | 32 | 0.757 | 0.765 | 0.041 | 0.692 | 0.824 | win-or-draw 0.88 (min 0.85); stochastic +0.608; vs strategic opp -0.931 |
| onehot_it8 | 10 | 0 | 24 | 8 | 0.778 | 0.783 | 0.029 | 0.702 | 0.808 | win-or-draw 0.89 (min 0.85); stochastic +0.565; vs strategic opp -0.900 |
| onehot_pop48 | 1 | 0 | 48 | 2 | 0.786 | 0.786 | 0.000 | 0.786 | 0.786 | win-or-draw 0.89 (min 0.89); stochastic +0.145; vs strategic opp -0.878 |
| onehot_pop6 | 10 | 0 | 6 | 2 | 0.770 | 0.780 | 0.040 | 0.684 | 0.824 | win-or-draw 0.89 (min 0.84); stochastic +0.700; vs strategic opp -0.889 |
| repr_fixed | 10 | 0 | 24 | 2 | 0.742 | 0.756 | 0.061 | 0.640 | 0.820 | win-or-draw 0.87 (min 0.82); stochastic +0.444; vs strategic opp -0.899 |
| repr_onehot | 10 | 0 | 24 | 2 | 0.777 | 0.774 | 0.023 | 0.740 | 0.812 | win-or-draw 0.89 (min 0.87); stochastic +0.496; vs strategic opp -0.892 |
| std_p3_it4 | 10 | 0 | 3 | 4 | 0.793 | 0.805 | 0.033 | 0.704 | 0.826 | win-or-draw 0.90 (min 0.85); stochastic +0.748; vs strategic opp -0.865 |

## nim

| config tag | ok | failed | pop | PPO iters | mean | median | sd | worst | best | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| aid_actoh_stateoh | 10 | 0 | 6 | 2 | 0.799 | 0.787 | 0.049 | 0.760 | 0.938 | win-or-draw 0.90 (min 0.88); stochastic +0.747; vs strategic opp -0.815 |
| aid_actoh_stateoh_greedyfit | 10 | 0 | 6 | 2 | 0.807 | 0.793 | 0.046 | 0.768 | 0.938 | win-or-draw 0.90 (min 0.88); stochastic +0.731; vs strategic opp -0.799 |
| base_aid | 10 | 0 | 24 | 2 | 0.251 | 0.252 | 0.037 | 0.162 | 0.316 | win-or-draw 0.63 (min 0.58); stochastic +0.217; vs strategic opp -0.925 |
| confirm_std | 10 | 0 | 3 | 4 | 0.852 | 0.817 | 0.064 | 0.786 | 0.952 | win-or-draw 0.93 (min 0.89); stochastic +0.817; vs strategic opp -0.800 |
| nim_onehotstate | 10 | 0 | 6 | 2 | 0.778 | 0.783 | 0.028 | 0.728 | 0.818 | win-or-draw 0.89 (min 0.86); stochastic +0.716; vs strategic opp -0.817 |
| nim_p6_budget_600000 | 10 | 0 | 6 | 2 | 0.267 | 0.242 | 0.117 | 0.162 | 0.602 | win-or-draw 0.63 (min 0.58); stochastic +0.268; vs strategic opp -0.934 |
| nim_p6_ent_0.001 | 10 | 0 | 6 | 2 | 0.251 | 0.252 | 0.037 | 0.162 | 0.316 | win-or-draw 0.63 (min 0.58); stochastic +0.231; vs strategic opp -0.925 |
| nim_p6_hidden_32 | 10 | 0 | 6 | 2 | 0.251 | 0.252 | 0.037 | 0.162 | 0.316 | win-or-draw 0.63 (min 0.58); stochastic +0.256; vs strategic opp -0.925 |
| nim_p6_lr_0.005 | 10 | 0 | 6 | 2 | 0.239 | 0.249 | 0.046 | 0.106 | 0.282 | win-or-draw 0.62 (min 0.55); stochastic +0.256; vs strategic opp -0.930 |
| nim_p6_ppo_iters_8 | 10 | 0 | 6 | 8 | 0.267 | 0.253 | 0.061 | 0.162 | 0.410 | win-or-draw 0.63 (min 0.58); stochastic +0.236; vs strategic opp -0.928 |
| onehot_it32 | 10 | 0 | 24 | 32 | 0.217 | 0.253 | 0.088 | 0.056 | 0.316 | win-or-draw 0.61 (min 0.53); stochastic +0.172; vs strategic opp -0.931 |
| onehot_it8 | 10 | 0 | 24 | 8 | 0.135 | 0.121 | 0.093 | 0.006 | 0.282 | win-or-draw 0.57 (min 0.50); stochastic +0.170; vs strategic opp -0.939 |
| onehot_pop48 | 10 | 0 | 48 | 2 | 0.134 | 0.119 | 0.123 | -0.106 | 0.316 | win-or-draw 0.57 (min 0.45); stochastic +0.102; vs strategic opp -0.930 |
| onehot_pop6 | 10 | 0 | 6 | 2 | 0.256 | 0.252 | 0.027 | 0.210 | 0.316 | win-or-draw 0.63 (min 0.60); stochastic +0.256; vs strategic opp -0.932 |
| repr_fixed | 10 | 0 | 24 | 2 | 0.213 | 0.238 | 0.068 | 0.076 | 0.316 | win-or-draw 0.61 (min 0.54); stochastic +0.190; vs strategic opp -0.927 |
| repr_onehot | 10 | 0 | 24 | 2 | 0.170 | 0.155 | 0.091 | 0.034 | 0.316 | win-or-draw 0.59 (min 0.52); stochastic +0.176; vs strategic opp -0.928 |
| std_p3_it4 | 10 | 0 | 3 | 4 | 0.845 | 0.819 | 0.075 | 0.762 | 0.968 | win-or-draw 0.92 (min 0.88); stochastic +0.819; vs strategic opp -0.771 |

## ttt

| config tag | ok | failed | pop | PPO iters | mean | median | sd | worst | best | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| aid_actoh | 10 | 0 | 6 | 2 | 0.627 | 0.634 | 0.054 | 0.542 | 0.704 | win-or-draw 0.84 (min 0.80); stochastic +0.569; vs strategic opp -0.552 |
| aid_actoh_greedyfit | 10 | 0 | 6 | 2 | 0.654 | 0.657 | 0.050 | 0.592 | 0.722 | win-or-draw 0.85 (min 0.82); stochastic +0.560; vs strategic opp -0.646 |
| base_aid | 10 | 0 | 24 | 2 | 0.259 | 0.262 | 0.032 | 0.177 | 0.305 | win-or-draw 0.65 (min 0.61); stochastic +0.010; vs strategic opp -0.882 |
| base_aid_tactical | 10 | 0 | 24 | 2 | 0.895 | 0.894 | 0.013 | 0.881 | 0.928 | win-or-draw 0.99 (min 0.98); stochastic +0.819; vs strategic opp +0.149 |
| confirm_std | 10 | 0 | 3 | 4 | 0.740 | 0.738 | 0.022 | 0.712 | 0.773 | win-or-draw 0.90 (min 0.88); stochastic +0.683; vs strategic opp -0.429 |
| fixed_it32 | 10 | 0 | 24 | 32 | 0.588 | 0.593 | 0.041 | 0.511 | 0.673 | win-or-draw 0.82 (min 0.80); stochastic +0.459; vs strategic opp -0.650 |
| fixed_it8 | 10 | 0 | 24 | 8 | 0.589 | 0.595 | 0.036 | 0.517 | 0.644 | win-or-draw 0.82 (min 0.79); stochastic +0.419; vs strategic opp -0.570 |
| fixed_pop6 | 10 | 0 | 6 | 2 | 0.724 | 0.733 | 0.037 | 0.663 | 0.764 | win-or-draw 0.88 (min 0.84); stochastic +0.621; vs strategic opp -0.557 |
| onehot_it32 | 10 | 0 | 24 | 32 | 0.532 | 0.540 | 0.036 | 0.476 | 0.594 | win-or-draw 0.80 (min 0.76); stochastic +0.440; vs strategic opp -0.599 |
| onehot_it8 | 10 | 0 | 24 | 8 | 0.516 | 0.509 | 0.030 | 0.460 | 0.566 | win-or-draw 0.79 (min 0.75); stochastic +0.380; vs strategic opp -0.670 |
| onehot_pop48 | 10 | 0 | 48 | 2 | 0.406 | 0.377 | 0.109 | 0.260 | 0.548 | win-or-draw 0.75 (min 0.67); stochastic +0.145; vs strategic opp -0.702 |
| onehot_pop6 | 10 | 0 | 6 | 2 | 0.644 | 0.646 | 0.046 | 0.549 | 0.735 | win-or-draw 0.84 (min 0.79); stochastic +0.574; vs strategic opp -0.636 |
| repr_fixed | 10 | 0 | 24 | 2 | 0.530 | 0.510 | 0.064 | 0.471 | 0.695 | win-or-draw 0.79 (min 0.76); stochastic +0.314; vs strategic opp -0.641 |
| repr_onehot | 10 | 0 | 24 | 2 | 0.519 | 0.530 | 0.056 | 0.365 | 0.563 | win-or-draw 0.80 (min 0.70); stochastic +0.325; vs strategic opp -0.696 |
| ttt_p3_it4 | 10 | 0 | 3 | 4 | 0.719 | 0.723 | 0.018 | 0.687 | 0.746 | win-or-draw 0.88 (min 0.86); stochastic +0.675; vs strategic opp -0.591 |
| ttt_p3_it4_b450k | 10 | 0 | 3 | 4 | 0.850 | 0.852 | 0.024 | 0.794 | 0.876 | win-or-draw 0.94 (min 0.92); stochastic +0.803; vs strategic opp -0.200 |
| ttt_p6_ent003 | 10 | 0 | 6 | 2 | 0.641 | 0.627 | 0.048 | 0.562 | 0.727 | win-or-draw 0.84 (min 0.81); stochastic +0.581; vs strategic opp -0.603 |
| ttt_p6_h32 | 10 | 0 | 6 | 2 | 0.676 | 0.691 | 0.048 | 0.543 | 0.716 | win-or-draw 0.86 (min 0.81); stochastic +0.607; vs strategic opp -0.594 |
