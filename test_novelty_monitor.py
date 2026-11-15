import numpy as np
from gp_transfer import NoveltyMonitor

# New structures with identical behavior must eventually trigger stagnation.
probes=np.random.default_rng(7).normal(size=(6,128))
monitor=NoveltyMonitor(probes,window=8)
for _ in range(9):
    stats=monitor.observe(['action_norm']*20)
assert monitor.stagnant >= 8
assert stats['behaviorally_new'] == 0
print('PASS: behavioral novelty stagnation detected after 8 consecutive generations.')

# A behaviorally different expression must reset the stagnation counter.
monitor.observe(['state_mean']*20)
assert monitor.stagnant == 0
print('PASS: new behavior resets stagnation counter.')
