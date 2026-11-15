#!/usr/bin/env python3
"""Feature-set files for the v9 transfer evaluation (one expression per line, K=4 each).
  source_<pair>.txt         : v9 GP shortlist discovered on the SOURCE game (no target-performance filtering)
  target_trained_<tgt>.txt  : v9 GP shortlist discovered on the TARGET game (control)
  random_<tgt>_s<seed>.txt  : 4 random trees over the SAME v7 vocabulary available in the target, same admission filter
                              (depth<=5, target output std >= 0.01, no near-duplicates)
plus feature_stats.json (target-game output std of every feature)."""
import sys, json, random, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gp_transfer as G
OUT = 'results_v9/features'; os.makedirs(OUT, exist_ok=True)
SEEDS = [11,22,33,44,55,66,77,88,99,111]; K = 4
def parse(e):
    def p(i):
        j = i
        while j < len(e) and (e[j].isalnum() or e[j] == '_'): j += 1
        name = e[i:j]
        if j < len(e) and e[j] == '(':
            args = []; j += 1
            while True:
                a, j = p(j); args.append(a)
                if e[j] == ',': j += 1; continue
                assert e[j] == ')'; j += 1; break
            return (name, *args), j
        return name, j
    return p(0)[0]
def read(path): return [l.strip() for l in open(path) if l.strip() and not l.startswith('#')][:K]
PAIRS = [  # (pair name, target game, source shortlist dir, target-trained dir)
 ('ttt_to_cartpole_rollout', 'cartpole', 'tt_cartpole_rollout'), ('ttt_to_cartpole_oracle', 'cartpole', 'tt_cartpole_rollout'),
 ('ttt_to_c4_rollout', 'connect_four', 'tt_connect_four_rollout'), ('ttt_to_c4_oracle', 'connect_four', 'tt_connect_four_rollout'),
 ('c4_to_nim_rollout', 'nim', 'tt_nim_rollout')]
stats = {}; made_random = set()
for pair, tgt, tt in PAIRS:
    Xt = G.make_dataset(tgt, 2026 + 912, nstates=35, rollouts=3)[0]
    prims = G.available_primitives(tgt)
    def std_of(e): v = np.asarray(G.ev(parse(e), Xt), float); return float(np.nanstd(v)) if np.all(np.isfinite(v)) else float('nan')
    src = read(f'results_v9/gp/{pair}/downstream_shortlist.txt'); ttl = read(f'results_v9/gp/{tt}/downstream_shortlist.txt')
    open(f'{OUT}/source_{pair}.txt', 'w').write('# v9 GP concepts discovered on the source game\n' + ''.join(e + '\n' for e in src))
    open(f'{OUT}/target_trained_{tgt}.txt', 'w').write('# v9 GP concepts discovered on the target game itself\n' + ''.join(e + '\n' for e in ttl))
    stats[pair] = {'source': {e: std_of(e) for e in src}, 'target_trained': {e: std_of(e) for e in ttl}}
    if tgt in made_random: continue
    made_random.add(tgt); stats['random_' + tgt] = {}
    for s in SEEDS:
        rng = random.Random(888000 + s); chosen = []; vecs = []; tries = 0
        while len(chosen) < K and tries < 20000:
            tries += 1; t = G.randtree(rng, primitives=prims)
            if G.depth(t) > 5: continue
            v = np.asarray(G.ev(t, Xt), float)
            if not np.all(np.isfinite(v)) or np.std(v) < .01: continue
            if any(abs(G.corr(v, w)) >= .995 for w in vecs): continue
            chosen.append(G.render(t)); vecs.append(v)
        assert len(chosen) == K, (tgt, s, len(chosen))
        open(f'{OUT}/random_{tgt}_s{s}.txt', 'w').write('# random v7-vocabulary features (control)\n' + ''.join(e + '\n' for e in chosen))
        stats['random_' + tgt][str(s)] = {e: float(np.std(v)) for e, v in zip(chosen, vecs)}
json.dump(stats, open(f'{OUT}/feature_stats.json', 'w'), indent=1)
for p, s in stats.items():
    if 'source' in s: print(p, 'source std', [round(x, 3) for x in s['source'].values()], '| target-trained std', [round(x, 3) for x in s['target_trained'].values()])
