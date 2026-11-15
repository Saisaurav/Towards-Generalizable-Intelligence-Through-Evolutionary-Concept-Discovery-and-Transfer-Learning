#!/usr/bin/env python3
"""Build the feature-set files used by the transfer matrix (one expression per line).
  source_<pair>.txt         : GP shortlist discovered on the SOURCE game (no target-performance filtering)
  target_trained_<tgt>.txt  : GP shortlist discovered on the TARGET game itself (control)
  random_<tgt>_s<seed>.txt  : K random algebraic expressions, same primitives/depth limit, same admission filter
                              (target-compatible primitives, depth<=5, target output std >= 0.01, not near-duplicate)
plus feature_stats.json with the target-game output std of every feature."""
import sys, json, random, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gp_transfer as G
OUT = 'results_v2/features'; os.makedirs(OUT, exist_ok=True)
SEEDS = [11,22,33,44,55,66,77,88,99,111]; K = 4
def parse(e):  # expression string -> tree
    def p(i):
        j = i
        while j < len(e) and (e[j].isalnum() or e[j] == '_'): j += 1
        name = e[i:j]
        if j < len(e) and e[j] == '(':
            a, j = p(j + 1); assert e[j] == ','; b, j = p(j + 1); assert e[j] == ')'; return (name, a, b), j + 1
        return name, j
    return p(0)[0]
def read(path): return [l.strip() for l in open(path) if l.strip() and not l.startswith('#')][:K]
stats = {}
for pair, tgt, src_file, tt_dir in [('ttt_to_cartpole', 'cartpole', 'results_v2/gp/ttt_to_cartpole/downstream_shortlist.txt', 'results_v2/gp/target_trained_cartpole'),
                                    ('c4_to_nim', 'nim', 'results_v2/gp/c4_to_nim/downstream_shortlist.txt', 'results_v2/gp/target_trained_nim')]:
    Xt, _ = G.make_dataset(tgt, 2026 + 912, nstates=35, rollouts=3)   # target STATES only (no returns used), same as discovery screening
    prims = G.available_primitives(tgt)
    def std_of(e): return float(np.std(G.ev(parse(e), Xt)))
    src = read(src_file); tt = read(tt_dir + '/downstream_shortlist.txt')
    open(f'{OUT}/source_{pair}.txt', 'w').write('# GP concepts discovered on the source game\n' + ''.join(e + '\n' for e in src))
    open(f'{OUT}/target_trained_{tgt}.txt', 'w').write('# GP concepts discovered on the target game itself\n' + ''.join(e + '\n' for e in tt))
    stats[pair] = {'source': {e: std_of(e) for e in src}, 'target_trained': {e: std_of(e) for e in tt}, 'random': {}}
    for s in SEEDS:
        rng = random.Random(777000 + s); chosen = []; vecs = []; tries = 0
        while len(chosen) < K and tries < 5000:
            tries += 1; t = G.randtree(rng, primitives=prims)
            if G.depth(t) > 5: continue
            v = G.ev(t, Xt)
            if np.std(v) < .01: continue
            if any(abs(G.corr(v, w)) >= .995 for w in vecs): continue
            chosen.append(G.render(t)); vecs.append(v)
        open(f'{OUT}/random_{tgt}_s{s}.txt', 'w').write('# random algebraic features (control)\n' + ''.join(e + '\n' for e in chosen))
        stats[pair]['random'][str(s)] = {e: float(np.std(v)) for e, v in zip(chosen, vecs)}
json.dump(stats, open(f'{OUT}/feature_stats.json', 'w'), indent=1)
print({p: {k: (len(v) if k != 'random' else len(v)) for k, v in s.items()} for p, s in stats.items()})
for p, s in stats.items(): print(p, 'source std', [round(x, 3) for x in s['source'].values()], 'target-trained std', [round(x, 3) for x in s['target_trained'].values()])
