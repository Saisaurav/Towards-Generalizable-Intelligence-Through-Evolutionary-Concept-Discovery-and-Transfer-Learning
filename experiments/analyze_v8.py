#!/usr/bin/env python3
"""TTT -> Connect Four with the v8 vocabulary. Held-out greedy return (1,000 episodes), vs-strategic-opponent return, AUC; unpaired permutation tests,
bootstrap CIs, Holm correction over source-vs-{scratch,random,sxa,primitives_only}. v7-vocabulary source rows are read from results_v9 for reference."""
import json, numpy as np
SEEDS = [11,22,33,44,55,66,77,88,99,111]
def load(d, g, c): return [json.load(open(f'{d}/{g}_{c}_s{s}.json')) for s in SEEDS]
R = {c: load('results_v8/transfer', 'connect_four', c) for c in ['scratch','sxa','random','target_trained','primitives_only','source_rollout','source_oracle']}
R['v7_source_rollout'] = load('results_v9/transfer', 'connect_four', 'source_rollout'); R['v7_source_oracle'] = load('results_v9/transfer', 'connect_four', 'source_oracle')
ret = lambda r: r['heldout']['vs_random_greedy']['mean_return']; strat = lambda r: r['heldout']['vs_strategic_greedy']['mean_return']
def auc(r):
    p = [(c['train_steps'], c['heldout_greedy']) for c in r['curve'] if 'heldout_greedy' in c]; x = np.array([a for a, _ in p], float); y = np.array([b for _, b in p], float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))
def ci(d):
    rng = np.random.default_rng(0); d = np.asarray(d); m = [rng.choice(d, len(d)).mean() for _ in range(10000)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def perm(a, b, n=20000):
    rng = np.random.default_rng(1); a = np.asarray(a); b = np.asarray(b); o = abs(a.mean() - b.mean()); z = np.concatenate([a, b]); c = 0
    for _ in range(n): rng.shuffle(z); c += abs(z[:len(a)].mean() - z[len(a):].mean()) >= o - 1e-12
    return (c + 1) / (n + 1)
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for k, i in enumerate(o): run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1, run)
    return adj
out = {'conditions': {}, 'comparisons': {}}
for c, rs in R.items():
    m = np.array([ret(r) for r in rs]); s = np.array([strat(r) for r in rs]); a = np.array([auc(r) for r in rs])
    out['conditions'][c] = dict(mean=float(m.mean()), sd=float(m.std(ddof=1)), median=float(np.median(m)), worst=float(m.min()), best=float(m.max()), per_seed=m.tolist(),
        vs_strategic=float(s.mean()), auc=float(a.mean()), runtime=float(np.mean([r['runtime_sec'] for r in rs])), failures=sum(r['status'] != 'ok' for r in rs))
fam = []
def cmp(x, y, fam_member=False):
    a = np.array(out['conditions'][x]['per_seed']); b = np.array(out['conditions'][y]['per_seed']); k = f'{x}_vs_{y}'
    out['comparisons'][k] = dict(mean_diff=float((a - b).mean()), ci95=ci(a - b), perm_p=perm(a, b))
    if fam_member: fam.append(k)
for s in ['source_rollout', 'source_oracle']:
    for b in ['scratch', 'random', 'sxa', 'primitives_only']: cmp(s, b, True)
    cmp(s, 'target_trained'); cmp(s, 'v7_' + s)
for b in ['random', 'primitives_only', 'target_trained']: cmp(b, 'scratch')
cmp('source_oracle', 'source_rollout')
for k, p in zip(fam, holm(np.array([out['comparisons'][k]['perm_p'] for k in fam]))): out['comparisons'][k]['holm_p'] = float(p)
json.dump(out, open('results_v8/transfer_analysis.json', 'w'), indent=1)
for c, v in out['conditions'].items(): print(f"{c:18s} {v['mean']:+.3f} ±{v['sd']:.3f} med {v['median']:+.3f} worst {v['worst']:+.3f} best {v['best']:+.3f} | vs strategic {v['vs_strategic']:+.3f} | AUC {v['auc']:.3f} | {v['runtime']:.1f}s | fail {v['failures']}")
for k, v in out['comparisons'].items(): print(f"  {k:36s} diff {v['mean_diff']:+.3f} CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['perm_p']:.4f}" + (f" holm={v['holm_p']:.4f}" if 'holm_p' in v else ''))
