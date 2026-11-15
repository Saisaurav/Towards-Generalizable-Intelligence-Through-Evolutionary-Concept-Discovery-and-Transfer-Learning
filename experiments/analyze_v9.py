#!/usr/bin/env python3
"""Statistics for the v9-concept transfer matrix (results_v9/transfer). Unpaired permutation tests + bootstrap CIs on the difference of means
(feature sets differ per condition, so seed-pairing carries little information); held-out greedy score on 1000 episodes; AUC over 100-episode
checkpoints. Also reports Holm-adjusted p-values across the source-vs-baseline comparisons."""
import json, numpy as np, os
D = 'results_v9/transfer'; SEEDS = [11,22,33,44,55,66,77,88,99,111]
GAMES = {'cartpole': dict(sources=['source_rollout', 'source_oracle'], thr=450, unit='mean episode length'),
         'connect_four': dict(sources=['source_rollout', 'source_oracle'], thr=0.7, unit='mean return vs random'),
         'nim': dict(sources=['source_rollout'], thr=0.8, unit='mean return vs random')}
BASE = ['scratch', 'random', 'sxa', 'target_trained']
def load(g, c): return [json.load(open(f'{D}/{g}_{c}_s{s}.json')) for s in SEEDS]
def met(r, g): h = r['heldout']['vs_random_greedy']; return h['mean_length'] if g == 'cartpole' else h['mean_return']
def ckpt(c, g): return c['heldout_len'] if g == 'cartpole' else c['heldout_greedy']
def auc(r, g):
    p = [(c['train_steps'], ckpt(c, g)) for c in r['curve'] if 'heldout_greedy' in c]
    x = np.array([a for a, _ in p], float); y = np.array([b for _, b in p], float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0])) if len(p) > 1 else float('nan')
def steps_to(r, g, thr):
    for c in r['curve']:
        if 'heldout_greedy' in c and ckpt(c, g) >= thr: return c['train_steps']
    return float('nan')
def ci(d, n=10000):
    rng = np.random.default_rng(0); d = np.asarray(d); m = [rng.choice(d, len(d)).mean() for _ in range(n)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def perm(a, b, n=20000):
    rng = np.random.default_rng(1); a = np.asarray(a); b = np.asarray(b); o = abs(a.mean() - b.mean()); z = np.concatenate([a, b]); c = 0
    for _ in range(n): rng.shuffle(z); c += abs(z[:len(a)].mean() - z[len(a):].mean()) >= o - 1e-12
    return (c + 1) / (n + 1)
def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(order): run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1, run)
    return adj
out = {}; family = []
for g, cfg in GAMES.items():
    conds = BASE + cfg['sources']; rows = {c: load(g, c) for c in conds}; out[g] = {'unit': cfg['unit'], 'conditions': {}, 'comparisons': {}}
    for c in conds:
        m = np.array([met(r, g) for r in rows[c]]); a = np.array([auc(r, g) for r in rows[c]]); ev = np.array([steps_to(r, g, cfg['thr']) for r in rows[c]])
        out[g]['conditions'][c] = dict(mean=float(m.mean()), median=float(np.median(m)), sd=float(m.std(ddof=1)), worst=float(m.min()), best=float(m.max()), per_seed=m.tolist(),
            auc=float(np.nanmean(a)), reached=int(np.sum(~np.isnan(ev))), median_steps_to_thr=(float(np.nanmedian(ev)) if np.any(~np.isnan(ev)) else None),
            stochastic=float(np.mean([r['heldout']['vs_random_stochastic']['mean_length' if g == 'cartpole' else 'mean_return'] for r in rows[c]])),
            runtime=float(np.mean([r['runtime_sec'] for r in rows[c]])), failures=sum(r['status'] != 'ok' for r in rows[c]))
    def cmp(x, y, key, tagfam=False):
        a = np.array(out[g]['conditions'][x]['per_seed']); b = np.array(out[g]['conditions'][y]['per_seed'])
        e = dict(mean_diff=float((a - b).mean()), ci95=ci(a - b), perm_p=perm(a, b)); out[g]['comparisons'][key] = e
        if tagfam: family.append((g, key))
    for s in cfg['sources']:
        for base in ['scratch', 'random', 'sxa']: cmp(s, base, f'{s}_vs_{base}', tagfam=True)
        cmp(s, 'target_trained', f'{s}_vs_target_trained')
    if len(cfg['sources']) == 2: cmp('source_oracle', 'source_rollout', 'oracle_vs_rollout')
    for b in ['random', 'sxa', 'target_trained']: cmp(b, 'scratch', f'{b}_vs_scratch')
ps = [out[g]['comparisons'][k]['perm_p'] for g, k in family]
for (g, k), p in zip(family, holm(np.array(ps))): out[g]['comparisons'][k]['holm_p'] = float(p)
json.dump(out, open('results_v9/transfer_analysis.json', 'w'), indent=1)
for g in out:
    print('\n==', g, '-', out[g]['unit'])
    for c, v in out[g]['conditions'].items(): print(f"{c:15s} {v['mean']:8.3f} ±{v['sd']:.3f} med {v['median']:.3f} worst {v['worst']:.3f} best {v['best']:.3f} | AUC {v['auc']:.3f} | reached thr {v['reached']}/10 | stoch {v['stochastic']:.3f} | {v['runtime']:.1f}s | fail {v['failures']}")
    for k, v in out[g]['comparisons'].items(): print(f"  {k:30s} diff {v['mean_diff']:+.3f} CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['perm_p']:.4f}" + (f" holm={v['holm_p']:.3f}" if 'holm_p' in v else ''))
