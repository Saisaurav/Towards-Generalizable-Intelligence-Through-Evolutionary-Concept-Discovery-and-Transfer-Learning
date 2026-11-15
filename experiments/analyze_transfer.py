#!/usr/bin/env python3
import json, glob, os, numpy as np, itertools, random
D = 'results_v2/transfer'; SEEDS = [11,22,33,44,55,66,77,88,99,111]
CONDS = ['scratch','source','random','sxa','target_trained']
def load(g, c):
    return [json.load(open(f'{D}/{g}_{c}_s{s}.json')) for s in SEEDS]
def metric(r, g):   # primary held-out metric (greedy, 1000 episodes)
    h = r['heldout']['vs_random_greedy']; return h['mean_length'] if g == 'cartpole' else h['mean_return']
def auc(r, g):      # area under held-out (greedy, 100-episode) checkpoint curve vs PPO steps, normalised by budget
    pts = [(c['train_steps'], c['heldout_len'] if g == 'cartpole' else c['heldout_greedy']) for c in r['curve'] if 'heldout_greedy' in c]
    if len(pts) < 2: return float('nan')
    x = np.array([p[0] for p in pts], float); y = np.array([p[1] for p in pts], float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))
def steps_to(r, g, thr):
    for c in r['curve']:
        if 'heldout_greedy' in c and (c['heldout_len'] if g == 'cartpole' else c['heldout_greedy']) >= thr: return c['train_steps']
    return float('nan')
def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d); m = [rng.choice(d, len(d)).mean() for _ in range(n)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def perm_p(a, b, n=20000, seed=0):  # two-sided permutation test on difference of means (unpaired; features differ per condition)
    rng = np.random.default_rng(seed); a = np.asarray(a); b = np.asarray(b); obs = abs(a.mean() - b.mean()); z = np.concatenate([a, b]); c = 0
    for _ in range(n):
        rng.shuffle(z); c += abs(z[:len(a)].mean() - z[len(a):].mean()) >= obs - 1e-12
    return (c + 1) / (n + 1)
out = {}
for g, unit, thr in [('cartpole', 'mean episode length (greedy, 1000 held-out eps)', 450), ('nim', 'mean return vs random (greedy, 1000 held-out eps)', 0.5)]:
    out[g] = {'metric': unit, 'conditions': {}}
    rows = {c: load(g, c) for c in CONDS}
    for c in CONDS:
        m = np.array([metric(r, g) for r in rows[c]]); a = np.array([auc(r, g) for r in rows[c]])
        stoch = np.array([r['heldout']['vs_random_stochastic']['mean_length' if g == 'cartpole' else 'mean_return'] for r in rows[c]])
        ev = np.array([steps_to(r, g, thr) for r in rows[c]])
        out[g]['conditions'][c] = dict(final_mean=m.mean(), final_median=float(np.median(m)), final_sd=m.std(ddof=1), worst=m.min(), best=m.max(), per_seed=m.tolist(),
            stochastic_mean=stoch.mean(), auc_mean=float(np.nanmean(a)), auc_sd=float(np.nanstd(a, ddof=1)), auc_per_seed=a.tolist(),
            reached_threshold=int(np.sum(~np.isnan(ev))), median_steps_to_threshold=(float(np.nanmedian(ev)) if np.any(~np.isnan(ev)) else None),
            runtime_sec_mean=float(np.mean([r['runtime_sec'] for r in rows[c]])), arch_hidden_mean=float(np.mean([r['architecture']['hidden'] for r in rows[c]])),
            n_conns_mean=float(np.mean([r['architecture']['connections'] for r in rows[c]])), failures=sum(r['status'] != 'ok' for r in rows[c]))
    out[g]['comparisons'] = {}
    for c in CONDS[1:]:
        a = np.array(out[g]['conditions'][c]['per_seed']); b = np.array(out[g]['conditions']['scratch']['per_seed']); d = a - b
        out[g]['comparisons'][f'{c}_vs_scratch'] = dict(mean_diff=float(d.mean()), ci95=boot_ci(d), perm_p=perm_p(a, b),
            auc_diff=float(np.nanmean(out[g]['conditions'][c]['auc_per_seed']) - np.nanmean(out[g]['conditions']['scratch']['auc_per_seed'])))
    for x, y in [('source', 'random'), ('source', 'sxa'), ('source', 'target_trained')]:
        a = np.array(out[g]['conditions'][x]['per_seed']); b = np.array(out[g]['conditions'][y]['per_seed'])
        out[g]['comparisons'][f'{x}_vs_{y}'] = dict(mean_diff=float((a - b).mean()), ci95=boot_ci(a - b), perm_p=perm_p(a, b))
json.dump(out, open('results_v2/transfer_analysis.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
for g in out:
    print('\n==', g, '-', out[g]['metric'])
    for c, v in out[g]['conditions'].items():
        print(f"{c:15s} final {v['final_mean']:8.3f} ±{v['final_sd']:.3f} med {v['final_median']:.3f} worst {v['worst']:.3f} best {v['best']:.3f} | stoch {v['stochastic_mean']:.3f} | AUC {v['auc_mean']:.3f} | reached thr {v['reached_threshold']}/10 | hid {v['arch_hidden_mean']:.1f} conns {v['n_conns_mean']:.0f} | {v['runtime_sec_mean']:.1f}s")
    for k, v in out[g]['comparisons'].items(): print(f"  {k:24s} diff {v['mean_diff']:+.3f}  95%CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]  perm p={v['perm_p']:.4f}" + (f"  AUCdiff {v['auc_diff']:+.3f}" if 'auc_diff' in v else ''))
