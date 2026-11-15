#!/usr/bin/env python3
"""Build llm_game_platform/web/data/*.json from the raw result JSON. Everything shown in the research page comes from these files;
nothing is synthesised. Every group is tagged data_kind='cached_experiment' (pre-computed, offline) vs live runs (none are produced here)."""
import json, glob, os, re, statistics as st, collections
OUT = 'llm_game_platform/web/data'; os.makedirs(OUT, exist_ok=True)
def mean(x): x = [v for v in x if v is not None]; return sum(x) / len(x) if x else None
def curve_of(runs):
    n = min(len(r['curve']) for r in runs)
    cur = []
    for i in range(n):
        pts = [r['curve'][i] for r in runs]
        e = {'gen': pts[0]['gen'], 'train_steps': mean([p['train_steps'] for p in pts]), 'best': mean([p['best'] for p in pts]), 'pop_mean': mean([p['mean'] for p in pts]),
             'diversity': mean([p['diversity'] for p in pts]), 'species': mean([p['species'] for p in pts]), 'champ_nodes': mean([p['champ_nodes'] for p in pts]),
             'champ_conns': mean([p['champ_conns'] for p in pts]), 'avg_nodes': mean([p['avg_nodes'] for p in pts]), 'avg_conns': mean([p['avg_conns'] for p in pts])}
        h = [p.get('heldout_greedy') for p in pts]
        if all(v is not None for v in h): e['heldout_greedy'] = mean(h); e['heldout_len'] = mean([p.get('heldout_len') for p in pts])
        cur.append(e)
    return cur
def smooth(vals, k=3):
    return [mean(vals[max(0, i - k + 1):i + 1]) for i in range(len(vals))]
def group(kind, files, label):
    runs = [json.load(open(f)) for f in sorted(files)]; ok = [r for r in runs if r.get('status') == 'ok']
    if not ok: return None
    g0 = ok[0]['config'].get('game')
    cur = curve_of(ok)
    for key in ('best', 'pop_mean'): 
        for c, s in zip(cur, smooth([c[key] for c in cur])): c[key + '_smoothed'] = s
    metric = 'mean_length' if g0 == 'cartpole' else 'mean_return'
    seed_final = {r['config']['seed']: r['heldout']['vs_random_greedy'][metric] for r in ok}
    d = ok[0]
    return {'group': label, 'data_kind': kind, 'game': g0, 'n_runs': len(runs), 'n_failed': len(runs) - len(ok), 'config': d['config'], 'resolved': d['resolved'],
            'repo_version': d['repo_version'], 'metric': metric + ' (greedy, 1000 held-out episodes)', 'seed_final': seed_final,
            'final_mean': mean(list(seed_final.values())), 'final_sd': st.pstdev(seed_final.values()) if len(seed_final) > 1 else 0,
            'stochastic_mean': mean([r['heldout']['vs_random_stochastic'][metric] for r in ok]), 'random_baseline': ok[0]['random_baseline']['vs_random'][metric],
            'architecture_mean': {k: mean([r['architecture'][k] for r in ok]) for k in ('hidden', 'nodes', 'connections', 'enabled_connections')},
            'runtime_sec_mean': mean([r['runtime_sec'] for r in ok]), 'curve_mean_over_seeds': cur,
            'detail_run': {'seed': ok[0]['config']['seed'], 'curve': [{k: c[k] for k in ('gen', 'train_steps', 'best', 'mean', 'species', 'diversity', 'champ_nodes', 'champ_conns', 'avg_nodes', 'avg_conns', 'fitness')} for c in ok[0]['curve']]}}
groups = []
def collect(dirname, kind, section=None):
    by = collections.defaultdict(list)
    for f in glob.glob(f'{dirname}/*_s*.json'):
        m = re.match(r'(.*)_s(\d+)\.json', os.path.basename(f)); by[m.group(1)].append(f)
    for label, files in sorted(by.items()):
        g = group(kind, files, label)
        if g: groups.append(dict(g, section=section or os.path.basename(dirname)))
collect('results_v2/standalone', 'cached_experiment'); collect('results_v2/transfer', 'cached_experiment'); collect('results_v2/demo', 'cached_experiment'); collect('results_v9/transfer', 'cached_experiment', 'v9_transfer'); collect('results_v2/learner_fix', 'cached_experiment', 'learner_fix'); collect('results_v8/transfer', 'cached_experiment', 'v8_transfer'); collect('results_v9/positive_control', 'cached_experiment', 'positive_control')
json.dump({'note': 'Pre-computed experiment results (cached). Not produced by a live training session.', 'groups': groups}, open(f'{OUT}/experiments.json', 'w'))
bank = json.load(open('results/concept_bank.json'))
concepts = []
for c in bank['concepts']:
    c = dict(c); c['curve_scope'] = 'pair-level (all 4 concepts of the pair together)'; concepts.append(c)
json.dump({'note': 'Concept bank v%s. Curves are pair-level; per-concept ablations were not run.' % bank['version'], 'version': bank['version'], 'created_utc': bank['created_utc'], 'concepts': concepts}, open(f'{OUT}/concepts.json', 'w'))
print(len(groups), 'groups,', len(concepts), 'concepts ->', OUT, sum(os.path.getsize(f) for f in glob.glob(OUT + '/*.json')) // 1024, 'KiB')
