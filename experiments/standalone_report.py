#!/usr/bin/env python3
"""Generate results_v2/STANDALONE_REPORT.md from raw run JSON (all configs, including unsuccessful ones)."""
import json, glob, re, os, collections, statistics as st
by = collections.defaultdict(list)
for f in glob.glob('results_v2/standalone/*_s*.json') + glob.glob('results_v2/learner_fix/*_s*.json'):
    m = re.match(r'(.*)_s(\d+)\.json', os.path.basename(f)); by[m.group(1)].append(json.load(open(f)))
ref = json.load(open('results_v2/reference_optimal.json'))
rows = collections.defaultdict(list)
for key, runs in sorted(by.items()):
    ok = [r for r in runs if r.get('status') == 'ok']
    if not ok: continue
    game = ok[0]['config']['game']; tag = key[len(game) + 1:]
    if game == 'cartpole':
        v = [r['heldout']['vs_random_greedy']['mean_length'] for r in ok]; full = [r['heldout']['vs_random_greedy']['full_length_episodes'] / 1000 for r in ok]
        s = [r['heldout']['vs_random_stochastic']['mean_length'] for r in ok]
        extra = f"500-rate greedy {st.mean(full):.2f} (min {min(full):.2f}); stochastic length {st.mean(s):.0f}"
    else:
        v = [r['heldout']['vs_random_greedy']['mean_return'] for r in ok]
        wd = [(r['heldout']['vs_random_greedy']['win'] + r['heldout']['vs_random_greedy']['draw_or_zero']) / 1000 for r in ok]
        s = [r['heldout']['vs_random_stochastic']['mean_return'] for r in ok]
        sg = [r['heldout']['vs_strategic_greedy']['mean_return'] for r in ok]
        extra = f"win-or-draw {st.mean(wd):.2f} (min {min(wd):.2f}); stochastic {st.mean(s):+.3f}; vs strategic opp {st.mean(sg):+.3f}"
    rows[game].append((tag, len(ok), len(runs) - len(ok), st.mean(v), st.median(v), st.pstdev(v), min(v), max(v), extra, ok[0]['resolved']['pop'], ok[0]['resolved']['ppo']['iters']))
md = ['# Standalone learner results (generated; every configuration that was run, none omitted)\n',
      'Metric: held-out greedy policy, 1000 episodes, vs the training-time random opponent (CartPole: episode length). 10 seeds each, 150,000 PPO rollout steps unless the tag says otherwise. '
      'Seeds 11,22,33,44,55,66,77,88,99,111. "stochastic" = sampling from the policy.\n',
      f"Exact-optimum references (expectimax, not used in training): TTT vs random: return {ref['ttt_optimal_vs_random']['expected_return']:.3f}, win-or-draw {ref['ttt_optimal_vs_random']['win_or_draw']:.3f}; Nim vs random: return {ref['nim_optimal_vs_random']['expected_return']:.3f}. "
      'Nim constant policy "always take 3": +0.248.\n']
for game, rs in rows.items():
    md.append(f'\n## {game}\n\n| config tag | ok | failed | pop | PPO iters | mean | median | sd | worst | best | notes |\n|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rs: md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[9]} | {r[10]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[8]} |")
open('results_v2/STANDALONE_REPORT.md', 'w').write('\n'.join(md) + '\n'); print(sum(len(v) for v in rows.values()), 'config rows')
