#!/usr/bin/env python3
"""Run a batch of runner.cpp experiments (sequential) and summarise. JSON per run is kept, failures included.
usage: batch.py OUTDIR TAG "key=val key=val ..." [--seeds 11,22,...]"""
import json, subprocess, sys, os, statistics as st
SEEDS = [11,22,33,44,55,66,77,88,99,111]
RUNNER = os.path.join(os.path.dirname(__file__), '..', 'src', 'runner')

def run(outdir, tag, args, seeds=SEEDS):
    os.makedirs(outdir, exist_ok=True); res = []
    game = [a.split('=')[1] for a in args.split() if a.startswith('game=')][0]
    for s in seeds:
        path = f"{outdir}/{game}_{tag}_s{s}.json"
        if not os.path.exists(path):
            r = subprocess.run([RUNNER, *args.replace("{seed}", str(s)).split(), f"seed={s}", f"tag={tag}", f"out={path}"], capture_output=True, text=True)
            if r.returncode != 0 and not os.path.exists(path):
                json.dump({"status": "failed: " + r.stderr[:300], "config": args, "seed": s}, open(path, "w"))
        res.append(json.load(open(path)))
    return res

def wd(o): return (o['win'] + o['draw_or_zero']) / o['episodes']
def summarize(res, game, key):
    ok = [r for r in res if r.get('status') == 'ok']
    out = {"n": len(res), "failed": len(res) - len(ok)}
    for k in ('vs_random_greedy', 'vs_random_stochastic', 'vs_strategic_greedy'):
        vals = [r['heldout'][k]['mean_return'] for r in ok if k in r['heldout']]
        if vals: out[k] = dict(mean=st.mean(vals), median=st.median(vals), sd=st.pstdev(vals), worst=min(vals), best=max(vals))
    if game == 'cartpole':
        out['len_greedy'] = st.mean(r['heldout']['vs_random_greedy']['mean_length'] for r in ok)
        out['full500_greedy'] = st.mean(r['heldout']['vs_random_greedy']['full_length_episodes'] / 1000 for r in ok)
        out['len_stoch'] = st.mean(r['heldout']['vs_random_stochastic']['mean_length'] for r in ok)
        out['full500_stoch'] = st.mean(r['heldout']['vs_random_stochastic']['full_length_episodes'] / 1000 for r in ok)
    else:
        out['wd_greedy'] = [wd(r['heldout']['vs_random_greedy']) for r in ok]
        out['wd_stoch'] = [wd(r['heldout']['vs_random_stochastic']) for r in ok]
    out['random_mean'] = st.mean(r['random_baseline']['vs_random']['mean_return'] for r in ok)
    out['runtime'] = st.mean(r['runtime_sec'] for r in ok)
    return out

def fmt(game, tag, s):
    g = s.get('vs_random_greedy'); t = s.get('vs_random_stochastic')
    line = f"{game:13s} {tag:28s} "
    if game == 'cartpole':
        line += f"len greedy {s['len_greedy']:.1f} (500-rate {s['full500_greedy']:.2f}) stoch {s['len_stoch']:.1f} (500-rate {s['full500_stoch']:.2f}) | random {s['random_mean']:.1f}"
    else:
        line += f"greedy ret {g['mean']:+.3f}±{g['sd']:.3f} [worst {g['worst']:+.3f} best {g['best']:+.3f}] W/D {st.mean(s['wd_greedy']):.2f} (min {min(s['wd_greedy']):.2f}) | stoch {t['mean']:+.3f} | random {s['random_mean']:+.3f}"
        if 'vs_strategic_greedy' in s: line += f" | vs strategic greedy {s['vs_strategic_greedy']['mean']:+.3f}"
    return line + f" | fail {s['failed']}/{s['n']}"

if __name__ == '__main__':
    outdir, tag, args = sys.argv[1:4]
    seeds = SEEDS
    if '--seeds' in sys.argv: seeds = [int(x) for x in sys.argv[sys.argv.index('--seeds') + 1].split(',')]
    game = [a.split('=')[1] for a in args.split() if a.startswith('game=')][0]
    res = run(outdir, tag, args, seeds)
    print(fmt(game, tag, summarize(res, game, tag)))
