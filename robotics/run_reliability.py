#!/usr/bin/env python3
"""Multi-seed reliability harness for the point-mass reach task (native C++ env, no Python relay).

For every (config, seed) it trains with src/brain, then evaluates the champion greedily on 100 held-out episodes
for the training env and a perturbation grid (obs noise, action delay, mass, force noise), and stores one JSON.

  python3 robotics/run_reliability.py run --configs ppo_full,neat_full --seeds 1-5 --steps 1500000
  python3 robotics/run_reliability.py summary            # median / worst success across seeds per perturbation
"""
import argparse, glob, json, os, re, statistics, subprocess, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAIN = os.path.join(ROOT, "src", "brain")
OUT = os.path.join(ROOT, "robotics", "results")
PPO = "--preset mlp --hidden 16 --iters 8 --train-fitness 1 --ent 0"
NEAT = "--hidden 16 --iters 16 --train-fitness 1 --persist-adam 1 --carry 0.5 --ent 0"
DR = "rnoise=0.15,rdelay=3,rmassx=2,rfnoise=0.3"
CONFIGS = {   # name: (pop, extra args, env args, stack)
    "ppo_full":    (1, PPO, "", 1),                 # velocity observed, plain PPO
    "neat_full":   (8, NEAT, "", 1),                # velocity observed, evolution on
    "ppo_pos":     (1, PPO, "posonly=1", 1),        # velocity hidden, memoryless policy
    "ppo_pos_s3":  (1, PPO, "posonly=1", 3),        # velocity hidden, 3-frame stack + previous action
    "neat_pos_s3": (8, NEAT, "posonly=1", 3),
    # domain randomization: per-episode noise<=0.15, delay<=3, mass x[1/2,2], force noise<=0.3 (eval points are fixed, not randomized)
    "ppo_full_dr":    (1, PPO, DR, 1),
    "neat_full_dr":   (8, NEAT, DR, 1),
    # filtered velocity estimates (f1 = EMA deltas, f2 = leaky action integral, f3 = both), velocity hidden, with randomization
    **{f"{a}_pos_f{m}_dr": (p, e, "posonly=1," + DR, f"f{m}") for m in (1, 2, 3) for a, p, e in (("ppo", 1, PPO), ("neat", 8, NEAT))},
    # same, with the action-integral input scaled to ~[-2,2] (--filt-k 0.02) so it does not saturate the network
    **{f"{a}_pos_f{m}k_dr": (p, e + " --filt-k 0.02", "posonly=1," + DR, f"f{m}") for m in (2, 3) for a, p, e in (("ppo", 1, PPO), ("neat", 8, NEAT))},
    "ppo_pos_f3_dr_lr": (1, PPO + " --lr 5e-4", "posonly=1," + DR, "f3"),
    "ppo_pos_s3_dr":  (1, PPO, "posonly=1," + DR, 3),
    "neat_pos_s3_dr": (8, NEAT, "posonly=1," + DR, 3),
}
ROB = re.compile(r"ROBUST (\w+) noise ([\d.]+) delay (\d+) mass ([\d.]+) fnoise ([\d.]+) success ([\d.]+) mean_return (-?[\d.]+)")


def seeds_of(s):
    a, _, b = s.partition("-"); return list(range(int(a), int(b or a) + 1))


def run_one(cfg, seed, steps, eval_eps, force=False):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{cfg}_s{seed}_{steps}.json")
    if os.path.exists(path) and not force: return json.load(open(path))
    pop, extra, envargs, stack = CONFIGS[cfg]
    cmd = [BRAIN, "pointmass", os.path.join(OUT, "none.txt"), str(steps), str(seed), str(pop)] + extra.split() + \
          (["--filt", stack[1:]] if isinstance(stack, str) else ["--stack", str(stack)]) + ["--robust", str(eval_eps)] + (["--env-args", envargs] if envargs else [])
    open(os.path.join(OUT, "none.txt"), "a").close()
    t0 = time.time()
    with tempfile.TemporaryDirectory() as wd:   # brain writes a CSV into cwd
        p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True)
    rob = [dict(kind=m[0], noise=float(m[1]), delay=int(m[2]), mass=float(m[3]), fnoise=float(m[4]), success=float(m[5]), ret=float(m[6])) for m in ROB.findall(p.stdout)]
    res = dict(config=cfg, seed=seed, steps=steps, sec=round(time.time() - t0, 1), rc=p.returncode, robust=rob, err=p.stderr[-300:])
    json.dump(res, open(path, "w"), indent=1); return res


def label(r):
    k = r["kind"]
    return {"train": "train env", "noise": f"obs noise {r['noise']}", "delay": f"delay {r['delay']}", "mass": f"mass x{r['mass']}", "fnoise": f"force noise {r['fnoise']}"}[k]


def summary(steps):
    rows, rets = {}, {}
    for f in sorted(glob.glob(os.path.join(OUT, f"*_{steps}.json"))):
        d = json.load(open(f))
        for r in d["robust"]:
            rows.setdefault(d["config"], {}).setdefault(label(r), []).append(r["success"])
            rets.setdefault(d["config"], {}).setdefault(label(r), []).append(r["ret"])
    if not rows: print("no results"); return
    cols = list(next(iter(rows.values())).keys())
    print("success rate: median [worst] over seeds   (n seeds in brackets after config)\n")
    print("| perturbation | " + " | ".join(f"{c} (n={len(next(iter(v.values())))})" for c, v in rows.items()) + " |")
    print("|---|" + "---|" * len(rows))
    for col in cols:
        print(f"| {col} | " + " | ".join(f"{statistics.median(v[col]):.2f} [{min(v[col]):.2f}]" if col in v else "-" for v in rows.values()) + " |")
    print("\nmean return (higher is better; PD controller = -1.3, do-nothing = -10): median [worst] over seeds\n")
    print("| perturbation | " + " | ".join(rets) + " |"); print("|---|" + "---|" * len(rets))
    for col in cols:
        print(f"| {col} | " + " | ".join(f"{statistics.median(v[col]):.2f} [{min(v[col]):.2f}]" if col in v else "-" for v in rets.values()) + " |")
    print("\nseeds solved on train env (return > -3.0):", {c: f"{sum(x > -3.0 for x in v['train env'])}/{len(v['train env'])}" for c, v in rets.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--configs", default=",".join(CONFIGS)); ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--steps", type=int, default=1_500_000); ap.add_argument("--eval_eps", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "summary": summary(a.steps); sys.exit()
    for c in a.configs.split(","):
        for s in seeds_of(a.seeds):
            r = run_one(c, s, a.steps, a.eval_eps, a.force)
            tr = [x for x in r["robust"] if x["kind"] == "train"]
            print(f"{c} s{s}: {r['sec']}s rc={r['rc']} train-env success {tr[0]['success'] if tr else 'n/a'}", flush=True)
