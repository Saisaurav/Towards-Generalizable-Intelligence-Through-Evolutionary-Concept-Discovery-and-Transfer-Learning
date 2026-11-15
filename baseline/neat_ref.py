#!/usr/bin/env python3
"""Run the repo's actual NEAT+PPO learner (src/brain ext, --head fixed) on a fixture with an in-process relay.

Same protocol as llm_game_platform/env_trainer.py, minus the sandbox processes (faster, same semantics).
  python3 baseline/neat_ref.py --game pong --opp random --budget 1000000 --seed 1
"""
import argparse, json, os, re, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "baseline"))
from ppo_baseline import load_runner, success_bar   # noqa: E402

QUALITY = re.compile(r"QUALITY gen\s+(\d+)\s+steps\s+(\d+)\s+win_rate\s+([\d.]+)\s+wins\s+(\d+)\s+losses\s+(\d+)\s+draws\s+(\d+)\s+mean_return\s+(-?[\d.]+)")
PROG = re.compile(r"gen\s+(\d+)\s+steps\s+(\d+)\s+best\s+(-?[\d.]+)\s+mean\s+(-?[\d.]+).*?train\s+(\d+)\s+eval\s+(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="pong"); ap.add_argument("--opp", default="random")
    ap.add_argument("--budget", type=int, default=1_000_000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pop", type=int, default=24); ap.add_argument("--head", default="fixed")
    ap.add_argument("--eval_eps", type=int, default=100); ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--max_seconds", type=float, default=240)
    ap.add_argument("--concepts", default=""); ap.add_argument("--extra", default=""); ap.add_argument("--tag", default="neat")
    a = ap.parse_args()
    bar, mode, cal = success_bar(a.game, a.opp)
    wd = tempfile.mkdtemp()
    cf = os.path.join(wd, "concepts.txt"); open(cf, "w").write(a.concepts)
    cmd = [os.path.join(ROOT, "src", "brain"), "ext", cf, str(a.budget), str(a.seed), str(a.pop), os.path.join(wd, "m.json"),
           "--target-win-rate", "1.0", "--eval-episodes", str(a.eval_eps), "--eval-interval", str(a.eval_every),
           "--max-seconds", str(a.max_seconds)] + (["--head", "fixed"] if a.head == "fixed" else []) + a.extra.split()
    p = subprocess.Popen(cmd, cwd=wd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 20)
    envs, nxt, t0, curve, gens = {}, 0, time.time(), [], {}
    n_env_steps = 0
    for raw in p.stdout:
        line = raw.decode().rstrip()
        if not line.startswith("@"):
            m = PROG.search(line)
            if m: gens[int(m.group(1))] = dict(steps=int(m.group(2)), best=float(m.group(3)), mean=float(m.group(4)), train=int(m.group(5)), eval=int(m.group(6)), sec=round(time.time() - t0, 1))
            q = QUALITY.search(line)
            if q:
                g, st, wr, w, l, d, mr = q.groups()
                curve.append(dict(gen=int(g), steps=int(st), win_rate=float(wr), wins=int(w), losses=int(l), draws=int(d), mean_return=float(mr), sec=round(time.time() - t0, 1)))
                tr = gens.get(int(g), {}); print(f"[{a.tag} {a.game}/{a.opp} s{a.seed}] train {tr.get('train','?')} gen {int(g):3d} steps {int(st):8d} win {wr} (W{w}/L{l}/D{d}) ret {mr} {time.time()-t0:5.0f}s", flush=True)
            continue
        parts = line.split()
        try:
            if parts[0] == "@NEW":
                r = load_runner(a.game, a.opp); r.success_return = bar
                envs[nxt] = r; reply = f"OK {nxt} {r.spec.observation_size} {r.spec.n_actions}"; nxt += 1
            elif parts[0] == "@R":
                r = envs[int(parts[1])]; r.learner_reset(int(parts[2])); reply = r.brain_line(0.0, False)
            elif parts[0] == "@A":
                r = envs[int(parts[1])]; rew, done = r.learner_step(int(parts[2])); reply = r.brain_line(rew, done); n_env_steps += 1
            elif parts[0] == "@DEL":
                envs.pop(int(parts[1]), None); continue
            else: reply = "ERR unknown"
        except Exception as e:   # noqa: BLE001
            reply = "ERR " + str(e).replace("\n", " ")
        p.stdin.write(reply.encode() + b"\n"); p.stdin.flush()
    p.wait()
    out = dict(game=a.game, opp=a.opp, seed=a.seed, pop=a.pop, head=a.head, budget=a.budget, mode=mode, bar=bar, curve=curve, gens=gens,
               env_steps_relayed=n_env_steps, sec=time.time() - t0, rc=p.returncode, stderr=p.stderr.read().decode()[-500:])
    os.makedirs(os.path.join(ROOT, "baseline", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "baseline", "results", f"{a.tag}_{a.game}_{a.opp}_s{a.seed}.json"), "w"), indent=1)
    print("done", out["sec"], "s  rc", p.returncode, out["stderr"])


if __name__ == "__main__":
    main()
