#!/usr/bin/env python3
"""Plain PPO baseline (fixed MLP, no evolution) on the repo's generated-env fixtures.

Same env source, same Runner (opponent, seat randomisation, truncation, legal-action masking), same reward as
the NEAT+PPO learner.  Shared tanh trunk with [n_actions policy logits | value] outputs, like the C++ net.

  python3 baseline/ppo_baseline.py --game snake --seeds 3 --steps 1000000
  python3 baseline/ppo_baseline.py --game pong --opp rule_based --seeds 3 --steps 1000000
"""
import argparse, json, math, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "llm_game_platform"))
import sandbox_worker as sw            # noqa: E402  (Runner only; nothing runs at import)
from env_interface import EnvSpec      # noqa: E402
import learnability                    # noqa: E402


def load_runner(game, opp):
    ns = {"__name__": "fixture_" + game}
    exec(compile(open(os.path.join(ROOT, "generated_env_fixtures", game + ".py")).read(), game, "exec"), ns)
    spec = EnvSpec.from_mapping(ns["SPEC"])
    r = sw.Runner(ns["Environment"], spec)
    r.default_controller = r.set_controller({"kind": opp})
    return r


def success_bar(game, opp, episodes=100):
    r = load_runner(game, opp)
    cal = r.calibrate(episodes, 777)
    mode = learnability.quality_mode({"agents": r.spec.agents, "cooperative": r.spec.cooperative}, cal)
    return (learnability.success_bar(cal) if mode == "return" else None), mode, cal


class Game:
    """One env instance; mirrors the brain-mode protocol (learner_reset / learner_step / brain_line)."""
    def __init__(self, game, opp, bar):
        self.r = load_runner(game, opp)
        self.r.success_return = bar
        self.game = game

    def reset(self, seed):
        self.r.learner_reset(seed)
        return self._obs()

    def _obs(self):
        return np.asarray(self.r.observe(self.r.learner), np.float32), self.r.legal(self.r.learner)

    def step(self, a):
        rew, done = self.r.learner_step(int(a))
        info = None
        if done:
            r = self.r
            won = r.result(r.learner) == "win"
            ok = won or (r.success_return is not None and r.cum[r.learner] > r.success_return)
            res = r.result(r.learner)
            info = dict(ret=r.cum[r.learner], len=r.steps, win=int(res == "win"), loss=int(res == "loss"),
                        success=int(ok), score=getattr(r.env, "score", 0))
            return rew, True, info
        return rew, False, None


class MLP:
    def __init__(self, sizes, rng, out_gain=0.01):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            g = math.sqrt(2.0) if i < len(sizes) - 2 else out_gain
            w = rng.standard_normal((sizes[i], sizes[i + 1])).astype(np.float32) * (g / math.sqrt(sizes[i]))
            self.W.append(w); self.b.append(np.zeros(sizes[i + 1], np.float32))
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]; self.v = [np.zeros_like(p) for p in self.params]; self.t = 0

    def forward(self, x):
        acts = [x]
        for i in range(len(self.W)):
            x = x @ self.W[i] + self.b[i]
            if i < len(self.W) - 1: x = np.tanh(x)
            acts.append(x)
        return x, acts

    def backward(self, acts, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d; gb[i] = d.sum(0)
            if i > 0: d = (d @ self.W[i].T) * (1 - acts[i] ** 2)
        return gW + gb

    def adam(self, grads, lr, max_grad):
        n = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
        sc = max_grad / n if n > max_grad else 1.0
        self.t += 1; b1, b2 = 0.9, 0.999
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * sc
            m *= b1; m += (1 - b1) * g; v *= b2; v += (1 - b2) * g * g
            p -= lr * (m / c1) / (np.sqrt(v / c2) + 1e-5)


def policy(net, obs, mask, rng, greedy=False):
    out, _ = net.forward(obs)
    A = mask.shape[1]
    z = np.where(mask, out[:, :A], -1e9)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    if greedy: a = z.argmax(1)
    else:
        u = rng.random(len(p))[:, None]; a = np.minimum((p.cumsum(1) < u).sum(1), A - 1)
    lp = np.log(np.maximum(p[np.arange(len(a)), a], 1e-30))
    return a, lp, out[:, A]


def evaluate(net, game, opp, bar, episodes, seed, greedy, n_act):
    g = Game(game, opp, bar); rng = np.random.default_rng(seed); R = []
    for e in range(episodes):
        obs, legal = g.reset(10_000_000 + seed * 1009 + e * 7919)
        while True:
            mask = np.zeros((1, n_act), bool); mask[0, legal] = True
            a, _, _ = policy(net, obs[None], mask, rng, greedy)
            _, done, info = g.step(a[0])
            if done: R.append(info); break
            obs, legal = g._obs()
    f = lambda k: float(np.mean([x[k] for x in R]))
    return dict(win=f("win"), loss=f("loss"), success=f("success"), ret=f("ret"), len=f("len"), score=f("score"))


def run(args, seed):
    bar, mode, cal = success_bar(args.game, args.opp)
    rng = np.random.default_rng(seed)
    games = [Game(args.game, args.opp, bar) for _ in range(args.envs)]
    n_obs = games[0].r.spec.observation_size; n_act = games[0].r.spec.n_actions
    sizes = [n_obs] + [int(h) for h in args.hidden.split(",") if h] + [n_act + 1]
    net = MLP(sizes, rng)
    obs = np.zeros((args.envs, n_obs), np.float32); mask = np.zeros((args.envs, n_act), bool)
    seed_ctr = seed * 1_000_003
    for i, g in enumerate(games):
        seed_ctr += 1; o, l = g.reset(seed_ctr); obs[i] = o; mask[i, l] = True
    T = args.rollout; N = args.envs
    steps = 0; t0 = time.time(); curve = []; train_eps = []
    next_ckpt = list(args.ckpts)
    while steps < args.steps:
        B_obs = np.zeros((T, N, n_obs), np.float32); B_mask = np.zeros((T, N, n_act), bool)
        B_a = np.zeros((T, N), np.int64); B_lp = np.zeros((T, N), np.float32); B_v = np.zeros((T, N), np.float32)
        B_r = np.zeros((T, N), np.float32); B_d = np.zeros((T, N), np.float32)
        for t in range(T):
            a, lp, v = policy(net, obs, mask, rng)
            B_obs[t] = obs; B_mask[t] = mask; B_a[t] = a; B_lp[t] = lp; B_v[t] = v
            for i, g in enumerate(games):
                rew, done, info = g.step(a[i]); B_r[t, i] = rew; B_d[t, i] = done
                if done:
                    train_eps.append(info); seed_ctr += 1; o, l = g.reset(seed_ctr)
                else: o, l = g._obs()
                obs[i] = o; mask[i] = False; mask[i, l] = True
        steps += T * N
        _, _, boot = policy(net, obs, mask, rng)
        adv = np.zeros((T, N), np.float32); gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = boot if t == T - 1 else B_v[t + 1]; nt = 1.0 - B_d[t]
            delta = B_r[t] + args.gamma * nv * nt - B_v[t]
            gae = delta + args.gamma * args.lam * nt * gae; adv[t] = gae
        ret = adv + B_v
        F = lambda x: x.reshape(T * N, *x.shape[2:])
        fo, fm, fa, flp, fadv, fret = F(B_obs), F(B_mask), F(B_a), F(B_lp), F(adv), F(ret)
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-6)
        K = T * N
        for ep in range(args.epochs):
            perm = rng.permutation(K)
            for s in range(0, K, args.minibatch):
                ix = perm[s:s + args.minibatch]; M = len(ix)
                out, acts = net.forward(fo[ix])
                z = np.where(fm[ix], out[:, :n_act], -1e9); z -= z.max(1, keepdims=True)
                p = np.exp(z); p /= p.sum(1, keepdims=True)
                lpa = np.log(np.maximum(p[np.arange(M), fa[ix]], 1e-30))
                ratio = np.exp(np.minimum(20.0, lpa - flp[ix])); A = fadv[ix]
                clipped = ((A >= 0) & (ratio > 1 + args.clip)) | ((A < 0) & (ratio < 1 - args.clip))
                g = np.where(clipped, 0.0, -A * ratio)
                lp_all = np.log(np.maximum(p, 1e-30)); H = -(p * lp_all).sum(1, keepdims=True)
                onehot = np.zeros_like(p); onehot[np.arange(M), fa[ix]] = 1
                dz = (g[:, None] * (onehot - p) + args.ent * p * (lp_all + H)) / M
                dz = np.where(fm[ix], dz, 0.0)
                dv = args.vf * (out[:, n_act] - fret[ix]) / M
                dout = np.concatenate([dz, dv[:, None]], 1).astype(np.float32)
                net.adam(net.backward(acts, dout), args.lr, args.max_grad)
        recent = train_eps[-50:]
        if next_ckpt and steps >= next_ckpt[0]:
            c = next_ckpt.pop(0)
            ev = {m: evaluate(net, args.game, args.opp, bar, args.eval_eps, seed, m == "greedy", n_act) for m in ("stochastic", "greedy")}
            curve.append(dict(steps=steps, sec=round(time.time() - t0, 1), train_recent_ret=float(np.mean([e["ret"] for e in recent])) if recent else None, **{k: v for k, v in ev.items()}))
            e = ev["stochastic"]; gdy = ev["greedy"]
            print(f"[{args.game} s{seed}] {steps:>8} steps {time.time()-t0:6.0f}s  stoch win {e['win']:.2f} succ {e['success']:.2f} ret {e['ret']:+.2f} score {e['score']:.2f} | greedy win {gdy['win']:.2f} succ {gdy['success']:.2f} ret {gdy['ret']:+.2f}", flush=True)
    return dict(game=args.game, opp=args.opp, seed=seed, mode=mode, bar=bar, hidden=args.hidden, curve=curve,
                steps=steps, sec=time.time() - t0, hp={k: getattr(args, k) for k in ("lr", "gamma", "lam", "clip", "ent", "vf", "epochs", "minibatch", "rollout", "envs")},
                calibration=dict(rand_ret=cal["random"]["mean_return"], rand_win=cal["random"]["win"],
                                 scripted_ret=cal["scripted"]["mean_return"] if cal["scripted"] else None,
                                 scripted_win=cal["scripted"]["win"] if cal["scripted"] else None))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="snake"); ap.add_argument("--opp", default="random")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1_000_000); ap.add_argument("--hidden", default="16")
    ap.add_argument("--envs", type=int, default=16); ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01); ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--max_grad", type=float, default=0.5); ap.add_argument("--eval_eps", type=int, default=100)
    ap.add_argument("--ckpts", type=int, nargs="*", default=[100_000, 250_000, 500_000, 1_000_000])
    ap.add_argument("--out", default="baseline/results")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    for s in range(a.seed0, a.seed0 + a.seeds):
        res = run(a, s)
        tag = f"{a.game}_{a.opp}_h{a.hidden.replace(',', '-')}_s{s}"
        json.dump(res, open(os.path.join(ROOT, a.out, tag + ".json"), "w"), indent=1)
