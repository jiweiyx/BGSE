"""Budget-aware search on NATS-Bench (size space), scored on CIFAR-10.

Setting
  space   : 5 stages x 8 widths -> 32768 architectures (NATS-Bench SSS)
  metric  : CIFAR-10 test accuracy after 90 epochs, precomputed by the benchmark
  budget  : a fixed number of architecture evaluations (table lookups)
  target  : minimise parameters subject to accuracy >= tau (lexicographic)

Methods compared
  bgse       budget-gated (1+lambda) evolution: mutation direction follows
             feasibility, selection is lexicographic
  rea        regularised evolution (Real et al. 2019): tournament + aging,
             same objective, no feasibility gating
  tpe        tree-structured Parzen estimator (Bergstra et al. 2011)
  reinforce  policy gradient over per-stage categorical distributions
  random     uniform sampling
  uniform    static reference: the best all-stages-equal width

All methods see the same budget, the same objective and the same table.
No training happens here: the benchmark supplies the ground truth.

Usage:
  python3 bench/search_nats.py --tau 93 --budget 100 --seeds 30
"""
import argparse
import json
import os
import random

import numpy as np

WIDTHS = [8, 16, 24, 32, 40, 48, 56, 64]
HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "results", "nats_sss_cifar10.npz")
OUT = os.path.join(HERE, "results", "nats_search.json")


class Bench:
    def __init__(self, path, tau):
        d = np.load(path)
        self.widths = d["widths"].tolist()          # width values, (N, 5)
        self.params = d["params"].astype(float)
        self.acc = d["acc90"].astype(float)
        self.tau = tau
        self.index = {tuple(w): i for i, w in enumerate(self.widths)}
        self.N = len(self.acc)

    def ev(self, cfg):
        i = self.index[tuple(cfg)]
        return float(self.params[i]), float(self.acc[i])

    def key(self, params, acc):
        """Lexicographic: feasible first, then fewest parameters."""
        return (1.0, -params) if acc >= self.tau else (0.0, acc)

    def best_feasible(self):
        ok = self.acc >= self.tau
        if not ok.any():
            return None
        i = np.where(ok)[0][np.argmin(self.params[ok])]
        return self.widths[i], float(self.params[i]), float(self.acc[i])


def rand_cfg(rng):
    return [WIDTHS[rng.randrange(8)] for _ in range(5)]


def mutate(cfg, rng, feasible=None, gated=False):
    """Change one stage: one step in the budget-gated direction, else random."""
    c = list(cfg)
    s = rng.randrange(5)
    pos = WIDTHS.index(c[s])
    if gated:
        # The demo's probabilities: feasible -> 0.55 shrink / 0.05 grow;
        # infeasible -> 0.05 shrink / 0.55 grow; the rest is a free move.
        p_shrink = 0.55 if feasible else 0.05
        p_grow = 0.05 if feasible else 0.55
        u = rng.random()
        if u < p_shrink:
            new = pos - 1
        elif u < p_shrink + p_grow:
            new = pos + 1
        else:
            new = rng.randrange(8)
        if new < 0:
            new = pos + 1
        elif new > 7:
            new = pos - 1
    else:
        new = rng.randrange(8)
    if new == pos:
        new = (pos + 1) % 8
    c[s] = WIDTHS[new]
    return c


class Run:
    """Counts evaluations so every method gets exactly the same budget."""

    def __init__(self, bench, budget):
        self.bench, self.budget, self.hist = bench, budget, []

    def ask(self, cfg):
        p, a = self.bench.ev(cfg)
        self.hist.append((list(cfg), p, a))
        return p, a

    @property
    def full(self):
        return len(self.hist) >= self.budget

    def best(self):
        return max(self.hist, key=lambda h: self.bench.key(h[1], h[2]))


def bgse(run, rng, lam=8):
    run.ask(rand_cfg(rng))
    while not run.full:
        best = run.best()
        feasible = best[2] >= run.bench.tau
        for _ in range(lam):
            if run.full:
                break
            run.ask(mutate(best[0], rng, feasible=feasible, gated=True))


def rea(run, rng, pop=40, tour=5):
    archive = []
    for _ in range(min(pop, run.budget)):
        cfg = rand_cfg(rng)
        p, a = run.ask(cfg)
        archive.append((cfg, p, a))
    while not run.full:
        parent = max(rng.sample(archive, tour),
                     key=lambda h: run.bench.key(h[1], h[2]))
        cfg = mutate(parent[0], rng)
        p, a = run.ask(cfg)
        archive.append((cfg, p, a))
        if len(archive) > pop:
            archive.pop(0)


def tpe(run, rng, n_init=10, n_cand=48, gamma=0.25):
    for _ in range(min(n_init, run.budget)):
        run.ask(rand_cfg(rng))
    while not run.full:
        scored = sorted(run.hist, key=lambda h: run.bench.key(h[1], h[2]),
                        reverse=True)
        k = max(2, int(len(scored) * gamma))
        good = [[WIDTHS.index(v) for v in h[0]] for h in scored[:k]]
        bad = [[WIDTHS.index(v) for v in h[0]] for h in scored[k:]]
        best_cfg, best_lp = None, -1e18
        for _ in range(n_cand):
            cand = [rng.randrange(8) for _ in range(5)]
            lp = 0.0
            for s in range(5):
                pg = (sum(g[s] == cand[s] for g in good) + 1.0) / (len(good) + 8)
                pb = (sum(b[s] == cand[s] for b in bad) + 1.0) / (max(1, len(bad)) + 8)
                lp += np.log(pg / pb)
            if lp > best_lp:
                best_cfg, best_lp = cand, lp
        run.ask([WIDTHS[x] for x in best_cfg])


def reinforce(run, rng, batch=8, lr=0.4):
    logits = np.zeros((5, 8))
    baseline = 0.0
    while not run.full:
        cfgs = []
        for _ in range(min(batch, run.budget - len(run.hist))):
            idx = []
            for s in range(5):
                p = np.exp(logits[s] - logits[s].max())
                p /= p.sum()
                idx.append(int(rng.choices(range(8), weights=p)[0]))
            cfgs.append(idx)
        rewards = []
        for idx in cfgs:
            p, a = run.ask([WIDTHS[x] for x in idx])
            rewards.append((1.0 - p / 20.0) if a >= run.bench.tau
                           else (a / run.bench.tau - 1.0))
        adv = np.array(rewards) - baseline
        baseline = 0.9 * baseline + 0.1 * float(np.mean(rewards))
        for idx, g in zip(cfgs, adv):
            for s in range(5):
                logits[s, idx[s]] += lr * g


def random_search(run, rng):
    while not run.full:
        run.ask(rand_cfg(rng))


def rea_gated(run, rng, pop=40, tour=5):
    """The same population search as `rea`, with the budget gate switched on.

    This isolates the gate from the (1+lambda) elitism: if gating helps, it
    should help inside a population search too.
    """
    archive = []
    for _ in range(min(pop, run.budget)):
        cfg = rand_cfg(rng)
        p, a = run.ask(cfg)
        archive.append((cfg, p, a))
    while not run.full:
        parent = max(rng.sample(archive, tour),
                     key=lambda h: run.bench.key(h[1], h[2]))
        cfg = mutate(parent[0], rng, feasible=parent[2] >= run.bench.tau,
                     gated=True)
        p, a = run.ask(cfg)
        archive.append((cfg, p, a))
        if len(archive) > pop:
            archive.pop(0)


METHODS = {
    "bgse": lambda r, g: bgse(r, g),
    "rea_gated": lambda r, g: rea_gated(r, g),
    "rea": lambda r, g: rea(r, g),
    "tpe": lambda r, g: tpe(r, g),
    "reinforce": lambda r, g: reinforce(r, g),
    "random": lambda r, g: random_search(r, g),
}


def summarize(run):
    tau = run.bench.tau
    first = next((i for i, h in enumerate(run.hist) if h[2] >= tau), None)
    feasible = [h for h in run.hist if h[2] >= tau]
    if feasible:
        best = min(feasible, key=lambda h: h[1])
        return {"ok": 1, "params": best[1], "acc": best[2],
                "eval_to_first": first, "cfg": best[0]}
    best = max(run.hist, key=lambda h: h[2])
    return {"ok": 0, "params": None, "acc": best[2],
            "eval_to_first": None, "cfg": best[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=93.0)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench = Bench(TABLE, args.tau)
    opt = bench.best_feasible()
    print(f"space {bench.N} archs | tau {args.tau} | budget {args.budget} | "
          f"seeds {args.seeds}")
    print(f"global optimum: {opt[0]} params {opt[1]:.4f}M acc {opt[2]:.2f}%")

    rows = {}
    for name, fn in METHODS.items():
        runs = []
        for seed in range(args.seeds):
            run = Run(bench, args.budget)
            fn(run, random.Random(10_000 + seed))
            runs.append(summarize(run))
        rows[name] = runs
        ok = [r for r in runs if r["ok"]]
        ps = [r["params"] for r in ok]
        print(f"{name:10s} success {len(ok):2d}/{args.seeds}  "
              f"median {np.median(ps) if ps else float('nan'):.4f}M  "
              f"mean {np.mean(ps) if ps else float('nan'):.4f}M")

    out = args.out or OUT
    with open(out, "w") as f:
        json.dump({"tau": args.tau, "budget": args.budget,
                   "seeds": args.seeds,
                   "optimum": {"cfg": opt[0], "params": opt[1], "acc": opt[2]},
                   "runs": rows}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
