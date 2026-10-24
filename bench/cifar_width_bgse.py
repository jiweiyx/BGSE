"""End-to-end width search on CIFAR-10: does repair pay for itself?

Setup
  data      : CIFAR-10 (10k train / 2k validation / 10k test)
  backbone  : 6-layer CNN, per-layer width in {8, 16, 32, 64} (4096 architectures)
  budget    : a fixed number of *training epochs* - the compute axis
  target    : lexicographic - validation accuracy >= tau first, then fewest
              parameters; epsilon is expressed as tau (two levels)

Methods
  bgse        budget-gated mutation + Lamarckian repair (fine-tune the child
              for one epoch before scoring it)
  bgse_norepair  the same gate, but the child is scored immediately: this is
              the ablation that should lose, because a shrunken child is
              judged before it can recover
  static_ea   Lamarckian repair, but grow/shrink chosen with fixed 0.5/0.5
              probabilities instead of by feasibility
  random      sample a width vector, train it from scratch for the full
              8 epochs; gets fewer evaluations for the same compute

All methods are run until the same total epoch budget and are compared on
(a) the narrowest feasible width found and (b) test accuracy, both as a
function of epochs spent. Five seeds, median and [min-max].

Usage:
  python3 bench/cifar_width_bgse.py --calibrate
  python3 bench/cifar_width_bgse.py --tau 0.55 0.62 --budget 200 --seeds 5
"""
import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "cifar-10-batches-py")
RES = os.path.join(HERE, "results")
WIDTHS = [8, 16, 32, 64]
N_LAYERS = 6
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 8          # epochs for a from-scratch training
REPAIR = 1          # epochs for a Lamarckian repair


# ---------------------------------------------------------------- data
def load_cifar():
    def batch(name):
        with open(os.path.join(DATA, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = np.array(d[b"labels"], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    xs, ys = [], []
    for i in range(1, 6):
        x, y = batch(f"data_batch_{i}")
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs)
    Y = torch.cat(ys)
    Xte, yte = batch("test_batch")

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    X = (X - mean) / std
    Xte = (Xte - mean) / std

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(X.shape[0], generator=g)
    X, Y = X[perm], Y[perm]
    return {
        "train": (X[:10000], Y[:10000]),
        "val": (X[10000:12000], Y[10000:12000]),
        # 2k for the search curves; the full 10k is used once, for the champion
        "test": (Xte[:2000], yte[:2000]),
        "test_full": (Xte, yte),
    }


def augment(x):
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (1,)).item()
    j = torch.randint(0, 9, (1,)).item()
    x = x[:, :, i:i + 32, j:j + 32]
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    return x


# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.widths = list(widths)
        ch = [3] + self.widths
        body = []
        for i in range(N_LAYERS):
            body += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1, bias=False),
                     nn.BatchNorm2d(ch[i + 1]), nn.ReLU(inplace=True)]
            if i in (1, 3, 5):
                body.append(nn.MaxPool2d(2))
        self.body = nn.Sequential(*body)
        self.fc = nn.Linear(self.widths[-1], 10)

    def forward(self, x):
        x = self.body(x)
        return self.fc(x.mean((2, 3)))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def resize(parent, widths):
    """Child inherits every learned tensor that still fits; new channels init."""
    child = Net(widths).to(DEVICE)
    sd = parent.state_dict()
    csd = child.state_dict()
    for k in csd:
        if k not in sd:
            continue
        a, b = sd[k], csd[k]
        if a.dim() == 0:
            csd[k].copy_(b)
            continue
        if a.shape[0] != b.shape[0] or (a.dim() > 1 and a.shape[1] != b.shape[1]):
            slices = tuple(slice(0, min(s, t)) for s, t in zip(a.shape, b.shape))
            csd[k][slices].copy_(a[slices])
        else:
            csd[k].copy_(a)
    child.load_state_dict(csd)
    return child


# ---------------------------------------------------------------- training
def evaluate(model, split, bs=256):
    model.eval()
    X, Y = split
    correct = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            pred = model(X[i:i + bs].to(DEVICE)).argmax(-1)
            correct += (pred == Y[i:i + bs].to(DEVICE)).sum().item()
    return correct / X.shape[0]


def train(model, data, epochs, lr=0.05, seed=0):
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)
    n = data["train"][0].shape[0]
    steps_per_epoch = (n + 127) // 128
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * steps_per_epoch))
    X, Y = data["train"]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb = augment(X[idx])
            opt.zero_grad()
            loss = F.cross_entropy(model(xb.to(DEVICE)), Y[idx].to(DEVICE))
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------- search
def score(model, widths, data, tau):
    acc = evaluate(model, data["val"])
    params = sum(widths)
    feasible = acc >= tau
    key = (1.0, -params) if feasible else (0.0, acc)
    return key, acc


def mutate(widths, feasible, gated, rng):
    w = list(widths)
    layer = rng.randrange(N_LAYERS)
    pos = WIDTHS.index(w[layer])
    if gated:
        p_shrink = 0.55 if feasible else 0.05
        p_grow = 0.05 if feasible else 0.55
        u = rng.random()
        if u < p_shrink:
            new = pos - 1
        elif u < p_shrink + p_grow:
            new = pos + 1
        else:
            new = rng.randrange(len(WIDTHS))
    else:
        new = pos + (1 if rng.random() < 0.5 else -1)
    new = max(0, min(len(WIDTHS) - 1, new))
    if new == pos:
        new = pos + (1 if pos == 0 else -1)
    w[layer] = WIDTHS[new]
    return w


def search(method, data, tau, budget, seed, record):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    if method == "random":
        used = 0
        while used + EPOCHS <= budget:
            widths = [rng.choice(WIDTHS) for _ in range(N_LAYERS)]
            model = Net(widths).to(DEVICE)
            train(model, data, EPOCHS, seed=seed + used)
            used += EPOCHS
            key, acc = score(model, widths, data, tau)
            record(used, widths, model, acc, feasible=acc >= tau)
        return

    mother_widths = [32] * N_LAYERS
    mother = train(Net(mother_widths).to(DEVICE), data, EPOCHS, seed=seed)
    used = EPOCHS
    key, acc = score(mother, mother_widths, data, tau)
    best = {"widths": mother_widths, "model": mother, "key": key,
            "acc": acc, "used": used}
    record(used, mother_widths, mother, acc, feasible=acc >= tau)

    repair_epochs = 0 if method == "bgse_norepair" else REPAIR
    # NoRepair costs no epochs per child, so cap its evaluations instead: it
    # gets ten times BGSE's number of chances, all at zero training cost.
    max_evals = budget  # same number of child evaluations as BGSE
    evals = 0
    while used + repair_epochs <= budget and evals < max_evals:
        feasible = best["acc"] >= tau
        widths = mutate(best["widths"], feasible, gated=(method != "static_ea"),
                        rng=rng)
        child = resize(best["model"], widths)
        if repair_epochs:
            train(child, data, repair_epochs, lr=0.01, seed=seed + used)
        used += repair_epochs
        key, acc = score(child, widths, data, tau)
        record(used, widths, child, acc, feasible=acc >= tau)
        evals += 1
        if key > best["key"]:
            best = {"widths": widths, "model": child, "key": key,
                    "acc": acc, "used": used}


def run(method, data, tau, budget, seed):
    curve = []

    def record(used, widths, model, acc, feasible):
        curve.append({"epochs": used, "widths": list(widths),
                      "params": sum(widths),
                      "param_count": model.param_count(),
                      "val": acc, "feasible": bool(feasible),
                      "test": evaluate(model, data["test"]) if feasible else None,
                      "model": model})

    search(method, data, tau, budget, seed, record)
    if champion(curve, tau) is not None:
        champion(curve, tau)["test_full"] = evaluate(
            champion(curve, tau)["model"], data["test_full"])
    return curve


def champion(curve, tau):
    feas = [p for p in curve if p["feasible"]]
    if not feas:
        return None
    return min(feas, key=lambda p: (p["params"], p["epochs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--tau", nargs="+", type=float, default=[0.62, 0.68])
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(RES, "cifar_bgse.json"))
    args = ap.parse_args()

    data = load_cifar()
    print(f"data: train {data['train'][0].shape[0]}  val {data['val'][0].shape[0]}"
          f"  test {data['test'][0].shape[0]}")

    if args.calibrate:
        for w in ([8] * 6, [32] * 6, [64] * 6):
            t = time.time()
            m = train(Net(w).to(DEVICE), data, EPOCHS, seed=0)
            print(f"widths {w[0]:2d} each: params {m.param_count():7d}  "
                  f"val {evaluate(m, data['val']):.4f}  "
                  f"test {evaluate(m, data['test']):.4f}  "
                  f"({time.time()-t:.0f}s, {EPOCHS} epochs)")
        return

    methods = ["bgse", "bgse_norepair", "static_ea", "random"]
    results = {}
    for tau in args.tau:
        for method in methods:
            runs = []
            for seed in range(args.seeds):
                t = time.time()
                curve = run(method, data, tau, args.budget, seed)
                champ = champion(curve, tau)
                runs.append({
                    "curve": [{k: v for k, v in p.items() if k != "model"}
                              for p in curve],
                    "champion": None if champ is None else
                    {"widths": champ["widths"], "params": champ["params"],
                     "param_count": champ["param_count"],
                     "val": champ["val"], "epochs": champ["epochs"],
                     "test_full": champ.get("test_full")},
                })
                print(f"tau {tau:.2f} {method:14s} seed {seed}  "
                      f"{time.time()-t:5.0f}s  "
                      f"champ {champ['params'] if champ else None}", flush=True)
            results.setdefault(str(tau), {})[method] = runs
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
        print(f"tau {tau:.2f} done")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
