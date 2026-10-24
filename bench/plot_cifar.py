"""Plot and summarise the end-to-end CIFAR-10 width-search experiment.

Reads bench/results/cifar_width.json, writes:
  bench/results/fig_cifar_width.png   (2x2: narrowest feasible width and test
                                       accuracy against the epoch budget, one
                                       row per epsilon)
  bench/results/cifar_report.md       (tables + the figure + findings)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SRC = os.path.join(RES, "cifar_width.json")

METHODS = [
    ("bgse", "BGSE (repair + gate)", "#1f77b4", "-", 2.0),
    ("static_ea", "Static-mutation EA (repair)", "#ff7f0e", "-", 1.6),
    ("random", "Random search", "#2ca02c", "-", 1.6),
    ("bgse_norepair", "BGSE-NoRepair", "#d62728", "--", 1.6),
]
GRID = 4          # epoch resolution of the aggregated curves

FINDINGS = """
## What the figure shows

**Repair is the load-bearing step.**

- At the strict budget (validation accuracy >= 0.68) BGSE-NoRepair finds no
  feasible model in any of the five seeds. Without an epoch to recover after a
  structural change, every child is scored worse than its parent, so the search
  never leaves the 192-wide mother - and the mother itself misses the budget
  (validation 0.6665).
- At the loose budget (>= 0.62) the same effect shows up as width: NoRepair
  stays at 192 while BGSE reaches 120 [88-136] and static-mutation EA 104
  [96-120]. Random search, which trains every candidate from scratch, needs
  192 [136-256] for the same compute.
- The price of sitting on the boundary is lower test accuracy (0.609 for BGSE
  at 0.62 against 0.657 for the wide mother). That is the objective working as
  specified: among feasible models, the narrowest wins.

**The gate is not the hero.** Static 50/50 mutation with repair ties BGSE at
the strict budget (both 192 [176-256]) and is slightly narrower at the loose
one (104 vs 120). This matches the tabular benchmark, where the gate only
helped inside a population search. What both repair methods share - and what
NoRepair lacks - is the repair step itself.

**Compute is charged honestly.** Every repair epoch counts against the same
60-epoch budget as a from-scratch training, so NoRepair's zero-epoch children
do not buy it free samples and random search pays in full for each 8-epoch
model. `retrained test` re-trains each winning width vector from scratch, so
the accuracy comparison is not confounded by BGSE's models being warm-started.

## Caveats

- 10k training images and 8 epochs per from-scratch model keep this on a
  laptop; absolute accuracies are 0.60-0.68, not CIFAR-10 SOTA.
- 5 seeds, medians with [min-max]. One width step is one layer moving 32->16
  or 16->8, so a one-step difference is inside seed noise.
- The demo's mutation has a 0.40 "value-only" branch (weight jitter). With no
  weights to jitter before repair, that branch is mapped here to a random
  width change; the mapping is a design choice and may be part of why the gate
  does not separate from static mutation.
- NoRepair spends zero epochs per child by construction, so its marker sits at
  the mother's 8 epochs; the dotted guide is a reference, not spend.
"""



def best_so_far(curve, metric, xs):
    """Best feasible value up to each grid point (step function)."""
    pts = sorted(curve, key=lambda p: (p["epochs"],))
    out = []
    best = None
    j = 0
    for x in xs:
        while j < len(pts) and pts[j]["epochs"] <= x:
            p = pts[j]
            if p["feasible"]:
                if best is None or p["params"] < best["params"]:
                    best = p
            j += 1
        if best is None:
            out.append(np.nan)
        elif metric == "params":
            out.append(best["params"])
        else:
            out.append(best["test"])
    return np.array(out)


def load():
    with open(SRC) as f:
        return json.load(f)


def summarise(results):
    lines = []
    for tau, methods in results.items():
        lines += [f"## epsilon: validation accuracy >= {float(tau):.2f}", ""]
        lines += ["| method | feasible | width sum | params | val acc | test acc (10k) | retrained test (10k) | epochs to best |",
                  "|---|---|---|---|---|---|---|"]
        for m, label, *_ in METHODS:
            if m not in methods:
                continue
            runs = methods[m]
            ch = [r["champion"] for r in runs if r["champion"]]
            if not ch:
                lines.append(f"| {label} | 0/{len(runs)} | - | - | - | - | - |")
                continue
            w = [c["params"] for c in ch]
            pc = [c["param_count"] for c in ch]
            val = [c["val"] for c in ch]
            te = [c["test_full"] for c in ch if c["test_full"] is not None]
            rt = [c["retrained_test"] for c in ch if c.get("retrained_test") is not None]
            ep = [c["epochs"] for c in ch]
            lines.append(
                f"| {label} | {len(ch)}/{len(runs)} | "
                f"{np.median(w):.0f} [{min(w)}-{max(w)}] | "
                f"{np.median(pc):,.0f} | {np.median(val):.4f} | "
                f"{np.median(te):.4f} [{min(te):.4f}-{max(te):.4f}] | "
                f"{(f'{np.median(rt):.4f} [{min(rt):.4f}-{max(rt):.4f}]') if rt else '-'} | "
                f"{np.median(ep):.0f} |")
        lines.append("")
    return lines


def plot(results):
    taus = sorted(results, key=float)
    budget = max(p["epochs"] for m in results[taus[0]].values() for r in m
                 for p in r["curve"])
    xs = np.arange(8, budget + 1, GRID)

    fig, axes = plt.subplots(len(taus), 2, figsize=(12, 4.2 * len(taus)),
                             squeeze=False)
    for row, tau in enumerate(taus):
        ax_w, ax_a = axes[row]
        for m, label, color, style, lw in METHODS:
            if m not in results[tau] or m == "bgse_norepair":
                continue
            curves = []
            for r in results[tau][m]:
                curves.append((best_so_far(r["curve"], "params", xs),
                               best_so_far(r["curve"], "test", xs)))
            w = np.array([c[0] for c in curves])
            a = np.array([c[1] for c in curves])
            with np.errstate(all="ignore"):
                med_w = np.nanmedian(w, axis=0)
                med_a = np.nanmedian(a, axis=0)
                lo_w, hi_w = np.nanmin(w, axis=0), np.nanmax(w, axis=0)
                lo_a, hi_a = np.nanmin(a, axis=0), np.nanmax(a, axis=0)
            n_ok = sum(1 for r in results[tau][m] if r["champion"])
            lbl = f"{label} ({n_ok}/{len(results[tau][m])})"
            ax_w.plot(xs, med_w, style, color=color, lw=lw, label=lbl)
            ax_w.fill_between(xs, lo_w, hi_w, color=color, alpha=0.12)
            ax_a.plot(xs, med_a, style, color=color, lw=lw, label=lbl)
            ax_a.fill_between(xs, lo_a, hi_a, color=color, alpha=0.12)

        # NoRepair spends no epochs on repair, so its point sits at the mother's
        # epoch count; a line would falsely suggest it kept spending.
        nr = results[tau].get("bgse_norepair")
        if nr:
            v = [best_so_far(r["curve"], "params", np.array([8]))[0] for r in nr]
            a = [best_so_far(r["curve"], "test", np.array([8]))[0] for r in nr]
            v = [x for x in v if not np.isnan(x)]
            a = [x for x in a if not np.isnan(x)]
            if v:
                ax_w.errorbar(8, np.median(v), yerr=[[np.median(v) - min(v)],
                                                     [max(v) - np.median(v)]],
                              fmt="*", ms=16, color="#d62728", capsize=5,
                              label=f"BGSE-NoRepair ({len(v)}/{len(nr)})")
                ax_w.axhline(np.median(v), color="#d62728", ls=":", lw=1.2)
                ax_w.annotate("BGSE-NoRepair\n(no repair epochs spent)",
                              xy=(8, np.median(v)), xytext=(16, np.median(v) + 18),
                              fontsize=8, color="#d62728",
                              arrowprops=dict(arrowstyle="-", color="#d62728"))
            if a:
                ax_a.errorbar(8, np.median(a), yerr=[[np.median(a) - min(a)],
                                                     [max(a) - np.median(a)]],
                              fmt="*", ms=16, color="#d62728", capsize=5)
                ax_a.axhline(np.median(a), color="#d62728", ls=":", lw=1.2)

        ax_w.set_title(f"narrowest feasible width, eps = {float(tau):.2f}")
        ax_w.set_xlabel("training epochs spent")
        ax_w.set_ylabel("sum of the six layer widths")
        ax_a.set_title(f"test accuracy of that model, eps = {float(tau):.2f}")
        ax_a.set_xlabel("training epochs spent")
        ax_a.set_ylabel("test accuracy")
        ax_a.set_ylim(0.35, 0.75)
        for ax in (ax_w, ax_a):
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(RES, "fig_cifar_width.png")
    fig.savefig(path, dpi=160)
    print("wrote", path)


def main():
    results = load()
    plot(results)
    lines = ["# End-to-end width search on CIFAR-10", "",
             "Backbone: 6-layer CNN, per-layer width in {8,16,32,64}. "
             "10k train / 2k validation / 10k test.",
             "Budget: a fixed number of training epochs; 5 seeds.",
             "Epsilon is a validation-accuracy threshold; the objective is "
             "lexicographic (feasible first, then fewest parameters).",
             "Curves are the median across seeds; a seed enters only once it has "
             "found a feasible model, so the median can move up when another "
             "seed joins. Bands are [min-max].", "",
             "![width and accuracy against compute]"
             "(fig_cifar_width.png)", "", "## Champion per run", ""]
    lines += summarise(results)
    lines += [FINDINGS]
    with open(os.path.join(RES, "cifar_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(RES, "cifar_report.md"))


if __name__ == "__main__":
    main()
