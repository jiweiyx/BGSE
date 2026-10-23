# BGSE — Budget-Gated Structural Evolution

**Let a model's size be measured, not guessed.**

BGSE decides *how large a model should be* instead of assuming it. You hand it a precision
budget — "error must be at most this much" — and it returns the smallest architecture that
meets it. Width becomes an output of search, not a hyperparameter.

```
                    ┌──────────────┐
   precision budget │              │
        ε  ────────►│     BGSE     │────────►  width, parameters, measured error
                    │              │
                    └──────────────┘
                     no hidden_dim=...
```

---

## 1. The question it answers

Every compression pipeline starts from a number nobody can defend:

```python
hidden_dim = 4     # why 4?
hidden_dim = 32    # why 32?
```

Structured pruning starts from an already-trained large model and a hand-picked pruning
ratio; NAS searches a fixed design space at fixed cost. In both cases *the scale itself*
remains an input. BGSE makes it an output, and exposes exactly one knob: **ε**, the error
you are willing to tolerate.

---

## 2. Core idea

Each candidate model is judged **lexicographically**:

1. **Feasibility first.** Does it satisfy `MSE ≤ ε` on held-out validation data?
2. **Then frugality.** Among feasible candidates, the one with the fewest parameters wins.

Structural mutation is **gated by feasibility**: while the incumbent is infeasible, offspring
are biased toward *adding* a channel; once feasible, toward *removing* one. The search
therefore expands to earn accuracy, then contracts to save size — it does not need you to
tell it which direction to move.

---

## 3. The four rules

These are not stylistic choices. Each one fixes a failure mode that was observed and
measured.

| # | Rule | Failure it prevents |
|---|------|---------------------|
| 1 | **Lexicographic budget fitness** — feasibility first, then parameter count | Scalarised objectives like `-(MSE + λ·params)` create a *shrinkage ratchet*: on 784-dim inputs, dropping one channel earns `785λ`, while the accuracy it costs is far smaller. The model gets smaller **and worse**, while the score keeps improving. |
| 2 | **Lamarckian repair** — every structural child takes a few Adam steps *before* it is scored | Removing a channel raises MSE immediately. Without a repair step, shrinkage is always judged worse, so the search can never move downward — it loses the ability to get smaller at all. |
| 3 | **The scoreboard must not sit on the training set** | With 5 training points and thousands of parameters, *every* width "passes". Data is split into **train** (gradients) / **validation** (fitness) / **test** (reporting only). |
| 4 | **One seed is an anecdote** | Width differences of one or two channels are inside the noise. Report several seeds as `mean [min–max]`, and deploy the **median** champion — not the luckiest one. |

Plus one operator detail: **mutation direction is gated by the budget**, not by fixed
probabilities. A static `0.2 add / 0.2 remove` leaves the search standing still, because it
is equally reluctant to grow and to shrink.

---

## 4. Demo: Heterogeneous-Scale MoE

[`HS-MoE.py`](HS-MoE.py) applies BGSE to a three-expert mixture whose experts deliberately
**do not share a size**. Each expert gets its own budget, and each evolves its own width.

| # | Expert | Task | Budget ε |
|---|--------|------|----------|
| 0 | Image | MNIST digit regression | `MSE ≤ 2.5` (RMSE ≈ 1.58) |
| 1 | Linear | `y = 2x + 1` | `MSE ≤ 1e-2` (error ≈ 0.1) |
| 2 | Multiplicative | `y = x₁·x₂` | `MSE ≤ 1e-3` (error ≈ 0.03) |

Nobody types `hidden_dim`. The router (784 → 32 → 3) dispatches each input to exactly one
expert; the others are not executed at all.

### Measured widths

| Expert | Hand-set | BGSE | Parameters | Test MSE | Saved |
|---|---|---|---|---|---|
| Linear | 4 | **1** | 3140 → 785 | 0.0000 → 0.0000 | −75.0 % |
| Multiplicative | 16 | **2** | 12560 → 1570 | 0.0019 → 0.0010 | −87.5 % |
| Image | 32 | **19** | 25120 → 14915 | 2.9094 → 2.5048 | −40.6 % |

The width ratio comes out as **1 : 2 : 19**, matching the task-difficulty ordering
(linear < multiplicative < image). *No one chose these three numbers.*

Note: because each expert also has a different ε, this is a **budget → scale** result rather
than a clean difficulty → scale result — see [§8](#8-honest-limitations).

### Three checks worth reading

**a) The gain comes from structure, not from extra compute.**
Running the same amount of training but *forbidding* width changes leaves every expert
exactly where it started — 4, 16 and 32 — with no reduction at all.

**b) Minimum width ≠ trainable width.** This is the most important caveat:

| Expert | BGSE width (validation) | Same width, trained from scratch |
|---|---|---|
| Linear | 1 dim — 0.0000 | 1 dim — **4.5893** |
| Multiplicative | 2 dims — 0.0008 | 2 dims — **0.0229** |
| Image | 19 dims — 2.4333 | 19 dims — 3.0207 |

The 1-dim solution is *unreachable by random initialisation* — a single ReLU unit collapses
into the dead-ReLU regime and outputs a constant. It exists only because the search started
at 4 dims and shrank down, repairing after every removal. **A width is not a deliverable on
its own; it must ship with the procedure that found it.**

**c) Below the knee, width is not what matters.** Sweeping width from scratch on the image
expert:

| Width | 4 | 8 | 16 | **19** | 32 | 64 |
|---|---|---|---|---|---|---|
| Test MSE | 3.17 | 3.03 | 3.19 | **2.97** | 2.47 | 2.10 |

4–16 dims all land around 3.0 — that band is noise. The real decision point is *whether to
go beyond 32*, not whether to trim from 32 to 19.

---

## 5. Quick start

```bash
git clone https://github.com/jiweiyx/BGSE
cd BGSE
pip install torch torchvision        # CPU is enough
python HS-MoE.py                     # ~110 s; MNIST downloads to ./data once
```

Output: per-seed search traces, the width sweep, a same-conditions comparison against the
hand-set values, the from-scratch reproducibility check, router accuracy, and live dispatch
examples for all three modalities.

```
Requirements : Python 3.10+, PyTorch (tested on 3.14 / torch 2.14.1)
Hardware     : CPU only, single machine
Determinism  : fixed seeds throughout; multi-seed by design
```

---

## 6. Repository layout

| Path | Contents |
|---|---|
| `HS-MoE.py` | The demo in this README: three experts, three budgets, one router |
| `12.py` | The BGSE study: 9-seed control, budget-vs-width behaviour, ablations, and the methodological checks behind rules 1–4 |
| `README.md` | This document |

---

## 7. What BGSE is not

| | BGSE |
|---|---|
| **Structured pruning** | One-way, starts from a trained model, requires a hand-picked ratio. BGSE is bidirectional (grow *and* shrink) and starts from a budget. |
| **NAS** | Searches architectures at fixed cost. In BGSE, scale is an output and ε is the only knob. |
| **Adaptive / conditional MoE** | Adapts *per input at inference time*. BGSE adapts *per expert at design time* — the deployed network is static. |
| **Genetic algorithms** | No crossover. Strictly mutation + elitist selection, i.e. a `(1+λ)` evolution strategy. "Genetic" would be the wrong word. |

---

## 8. Honest limitations

- **The demo is tiny and synthetic** — a 3-expert MoE on MNIST plus two toy functions. It
  demonstrates the mechanism, not a production compression result.
- **Widths and budgets are confounded.** In the demo table the three experts differ in both
  task difficulty *and* ε, so the `1 : 2 : 19` ordering reflects the two jointly. It is a
  budget→scale result, not a clean difficulty→scale result.
- **The multiplicative expert sits on the boundary** (validation 0.0008 against ε = 1e-3).
  That is a borderline solution; if you need engineering margin, take 4 dims.
- **Deployed widths are not reproducible from scratch** (§4b). This is a property of the
  method, not a bug, but it means BGSE must ship as *search + artefact*.
- **Linearity of cost assumed** — evaluation cost is linear in width, which holds for these
  MLPs and would need rework for attention-based blocks.

---

## 9. Terminology

| Term | Meaning |
|---|---|
| **BGSE** | Budget-Gated Structural Evolution — the method. "Gated" refers to mutation direction being gated by budget feasibility. |
| **HS-MoE** | Heterogeneous-Scale MoE — the deployment pattern in which each expert carries a different, independently derived width. |

Deliberately avoided: `genetic` (no crossover), `adaptive MoE` (means runtime routing in the
MoE literature), `genetic improvement` (an established term for automated program repair).

---

## Author

**Jiwei Li** — jiweiyx@gmail.com — https://github.com/jiweiyx/BGSE
