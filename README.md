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

[`HS-MoE-Demo.py`](HS-MoE-Demo.py) applies BGSE to a three-expert mixture whose experts
deliberately **do not share a size**. Each expert gets its own budget, and each evolves its
own width.

| # | Expert | Task | Budget ε |
|---|--------|------|----------|
| 0 | Image | MNIST digit regression | `MSE ≤ 2.5` (RMSE ≈ 1.58) |
| 1 | Linear | `y = 2x + 1` | `MSE ≤ 1e-2` (error ≈ 0.1) |
| 2 | Multiplicative | `y = x₁·x₂` | `MSE ≤ 1e-3` (error ≈ 0.03) |

Nobody types `hidden_dim`, including for the gate. The router (784 → h → 3) dispatches each
input to exactly one expert; the others are not executed at all. `h` is searched under a
misrouting budget, exactly like the experts' widths.

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

### The router is searched too

The gate has the same problem the experts had: its hidden width was a number nobody could
defend. It gets the same procedure, with **misrouting rate** as the budget instead of MSE.

| Component | Hand-set | BGSE | Parameters | Held-out misrouting | Saved |
|---|---|---|---|---|---|
| Router | 32 | **2** | 25184 → 1574 | 0.0000 → 0.0000 | −93.8 % |

The budget is ≤ 2 % misrouting on a held-out quarter of the routing pool, and all five seeds
land on 2 units. Unlike the experts' widths, this one **is** reproducible from scratch — the
sweep gives 6.8 % misrouting at 1 unit and 0.0 % at 2 units — so the gate is the one component
whose searched size could have been found by trial and error as well. Deployed routing
accuracy is 100 % image, 100 % linear, 99.8 % multiplicative: one input in 640 is sent to the
wrong numeric expert.

The gate is now 2 units wide and 94 % smaller, and that number came out of the search. It is
far below the hand-set 32 even though the router is the component that separates the two
numeric formats, which are nearly identical (one non-zero input dimension vs two) — the old
32 was not buying accuracy: 2 units misroute nothing on the held-out pool and lose one input
out of 640 in deployment.

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
python HS-MoE-Demo.py                # ~130 s; MNIST downloads to ./data once
python BGSE-Study.py                 # ~80 s; the five methodological experiments
```

`HS-MoE-Demo.py` prints per-seed search traces for the three experts and the router, the
width sweeps, a same-conditions comparison against the hand-set values, the from-scratch
reproducibility checks, routing accuracy, and live dispatch examples for all three
modalities.

`BGSE-Study.py` reproduces the evidence behind §3: the nine-seed control, the
single-variable task comparison, the repair-timing ablation, and the scoreboard
comparison. It prints one table per experiment and closes with what the data does — and
does not — support.

```
Requirements : Python 3.10+, PyTorch (tested on 3.14 / torch 2.14.1)
Hardware     : CPU only, single machine
Determinism  : fixed seeds throughout; multi-seed by design
```

---

## 6. Repository layout

| Path | Contents |
|---|---|
| `HS-MoE-Demo.py` | The demo in this README: three experts and a router, each with a searched width |
| `BGSE-Study.py` | The study behind the rules: nine-seed budget-vs-width control, single-variable task comparison, repair-timing ablation, and the scoreboard comparison |
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
- **Deployed expert widths are not reproducible from scratch** (§4b). This is a property of
  the method, not a bug, but it means BGSE must ship as *search + artefact*. The router is
  the exception: its 2-unit width is reachable from scratch.
- **Only width is searched** — depth, activation and the number of experts are fixed by hand.
  The router is a single hidden layer by construction; a deeper gate would need a different
  mutation operator.
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

## 10. Standard benchmarks

The demo is deliberately small. [`bench/`](bench/) carries two standard-benchmark studies
with the scripts that regenerate them; raw records, figures and generated reports are in
`bench/results/`.

**NATS-Bench, size space** — 32,768 width configurations, each with CIFAR-10 accuracy after
90 epochs precomputed by the benchmark. Methods get 100 or 300 architecture evaluations and
the same lexicographic objective, 30 seeds each: BGSE, REA, REA + gate, TPE, REINFORCE and
random search.

| method | τ = 92, 300 evals | τ = 93, 300 evals |
|---|---|---|
| TPE | **0.2227 M** | 0.4311 M |
| REA + gate | 0.2300 M | 0.4294 M |
| REA | 0.2353 M | **0.4286 M** |
| Random | 0.2503 M | 0.4578 M |
| BGSE (1+λ) | 0.2903 M | 0.4682 M |

Median parameters of the cheapest feasible architecture (lower is better; τ is the accuracy
floor). On this table **the gate helps a little and the elitist loop hurts**: BGSE's
`(1+λ)` stalls at the first feasibility boundary it reaches, while the same gate inside REA
beats plain REA by about 2–3 % at τ = 92 and ties it at τ = 93. TPE wins overall. Accuracy here
is a table lookup, so there is no repair step — this measures the search rule alone.

**CIFAR-10, end to end** — a 6-layer CNN with per-layer widths in `{8, 16, 32, 64}`, 10k
training images, two accuracy budgets and 5 seeds; the comparison is plotted against training
epochs actually spent, so a warm-started repair is charged what it costs.

| method | ε = 0.62 width | ε = 0.68 width | feasible at 0.68 |
|---|---|---|---|
| BGSE (repair + gate) | 120 [88–136] | 192 [176–256] | 5/5 |
| Static-mutation EA (repair) | **104 [96–120]** | 192 [176–256] | 5/5 |
| Random search | 192 [136–256] | 288 (one seed) | 1/5 |
| BGSE-NoRepair | 192 [192–192] | – | **0/5** |

Sum of the six layer widths, median over five seeds. **Repair is the load-bearing step**:
without it the strict budget is never met (0/5) and the loose one never shrinks below the
mother. The budget gate is not what carries the method — static mutation with repair ties
BGSE. Table with test accuracy and the figure:
[`bench/results/cifar_report.md`](bench/results/cifar_report.md).

### Why the single-lineage loop is the right default end to end

The two tables above are honest about where the search rule loses: when an architecture
evaluation is free (a lookup table), a population (REA) and a model-based sampler (TPE) beat
the `(1+λ)` loop. End-to-end training is the opposite regime, and there the trade-off flips:

- **One model resident instead of N.** A population has to keep every individual alive and,
  with repair, fine-tune each of them; `(1+λ)` keeps the incumbent plus one child. On a
  single GPU or a laptop that difference is the whole memory budget.
- **The epoch budget goes to the incumbent.** A population either spends `N × 8` epochs
  before it can make its first comparison, or gives each individual a fraction of the
  training. `(1+λ)` spends the budget on the model it is about to ship.
- **Repair removes the need for a population.** A population helps because a shrunken child
  usually scores worse; a few warm-start steps fix that directly, so the search does not need
  many lineages to recover a good region.

So the end-to-end claim is deliberately modest: `(1+λ)` plus repair is **enough** to meet the
budget, with one model in memory and no population bookkeeping. It is a cost argument rather
than a head-to-head against REA at equal epochs; the measured support is the repair ablation
above, which is where the method's value actually shows.

---

## Author

**Jiwei Li** — jiweiyx@gmail.com — https://github.com/jiweiyx/BGSE
