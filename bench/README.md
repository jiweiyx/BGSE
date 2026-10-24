# Benchmarks

Two standard-benchmark studies for BGSE. Everything here is scripted; raw
records and figures live in `results/`, and `DATA.md` says where the datasets
come from. `data/` and `vendor/` are gitignored.

## Reproduce

```bash
python3 bench/build_table.py                  # NATS-Bench -> results/nats_sss_cifar10.npz
for tau in 92 93; do for b in 100 300; do
  python3 bench/search_nats.py --tau $tau --budget $b --seeds 30 \
      --out bench/results/nats_t${tau}_b${b}.json
done; done
python3 bench/report.py                       # -> results/report.md

python3 bench/cifar_width_bgse.py --tau 0.62 0.68 --budget 60 --seeds 5
python3 bench/plot_cifar.py                   # -> results/fig_cifar_width.png
```

## 1. NATS-Bench, size space (CIFAR-10)

The size space is 5 stages x 8 widths (32,768 architectures); the benchmark
stores CIFAR-10 test accuracy after 90 epochs for every one of them, so a
search method can be scored without training. The objective is lexicographic:
feasible first (`accuracy >= tau`), then fewest parameters. All methods get the
same number of architecture evaluations, 30 seeds each.

| method | tau=92, 300 evals | tau=93, 300 evals |
|---|---|---|
| TPE | **0.2227 M** | 0.4311 M |
| REA + gate | 0.2300 M | 0.4294 M |
| REA | 0.2353 M | **0.4286 M** |
| Random | 0.2503 M | 0.4578 M |
| BGSE (1+lambda) | 0.2903 M | 0.4682 M |

(median parameters of the best feasible architecture; lower is better;
`results/report.md` has all four settings plus success rates and gaps to the
global optimum.)

What the table says:

- **The gate helps a little, the outer loop hurts a lot.** Turning the gate on
  inside a population search (`REA + gate`) improves on plain REA at tau=92 by
  about 2-3%, and ties it at tau=93/300. But the demo's
  `(1+lambda)` elitist wrapper stalls: at tau=92 it finds 0.291 M at 100
  evaluations and still 0.291 M at 300, because every shrink from the boundary
  is rejected and nothing escapes.
- **TPE wins** the easy budget and ties at the hard one, which is what one
  expects from a model-based method in five categorical dimensions.
- This benchmark has **no repair step** - accuracy is a table lookup - so it
  measures the search rule alone, not the full method. That is the motivation
  for the second benchmark.

## 2. CIFAR-10, end to end

A 6-layer CNN whose six layer widths are searched in {8, 16, 32, 64} (4,096
architectures), 10k training images from CIFAR-10, and two budgets expressed as
a validation-accuracy threshold. The x-axis is training epochs actually spent,
so a warm-started repair is charged what it costs.

Budget: 60 epochs, counted as training epochs actually spent.

| method | eps = 0.62 width | eps = 0.68 width | feasible at 0.68 |
|---|---|---|---|
| BGSE (repair + gate) | 120 [88-136] | 192 [176-256] | 5/5 |
| Static-mutation EA (repair) | **104 [96-120]** | 192 [176-256] | 5/5 |
| Random search | 192 [136-256] | 288 (one seed) | 1/5 |
| BGSE-NoRepair | 192 [192-192] | - | **0/5** |

Sum of the six layer widths, median across five seeds. The full table -
including test accuracy before and after a from-scratch retrain - and the
figure are in `results/cifar_report.md` and `results/fig_cifar_width.png`.

What it shows: repair is the load-bearing step. Without it the strict budget is
never met and the loose one never shrinks below the mother; static mutation
with repair ties BGSE, so the gate is not what carries the method.
