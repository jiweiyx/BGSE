"""Cache the NATS-Bench size space (CIFAR-10) as a flat table.

The size search space is 5 stages x 8 candidate widths {8,16,24,32,40,48,56,64},
i.e. 32768 architectures. Every architecture was trained on CIFAR-10 for 12 and
90 epochs; the benchmark stores the resulting test accuracy, parameter count and
FLOPs, so a search method can be scored without training anything.

Output: bench/results/nats_sss_cifar10.npz
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
import nats_bench  # noqa: E402

DATA = os.path.join(HERE, "data", "NATS-sss-v1_0-50262-simple")
OUT = os.path.join(HERE, "results", "nats_sss_cifar10.npz")


def main():
    api = nats_bench.create(DATA, "sss", fast_mode=True, verbose=False)
    n = len(api)
    widths = np.zeros((n, 5), dtype=np.int8)
    params = np.zeros(n, dtype=np.float32)
    flops = np.zeros(n, dtype=np.float32)
    acc12 = np.zeros(n, dtype=np.float32)
    acc90 = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    for i in range(n):
        widths[i] = [int(x) for x in api.arch(i).split(":")]
        cost = api.get_cost_info(i, "cifar10")
        params[i] = cost["params"]
        flops[i] = cost["flops"]
        acc12[i] = api.get_more_info(i, "cifar10", hp="12", is_random=True)["test-accuracy"]
        acc90[i] = api.get_more_info(i, "cifar10", hp="90", is_random=True)["test-accuracy"]
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:5d}/{n}  {time.time() - t0:5.1f}s", flush=True)

    np.savez_compressed(OUT, widths=widths, params=params, flops=flops,
                        acc12=acc12, acc90=acc90)
    print(f"wrote {OUT}  ({time.time() - t0:.1f}s)")

    for tau in (90.0, 92.0, 93.0, 93.5, 94.0):
        ok = acc90 >= tau
        best = params[ok].min() if ok.any() else float("nan")
        print(f"  tau={tau:5.1f}: feasible {ok.sum():5d}/{n}  "
              f"cheapest {best:7.4f}M  max acc {acc90.max():.2f}")


if __name__ == "__main__":
    main()
