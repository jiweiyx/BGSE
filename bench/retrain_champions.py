"""Retrain every champion from scratch and record its test accuracy.

The search-time model of a BGSE run is a warm-started, repaired network: it has
seen 8 + k epochs of training but never a clean full schedule. Comparing that
against a from-scratch baseline would understate it. This script takes each
run's winning width vector, trains it from scratch for the standard 8 epochs,
and stores `retrained_test` next to the champion.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cifar_width_bgse import (DEVICE, EPOCHS, Net, evaluate, load_cifar,  # noqa: E402
                              train)

SRC = os.path.join(HERE, "results", "cifar_width.json")


def main():
    data = load_cifar()
    d = json.load(open(SRC))
    for tau in d:
        for method in d[tau]:
            for i, run in enumerate(d[tau][method]):
                ch = run["champion"]
                if not ch:
                    continue
                net = Net(ch["widths"]).to(DEVICE)
                train(net, data, EPOCHS, seed=1000 + i)
                ch["retrained_test"] = evaluate(net, data["test_full"])
                print(f"tau {tau} {method:14s} seed {i}  "
                      f"width {ch['params']:3d}  retrained test {ch['retrained_test']:.4f}",
                      flush=True)
                with open(SRC, "w") as f:
                    json.dump(d, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
