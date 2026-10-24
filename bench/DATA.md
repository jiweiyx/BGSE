# Benchmark data (not committed)

`bench/data/` is gitignored. To reproduce the studies, fetch these two:

## NATS-Bench, size space (CIFAR-10 ground truth)

The size space is 5 stages x 8 widths = 32,768 architectures, each trained on
CIFAR-10 for 12 and 90 epochs by the benchmark authors.

```bash
# official file: NATS-sss-v1_0-50262.pickle.pbz2 / -simple.tar from
# https://github.com/D-X-Y/NATS-Bench  (Google Drive link in its README)
gdown 1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA -O bench/data/NATS-sss-simple.tar
tar -xf bench/data/NATS-sss-simple.tar -C bench/data
```

After extraction the directory `NATS-sss-v1_0-50262-simple/` must exist.

## CIFAR-10

```bash
curl -L -o bench/data/cifar-10-python.tar.gz \
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf bench/data/cifar-10-python.tar.gz -C bench/data
```

After extraction the directory `cifar-10-batches-py/` must exist.
