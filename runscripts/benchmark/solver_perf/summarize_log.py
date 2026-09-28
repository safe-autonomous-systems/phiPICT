"""Summarise a capture log: solves, iterations and wall time per solve tag."""
import collections
import sys

import torch

for path in sys.argv[1:]:
    log = torch.load(path, weights_only=False)
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    for e in log:
        a = agg[(e["tag"], e["kind"], e["n"], e["n_rhs"])]
        a[0] += 1
        a[1] += max(e["iters"])
        a[2] += e["seconds"]
    tot = sum(v[2] for v in agg.values())
    print(path)
    for k, v in agg.items():
        print(f"  {k[0]:>10s} {k[1]:>8s} n={k[2]:>8d} rhs={k[3]} solves={v[0]:4d} iters={v[1]:6d} "
              f"time={v[2]:8.3f}s ({100*v[2]/tot:4.1f}%) us/iter={1e6*v[2]/max(v[1],1):9.1f}")
