"""Benchmark tool: measures any index on the same four numbers.

    recall@k      how correct (1.0 = found every true neighbour)
    qps           queries per second, one query at a time
    build_s       seconds to add all base vectors
    memory_bytes  what the index reports it holds in RAM

Usage:
    python -m bench.run                      # synthetic data, all indexes
    python -m bench.run --dataset siftsmall  # after downloading SIFT10K
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from vecsearch import BruteForceIndex
from vecsearch import datasets

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# name -> list of (params, factory). Each later phase adds its index here,
# with one entry per parameter setting so the results trace a curve.
INDEXES = {
    "brute": [({}, lambda dim: BruteForceIndex(dim))],
}


def recall_at_k(found, truth, k):
    return len(set(np.asarray(found[:k]).tolist()) & set(np.asarray(truth[:k]).tolist())) / k


def run(name, index, base, queries, gt, k=10, params=None):
    t = time.perf_counter()
    index.add(base)
    build_s = time.perf_counter() - t

    for q in queries[:5]:  # warm-up, not timed
        index.search(q, k)

    t = time.perf_counter()
    results = [index.search(q, k)[0] for q in queries]
    qps = len(queries) / (time.perf_counter() - t)

    recall = float(np.mean([recall_at_k(r, g, k) for r, g in zip(results, gt)]))
    return {
        "index": name,
        "params": params or {},
        "k": k,
        "recall": round(recall, 4),
        "qps": round(qps, 1),
        "build_s": round(build_s, 4),
        "memory_bytes": int(index.memory_bytes()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic", choices=sorted(datasets.LOADERS))
    parser.add_argument("--index", default="all", choices=["all", *INDEXES])
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    base, queries, gt = datasets.load(args.dataset)
    print(f"dataset={args.dataset} base={base.shape} queries={queries.shape} k={args.k}")

    names = list(INDEXES) if args.index == "all" else [args.index]
    rows = []
    for name in names:
        for params, factory in INDEXES[name]:
            row = run(name, factory(base.shape[1]), base, queries, gt, args.k, params)
            print(
                f"{name:8} {json.dumps(params):24} recall@{args.k}={row['recall']:.4f}  "
                f"qps={row['qps']:>9.1f}  build={row['build_s']:.3f}s  "
                f"mem={row['memory_bytes'] / 1e6:.1f}MB"
            )
            rows.append(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.dataset}.json"
    out.write_text(json.dumps({
        "dataset": args.dataset,
        "machine": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "results": rows,
    }, indent=2))
    print(f"saved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
