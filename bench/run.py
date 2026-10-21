"""Benchmark tool: measures any index on the same four numbers.

    recall@k      how correct (1.0 = found every true neighbour)
    qps           queries per second, one query at a time (median of 3 runs)
    build_s       seconds to add all base vectors (includes training)
    memory_bytes  what the index reports it holds in RAM

Each index is built once per build setting, then every search setting is
measured on that same build (e.g. IVF: build with nlist, sweep nprobe).

Usage:
    python -m bench.run                      # synthetic data, all indexes
    python -m bench.run --dataset siftsmall  # after downloading SIFT10K
    python -m bench.run --index ivf
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from vecsearch import BruteForceIndex, IVFIndex
from vecsearch import datasets

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NPROBES = [1, 2, 4, 8, 16, 32, 64]

# name -> builds:  list of build settings (one index is built per setting)
#         make:    (dim, **build_setting) -> index
#         sweep:   search settings, applied with setattr on the built index
INDEXES = {
    "brute": {
        "builds": [{}],
        "make": lambda dim: BruteForceIndex(dim),
        "sweep": [{}],
    },
    "ivf": {
        "builds": [{"nlist": n} for n in (50, 100, 200)],
        "make": lambda dim, nlist: IVFIndex(dim, nlist),
        "sweep": [{"nprobe": p} for p in NPROBES],
    },
}


def recall_at_k(found, truth, k):
    return len(set(np.asarray(found[:k]).tolist()) & set(np.asarray(truth[:k]).tolist())) / k


def measure(index, queries, gt, k, repeats=3):
    for q in queries[:5]:  # warm-up, not timed
        index.search(q, k)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        results = [index.search(q, k)[0] for q in queries]
        times.append(time.perf_counter() - t)
    recall = float(np.mean([recall_at_k(r, g, k) for r, g in zip(results, gt)]))
    return recall, len(queries) / statistics.median(times)


def run(name, index, base, queries, gt, k=10, build=None, sweep=({},)):
    """Build `index` once, then measure it at every search setting. Returns rows."""
    t = time.perf_counter()
    index.add(base)
    build_s = time.perf_counter() - t

    rows = []
    for search in sweep:
        for key, value in search.items():
            setattr(index, key, value)
        recall, qps = measure(index, queries, gt, k)
        rows.append({
            "index": name,
            "build": build or {},
            "search": dict(search),
            "k": k,
            "recall": round(recall, 4),
            "qps": round(qps, 1),
            "build_s": round(build_s, 4),
            "memory_bytes": int(index.memory_bytes()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic", choices=sorted(datasets.LOADERS))
    parser.add_argument("--index", default="all", choices=["all", *INDEXES])
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    base, queries, gt = datasets.load(args.dataset)
    print(f"dataset={args.dataset} base={base.shape} queries={queries.shape} k={args.k}")

    out = RESULTS_DIR / f"{args.dataset}.json"
    names = list(INDEXES) if args.index == "all" else [args.index]
    # keep earlier results for indexes not re-run this time
    rows = []
    if out.exists() and args.index != "all":
        rows = [r for r in json.loads(out.read_text())["results"] if r["index"] not in names]

    for name in names:
        spec = INDEXES[name]
        for build in spec["builds"]:
            index = spec["make"](base.shape[1], **build)
            for row in run(name, index, base, queries, gt, args.k, build, spec["sweep"]):
                print(
                    f"{name:6} {json.dumps(build):15} {json.dumps(row['search']):15} "
                    f"recall@{args.k}={row['recall']:.4f}  qps={row['qps']:>9.1f}  "
                    f"build={row['build_s']:.3f}s  mem={row['memory_bytes'] / 1e6:.1f}MB"
                )
                rows.append(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "dataset": args.dataset,
        "machine": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "results": rows,
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
