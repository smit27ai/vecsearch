"""Filtered search: three ways to answer "top 10 among items matching X".

    post-filter   search the top 10 normally, then drop non-matching ones
    in-graph      HNSW walks every node but only collects matching ones
                  (HNSWIndex._search_layer0_filtered)
    exact scan    compute distances to every matching item, nothing else

A random fraction of SIFT10K items is marked as matching; the true answer
is the exact top 10 among those. Reports results returned per query,
recall@10 and distance computations per query.

Usage:
    python -m bench.filtered     # writes bench/results/filtered.json
"""

import json
from pathlib import Path

import numpy as np

from bench.run import recall_at_k
from vecsearch import HNSWIndex, datasets

OUT = Path(__file__).resolve().parent / "results" / "filtered.json"
SELECTIVITY = [0.5, 0.1, 0.01, 0.002]
K = 10


def main():
    base, queries, _ = datasets.siftsmall()
    index = HNSWIndex(base.shape[1], M=16, ef_construction=200, ef_search=40)
    index.add(base)
    norms = (base**2).sum(1)
    rng = np.random.default_rng(0)
    rows = []
    for sel in SELECTIVITY:
        allowed = rng.random(len(base)) < sel
        cand = np.flatnonzero(allowed)
        stats = {m: {"returned": 0, "recall": 0.0, "ndis": 0} for m in
                 ("post-filter", "in-graph", "exact scan")}
        for q in queries:
            d = norms[cand] - 2 * base[cand] @ q + q @ q
            truth = cand[np.argsort(d)[:K]]

            index.ndis = 0
            ids, _ = index.search(q, K)
            got = [i for i in ids if allowed[i]]
            stats["post-filter"]["ndis"] += index.ndis

            index.ndis = 0
            ids_in, _ = index.search(q, K, allowed=allowed)
            stats["in-graph"]["ndis"] += index.ndis

            for name, found in (("post-filter", got), ("in-graph", ids_in), ("exact scan", truth)):
                stats[name]["returned"] += len(found)
                stats[name]["recall"] += recall_at_k(found, truth, K)
            stats["exact scan"]["ndis"] += len(cand)

        nq = len(queries)
        for name, s in stats.items():
            row = {"selectivity": sel, "matching": int(allowed.sum()), "method": name,
                   "returned": round(s["returned"] / nq, 2), "recall": round(s["recall"] / nq, 4),
                   "dist_per_query": round(s["ndis"] / nq, 1)}
            rows.append(row)
            print(f"{sel:>6.1%} match ({row['matching']:>5})  {name:12} returned={row['returned']:5.2f}  "
                  f"recall@{K}={row['recall']:.3f}  dist/q={row['dist_per_query']:>7.0f}")
    OUT.write_text(json.dumps({"dataset": "siftsmall", "k": K, "hnsw": {"M": 16, "ef_search": 40},
                               "results": rows}, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
