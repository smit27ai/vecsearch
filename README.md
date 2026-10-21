# vecsearch

A vector search engine built from scratch, to learn what FAISS and HNSW-based
databases do underneath. Every index is measured on the same benchmark:
recall@10, queries per second, build time, and memory.

## Status

| Phase | Index | Status |
|---|---|---|
| 0 | dataset loaders, ground truth | done |
| 1 | brute force + benchmark tool | done |
| 2 | k-means + IVF | done |
| 3 | HNSW | next |
| 4 | Product Quantization | |
| 5 | persistence, delete, filter, API | |
| 6 | comparison with FAISS / hnswlib | |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux / macOS
```

## Commands

```bash
python -m pytest                          # run tests (synthetic data, no download)
python -m bench.run                       # benchmark on synthetic data
python -m vecsearch.datasets download     # download SIFT10K (~5 MB) into data/
python -m bench.run --dataset siftsmall   # benchmark on SIFT10K
python -m bench.plot --dataset siftsmall  # draw the recall vs QPS chart
```

Results are written to `bench/results/<dataset>.json` and the chart to
`bench/results/<dataset>.png`.

## Results so far (SIFT10K, k=10, one query at a time, single machine)

![recall vs QPS on SIFT10K](bench/results/siftsmall.png)

| index | setting | recall@10 | queries/sec |
|---|---|---|---|
| brute | | 1.000 | 6,212 |
| IVF nlist=50 | nprobe=4 | 0.916 | 23,902 |
| IVF nlist=50 | nprobe=8 | 0.985 | 10,348 |
| IVF nlist=50 | nprobe=16 | 1.000 | 6,880 |
| IVF nlist=100 | nprobe=8 | 0.951 | 17,971 |

Speeds vary by roughly ±20% between runs; recall is deterministic.

## What I learned

**IVF needs data with structure.** On random Gaussian vectors IVF is
nearly useless (nlist=100, nprobe=8 finds 25% of the true top 10); on SIFT,
the same setting finds 95%. Random high-dimensional points are all about
equally far apart, so clusters mean nothing. Real embeddings cluster, which
is why IVF works. Benchmarks on random data alone would be misleading.

**How you compute distance mattered more than the algorithm.**
`((v - q) ** 2).sum(1)` builds a temporary array as big as all the vectors
being checked. Past about 256 KB the allocation got ~4x slower on this
Windows machine, which made IVF's speed fall off a cliff between nprobe=4
and nprobe=8. Rewriting it as `||v||^2 - 2 v.q + ||q||^2` with the norms
stored at add time (one matrix-vector product, no big temporary) made
brute force 18x faster and removed the cliff. Recall did not change.

**At 10,000 vectors, Python overhead limits IVF.** Each probed list costs a
Python loop step, so with nprobe=64 IVF is slower than brute force, and the
best speed-up at recall >= 0.95 is about 3x. The gap should grow with
dataset size; that is for the 1M-vector SIFT run.

## Layout

```
vecsearch/datasets.py   .fvecs/.ivecs readers, synthetic data, exact ground truth
vecsearch/brute.py      exact search (the reference every other index is checked against)
vecsearch/kmeans.py     k-means clustering
vecsearch/ivf.py        IVF index: search only the nprobe nearest clusters
bench/run.py            the benchmark tool; each new index registers in INDEXES
bench/plot.py           recall vs QPS chart
tests/                  pytest
```

## Notes

- Distances are squared L2 (no `sqrt`: it does not change the ranking).
- Ground truth for synthetic data is computed in float64 with a full sort,
  separately from the brute-force index, so the index is tested against an
  independent reference.
