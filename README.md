# vecsearch

A vector search engine built from scratch, to learn what FAISS and HNSW-based
databases do underneath. Every index is measured on the same benchmark:
recall@10, queries per second, distance computations per query, build time,
and memory.

## Status

| Phase | Index | Status |
|---|---|---|
| 0 | dataset loaders, ground truth | done |
| 1 | brute force + benchmark tool | done |
| 2 | k-means + IVF | done |
| 3 | HNSW | done |
| 4 | Product Quantization | next |
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
python -m bench.plot --dataset siftsmall  # draw the charts
python -m bench.draw_graph                # draw a small 2D HNSW, layer by layer
```

Results are written to `bench/results/<dataset>.json` and the charts to
`bench/results/<dataset>.png`. A full SIFT10K run takes about 3 minutes,
mostly HNSW builds.

## Results so far (SIFT10K, k=10, one query at a time, single machine)

![benchmark charts on SIFT10K](bench/results/siftsmall.png)

Left chart: speed of this Python code. Right chart: distance computations
per query, which measures the algorithm itself, independent of language.

| index | setting | recall@10 | distances / query | queries/sec |
|---|---|---|---|---|
| brute | | 1.000 | 10,000 | 5,567 |
| IVF nlist=50 | nprobe=8 | 0.985 | 1,682 | 12,136 |
| IVF nlist=100 | nprobe=16 | 0.995 | 1,785 | 8,577 |
| HNSW M=16 | ef_search=20 | 0.980 | 300 | 3,689 |
| HNSW M=16 | ef_search=40 | 0.996 | 480 | 2,137 |

Speeds vary by roughly ±20% between runs; recall and distance counts are
deterministic.

## How HNSW searches

![HNSW layers on 400 random 2D points](docs/hnsw_layers.png)

The search starts at the single node on the top layer, takes long jumps
on the sparse upper layers, and drops a layer each time no neighbour is
closer to the query. By layer 0 it is already next to the target.

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

**HNSW does the least work but runs slowest here, and both are true.** At
about 98% recall, HNSW computes 300 distances per query, IVF 1,682 and
brute force 10,000. But brute force and IVF do a few large NumPy operations
(fast C loops), while HNSW does hundreds of small Python steps per query
(heap pushes, set lookups). At 10,000 vectors, Python overhead dominates,
so queries/sec mostly measures Python. That is why the benchmark also
counts distance computations: it is the number that would carry over to a
C++ or Rust implementation, and it is what the HNSW paper reports.

**HNSW can lose nodes.** When a node's link list is full, the paper's
algorithm trims it, and that can delete the only link into some other node.
That node can then never be found. On 1,000 random vectors:

| neighbour rule | M=4 | M=8 | M=16 |
|---|---|---|---|
| heuristic (Algorithm 4) | 8 unreachable | 1 | 0 |
| simple (closest M) | 40 | 8 | 0 |

Small M makes it worse; the heuristic, which spreads links in different
directions, makes it better. The test suite checks that M=16 reaches every
node.

**The heuristic helps less than expected on SIFT10K.** With M=16 the
heuristic and simple rules reach nearly the same recall for the same work,
and the simple rule builds about 1.8x faster. The heuristic matters more
for clustered data, where the simple rule tends to link only inside a
cluster; that is worth testing on GloVe.

## Layout

```
vecsearch/datasets.py   .fvecs/.ivecs readers, synthetic data, exact ground truth
vecsearch/brute.py      exact search (the reference every other index is checked against)
vecsearch/kmeans.py     k-means clustering
vecsearch/ivf.py        IVF index: search only the nprobe nearest clusters
vecsearch/hnsw.py       HNSW graph index (Malkov & Yashunin, Algorithms 1-5)
bench/run.py            the benchmark tool; each new index registers in INDEXES
bench/plot.py           recall vs QPS and recall vs distance-count charts
bench/draw_graph.py     2D picture of HNSW layers and a search path
docs/                   images used by this README
tests/                  pytest
```

## Notes

- Distances are squared L2 (no `sqrt`: it does not change the ranking).
- Ground truth for synthetic data is computed in float64 with a full sort,
  separately from the brute-force index, so the index is tested against an
  independent reference.
