# vecsearch

A vector search engine built from scratch, to learn what FAISS and HNSW-based
databases do underneath. Every index is measured on the same benchmark:
recall@10, queries per second, build time, and memory.

## Status

| Phase | Index | Status |
|---|---|---|
| 0 | dataset loaders, ground truth | done |
| 1 | brute force + benchmark tool | done |
| 2 | k-means + IVF | next |
| 3 | HNSW | |
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
```

Results are written to `bench/results/<dataset>.json`.

## Layout

```
vecsearch/datasets.py   .fvecs/.ivecs readers, synthetic data, exact ground truth
vecsearch/brute.py      exact search (the reference every other index is checked against)
bench/run.py            the benchmark tool; each new index registers in INDEXES
tests/                  pytest
```

## Notes

- Distances are squared L2 (no `sqrt`: it does not change the ranking).
- Ground truth for synthetic data is computed in float64 with a full sort,
  separately from the brute-force index, so the index is tested against an
  independent reference.
