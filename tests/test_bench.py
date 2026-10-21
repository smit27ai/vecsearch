from bench.run import recall_at_k, run
from vecsearch import BruteForceIndex
from vecsearch.datasets import synthetic


def test_recall_at_k_example():
    # the example from the roadmap: 8 of the true top 10 found
    assert recall_at_k([1, 2, 3, 4, 5, 6, 7, 8, 42, 99], list(range(1, 11)), 10) == 0.8


def test_run_reports_all_fields():
    base, queries, gt = synthetic(n=500, n_queries=10, dim=8, k=10)
    row = run("brute", BruteForceIndex(8), base, queries, gt, k=10)
    assert row["recall"] == 1.0
    assert row["qps"] > 0 and row["build_s"] >= 0
    assert row["memory_bytes"] == 500 * 8 * 4
