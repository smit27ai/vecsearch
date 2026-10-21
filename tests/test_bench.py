import json

from bench.plot import plot
from bench.run import recall_at_k, run
from vecsearch import BruteForceIndex, IVFIndex
from vecsearch.datasets import synthetic


def test_recall_at_k_example():
    # the example from the roadmap: 8 of the true top 10 found
    assert recall_at_k([1, 2, 3, 4, 5, 6, 7, 8, 42, 99], list(range(1, 11)), 10) == 0.8


def test_run_reports_all_fields():
    base, queries, gt = synthetic(n=500, n_queries=10, dim=8, k=10)
    [row] = run("brute", BruteForceIndex(8), base, queries, gt, k=10)
    assert row["recall"] == 1.0
    assert row["qps"] > 0 and row["build_s"] >= 0
    assert row["memory_bytes"] == 500 * 8 * 4 + 500 * 4  # vectors + norms


def test_run_sweeps_search_settings_on_one_build():
    base, queries, gt = synthetic(n=500, n_queries=10, dim=8, k=10)
    sweep = [{"nprobe": 1}, {"nprobe": 10}]
    rows = run("ivf", IVFIndex(8, nlist=10), base, queries, gt, 10, {"nlist": 10}, sweep)
    assert [r["search"] for r in rows] == sweep
    assert rows[0]["build_s"] == rows[1]["build_s"]  # built once
    assert rows[1]["recall"] == 1.0  # nprobe == nlist


def test_plot_writes_png(tmp_path):
    base, queries, gt = synthetic(n=500, n_queries=10, dim=8, k=10)
    rows = run("brute", BruteForceIndex(8), base, queries, gt, 10)
    rows += run("ivf", IVFIndex(8, nlist=10), base, queries, gt, 10, {"nlist": 10},
                [{"nprobe": 1}, {"nprobe": 4}])
    src = tmp_path / "r.json"
    src.write_text(json.dumps({"dataset": "test", "results": rows}))
    plot(src, tmp_path / "r.png")
    assert (tmp_path / "r.png").stat().st_size > 1000
