import numpy as np

from vecsearch.datasets import exact_ground_truth, read_fvecs, synthetic, write_fvecs


def test_fvecs_roundtrip(tmp_path):
    x = np.random.default_rng(0).standard_normal((7, 5)).astype(np.float32)
    path = tmp_path / "x.fvecs"
    write_fvecs(path, x)
    np.testing.assert_array_equal(read_fvecs(path), x)


def test_synthetic_shapes():
    base, queries, gt = synthetic(n=500, n_queries=10, dim=16, k=20)
    assert base.shape == (500, 16) and base.dtype == np.float32
    assert queries.shape == (10, 16)
    assert gt.shape == (10, 20)


def test_ground_truth_on_known_points():
    base = np.array([[0, 0], [1, 0], [5, 5], [0, 2]], dtype=np.float32)
    queries = np.array([[0.1, 0.1], [4, 4]], dtype=np.float32)
    gt = exact_ground_truth(base, queries, k=2)
    assert gt[0].tolist() == [0, 1]
    assert gt[1].tolist() == [2, 3]  # (0,2) is at 16+4=20, (1,0) at 9+16=25
