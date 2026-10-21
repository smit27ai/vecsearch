import numpy as np

from bench.run import recall_at_k
from vecsearch import BruteForceIndex
from vecsearch.datasets import synthetic


def test_hand_made_points():
    index = BruteForceIndex(dim=2)
    index.add([[0, 0], [10, 10], [1, 1], [-3, 0]])
    ids, dists = index.search(np.array([0.9, 0.9], dtype=np.float32), k=2)
    assert ids.tolist() == [2, 0]
    np.testing.assert_allclose(dists, [0.02, 1.62], rtol=1e-5)


def test_add_in_two_batches_keeps_ids():
    index = BruteForceIndex(dim=2)
    index.add([[0, 0], [5, 5]])
    index.add([[1, 1]])
    assert len(index) == 3
    assert index.search(np.array([1, 1], dtype=np.float32), k=1)[0].tolist() == [2]


def test_k_larger_than_index():
    index = BruteForceIndex(dim=2)
    index.add([[3, 3], [0, 0]])
    ids, _ = index.search(np.array([0, 0], dtype=np.float32), k=10)
    assert ids.tolist() == [1, 0]


def test_recall_is_one_against_ground_truth():
    base, queries, gt = synthetic(n=2000, n_queries=50, dim=32, k=10)
    index = BruteForceIndex(dim=32)
    index.add(base)
    recalls = [recall_at_k(index.search(q, 10)[0], g, 10) for q, g in zip(queries, gt)]
    assert np.mean(recalls) == 1.0


def test_batch_matches_single():
    base, queries, _ = synthetic(n=2000, n_queries=50, dim=32, k=10)
    index = BruteForceIndex(dim=32)
    index.add(base)
    batch = index.search_batch(queries, 10)
    single = np.stack([index.search(q, 10)[0] for q in queries])
    np.testing.assert_array_equal(batch, single)


def test_memory_bytes():
    index = BruteForceIndex(dim=128)
    index.add(np.zeros((100, 128)))
    assert index.memory_bytes() == 100 * 128 * 4
