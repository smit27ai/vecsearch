import numpy as np
import pytest

from vecsearch.kmeans import kmeans, nearest_center


def blobs(seed=0):
    """Three tight, far-apart groups of 100 points each."""
    rng = np.random.default_rng(seed)
    true_centers = np.array([[0, 0], [20, 0], [0, 20]], dtype=np.float32)
    x = np.vstack([c + rng.normal(0, 0.5, (100, 2)) for c in true_centers])
    return x.astype(np.float32), true_centers


def test_nearest_center():
    centers = np.array([[0, 0], [10, 10]], dtype=np.float32)
    x = np.array([[1, 1], [9, 8], [4, 4], [6, 6]], dtype=np.float32)
    assert nearest_center(x, centers).tolist() == [0, 1, 0, 1]


def test_finds_the_three_blobs():
    x, true_centers = blobs()
    centers = kmeans(x, 3, seed=1)
    # every true center has a found center within 1.0 of it
    for t in true_centers:
        assert np.min(np.linalg.norm(centers - t, axis=1)) < 1.0


def test_same_seed_same_result():
    x, _ = blobs()
    np.testing.assert_array_equal(kmeans(x, 3, seed=5), kmeans(x, 3, seed=5))


def test_duplicate_points_do_not_crash():
    # 4 identical points: some clusters end up empty and must keep their center
    x = np.zeros((4, 3), dtype=np.float32)
    centers = kmeans(x, 2)
    assert centers.shape == (2, 3) and np.isfinite(centers).all()


def test_too_many_clusters():
    with pytest.raises(ValueError):
        kmeans(np.zeros((3, 2)), 5)
