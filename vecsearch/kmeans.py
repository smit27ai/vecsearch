"""k-means clustering: split vectors into groups around "centers".

    1. Pick n random vectors as the starting centers.
    2. Assign every vector to its nearest center.
    3. Move each center to the average of the vectors assigned to it.
    4. Repeat 2-3.

IVF uses the centers as its "library sections"; Product Quantization
(Phase 4) uses k-means again to build its codebooks.
"""

import numpy as np


def nearest_center(x, centers):
    """Index of the closest center for every row of x (squared L2).

    Uses ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, so all distances come
    from one matrix multiply instead of a Python loop.
    """
    d = (x**2).sum(1)[:, None] - 2 * x @ centers.T + (centers**2).sum(1)[None, :]
    return d.argmin(1)


def kmeans(x, n_clusters, iters=20, seed=0):
    x = np.asarray(x, dtype=np.float32)
    if n_clusters > len(x):
        raise ValueError(f"need at least {n_clusters} vectors, got {len(x)}")
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), n_clusters, replace=False)].copy()
    for _ in range(iters):
        assign = nearest_center(x, centers)
        for c in range(n_clusters):
            members = x[assign == c]
            if len(members):  # an empty cluster keeps its old center
                centers[c] = members.mean(axis=0)
    return centers
