"""Exact search: compare the query with every vector.

Slow (O(N * d) per query) but always correct. Every approximate index in
this project is measured against it.
"""

import numpy as np


class BruteForceIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.norms = np.empty(0, dtype=np.float32)  # ||v||^2, computed once on add

    def __len__(self):
        return len(self.vectors)

    def add(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.dim)
        self.vectors = np.vstack([self.vectors, x])
        self.norms = np.concatenate([self.norms, (x**2).sum(1)])

    def search(self, q, k):
        """Return (ids, squared L2 distances) of the k closest vectors, closest first."""
        # ||v - q||^2 = ||v||^2 - 2 v.q + ||q||^2. The matrix-vector product
        # avoids building an (N, d) temporary like (vectors - q) ** 2 does.
        q = np.asarray(q, dtype=np.float32)
        d = self.norms - 2 * (self.vectors @ q) + q @ q
        k = min(k, len(d))
        if k < len(d):
            idx = np.argpartition(d, k)[:k]  # top k in O(N), unsorted
        else:
            idx = np.arange(len(d))
        idx = idx[np.argsort(d[idx])]  # sort only those k
        return idx, d[idx]

    def search_batch(self, Q, k):
        """Search many queries at once with one matrix multiply.

        Uses ||q - v||^2 = ||q||^2 - 2 q.v + ||v||^2. Returns ids only.
        """
        Q = np.asarray(Q, dtype=np.float32)
        d = (Q**2).sum(1)[:, None] - 2 * Q @ self.vectors.T + self.norms[None, :]
        k = min(k, d.shape[1])
        if k < d.shape[1]:
            idx = np.argpartition(d, k, axis=1)[:, :k]
        else:
            idx = np.tile(np.arange(d.shape[1]), (len(Q), 1))
        order = np.take_along_axis(d, idx, 1).argsort(1)
        return np.take_along_axis(idx, order, 1)

    def memory_bytes(self):
        return self.vectors.nbytes + self.norms.nbytes
