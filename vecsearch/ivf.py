"""IVF (inverted file) index: search only the nearest few clusters.

Library analogy: books are shelved by section; to find a science book you
only look in the science section.

    build:  k-means gives `nlist` centers; each vector goes into the list
            of its nearest center.
    search: find the `nprobe` centers nearest the query, then brute-force
            only the vectors in those lists.

It is approximate: a query near the border of two clusters can have its
true neighbour in a list that was not probed. Raising `nprobe` fixes that
at the cost of speed; nprobe == nlist searches everything (= brute force).
"""

import numpy as np

from .kmeans import kmeans, nearest_center


class IVFIndex:
    def __init__(self, dim, nlist, nprobe=1, seed=0):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self.centers = None
        self.list_ids = [np.empty(0, dtype=np.int64) for _ in range(nlist)]
        self.list_vectors = [np.empty((0, dim), dtype=np.float32) for _ in range(nlist)]
        self.list_norms = [np.empty(0, dtype=np.float32) for _ in range(nlist)]  # ||v||^2
        self.n = 0

    def __len__(self):
        return self.n

    def train(self, x):
        self.centers = kmeans(x, self.nlist, seed=self.seed)

    def add(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.dim)
        if self.centers is None:  # train on the first batch if not trained yet
            self.train(x)
        ids = np.arange(self.n, self.n + len(x))
        assign = nearest_center(x, self.centers)
        for c in range(self.nlist):
            mask = assign == c
            if mask.any():
                self.list_ids[c] = np.concatenate([self.list_ids[c], ids[mask]])
                self.list_vectors[c] = np.vstack([self.list_vectors[c], x[mask]])
                self.list_norms[c] = np.concatenate([self.list_norms[c], (x[mask] ** 2).sum(1)])
        self.n += len(x)

    def search(self, q, k):
        q = np.asarray(q, dtype=np.float32)
        cd = ((self.centers - q) ** 2).sum(1)
        nprobe = min(self.nprobe, self.nlist)
        probe = np.argpartition(cd, nprobe - 1)[:nprobe]  # nearest clusters
        ids = np.concatenate([self.list_ids[c] for c in probe])
        if len(ids) == 0:
            return ids, np.empty(0, dtype=np.float32)
        # ||v - q||^2 = ||v||^2 - 2 v.q + ||q||^2, one list at a time.
        # (v - q) ** 2 would allocate an array as big as all probed vectors on
        # every query; past ~256 KB that allocation alone costs ~4x.
        qq = q @ q
        d = np.concatenate([self.list_norms[c] - 2 * (self.list_vectors[c] @ q) + qq
                            for c in probe])
        top = np.argsort(d)[:k]
        return ids[top], d[top]

    def list_sizes(self):
        return np.array([len(ids) for ids in self.list_ids])

    def memory_bytes(self):
        return (
            (self.centers.nbytes if self.centers is not None else 0)
            + sum(v.nbytes for v in self.list_vectors)
            + sum(i.nbytes for i in self.list_ids)
            + sum(n.nbytes for n in self.list_norms)
        )
