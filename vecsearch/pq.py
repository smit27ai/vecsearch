"""Product Quantization: store each vector in m bytes instead of dim * 4.

Crayon-box analogy: instead of storing an exact colour, store "closest to
crayon #37" from a box of 256 crayons. That is one byte.

PQ does this piecewise:
    1. Split each vector into m chunks (sub-vectors) of dim/m numbers.
    2. For each chunk position, run k-means with 256 centers: the codebook.
    3. Store each chunk as the id (0-255) of its nearest center: 1 byte.
For 128-dim SIFT with m=8 that is 8 bytes per vector instead of 512.

Search uses asymmetric distance computation (ADC): the query stays exact.
For each chunk position j, a table holds the distance from the query's
chunk j to all 256 centers. The distance to a stored vector is then the
sum of m table lookups, with no multiplications.

PQ distances are approximate. Re-ranking fixes most of that: take the
best `rerank` candidates by PQ distance and re-score them with the original
vectors. Those originals must then be kept somewhere; here they stay in
RAM and count toward memory (real systems often keep them on disk).
"""

import numpy as np

from .kmeans import kmeans, nearest_center


def top_k(d, k):
    """Indices of the k smallest values of d, smallest first."""
    k = min(k, len(d))
    if k < len(d):
        idx = np.argpartition(d, k)[:k]
    else:
        idx = np.arange(len(d))
    return idx[np.argsort(d[idx])]


class ProductQuantizer:
    """The codec: train codebooks, encode, decode, and ADC distances."""

    def __init__(self, dim, m=8, ksub=256, seed=0):
        if dim % m:
            raise ValueError(f"dim={dim} is not divisible by m={m}")
        if ksub > 256:
            raise ValueError("ksub must be <= 256 so each code fits in one byte")
        self.dim, self.m, self.ksub, self.dsub = dim, m, ksub, dim // m
        self.seed = seed
        self.codebooks = None  # (m, ksub, dsub)

    def _chunk(self, x, j):
        return x[:, j * self.dsub:(j + 1) * self.dsub]

    def train(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.codebooks = np.stack([
            kmeans(self._chunk(x, j), self.ksub, seed=self.seed + j) for j in range(self.m)
        ])

    def encode(self, x):
        """(N, dim) float32 -> (N, m) uint8 codes."""
        x = np.asarray(x, dtype=np.float32)
        codes = np.empty((len(x), self.m), dtype=np.uint8)
        for j in range(self.m):
            codes[:, j] = nearest_center(self._chunk(x, j), self.codebooks[j])
        return codes

    def decode(self, codes):
        """(N, m) codes -> (N, dim) approximate vectors."""
        return np.concatenate([self.codebooks[j][codes[:, j]] for j in range(self.m)], axis=1)

    def distance_table(self, q):
        """(m, ksub): squared distance from each query chunk to each center.

        Costs m * ksub * dsub = ksub * dim multiply-adds, i.e. the same as
        ksub full distance computations.
        """
        chunks = np.asarray(q, dtype=np.float32).reshape(self.m, 1, self.dsub)
        return ((self.codebooks - chunks) ** 2).sum(2)

    @staticmethod
    def adc(table, codes_t):
        """Approximate distances to every stored vector.

        codes_t is (m, N): codes stored chunk-major, so each step reads one
        contiguous row. Adding one (N,) array per chunk avoids building an
        (N, m) temporary (the large-temporary slowdown found in Phase 2).
        """
        d = table[0][codes_t[0]]  # fancy indexing returns a new array
        for j in range(1, len(codes_t)):
            d += table[j][codes_t[j]]
        return d

    def memory_bytes(self):
        return self.codebooks.nbytes if self.codebooks is not None else 0


class PQIndex:
    """Flat PQ: scan every code with ADC, optionally re-rank with originals."""

    def __init__(self, dim, m=8, ksub=256, keep_vectors=False, rerank=0, seed=0):
        self.dim = dim
        self.pq = ProductQuantizer(dim, m, ksub, seed)
        self.keep_vectors = keep_vectors
        self.rerank = rerank  # 0 = no re-ranking
        self.codes_t = np.empty((m, 0), dtype=np.uint8)
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.ndis = 0  # scored candidates (ADC or exact), counted by the benchmark

    def __len__(self):
        return self.codes_t.shape[1]

    def train(self, x):
        self.pq.train(x)

    def add(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.dim)
        if self.pq.codebooks is None:
            self.train(x)
        self.codes_t = np.hstack([self.codes_t, self.pq.encode(x).T])
        if self.keep_vectors:
            self.vectors = np.vstack([self.vectors, x])

    def search(self, q, k):
        q = np.asarray(q, dtype=np.float32)
        d = self.pq.adc(self.pq.distance_table(q), self.codes_t)
        self.ndis += self.pq.ksub + len(d)  # table (= ksub distances) + one ADC per code
        if not self.rerank:
            top = top_k(d, k)
            return top, d[top]
        if not self.keep_vectors:
            raise ValueError("rerank needs keep_vectors=True")
        cand = top_k(d, max(self.rerank, k))
        exact = ((self.vectors[cand] - q) ** 2).sum(1)
        self.ndis += len(cand)
        order = top_k(exact, k)
        return cand[order], exact[order]

    def memory_bytes(self):
        return self.codes_t.nbytes + self.pq.memory_bytes() + self.vectors.nbytes


class IVFPQIndex:
    """IVF + PQ: probe the nearest clusters, score their codes with ADC.

    PQ encodes the residual (vector - its cluster center), not the vector.
    Residuals are small and centered on zero, so the same 256 centers per
    chunk describe them more precisely than they would raw vectors.
    For a probed list with center c, ||q - (c + r)||^2 = ||(q - c) - r||^2,
    so the distance table is built from q - c, once per probed list.
    """

    def __init__(self, dim, nlist, m=8, ksub=256, nprobe=1, seed=0):
        self.dim, self.nlist, self.nprobe, self.seed = dim, nlist, nprobe, seed
        self.pq = ProductQuantizer(dim, m, ksub, seed)
        self.centers = None
        self.list_ids = [np.empty(0, dtype=np.int64) for _ in range(nlist)]
        self.list_codes = [np.empty((m, 0), dtype=np.uint8) for _ in range(nlist)]  # (m, n_c)
        self.n = 0
        self.ndis = 0

    def __len__(self):
        return self.n

    def train(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.centers = kmeans(x, self.nlist, seed=self.seed)
        assign = nearest_center(x, self.centers)
        self.pq.train(x - self.centers[assign])

    def add(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.dim)
        if self.centers is None:
            self.train(x)
        ids = np.arange(self.n, self.n + len(x))
        assign = nearest_center(x, self.centers)
        codes = self.pq.encode(x - self.centers[assign])
        for c in range(self.nlist):
            mask = assign == c
            if mask.any():
                self.list_ids[c] = np.concatenate([self.list_ids[c], ids[mask]])
                self.list_codes[c] = np.hstack([self.list_codes[c], codes[mask].T])
        self.n += len(x)

    def search(self, q, k):
        q = np.asarray(q, dtype=np.float32)
        cd = ((self.centers - q) ** 2).sum(1)
        nprobe = min(self.nprobe, self.nlist)
        probe = np.argpartition(cd, nprobe - 1)[:nprobe]
        ids, dists = [], []
        for c in probe:
            if len(self.list_ids[c]) == 0:
                continue
            table = self.pq.distance_table(q - self.centers[c])
            ids.append(self.list_ids[c])
            dists.append(self.pq.adc(table, self.list_codes[c]))
        n_scored = sum(len(i) for i in ids)
        self.ndis += self.nlist + len(ids) * self.pq.ksub + n_scored
        if not ids:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        ids, d = np.concatenate(ids), np.concatenate(dists)
        top = top_k(d, k)
        return ids[top], d[top]

    def memory_bytes(self):
        return (
            (self.centers.nbytes if self.centers is not None else 0)
            + self.pq.memory_bytes()
            + sum(c.nbytes for c in self.list_codes)
            + sum(i.nbytes for i in self.list_ids)
        )
