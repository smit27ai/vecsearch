"""VectorStore: the database layer on top of an index.

An index only knows vectors numbered 0, 1, 2, ... A real app needs more:

    string ids     "doc3_chunk7" instead of 4211
    metadata       {"chapter": 3, "source": "dsa.pdf"} stored with each vector
    filters        "only chunks from chapter 3"
    delete         remove an item; update by adding the same id again
    save / load    keep the index on disk and reload it without rebuilding
    cosine         normalize vectors so L2 ranking equals cosine ranking

Deletes use tombstones: the item is marked dead and never returned, but its
vector stays in the index. In HNSW a node is also part of the roads between
other nodes, and removing it could cut them off. compact() rebuilds the
index from live items once many are dead.

Filters: the store turns the filter into a bool mask (one entry per
internal id) and passes it to the index. When few items match, it scans
just those exactly instead of walking the graph, because walking a graph
where 99% of nodes are filtered out wastes most of its steps.
"""

import json
import os
from pathlib import Path

import numpy as np

from .brute import BruteForceIndex
from .hnsw import HNSWIndex

INDEX_TYPES = {"flat": BruteForceIndex, "hnsw": HNSWIndex}
FORMAT_VERSION = 1


def _matches(meta, flt):
    """{"chapter": 3}: equals. {"chapter": [3, 4]}: one of."""
    for key, want in flt.items():
        have = meta.get(key)
        if isinstance(want, list):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class VectorStore:
    def __init__(self, dim, index="hnsw", normalize=True, exact_below=1000, **index_params):
        """
        index:        "hnsw" (fast, approximate) or "flat" (exact)
        normalize:    store unit-length vectors, so results rank by cosine
                      similarity (what sentence-transformers embeddings use)
        exact_below:  if a filter matches fewer items than this, scan them
                      exactly instead of searching the graph
        """
        self.dim = dim
        self.index_type = index
        self.normalize = normalize
        self.exact_below = exact_below
        self.index_params = index_params
        self.index = INDEX_TYPES[index](dim, **index_params)
        self.ids = []  # internal id -> external string id
        self.pos = {}  # external id -> internal id (live items only)
        self.meta = []  # internal id -> metadata dict
        self.alive = np.zeros(0, dtype=bool)

    def __len__(self):
        return len(self.pos)

    def __contains__(self, item_id):
        return item_id in self.pos

    # ---------- write ----------

    def _prepare(self, vectors):
        x = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        if self.normalize:
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("cannot normalize a zero vector")
            x = x / norms
        return x

    def add(self, ids, vectors, metadata=None):
        """Add items. An id that already exists is replaced (upsert)."""
        ids = [str(i) for i in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in one add() call")
        x = self._prepare(vectors)
        if len(x) != len(ids):
            raise ValueError(f"{len(ids)} ids but {len(x)} vectors")
        metadata = metadata or [{} for _ in ids]
        if len(metadata) != len(ids):
            raise ValueError(f"{len(ids)} ids but {len(metadata)} metadata entries")
        self.delete([i for i in ids if i in self.pos])  # upsert: old version becomes a tombstone

        start = len(self.ids)
        self.index.add(x)
        for offset, (item_id, m) in enumerate(zip(ids, metadata)):
            self.pos[item_id] = start + offset
            self.ids.append(item_id)
            self.meta.append(dict(m))
        self.alive = np.concatenate([self.alive, np.ones(len(ids), dtype=bool)])

    def delete(self, ids):
        """Mark items deleted. Returns how many existed."""
        found = 0
        for item_id in ids:
            internal = self.pos.pop(str(item_id), None)
            if internal is not None:
                self.alive[internal] = False
                found += 1
        return found

    def compact(self):
        """Rebuild the index from live items only. Returns how many tombstones were dropped."""
        dead = len(self.ids) - len(self.pos)
        if dead == 0:
            return 0
        live = np.flatnonzero(self.alive)
        vectors = self.index.stored_vectors()[live].copy()
        ids = [self.ids[i] for i in live]
        meta = [self.meta[i] for i in live]
        self.index = INDEX_TYPES[self.index_type](self.dim, **self.index_params)
        self.ids, self.pos, self.meta = [], {}, []
        self.alive = np.zeros(0, dtype=bool)
        normalize, self.normalize = self.normalize, False  # already normalized
        try:
            self.add(ids, vectors, meta)
        finally:
            self.normalize = normalize
        return dead

    # ---------- read ----------

    def get(self, item_id):
        internal = self.pos.get(str(item_id))
        if internal is None:
            return None
        return {"id": self.ids[internal], "metadata": self.meta[internal],
                "vector": self.index.stored_vectors()[internal].tolist()}

    def allowed_mask(self, flt=None):
        if not flt:
            return self.alive
        match = np.fromiter((_matches(m, flt) for m in self.meta), dtype=bool, count=len(self.meta))
        return self.alive & match

    def search(self, vector, k=10, filter=None):
        """Top k live items matching `filter`. Returns [{id, score, distance, metadata}]."""
        q = self._prepare(vector)[0]
        allowed = self.allowed_mask(filter)
        n_allowed = int(allowed.sum())
        if n_allowed == 0:
            return []
        if n_allowed < self.exact_below:
            # few matches: an exact scan over just those is cheap and never misses
            cand = np.flatnonzero(allowed)
            d = ((self.index.stored_vectors()[cand] - q) ** 2).sum(1)
            order = np.argsort(d)[:k]
            internal, dists = cand[order], d[order]
        else:
            internal, dists = self.index.search(q, k, allowed=allowed)
        return [self._result(int(i), float(d)) for i, d in zip(internal, dists)]

    def _result(self, internal, dist):
        out = {"id": self.ids[internal], "distance": dist, "metadata": self.meta[internal]}
        if self.normalize:
            out["score"] = 1 - dist / 2  # for unit vectors: ||a-b||^2 = 2 - 2 cos(a, b)
        return out

    def stats(self):
        return {"items": len(self), "tombstones": len(self.ids) - len(self),
                "index": self.index_type, "dim": self.dim, "normalize": self.normalize}

    # ---------- save / load ----------

    def save(self, path):
        """Write everything to one .npz file, atomically.

        The data goes to a temporary file first and is then renamed over the
        old one. A crash mid-save leaves the old file untouched instead of a
        half-written one.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        settings, arrays = self.index.state()
        header = {
            "format_version": FORMAT_VERSION,
            "store": {"dim": self.dim, "index": self.index_type, "normalize": self.normalize,
                      "exact_below": self.exact_below, "index_params": self.index_params},
            "index": settings,
            "ids": self.ids,
            "meta": self.meta,
        }
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, header=np.array(json.dumps(header)), alive=self.alive,
                     **{f"index_{k}": v for k, v in arrays.items()})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            header = json.loads(str(f["header"]))
            if header["format_version"] != FORMAT_VERSION:
                raise ValueError(f"unsupported file version {header['format_version']}")
            alive = f["alive"]
            arrays = {k[len("index_"):]: f[k] for k in f.files if k.startswith("index_")}
        s = header["store"]
        store = cls(s["dim"], index=s["index"], normalize=s["normalize"],
                    exact_below=s["exact_below"], **s["index_params"])
        store.index = INDEX_TYPES[s["index"]].from_state(header["index"], arrays)
        store.ids, store.meta, store.alive = header["ids"], header["meta"], alive.copy()
        store.pos = {i: n for n, i in enumerate(store.ids) if alive[n]}
        return store
