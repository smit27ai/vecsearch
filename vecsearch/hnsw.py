"""HNSW (Hierarchical Navigable Small World) graph index.

Follows Malkov & Yashunin, "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs" (2016).
Algorithm numbers in comments refer to that paper.

Road-trip analogy: the top layer is the highway (few nodes, long jumps),
lower layers are main roads, and layer 0 holds every vector (local streets).
A search drives greedily down from the top, then searches layer 0 carefully.

    M                links per node on upper layers (2*M on layer 0)
    ef_construction  search width while inserting (better graph, slower build)
    ef_search        search width while querying (better recall, slower search)
    heuristic        neighbour choice: True = Algorithm 4 (spread links out),
                     False = Algorithm 3 (just the M closest)

One deliberate change from a line-by-line port: when a node is expanded,
the distances to all its unvisited neighbours are computed in one NumPy call
instead of one call per neighbour. The search visits the same nodes in the
same order; it only cuts Python overhead.
"""

import heapq
import math
import random

import numpy as np


class HNSWIndex:
    def __init__(self, dim, M=16, ef_construction=200, ef_search=50, heuristic=True, seed=0):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M  # layer 0 holds every node, so it gets more links
        self.mL = 1 / math.log(M)  # level multiplier from the paper
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.heuristic = heuristic
        self.rng = random.Random(seed)

        self.data = np.empty((1024, dim), dtype=np.float32)  # grows by doubling
        self.norms = np.empty(1024, dtype=np.float32)  # ||v||^2
        self.n = 0
        self.levels = []  # top layer of each node
        self.graph = []  # graph[layer] = {node: [neighbour ids]}
        self.entry = None  # entry point: a node on the top layer
        self.max_level = -1
        self.ndis = 0  # distance computations, counted by the benchmark

    def __len__(self):
        return self.n

    # ---------- distances ----------

    def _dists(self, q, qq, ids):
        """Squared L2 from q (with qq = q.q) to each node in ids."""
        ids = np.asarray(ids)
        self.ndis += len(ids)
        return self.norms[ids] - 2 * (self.data[ids] @ q) + qq

    # ---------- Algorithm 2: search one layer ----------

    def _search_layer(self, q, qq, entry_points, ef, layer):
        """Best-first search on one layer.

        entry_points: [(dist, id)]. Returns up to ef (dist, id), closest first.
        candidates: min-heap of nodes still to expand (closest first)
        results:    max-heap (stored as -dist) of the best ef found so far
        """
        links = self.graph[layer]
        visited = {i for _, i in entry_points}
        candidates = list(entry_points)
        heapq.heapify(candidates)
        results = [(-d, i) for d, i in entry_points]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            d, c = heapq.heappop(candidates)
            if d > -results[0][0]:  # closest candidate is worse than worst result: done
                break
            fresh = [n for n in links[c] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for dn, n in zip(self._dists(q, qq, fresh).tolist(), fresh):
                if len(results) < ef or dn < -results[0][0]:
                    heapq.heappush(candidates, (dn, n))
                    heapq.heappush(results, (-dn, n))
                    if len(results) > ef:
                        heapq.heappop(results)  # drop the worst
        return sorted((-d, i) for d, i in results)

    def _search_layer0_filtered(self, q, qq, entry_points, ef, allowed):
        """Algorithm 2 on layer 0, returning only nodes where allowed[i] is True.

        Filtered-out nodes are still walked through: they are the roads to
        other nodes. Only allowed nodes enter `results`, and the search does
        not stop until it has ef allowed results (or runs out of graph).
        So a rare filter costs more steps but still returns k results,
        which plain "search, then drop non-matching" cannot promise.
        """
        links = self.graph[0]
        visited = {i for _, i in entry_points}
        candidates = list(entry_points)
        heapq.heapify(candidates)
        results = [(-d, i) for d, i in entry_points if allowed[i]]
        heapq.heapify(results)

        while candidates:
            d, c = heapq.heappop(candidates)
            if len(results) >= ef and d > -results[0][0]:
                break
            fresh = [n for n in links[c] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for dn, n in zip(self._dists(q, qq, fresh).tolist(), fresh):
                if len(results) < ef or dn < -results[0][0]:
                    heapq.heappush(candidates, (dn, n))
                    if allowed[n]:
                        heapq.heappush(results, (-dn, n))
                        if len(results) > ef:
                            heapq.heappop(results)
        return sorted((-d, i) for d, i in results)

    # ---------- Algorithms 3 and 4: choose neighbours ----------

    def _select(self, candidates, m):
        """candidates: [(dist to the new node, id)] sorted closest first."""
        if not self.heuristic:
            return [i for _, i in candidates[:m]]
        # Keep a candidate only if it is closer to the new node than to every
        # neighbour already kept. Links then point in different directions,
        # so the graph stays connected between clusters.
        selected = []
        for d, i in candidates:
            if len(selected) == m:
                break
            if selected:
                to_selected = self._dists(self.data[i], self.norms[i], selected)
                if (to_selected < d).any():
                    continue
            selected.append(i)
        return selected

    def _shrink(self, node, links, m):
        """Cut a node's link list back to m when a new link pushed it over."""
        d = self._dists(self.data[node], self.norms[node], links)
        return self._select(sorted(zip(d.tolist(), links)), m)

    # ---------- Algorithm 1: insert ----------

    def _random_level(self):
        # most nodes get level 0, about 1/M reach level 1, 1/M^2 level 2, ...
        return int(-math.log(1.0 - self.rng.random()) * self.mL)

    def _insert(self, i):
        q, qq = self.data[i], self.norms[i]
        level = self._random_level()
        self.levels.append(level)
        while len(self.graph) <= level:
            self.graph.append({})
        for layer in range(level + 1):
            self.graph[layer][i] = []

        if self.entry is None:  # first node
            self.entry, self.max_level = i, level
            return

        # 1) On layers above the new node's level, walk greedily (ef = 1).
        ep = [(float(self._dists(q, qq, [self.entry])[0]), self.entry)]
        for layer in range(self.max_level, level, -1):
            ep = self._search_layer(q, qq, ep, 1, layer)[:1]

        # 2) On each layer the node lives on, find neighbours and link both ways.
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(q, qq, ep, self.ef_construction, layer)
            neighbours = self._select(found, self.M)
            self.graph[layer][i] = neighbours
            mmax = self.M0 if layer == 0 else self.M
            for n in neighbours:
                links = self.graph[layer][n]
                links.append(i)
                if len(links) > mmax:
                    self.graph[layer][n] = self._shrink(n, links, mmax)
            ep = found

        if level > self.max_level:
            self.entry, self.max_level = i, level

    def add(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.dim)
        need = self.n + len(x)
        if need > len(self.data):
            cap = max(need, 2 * len(self.data))
            self.data = np.resize(self.data, (cap, self.dim))
            self.norms = np.resize(self.norms, cap)
        self.data[self.n:need] = x
        self.norms[self.n:need] = (x**2).sum(1)
        for i in range(self.n, need):
            self.n = i + 1
            self._insert(i)

    # ---------- Algorithm 5: search ----------

    def search(self, q, k, allowed=None):
        """allowed: optional bool array, one per node; False nodes are never returned."""
        if self.entry is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        q = np.asarray(q, dtype=np.float32)
        qq = float(q @ q)
        ep = [(float(self._dists(q, qq, [self.entry])[0]), self.entry)]
        for layer in range(self.max_level, 0, -1):  # highways: ef = 1
            ep = self._search_layer(q, qq, ep, 1, layer)[:1]
        ef = max(self.ef_search, k)
        if allowed is None:
            found = self._search_layer(q, qq, ep, ef, 0)[:k]
        else:
            found = self._search_layer0_filtered(q, qq, ep, ef, allowed)[:k]
        return (np.array([i for _, i in found], dtype=np.int64),
                np.array([d for d, _ in found], dtype=np.float32))

    # ---------- inspection ----------

    def stats(self):
        return {
            "nodes": self.n,
            "layers": len(self.graph),
            "nodes_per_layer": [len(g) for g in self.graph],
            "avg_links_layer0": round(
                sum(len(v) for v in self.graph[0].values()) / max(self.n, 1), 2),
        }

    def memory_bytes(self):
        # Vectors and norms are real NumPy bytes. Links are counted as 4-byte
        # ints, i.e. what a compact (C / NumPy) layout would need. Python
        # lists of ints actually use several times more.
        links = sum(len(v) for g in self.graph for v in g.values())
        return self.data[:self.n].nbytes + self.norms[:self.n].nbytes + 4 * links

    # ---------- save / load (used by VectorStore) ----------

    def stored_vectors(self):
        return self.data[:self.n]

    def state(self):
        """(settings, arrays) that fully describe the index.

        Each layer is stored in CSR form, like a sparse matrix:
            nodes[i]                       a node on this layer
            links[offsets[i]:offsets[i+1]] its neighbour list, in order
        The random generator's state is saved too, so adding vectors after
        a load gives exactly the same graph as adding them before the save.
        """
        settings = {
            "dim": self.dim, "M": self.M, "ef_construction": self.ef_construction,
            "ef_search": self.ef_search, "heuristic": self.heuristic,
            "entry": self.entry, "max_level": self.max_level,
            "rng_state": [self.rng.getstate()[0], list(self.rng.getstate()[1]),
                          self.rng.getstate()[2]],
        }
        arrays = {"vectors": self.data[:self.n], "levels": np.array(self.levels, dtype=np.int32)}
        for layer, g in enumerate(self.graph):
            nodes = np.fromiter(g.keys(), dtype=np.int64, count=len(g))
            counts = np.fromiter((len(v) for v in g.values()), dtype=np.int64, count=len(g))
            arrays[f"l{layer}_nodes"] = nodes
            arrays[f"l{layer}_offsets"] = np.concatenate([[0], np.cumsum(counts)])
            arrays[f"l{layer}_links"] = np.fromiter(
                (n for v in g.values() for n in v), dtype=np.int64, count=int(counts.sum()))
        settings["layers"] = len(self.graph)
        return settings, arrays

    @classmethod
    def from_state(cls, settings, arrays):
        index = cls(settings["dim"], M=settings["M"], ef_construction=settings["ef_construction"],
                    ef_search=settings["ef_search"], heuristic=settings["heuristic"])
        vectors = np.asarray(arrays["vectors"], dtype=np.float32)
        index.n = len(vectors)
        index.data = np.resize(index.data, (max(index.n, 1024), index.dim))
        index.norms = np.resize(index.norms, max(index.n, 1024))
        index.data[:index.n] = vectors
        index.norms[:index.n] = (vectors**2).sum(1)
        index.levels = arrays["levels"].tolist()
        index.graph = []
        for layer in range(settings["layers"]):
            nodes = arrays[f"l{layer}_nodes"].tolist()
            off = arrays[f"l{layer}_offsets"]
            links = arrays[f"l{layer}_links"].tolist()
            index.graph.append({n: links[off[i]:off[i + 1]] for i, n in enumerate(nodes)})
        index.entry, index.max_level = settings["entry"], settings["max_level"]
        s = settings["rng_state"]
        index.rng.setstate((s[0], tuple(s[1]), s[2]))
        return index
