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

    def search(self, q, k):
        if self.entry is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        q = np.asarray(q, dtype=np.float32)
        qq = float(q @ q)
        ep = [(float(self._dists(q, qq, [self.entry])[0]), self.entry)]
        for layer in range(self.max_level, 0, -1):  # highways: ef = 1
            ep = self._search_layer(q, qq, ep, 1, layer)[:1]
        found = self._search_layer(q, qq, ep, max(self.ef_search, k), 0)[:k]
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
