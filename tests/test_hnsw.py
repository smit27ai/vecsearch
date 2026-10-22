from collections import deque

import numpy as np
import pytest

from bench.run import measure
from vecsearch import HNSWIndex
from vecsearch.datasets import synthetic


def build(n=1000, dim=16, M=8, heuristic=True, seed=0):
    base, queries, gt = synthetic(n=n, n_queries=30, dim=dim, k=10, seed=seed)
    index = HNSWIndex(dim, M=M, ef_construction=64, heuristic=heuristic, seed=seed)
    index.add(base)
    return index, queries, gt


@pytest.fixture(scope="module")
def small():
    return build()


def test_empty_and_single():
    index = HNSWIndex(2)
    assert len(index.search(np.zeros(2, dtype=np.float32), 5)[0]) == 0
    index.add([[3, 4]])
    ids, dists = index.search(np.zeros(2, dtype=np.float32), 5)
    assert ids.tolist() == [0]
    np.testing.assert_allclose(dists, [25])


def test_hand_made_points():
    index = HNSWIndex(2, M=2)
    index.add([[0, 0], [10, 10], [1, 1], [-3, 0], [5, 5]])
    index.ef_search = 10
    ids, _ = index.search(np.array([0.9, 0.9], dtype=np.float32), 3)
    assert ids.tolist() == [2, 0, 3]  # (-3,0) is at 16.02, (5,5) at 33.62


def test_wide_search_is_exact(small):
    index, queries, gt = small
    index.ef_search = len(index)
    assert measure(index, queries, gt, 10, repeats=1)[0] == 1.0


def test_recall_improves_with_ef(small):
    index, queries, gt = small
    index.ef_search = 10
    low = measure(index, queries, gt, 10, repeats=1)[0]
    index.ef_search = 200
    high = measure(index, queries, gt, 10, repeats=1)[0]
    assert high >= low and high >= 0.95


@pytest.mark.parametrize("heuristic", [True, False])
def test_link_limits_and_targets(heuristic):
    index, _, _ = build(n=600, heuristic=heuristic)
    for layer, g in enumerate(index.graph):
        limit = index.M0 if layer == 0 else index.M
        for node, links in g.items():
            assert len(links) <= limit
            assert node not in links
            assert all(n in g for n in links)  # every link stays on its layer


def reachable_from_entry(index):
    seen, todo = {index.entry}, deque([index.entry])
    while todo:
        for n in index.graph[0][todo.popleft()]:
            if n not in seen:
                seen.add(n)
                todo.append(n)
    return len(seen)


def test_layer0_reachable_from_entry(small):
    # When _shrink trims a full link list it can drop the only link into some
    # node, leaving it unreachable. The paper's algorithm allows this; with
    # M=8 it hits about 1 node in 1000 here, with M=16 none.
    index, _, _ = small
    assert reachable_from_entry(index) >= 0.99 * len(index)
    index16, _, _ = build(M=16)
    assert reachable_from_entry(index16) == len(index16)


def test_layers_shrink_upwards(small):
    index, _, _ = small
    sizes = index.stats()["nodes_per_layer"]
    assert sizes[0] == len(index)
    assert sizes == sorted(sizes, reverse=True)
    # about 1/M of nodes reach layer 1
    assert 0.5 / index.M < sizes[1] / sizes[0] < 2 / index.M


def test_same_seed_same_graph():
    a, _, _ = build(n=300)
    b, _, _ = build(n=300)
    assert a.graph == b.graph and a.entry == b.entry


def test_add_in_batches_grows_storage():
    base, _, _ = synthetic(n=3000, n_queries=1, dim=8, k=1)
    index = HNSWIndex(8, M=4, ef_construction=20)
    index.add(base[:1500])
    index.add(base[1500:])
    assert len(index) == 3000
    index.ef_search = 50
    ids, _ = index.search(base[2999], 1)
    assert ids.tolist() == [2999]
