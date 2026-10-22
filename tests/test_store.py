import numpy as np
import pytest

from vecsearch import VectorStore
from vecsearch.datasets import synthetic


def make(index="hnsw", n=2000, exact_below=0, **kw):
    """A store of n items; chapter = i % 10, so each chapter is 10% of items."""
    base, queries, _ = synthetic(n=n, n_queries=20, dim=16, k=10)
    params = {"M": 8, "ef_construction": 64} if index == "hnsw" else {}
    store = VectorStore(16, index=index, exact_below=exact_below, **params, **kw)
    store.add([f"doc{i}" for i in range(n)], base,
              [{"chapter": i % 10, "even": i % 2 == 0} for i in range(n)])
    return store, base, queries


def exact_top(base, q, k, allowed=None):
    unit = base / np.linalg.norm(base, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q)
    d = ((unit - qn) ** 2).sum(1)
    if allowed is not None:
        d = np.where(allowed, d, np.inf)
    return [f"doc{i}" for i in np.argsort(d)[:k]]


@pytest.fixture(scope="module")
def hnsw_store():
    return make()


# ---------- basic search ----------

def test_flat_store_matches_exact_cosine():
    store, base, queries = make("flat")
    for q in queries[:5]:
        assert [r["id"] for r in store.search(q, 10)] == exact_top(base, q, 10)


def test_scores_are_cosine_similarity():
    store = VectorStore(2, index="flat")
    store.add(["right", "up", "left"], [[1, 0], [0, 5], [-2, 0]])
    res = store.search([3, 0], 3)
    assert [r["id"] for r in res] == ["right", "up", "left"]
    np.testing.assert_allclose([r["score"] for r in res], [1, 0, -1], atol=1e-6)


def test_hnsw_store_recall(hnsw_store):
    store, base, queries = hnsw_store
    hits = sum(len(set(r["id"] for r in store.search(q, 10)) & set(exact_top(base, q, 10)))
               for q in queries)
    assert hits / (10 * len(queries)) >= 0.9


# ---------- filters ----------

@pytest.mark.parametrize("index", ["flat", "hnsw"])
def test_filter_returns_only_matches(index):
    store, _, queries = make(index)
    for r in store.search(queries[0], 10, filter={"chapter": 3}):
        assert r["metadata"]["chapter"] == 3


def test_filter_with_list_means_one_of(hnsw_store):
    store, _, queries = hnsw_store
    res = store.search(queries[0], 20, filter={"chapter": [1, 2]})
    assert {r["metadata"]["chapter"] for r in res} <= {1, 2}


def test_rare_filter_still_returns_k(hnsw_store):
    # 1 item in 200 matches; graph search must still find 10 of them
    store, base, queries = hnsw_store
    for i in range(0, 2000, 200):
        store.meta[i]["rare"] = True
    try:
        res = store.search(queries[0], 10, filter={"rare": True})
        assert len(res) == 10
        allowed = np.array([m.get("rare", False) for m in store.meta])
        assert [r["id"] for r in res] == exact_top(base, queries[0], 10, allowed)
    finally:
        for i in range(0, 2000, 200):
            del store.meta[i]["rare"]


def test_small_filter_uses_exact_scan():
    store, base, queries = make(exact_below=1000)
    store.index.ndis = 0
    allowed = np.array([m["chapter"] == 5 for m in store.meta])  # 200 items < 1000
    res = store.search(queries[0], 10, filter={"chapter": 5})
    assert [r["id"] for r in res] == exact_top(base, queries[0], 10, allowed)
    assert store.index.ndis == 0  # graph not touched


def test_filter_matching_nothing():
    store, _, queries = make("flat", n=100)
    assert store.search(queries[0], 5, filter={"chapter": 99}) == []


# ---------- delete, upsert, compact ----------

@pytest.mark.parametrize("index", ["flat", "hnsw"])
def test_deleted_items_never_returned(index):
    store, _, queries = make(index)
    top = [r["id"] for r in store.search(queries[0], 5)]
    assert store.delete(top[:3]) == 3
    after = [r["id"] for r in store.search(queries[0], 5)]
    assert not set(top[:3]) & set(after)
    assert len(store) == 1997 and store.get(top[0]) is None


def test_delete_unknown_id():
    store, _, _ = make("flat", n=10)
    assert store.delete(["nope"]) == 0


def test_upsert_replaces_vector_and_metadata():
    store = VectorStore(2, index="flat")
    store.add(["a", "b"], [[1, 0], [0, 1]], [{"v": 1}, {"v": 1}])
    store.add(["a"], [[-1, 0]], [{"v": 2}])
    assert len(store) == 2
    assert store.get("a")["metadata"] == {"v": 2}
    assert store.search([-1, 0], 1)[0]["id"] == "a"
    assert store.stats()["tombstones"] == 1


def test_compact_drops_tombstones_and_keeps_results():
    store, _, queries = make("flat")
    store.delete([f"doc{i}" for i in range(0, 2000, 2)])  # delete half
    before = [r["id"] for r in store.search(queries[0], 10)]
    assert store.compact() == 1000
    assert store.stats() == {"items": 1000, "tombstones": 0, "index": "flat", "dim": 16,
                             "normalize": True}
    assert [r["id"] for r in store.search(queries[0], 10)] == before
    np.testing.assert_allclose(np.linalg.norm(store.index.stored_vectors(), axis=1), 1, rtol=1e-5)


# ---------- input checks ----------

def test_bad_input():
    store = VectorStore(2, index="flat")
    with pytest.raises(ValueError):
        store.add(["a", "a"], [[1, 0], [0, 1]])  # duplicate ids
    with pytest.raises(ValueError):
        store.add(["a"], [[1, 0], [0, 1]])  # count mismatch
    with pytest.raises(ValueError):
        store.add(["z"], [[0, 0]])  # zero vector cannot be normalized


# ---------- save / load ----------

@pytest.mark.parametrize("index", ["flat", "hnsw"])
def test_save_load_same_results(tmp_path, index):
    store, _, queries = make(index, n=1000)
    store.delete(["doc1", "doc2"])
    path = tmp_path / "store.npz"
    store.save(path)
    loaded = VectorStore.load(path)
    assert loaded.stats() == store.stats()
    for q in queries[:5]:
        assert loaded.search(q, 10) == store.search(q, 10)
        assert loaded.search(q, 10, filter={"even": True}) == store.search(q, 10, filter={"even": True})
    assert not (tmp_path / "store.npz.tmp").exists()


def test_add_after_load_matches_add_without_save(tmp_path):
    # the saved random-generator state makes the graph identical either way
    a, _, _ = make("hnsw", n=500)
    a.save(tmp_path / "s.npz")
    b = VectorStore.load(tmp_path / "s.npz")
    extra = np.random.default_rng(9).standard_normal((50, 16)).astype(np.float32)
    ids = [f"new{i}" for i in range(50)]
    a.add(ids, extra)
    b.add(ids, extra)
    assert a.index.graph == b.index.graph


def test_failed_save_keeps_old_file(tmp_path, monkeypatch):
    store, _, _ = make("flat", n=50)
    path = tmp_path / "s.npz"
    store.save(path)
    old = path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", boom)
    with pytest.raises(OSError):
        store.save(path)
    assert path.read_bytes() == old
