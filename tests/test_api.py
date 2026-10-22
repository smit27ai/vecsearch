import pytest
from fastapi.testclient import TestClient

from vecsearch import VectorStore
from vecsearch.api import create_app


@pytest.fixture
def client(tmp_path):
    store = VectorStore(3, index="flat")
    client = TestClient(create_app(store, tmp_path / "s.npz"))
    client.post("/items", json={"items": [
        {"id": "a", "vector": [1, 0, 0], "metadata": {"chapter": 1}},
        {"id": "b", "vector": [0, 1, 0], "metadata": {"chapter": 2}},
        {"id": "c", "vector": [0.9, 0.1, 0], "metadata": {"chapter": 2}},
    ]})
    return client, tmp_path


def test_search(client):
    c, _ = client
    res = c.post("/search", json={"vector": [1, 0, 0], "k": 2}).json()["results"]
    assert [r["id"] for r in res] == ["a", "c"]
    assert res[0]["score"] == pytest.approx(1.0)


def test_search_with_filter(client):
    c, _ = client
    res = c.post("/search", json={"vector": [1, 0, 0], "k": 2, "filter": {"chapter": 2}})
    assert [r["id"] for r in res.json()["results"]] == ["c", "b"]


def test_get_and_delete(client):
    c, _ = client
    assert c.get("/items/a").json()["metadata"] == {"chapter": 1}
    assert c.delete("/items/a").status_code == 200
    assert c.get("/items/a").status_code == 404
    assert c.delete("/items/a").status_code == 404
    assert c.get("/stats").json()["items"] == 2


def test_wrong_vector_size(client):
    c, _ = client
    assert c.post("/search", json={"vector": [1, 0]}).status_code == 422
    bad = {"items": [{"id": "x", "vector": [1, 2]}]}
    assert c.post("/items", json=bad).status_code == 422


def test_save_then_reload(client):
    c, tmp_path = client
    assert c.post("/save").status_code == 200
    reloaded = VectorStore.load(tmp_path / "s.npz")
    assert len(reloaded) == 3


def test_compact(client):
    c, _ = client
    c.delete("/items/b")
    assert c.post("/compact").json() == {"removed": 1}
    assert c.get("/stats").json()["tombstones"] == 0
