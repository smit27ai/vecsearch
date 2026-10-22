"""HTTP API for a VectorStore, built with FastAPI.

    POST   /items          add or replace items        {"items": [{id, vector, metadata}]}
    GET    /items/{id}     one item (vector and metadata)
    DELETE /items/{id}     delete one item
    POST   /search         {"vector": [...], "k": 5, "filter": {"chapter": 3}}
    POST   /compact        drop tombstones
    POST   /save           write the store to disk
    GET    /stats          item and tombstone counts

Run:
    python -m vecsearch.api --path data/store.npz --dim 384
then open http://127.0.0.1:8000/docs for an interactive page that can call
every endpoint.
"""

import argparse
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .store import VectorStore


class Item(BaseModel):
    id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddRequest(BaseModel):
    items: list[Item]


class SearchRequest(BaseModel):
    vector: list[float]
    k: int = Field(10, ge=1, le=1000)
    filter: dict[str, Any] | None = None


def create_app(store, path=None):
    """path: where /save writes. None disables /save."""
    app = FastAPI(title="vecsearch", description="A vector search engine built from scratch.")
    lock = Lock()  # the store is not thread-safe; FastAPI may run handlers in threads

    def check_dim(vector):
        if len(vector) != store.dim:
            raise HTTPException(422, f"vector has {len(vector)} numbers, expected {store.dim}")

    @app.post("/items")
    def add(req: AddRequest):
        for item in req.items:
            check_dim(item.vector)
        try:
            with lock:
                store.add([i.id for i in req.items], [i.vector for i in req.items],
                          [i.metadata for i in req.items])
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"added": len(req.items), "items": len(store)}

    @app.get("/items/{item_id}")
    def get(item_id: str):
        with lock:
            item = store.get(item_id)
        if item is None:
            raise HTTPException(404, f"no item {item_id!r}")
        return item

    @app.delete("/items/{item_id}")
    def delete(item_id: str):
        with lock:
            if not store.delete([item_id]):
                raise HTTPException(404, f"no item {item_id!r}")
        return {"deleted": item_id}

    @app.post("/search")
    def search(req: SearchRequest):
        check_dim(req.vector)
        try:
            with lock:
                return {"results": store.search(req.vector, req.k, req.filter)}
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.post("/compact")
    def compact():
        with lock:
            return {"removed": store.compact()}

    @app.post("/save")
    def save():
        if path is None:
            raise HTTPException(400, "server was started without a save path")
        with lock:
            store.save(path)
        return {"saved": str(path), "items": len(store)}

    @app.get("/stats")
    def stats():
        with lock:
            return store.stats()

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="store file; loaded if it exists")
    parser.add_argument("--dim", type=int, help="vector size, needed for a new store")
    parser.add_argument("--index", default="hnsw", choices=["hnsw", "flat"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    path = Path(args.path)
    if path.exists():
        store = VectorStore.load(path)
        print(f"loaded {len(store)} items from {path}")
    elif args.dim:
        store = VectorStore(args.dim, index=args.index)
        print(f"new empty store, dim={args.dim}")
    else:
        parser.error("--dim is required when the store file does not exist yet")
    uvicorn.run(create_app(store, path), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
