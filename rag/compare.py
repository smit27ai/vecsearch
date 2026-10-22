"""Does VectorStore retrieve the same passages FAISS would?

Embeds the sample chunks and questions once with MiniLM, then searches
them with:
    FAISS IndexFlatIP     exact inner product (= cosine on unit vectors),
                          what the original chatbot used
    VectorStore flat      this project's exact index
    VectorStore hnsw      this project's graph index

Reports, per index, how often its top-k list matches FAISS exactly (same
ids, same order), the average overlap of the top-k sets, and the largest
score difference. Timings are recorded but mean little at this size: every
search over a few dozen chunks takes well under a millisecond.

Usage:
    python -m rag.compare
"""

import json
import time
from pathlib import Path

import numpy as np

from .pdf import chunk_pages, extract_pages
from .pipeline import build_store

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "compare.json"


class _Fixed:
    """Embedder stand-in that returns precomputed vectors, so every index sees the same input."""
    dim = 384

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return self.vectors


def main(k=4):
    import faiss

    from .embed import MiniLMEmbedder

    chunks = chunk_pages(extract_pages(HERE / "sample" / "dsa_notes.pdf"), "dsa_notes.pdf")
    qs = json.loads((HERE / "sample" / "questions.json").read_text())
    questions = [q["q"] for q in qs["in_scope"]] + qs["out_of_scope"]

    embedder = MiniLMEmbedder()
    chunk_vecs = embedder.encode([c["text"] for c in chunks])
    query_vecs = embedder.encode(questions)
    ids = [c["id"] for c in chunks]

    flat = faiss.IndexFlatIP(chunk_vecs.shape[1])
    flat.add(chunk_vecs)
    t = time.perf_counter()
    f_scores, f_idx = flat.search(query_vecs, k)
    faiss_ms = (time.perf_counter() - t) * 1000 / len(questions)
    reference = [[ids[i] for i in row] for row in f_idx]

    report = {"chunks": len(chunks), "questions": len(questions), "k": k,
              "faiss_ms_per_query": round(faiss_ms, 4), "indexes": {}}
    for kind in ("flat", "hnsw"):
        store = build_store(chunks, _Fixed(chunk_vecs), index=kind)
        # With only a few dozen chunks the store would scan them exactly
        # (fewer than exact_below match); force the graph so HNSW is tested.
        store.exact_below = 0
        t = time.perf_counter()
        results = [store.search(q, k) for q in query_vecs]
        ms = (time.perf_counter() - t) * 1000 / len(questions)
        same_order = sum([r["id"] for r in res] == ref for res, ref in zip(results, reference))
        overlap = np.mean([len({r["id"] for r in res} & set(ref)) / k
                           for res, ref in zip(results, reference)])
        score_diff = max(abs(r["score"] - s) for res, row in zip(results, f_scores)
                         for r, s in zip(res, row))
        report["indexes"][kind] = {"identical_top_k": f"{same_order}/{len(questions)}",
                                   "mean_overlap": round(float(overlap), 4),
                                   "max_score_diff": float(f"{score_diff:.2e}"),
                                   "ms_per_query": round(ms, 4)}
        print(f"VectorStore {kind:4}: identical top-{k} {same_order}/{len(questions)}  "
              f"overlap={overlap:.3f}  max score diff={score_diff:.1e}  {ms:.3f} ms/query")
    print(f"FAISS IndexFlatIP: {faiss_ms:.3f} ms/query (batched)")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
