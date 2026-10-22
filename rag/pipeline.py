"""Ingest PDFs into a VectorStore, retrieve chunks, and answer from them.

The store replaces FAISS. Compared with a bare FAISS index it keeps each
chunk's text and page with its vector, returns string ids and cosine
scores, can filter by source or page, and saves to one file.
"""

from pathlib import Path

from vecsearch import VectorStore

from .pdf import chunk_pages, extract_pages

REFUSAL = "I don't know based on the provided material."

SYSTEM_PROMPT = f"""You are a study assistant for data structures.
Answer the question using ONLY the numbered context passages below.
Rules:
- If the passages do not contain the answer, reply exactly: {REFUSAL}
- Do not use outside knowledge, even if you know the answer.
- Cite the passages you used, like [1] or [2][3].
- Keep the answer short: at most 5 sentences."""


def ingest(pdf_paths, embedder, index="hnsw", size=120, overlap=30):
    """Parse, chunk and embed PDFs into a new VectorStore."""
    chunks = []
    for path in pdf_paths:
        chunks += chunk_pages(extract_pages(path), path, size, overlap)
    return build_store(chunks, embedder, index)


def build_store(chunks, embedder, index="hnsw"):
    params = {"M": 16, "ef_construction": 200, "ef_search": 64} if index == "hnsw" else {}
    store = VectorStore(embedder.dim, index=index, **params)
    if chunks:
        vectors = embedder.encode([c["text"] for c in chunks])
        store.add([c["id"] for c in chunks], vectors,
                  [{k: c[k] for k in ("text", "source", "page", "chunk")} for c in chunks])
    return store


def retrieve(store, embedder, question, k=4, filter=None):
    """Top-k chunks: [{"id", "score", "text", "source", "page"}]."""
    q = embedder.encode([question])[0]
    return [{"id": r["id"], "score": round(r["score"], 4), **r["metadata"]}
            for r in store.search(q, k, filter=filter)]


def build_messages(question, passages):
    context = "\n\n".join(
        f"[{n}] ({p['source']}, page {p['page']})\n{p['text']}" for n, p in enumerate(passages, 1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def is_refusal(answer):
    return answer.strip().lower().startswith("i don't know")


def answer(question, store, embedder, llm, k=4, filter=None):
    passages = retrieve(store, embedder, question, k, filter)
    if not passages:
        return {"question": question, "answer": REFUSAL, "passages": [], "refused": True}
    reply = llm.complete(build_messages(question, passages))
    return {"question": question, "answer": reply, "passages": passages,
            "refused": is_refusal(reply)}


def main():
    import argparse

    from .embed import MiniLMEmbedder

    parser = argparse.ArgumentParser(description="Ask questions about PDFs.")
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--store", type=Path, default=Path("data/rag_store.npz"))
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    embedder = MiniLMEmbedder()
    if args.store.exists():
        store = VectorStore.load(args.store)
        print(f"loaded {len(store)} chunks from {args.store}")
    else:
        store = ingest(args.pdfs, embedder)
        store.save(args.store)
        print(f"indexed {len(store)} chunks into {args.store}")

    from .llm import HFChat
    llm = HFChat()
    while True:
        try:
            question = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        result = answer(question, store, embedder, llm, args.k)
        print("\n" + result["answer"])
        for n, p in enumerate(result["passages"], 1):
            print(f"  [{n}] {p['source']} p.{p['page']}  score={p['score']}")


if __name__ == "__main__":
    main()
