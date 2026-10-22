"""A small RAG pipeline that uses vecsearch's VectorStore instead of FAISS.

    PDF -> pages -> chunks -> MiniLM embeddings -> VectorStore
    question -> embedding -> top-k chunks -> grounded prompt -> LLM answer
    answer + chunks -> LLM judge -> faithfulness score (1-5)

Heavy libraries (sentence-transformers, huggingface_hub, pypdf) are imported
inside the functions that need them, so the core tests run without them.
"""
