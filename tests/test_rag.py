"""RAG pipeline tests. Offline: a word-hashing embedder and a scripted LLM
stand in for MiniLM and Gemma, so no model download or token is needed."""

import re
import zlib

import numpy as np
import pytest

from rag.judge import build_judge_messages, judge, parse_judgement
from rag.make_pdf import text_to_pages, write_pdf
from rag.pdf import chunk_pages, clean
from rag.pipeline import REFUSAL, answer, build_messages, build_store, is_refusal, retrieve


class WordEmbedder:
    """Bag of words hashed into 64 buckets: texts sharing words are similar."""
    dim = 64

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % self.dim] += 1
        out[out.sum(1) == 0, 0] = 1  # avoid zero vectors
        return out


class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def complete(self, messages):
        self.seen.append(messages)
        return self.reply


CHUNKS = [
    {"id": "n_p1_c0", "text": "a stack is last in first out push pop", "source": "n.pdf", "page": 1, "chunk": 0},
    {"id": "n_p1_c1", "text": "a queue is first in first out enqueue dequeue", "source": "n.pdf", "page": 1, "chunk": 1},
    {"id": "n_p2_c0", "text": "a hash table resizes when the load factor grows", "source": "n.pdf", "page": 2, "chunk": 0},
]


@pytest.fixture
def store():
    return build_store(CHUNKS, WordEmbedder(), index="flat")


# ---------- chunking ----------

def test_chunks_overlap_and_cover_every_word():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_pages([(3, " ".join(words))], "book.pdf", size=10, overlap=3)
    # starts at words 0, 7, 14, 21; the third chunk ends at word 23, so word 24 needs a fourth
    assert [c["id"] for c in chunks] == ["book_p3_c0", "book_p3_c1", "book_p3_c2", "book_p3_c3"]
    assert chunks[0]["text"].split()[-3:] == chunks[1]["text"].split()[:3]
    seen = {w for c in chunks for w in c["text"].split()}
    assert seen == set(words)
    assert all(c["page"] == 3 and c["source"] == "book.pdf" for c in chunks)


def test_short_page_is_one_chunk():
    assert len(chunk_pages([(1, "only five words in here")], "x.pdf", size=10, overlap=3)) == 1


def test_bad_overlap():
    with pytest.raises(ValueError):
        chunk_pages([(1, "a b c")], "x.pdf", size=5, overlap=5)


def test_clean_joins_hyphenated_line_breaks():
    assert clean("data struc-\ntures  are\n\nfun") == "data structures are fun"


# ---------- PDF round trip ----------

def test_pdf_round_trip(tmp_path):
    pytest.importorskip("pypdf")
    from rag.pdf import extract_pages

    text = "# Stacks\nA stack (LIFO) uses push and pop.\n\n# Queues\nA queue is FIFO; see C:\\notes."
    path = tmp_path / "notes.pdf"
    write_pdf(text_to_pages(text), path)
    pages = extract_pages(path)
    assert len(pages) == 1
    assert "A stack (LIFO) uses push and pop." in pages[0][1]
    assert "C:\\notes" in pages[0][1]


def test_long_text_makes_several_pages():
    pages = text_to_pages("\n".join(f"line {i}" for i in range(200)))
    assert len(pages) == 4 and all(len(p) <= 54 for p in pages)


# ---------- retrieval and answering ----------

def test_retrieve_returns_text_page_and_score(store):
    top = retrieve(store, WordEmbedder(), "what is a stack push pop", k=2)
    assert top[0]["id"] == "n_p1_c0"
    assert top[0]["page"] == 1 and "stack" in top[0]["text"]
    assert top[0]["score"] >= top[1]["score"]


def test_retrieve_with_filter(store):
    top = retrieve(store, WordEmbedder(), "stack", k=3, filter={"page": 2})
    assert [p["id"] for p in top] == ["n_p2_c0"]


def test_prompt_numbers_passages_and_states_rules(store):
    passages = retrieve(store, WordEmbedder(), "queue", k=2)
    messages = build_messages("What is a queue?", passages)
    assert REFUSAL in messages[0]["content"]
    assert "[1] (n.pdf, page 1)" in messages[1]["content"]
    assert messages[1]["content"].endswith("Question: What is a queue?")


def test_answer_passes_passages_to_llm(store):
    llm = ScriptedLLM("A queue is first in, first out [1].")
    res = answer("what is a queue", store, WordEmbedder(), llm, k=2)
    assert not res["refused"] and res["passages"][0]["id"] == "n_p1_c1"
    assert "first in first out" in llm.seen[0][1]["content"]


def test_refusal_detected(store):
    res = answer("who won the world cup", store, WordEmbedder(), ScriptedLLM(REFUSAL))
    assert res["refused"]
    assert is_refusal("I don't know based on the provided material.")
    assert not is_refusal("A stack is LIFO.")


def test_empty_store_refuses_without_calling_llm():
    empty = build_store([], WordEmbedder(), index="flat")
    llm = ScriptedLLM("should not be called")
    res = answer("anything", empty, WordEmbedder(), llm)
    assert res["refused"] and llm.seen == []


def test_store_round_trip_keeps_passage_text(store, tmp_path):
    from vecsearch import VectorStore

    store.save(tmp_path / "s.npz")
    loaded = VectorStore.load(tmp_path / "s.npz")
    assert retrieve(loaded, WordEmbedder(), "hash table") == retrieve(store, WordEmbedder(), "hash table")


# ---------- judge output parsing ----------

@pytest.mark.parametrize("text, score", [
    ('{"score": 4, "reason": "ok"}', 4),
    ('```json\n{"score": 5, "reason": "all supported"}\n```', 5),
    ('Here is my grade:\n{"score": 3, "reason": "one claim unsupported"} Hope this helps.', 3),
    ("{'score': 2, 'reason': 'single quotes'}", 2),
    ('{"score": 4, "reason": "trailing comma",}', 4),
    ('{"score": "5", "reason": "score as a string"}', 5),
    ('{"score": 4.0, "reason": "float"}', 4),
    ('{"note": "no score here"} but Score: 2', 2),
    ("Faithfulness score - 3 out of 5", 3),
])
def test_parse_judgement_handles_messy_output(text, score):
    assert parse_judgement(text)["score"] == score


@pytest.mark.parametrize("text", ["", "I cannot grade this.", '{"score": 9}', '{"score": "high"}'])
def test_parse_judgement_returns_none_when_unusable(text):
    assert parse_judgement(text) is None


def test_parse_judgement_keeps_reason():
    assert parse_judgement('{"score": 5, "reason": " fine "}')["reason"] == "fine"


def test_judge_sends_passages_and_answer():
    llm = ScriptedLLM('{"score": 5, "reason": "supported"}')
    verdict = judge("q?", "the answer", [{"text": "passage one"}], llm)
    assert verdict == {"score": 5, "reason": "supported"}
    prompt = llm.seen[0][0]["content"]
    assert "[1] passage one" in prompt and "the answer" in prompt
    assert build_judge_messages("q", "a", [])[0]["role"] == "user"
