"""Evaluate the RAG pipeline on rag/sample (or your own PDFs and questions).

Always runs (no API key needed):
    retrieval   hit@1, hit@4 and MRR: is a passage containing the expected
                phrase retrieved, and how high is it ranked?
    score gap   top-1 cosine score for in-scope vs out-of-scope questions

Runs only when HF_TOKEN is set:
    answers       Gemma answers every question from its retrieved passages
    refusals      out-of-scope questions should be refused, in-scope not
    faithfulness  an LLM judge scores each non-refused answer 1-5

Usage:
    python -m rag.eval
    python -m rag.eval --pdf my.pdf --questions my_questions.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from .pipeline import answer, ingest, retrieve

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "eval.json"


def retrieval_eval(store, embedder, questions, k=4):
    rows = []
    for item in questions:
        passages = retrieve(store, embedder, item["q"], k)
        rank = next((n for n, p in enumerate(passages, 1)
                     if item["expect"].lower() in p["text"].lower()), None)
        rows.append({"q": item["q"], "rank": rank, "top_score": passages[0]["score"],
                     "top_id": passages[0]["id"]})
    n = len(rows)
    return {
        "hit@1": round(sum(r["rank"] == 1 for r in rows) / n, 3),
        f"hit@{k}": round(sum(r["rank"] is not None for r in rows) / n, 3),
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
        "misses": [r["q"] for r in rows if r["rank"] is None],
        "rows": rows,
    }


def score_gap(store, embedder, in_scope, out_of_scope):
    top = lambda q: retrieve(store, embedder, q, 1)[0]["score"]
    ins = [top(item["q"]) for item in in_scope]
    outs = [top(q) for q in out_of_scope]
    return {"in_scope_top1": {"min": min(ins), "mean": round(statistics.mean(ins), 3)},
            "out_of_scope_top1": {"max": max(outs), "mean": round(statistics.mean(outs), 3)}}


def llm_eval(store, embedder, in_scope, out_of_scope, k=4):
    from .judge import judge
    from .llm import HFChat

    llm = HFChat()
    answers, scores, unparsed = [], [], 0
    for item in in_scope:
        res = answer(item["q"], store, embedder, llm, k)
        res["in_scope"] = True
        if not res["refused"]:
            verdict = judge(item["q"], res["answer"], res["passages"], llm)
            res["faithfulness"] = verdict
            if verdict:
                scores.append(verdict["score"])
            else:
                unparsed += 1
        answers.append(res)
        time.sleep(0.5)  # stay under free-tier rate limits
    for q in out_of_scope:
        res = answer(q, store, embedder, llm, k)
        res["in_scope"] = False
        answers.append(res)
        time.sleep(0.5)
    return {
        "model": llm.model,
        "faithfulness_mean": round(statistics.mean(scores), 2) if scores else None,
        "faithfulness_counts": {s: scores.count(s) for s in range(1, 6)},
        "judge_unparsed": unparsed,
        "false_refusals": sum(a["refused"] for a in answers if a["in_scope"]),
        "correct_refusals": sum(a["refused"] for a in answers if not a["in_scope"]),
        "out_of_scope_total": len(out_of_scope),
        "answers": [{k: v for k, v in a.items() if k != "passages"} |
                    {"passage_ids": [p["id"] for p in a["passages"]]} for a in answers],
    }


def main():
    import os

    from .embed import MiniLMEmbedder

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=HERE / "sample" / "dsa_notes.pdf")
    parser.add_argument("--questions", type=Path, default=HERE / "sample" / "questions.json")
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    qs = json.loads(args.questions.read_text())
    embedder = MiniLMEmbedder()
    store = ingest([args.pdf], embedder)
    print(f"indexed {len(store)} chunks from {args.pdf.name}")

    report = {"pdf": args.pdf.name, "chunks": len(store), "k": args.k}
    report["retrieval"] = retrieval_eval(store, embedder, qs["in_scope"], args.k)
    r = report["retrieval"]
    print(f"retrieval: hit@1={r['hit@1']}  hit@{args.k}={r[f'hit@{args.k}']}  MRR={r['mrr']}"
          f"  misses={r['misses']}")
    report["score_gap"] = score_gap(store, embedder, qs["in_scope"], qs["out_of_scope"])
    print(f"top-1 cosine: {report['score_gap']}")

    if os.environ.get("HF_TOKEN"):
        report["llm"] = llm_eval(store, embedder, qs["in_scope"], qs["out_of_scope"], args.k)
        m = report["llm"]
        print(f"faithfulness mean={m['faithfulness_mean']}  counts={m['faithfulness_counts']}  "
              f"unparsed={m['judge_unparsed']}")
        print(f"refusals: {m['correct_refusals']}/{m['out_of_scope_total']} out-of-scope refused, "
              f"{m['false_refusals']} in-scope wrongly refused")
    else:
        print("HF_TOKEN not set: skipped answer generation and faithfulness judging")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
