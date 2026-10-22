"""LLM-as-judge: score how faithful an answer is to its retrieved passages.

Faithfulness asks "is every claim in the answer supported by the passages?",
not "is the answer correct?". An answer can be true and still unfaithful if
it comes from the model's own knowledge instead of the passages.

LLMs do not reliably return clean JSON: they wrap it in ```json fences, add
a sentence before it, use single quotes, leave trailing commas, or write the
score as a string. parse_judgement() handles all of these and returns None
instead of crashing when nothing usable is there.
"""

import json
import re

JUDGE_PROMPT = """You are grading a study assistant's answer for FAITHFULNESS.
Faithfulness means every claim in the answer is supported by the passages.
Ignore whether the answer is complete; only check that it adds nothing
the passages do not say.

Scale:
5 = every claim is directly supported by the passages
4 = all main claims supported; a minor detail is a small stretch
3 = mostly supported, but at least one claim is not in the passages
2 = several claims are not in the passages
1 = the answer mostly contradicts or ignores the passages

Passages:
{passages}

Question: {question}

Answer to grade:
{answer}

Reply with JSON only, like: {{"score": 4, "reason": "one short sentence"}}"""


def build_judge_messages(question, answer, passages):
    text = "\n\n".join(f"[{n}] {p['text']}" for n, p in enumerate(passages, 1))
    return [{"role": "user", "content": JUDGE_PROMPT.format(
        passages=text, question=question, answer=answer)}]


def parse_judgement(text):
    """{"score": int 1-5, "reason": str} or None."""
    if not text:
        return None
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for raw in candidates:
        for attempt in (raw, _repair(raw)):
            try:
                data = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            score = _to_score(data.get("score") if isinstance(data, dict) else None)
            if score is not None:
                return {"score": score, "reason": str(data.get("reason", "")).strip()}
    # no usable JSON: fall back to a bare "score: 4" anywhere in the text
    m = re.search(r"score\W{0,4}([1-5])\b", text, flags=re.IGNORECASE)
    return {"score": int(m.group(1)), "reason": ""} if m else None


def _repair(raw):
    fixed = re.sub(r",\s*([}\]])", r"\1", raw)  # trailing commas
    if '"' not in fixed:
        fixed = fixed.replace("'", '"')  # single-quoted keys and strings
    return fixed


def _to_score(value):
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def judge(question, answer, passages, llm):
    return parse_judgement(llm.complete(build_judge_messages(question, answer, passages)))
