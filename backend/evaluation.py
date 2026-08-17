"""Bonus 3 — Evaluation harness.

Runs a fixed set of question/expected-keyword pairs against an already
uploaded document and reports:

  - Retrieval accuracy : did at least one retrieved chunk contain an
                          expected keyword/phrase for that question?
  - Answer relevance   : does the generated answer contain an expected
                          keyword/phrase? (cheap proxy; an LLM-judge could
                          replace this, noted in the README)
  - Hallucination rate  : is the answer grounded in the retrieved chunks?
                          approximated via token overlap between the answer
                          and the retrieved context.
  - Latency             : wall-clock time per question (ms).

Usage:
    python evaluation.py <doc_id> [--questions questions.json]

The server must already be running (this hits the HTTP API) OR you can
import and call the pipeline functions directly if run standalone.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from typing import List

import requests

DEFAULT_QUESTIONS = [
    {
        "question": "What is the objective of the system described in this document?",
        "expect_any": ["document intelligence", "extract", "answer", "evidence"],
    },
    {
        "question": "What file types can be uploaded?",
        "expect_any": ["pdf", "txt"],
    },
    {
        "question": "What backend framework is recommended?",
        "expect_any": ["fastapi", "python"],
    },
    {
        "question": "What vector databases are recommended?",
        "expect_any": ["faiss", "chroma", "chromadb"],
    },
    {
        "question": "What is required for source attribution?",
        "expect_any": ["page", "source", "chunk"],
    },
    {
        "question": "What does Bonus 1 ask you to implement?",
        "expect_any": ["conversation memory", "follow-up", "follow up"],
    },
    {
        "question": "What does Bonus 2 ask you to implement?",
        "expect_any": ["hybrid search", "keyword"],
    },
    {
        "question": "What is the submission deadline?",
        "expect_any": ["18 august", "august 2026", "midnight"],
    },
    {
        "question": "What is the capital of France?",  # out-of-scope probe
        "expect_any": [],
        "expect_refusal": True,
    },
]


def _contains_any(text: str, phrases: List[str]) -> bool:
    text_l = text.lower()
    return any(p.lower() in text_l for p in phrases)


def _token_overlap_ratio(answer: str, context: str) -> float:
    ans_tokens = set(re.findall(r"[a-z]{4,}", answer.lower()))
    ctx_tokens = set(re.findall(r"[a-z]{4,}", context.lower()))
    if not ans_tokens:
        return 1.0
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)


def run_eval(base_url: str, doc_id: str, questions: List[dict]) -> dict:
    results = []
    for q in questions:
        start = time.time()
        resp = requests.post(
            f"{base_url}/api/ask",
            json={"doc_id": doc_id, "question": q["question"], "search_mode": "hybrid"},
            timeout=60,
        )
        latency_ms = (time.time() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        answer = data["answer"]
        context = " ".join(s["snippet"] for s in data["sources"])

        retrieval_hit = (
            _contains_any(context, q["expect_any"]) if q.get("expect_any") else None
        )
        answer_relevant = (
            _contains_any(answer, q["expect_any"]) if q.get("expect_any") else None
        )
        overlap = _token_overlap_ratio(answer, context)

        if q.get("expect_refusal"):
            # For out-of-scope questions we WANT the model to say it can't
            # find the answer, rather than confidently answering.
            refused = any(
                p in answer.lower()
                for p in ["does not appear", "not contain", "cannot find", "no information", "don't know", "not mentioned"]
            )
            results.append(
                {
                    "question": q["question"],
                    "type": "refusal_probe",
                    "correctly_refused": refused,
                    "answer": answer,
                    "latency_ms": round(latency_ms, 1),
                }
            )
            continue

        results.append(
            {
                "question": q["question"],
                "retrieval_hit": retrieval_hit,
                "answer_relevant": answer_relevant,
                "grounding_overlap": round(overlap, 2),
                "answer": answer,
                "latency_ms": round(latency_ms, 1),
            }
        )

    scored = [r for r in results if "retrieval_hit" in r]
    refusal_probes = [r for r in results if r.get("type") == "refusal_probe"]

    summary = {
        "num_questions": len(results),
        "retrieval_accuracy": round(
            sum(1 for r in scored if r["retrieval_hit"]) / len(scored), 2
        )
        if scored
        else None,
        "answer_relevance": round(
            sum(1 for r in scored if r["answer_relevant"]) / len(scored), 2
        )
        if scored
        else None,
        "avg_grounding_overlap": round(
            statistics.mean(r["grounding_overlap"] for r in scored), 2
        )
        if scored
        else None,
        "estimated_hallucination_rate": round(
            1 - statistics.mean(r["grounding_overlap"] for r in scored), 2
        )
        if scored
        else None,
        "refusal_probe_pass_rate": round(
            sum(1 for r in refusal_probes if r["correctly_refused"]) / len(refusal_probes),
            2,
        )
        if refusal_probes
        else None,
        "avg_latency_ms": round(statistics.mean(r["latency_ms"] for r in results), 1),
        "p95_latency_ms": round(
            sorted(r["latency_ms"] for r in results)[
                max(0, int(len(results) * 0.95) - 1)
            ],
            1,
        ),
    }

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_id", help="doc_id returned by /api/upload")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--questions", default=None, help="path to a JSON file of questions")
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    if args.questions:
        with open(args.questions) as f:
            questions = json.load(f)
    else:
        questions = DEFAULT_QUESTIONS

    report = run_eval(args.base_url, args.doc_id, questions)
    print(json.dumps(report["summary"], indent=2))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull results written to {args.out}")
