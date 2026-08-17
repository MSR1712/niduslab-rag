"""Retrieval-Augmented Generation pipeline built on Google Gemini
(free tier). Two responsibilities live here:

1. Resolve follow-up questions against conversation history (Bonus 1) so
   retrieval uses a self-contained query instead of a bare pronoun.
2. Build a grounded prompt and call the LLM, instructing it to answer only
   from the provided chunks and say so explicitly when the answer isn't
   present (this is what keeps hallucination down).
"""
from __future__ import annotations

import os
from typing import Iterable, List

import google.generativeai as genai

from memory import ConversationMemory
from vectorstore import RetrievedChunk

GENERATION_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

SYSTEM_INSTRUCTION = (
    "You are a document question-answering assistant. Answer the user's "
    "question using ONLY the information in the numbered source excerpts "
    "below. Do not use outside knowledge and do not guess. "
    "If the excerpts do not contain the answer, say clearly that the "
    "document does not appear to contain that information instead of "
    "inventing one. Keep answers concise. When you use a fact from an "
    "excerpt, you don't need inline citation markers — sources are shown "
    "separately to the user."
)


def _configure():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and put it in your .env file."
        )
    genai.configure(api_key=api_key)


def resolve_followup(question: str, history_text: str) -> str:
    """Rewrite a possibly-elliptical follow-up question into a standalone
    one using the recent conversation, e.g. 'Who proposed this approach?'
    -> 'Who proposed the deep learning approach used in the study?'."""
    if not history_text:
        return question

    _configure()
    model = genai.GenerativeModel(GENERATION_MODEL)
    prompt = (
        "Given this recent conversation:\n"
        f"{history_text}\n\n"
        f"Rewrite the follow-up question so it is a fully standalone "
        f"question that does not depend on the conversation above. "
        f"Only output the rewritten question, nothing else.\n\n"
        f"Follow-up question: {question}"
    )
    try:
        resp = model.generate_content(prompt)
        rewritten = (resp.text or "").strip().strip('"')
        return rewritten if rewritten else question
    except Exception:
        # If rewriting fails for any reason, fall back to the raw question
        # rather than breaking the whole request.
        return question


def _build_prompt(question: str, chunks: List[RetrievedChunk], history_text: str) -> str:
    excerpt_blocks = []
    for i, c in enumerate(chunks, start=1):
        excerpt_blocks.append(f"[{i}] (page {c.page}) {c.text}")
    excerpts = "\n\n".join(excerpt_blocks) if excerpt_blocks else "(no excerpts retrieved)"

    history_block = f"Previous conversation:\n{history_text}\n\n" if history_text else ""

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"{history_block}"
        f"Source excerpts:\n{excerpts}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def generate_answer(question: str, chunks: List[RetrievedChunk], history_text: str = "") -> str:
    _configure()
    model = genai.GenerativeModel(GENERATION_MODEL)
    prompt = _build_prompt(question, chunks, history_text)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def stream_answer(question: str, chunks: List[RetrievedChunk], history_text: str = "") -> Iterable[str]:
    _configure()
    model = genai.GenerativeModel(GENERATION_MODEL)
    prompt = _build_prompt(question, chunks, history_text)
    for event in model.generate_content(prompt, stream=True):
        if event.text:
            yield event.text
