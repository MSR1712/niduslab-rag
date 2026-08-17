"""Pydantic schemas shared across the API."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    text: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int


class SourceChunk(BaseModel):
    chunk_id: str
    page: int
    snippet: str
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    combined_score: Optional[float] = None
    matched_by: List[str] = []  # ["semantic"], ["keyword"], or both


class AskRequest(BaseModel):
    doc_id: str
    question: str
    session_id: Optional[str] = "default"
    top_k: int = 4
    search_mode: str = "hybrid"  # "semantic" | "keyword" | "hybrid"


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    resolved_question: str  # question after coreference resolution w/ history
    latency_ms: float
