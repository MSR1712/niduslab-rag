"""Extract text from PDF/TXT files, clean it, and split it into overlapping,
page-aware chunks with metadata (page number, chunk id).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple

from pypdf import PdfReader


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class DocChunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str


def extract_text(file_path: str, filename: str) -> List[PageText]:
    """Return a list of (page_number, raw_text). TXT files are treated as a
    single 'page' (page=1)."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            pages.append(PageText(page=i, text=raw))
        return pages
    elif filename.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        return [PageText(page=1, text=raw)]
    else:
        raise ValueError("Unsupported file type. Only PDF and TXT are accepted.")


def clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, drop control characters, fix
    hyphenated line-breaks, strip page-number-only lines."""
    if not text:
        return ""
    # Remove null / control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # Fix words broken across a line by a hyphen, e.g. "docu-\nment" -> "document"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse remaining newlines into spaces
    text = re.sub(r"\s*\n\s*", " ", text)
    # Collapse runs of whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Drop lines that are just a number (stray page numbers)
    text = re.sub(r"\b\d{1,4}\b(?=\s*$)", "", text)
    return text.strip()


def chunk_pages(
    doc_id: str,
    pages: List[PageText],
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[DocChunk]:
    """Split each page's cleaned text into overlapping character-window
    chunks, preserving the source page number as metadata. Splits on
    sentence boundaries where possible so chunks stay 'meaningful'."""
    chunks: List[DocChunk] = []

    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= chunk_size:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(
                        DocChunk(
                            chunk_id=f"{doc_id}-{uuid.uuid4().hex[:8]}",
                            doc_id=doc_id,
                            page=page.page,
                            text=current,
                        )
                    )
                # start new chunk, carry over overlap tail of previous chunk
                tail = current[-overlap:] if overlap and current else ""
                current = f"{tail} {sentence}".strip()

        if current:
            chunks.append(
                DocChunk(
                    chunk_id=f"{doc_id}-{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    page=page.page,
                    text=current,
                )
            )

    return chunks


def process_document(file_path: str, filename: str, doc_id: str) -> List[DocChunk]:
    pages = extract_text(file_path, filename)
    return chunk_pages(doc_id, pages)
