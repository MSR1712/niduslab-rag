from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from document_processor import process_document
from memory import ConversationMemory
from models import AskRequest, AskResponse, SourceChunk, UploadResponse
from rag import generate_answer, resolve_followup, stream_answer
from vectorstore import StoreRegistry

app = FastAPI(title="NidusLab Document Intelligence (RAG) API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

registry = StoreRegistry()
memory = ConversationMemory()

# doc_id -> {"filename": str, "num_pages": int, "num_chunks": int}
DOCS: Dict[str, dict] = {}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF and TXT files are supported.")

    doc_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    content = await file.read()
    dest.write_bytes(content)

    try:
        chunks = process_document(str(dest), file.filename, doc_id)
    except Exception as e:
        raise HTTPException(400, f"Failed to process document: {e}")

    if not chunks:
        raise HTTPException(400, "No extractable text found in the document.")

    store = registry.get_or_create(doc_id)
    store.add_chunks(chunks)

    num_pages = max(c.page for c in chunks)
    DOCS[doc_id] = {
        "filename": file.filename,
        "num_pages": num_pages,
        "num_chunks": len(chunks),
    }

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_pages=num_pages,
        num_chunks=len(chunks),
    )


@app.get("/api/documents")
async def list_documents():
    return [{"doc_id": k, **v} for k, v in DOCS.items()]


def _snippet(text: str, length: int = 220) -> str:
    text = text.strip()
    return text if len(text) <= length else text[:length].rsplit(" ", 1)[0] + "..."


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    start = time.time()
    store = registry.get(req.doc_id)
    if store is None:
        raise HTTPException(404, "Unknown doc_id. Upload the document first.")

    history_text = memory.as_prompt_context(req.session_id)
    resolved_question = resolve_followup(req.question, history_text)

    retrieved = store.search(resolved_question, top_k=req.top_k, mode=req.search_mode)
    answer = generate_answer(resolved_question, retrieved, history_text)

    memory.add_turn(req.session_id, req.question, answer)

    sources = [
        SourceChunk(
            chunk_id=c.chunk_id,
            page=c.page,
            snippet=_snippet(c.text),
            semantic_score=c.semantic_score,
            keyword_score=c.keyword_score,
            combined_score=c.combined_score,
            matched_by=c.matched_by or [],
        )
        for c in retrieved
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        resolved_question=resolved_question,
        latency_ms=round((time.time() - start) * 1000, 1),
    )


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    store = registry.get(req.doc_id)
    if store is None:
        raise HTTPException(404, "Unknown doc_id. Upload the document first.")

    history_text = memory.as_prompt_context(req.session_id)
    resolved_question = resolve_followup(req.question, history_text)
    retrieved = store.search(resolved_question, top_k=req.top_k, mode=req.search_mode)

    def event_gen():
        sources_payload = [
            {
                "chunk_id": c.chunk_id,
                "page": c.page,
                "snippet": _snippet(c.text),
                "matched_by": c.matched_by or [],
            }
            for c in retrieved
        ]
        yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"

        full_answer = []
        for token in stream_answer(resolved_question, retrieved, history_text):
            full_answer.append(token)
            yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

        memory.add_turn(req.session_id, req.question, "".join(full_answer))
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# Serve the simple frontend (index.html + app.js) as static files.
# Locally: backend/main.py -> ../frontend. In the Docker image the frontend
# is copied to /frontend (sibling of /app), which resolves the same way.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
