# Document Intelligence (RAG) — NidusLab AI SWE Intern Assessment

A small web app that lets you upload a PDF/TXT, ask questions about it, and
get an answer with the exact source passages (page + excerpt) it was based
on.

## Features

- **Document processing**: PDF/TXT upload → text extraction → cleanup
  (dehyphenation, whitespace/control-char normalization) → page-aware,
  sentence-boundary chunking with overlap.
- **Semantic search**: local `sentence-transformers` (`all-MiniLM-L6-v2`)
  embeddings, stored and queried via **ChromaDB** (persisted to disk).
- **RAG pipeline**: retrieved chunks are injected into a grounded prompt
  sent to **Gemini** (free tier); the model is explicitly instructed to
  answer only from the provided excerpts and to say when the document
  doesn't contain the answer.
- **Source attribution**: every answer is returned with the page number,
  a snippet of the source chunk, and which retrieval method(s) matched it.
- **Bonus 1 — Conversation memory**: per-session history is used to rewrite
  follow-up questions (e.g. "Who proposed this approach?") into standalone
  questions before retrieval, and is included in the generation prompt.
- **Bonus 2 — Hybrid search**: semantic search (Chroma) is combined with
  keyword search (BM25 via `rank_bm25`) using **Reciprocal Rank Fusion**.
  You can switch between semantic / keyword / hybrid in the UI to compare
  results directly.
- **Bonus 3 — Evaluation harness**: `backend/evaluation.py` runs a fixed
  set of questions against an uploaded document and reports retrieval
  accuracy, answer relevance, an estimated hallucination rate (answer/
  context token-overlap), a refusal-probe pass rate (does the model
  correctly decline to answer an out-of-scope question), and latency.
- **Bonus 4 — Streaming**: `/api/ask/stream` is a Server-Sent-Events
  endpoint; the frontend renders tokens as they arrive.
- **Bonus 5 — Docker**: `docker-compose up` builds and runs the whole app.

## Why this stack

- **Local embeddings, hosted generation.** Embedding is the
  highest-volume call in a RAG system (once per chunk, once per query).
  Running that locally with `sentence-transformers` avoids burning a
  limited free-tier quota and keeps retrieval fast and available even if
  the LLM API is rate-limited. Gemini is used only for the (much lower
  volume) generation step, where its free tier is generous.
- **RRF over score-normalization for hybrid search.** BM25 scores and
  cosine-similarity scores live on incomparable scales. Reciprocal Rank
  Fusion sidesteps that by fusing on *rank* rather than raw score, which
  is simple, has no tunable normalization constant, and is the standard
  approach for hybrid retrieval.
- **Page-aware, sentence-boundary chunking.** Splitting on sentence
  boundaries (instead of a hard character cut) keeps chunks semantically
  coherent, which measurably helps both embedding quality and how
  readable the "source excerpt" shown to the user is.
- **Hallucination control via prompt constraints + a measurable proxy.**
  Rather than a hallucination *filter*, the system prompt makes refusal
  the expected behavior when context is insufficient, and the eval
  harness includes an explicit out-of-scope probe question ("What is the
  capital of France?" against a technical document) to check that the
  model actually does refuse instead of answering from world knowledge.

## Project structure

```
niduslab-rag/
├── backend/
│   ├── main.py                # FastAPI app: upload / ask / ask/stream
│   ├── document_processor.py  # extract → clean → chunk
│   ├── embeddings.py          # local sentence-transformers wrapper
│   ├── vectorstore.py         # Chroma (semantic) + BM25 (keyword) + RRF
│   ├── rag.py                 # prompt construction + Gemini calls
│   ├── memory.py              # per-session conversation history
│   ├── models.py              # Pydantic request/response schemas
│   ├── evaluation.py          # Bonus 3 eval harness
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── app.js                 # upload, ask, SSE streaming
│   └── style.css
├── docker-compose.yml
├── .env.example
└── README.md
```

## Setup

### 1. Get a free Gemini API key
Visit https://aistudio.google.com/apikey, create a key, then:
```bash
cp .env.example .env
# edit .env and paste your key into GEMINI_API_KEY
```

### 2a. Run with Docker
```bash
docker-compose up --build
```
Then open http://localhost:8000

### 2b. Run locally without Docker
```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env   # then edit ../.env with your key
uvicorn main:app --reload --port 8000
```
Then open http://localhost:8000 (the backend also serves the frontend as
static files).

> Note: `sentence-transformers` pulls in PyTorch (~1–2 GB) on first
> install — this is expected and only needs to happen once.

### 3. Use it
1. Upload a PDF or TXT file.
2. Ask a question. Toggle "Stream response" and the search mode
   (hybrid / semantic / keyword) to compare behavior.

### 4. Run the evaluation harness 
With the server running and a document already uploaded (grab the
`doc_id` from the upload response or the browser network tab):
```bash
cd backend
python3 evaluation.py <doc_id>
```
This prints a summary (retrieval accuracy, answer relevance, estimated
hallucination rate, refusal-probe pass rate, avg/p95 latency) and writes
per-question detail to `eval_results.json`. The default question set is
written for this assessment PDF itself — pass `--questions your.json` to
evaluate against a different document (format: a JSON list of
`{"question": ..., "expect_any": [...]}` objects).

Tested end-to-end locally (upload → chunk → embed → hybrid retrieval →
Gemini generation → streaming → source attribution) against the
assessment PDF itself, as well as via `docker-compose up --build`.
