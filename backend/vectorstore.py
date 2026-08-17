"""Storage and retrieval layer.

- Semantic search: ChromaDB, persisted to disk, one collection per doc_id.
- Keyword search: an in-memory BM25 index per doc_id (rank_bm25).
- Hybrid search: Reciprocal Rank Fusion (RRF) of the two ranked lists, so
  results from either method can contribute even if their raw scores are on
  different scales.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import chromadb
from rank_bm25 import BM25Okapi

from document_processor import DocChunk
from embeddings import embed_texts, embed_query

CHROMA_DIR = "data/chroma"


@dataclass
class RetrievedChunk:
    chunk_id: str
    page: int
    text: str
    semantic_score: float | None = None
    keyword_score: float | None = None
    combined_score: float | None = None
    matched_by: List[str] | None = None


class DocumentStore:
    """Holds everything needed to search one uploaded document."""

    def __init__(self, doc_id: str, client: "chromadb.ClientAPI"):
        self.doc_id = doc_id
        self.collection = client.get_or_create_collection(name=f"doc_{doc_id}")
        self.bm25: BM25Okapi | None = None
        self.chunks_by_id: Dict[str, DocChunk] = {}
        self.bm25_ids: List[str] = []

    def add_chunks(self, chunks: List[DocChunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[{"page": c.page, "doc_id": c.doc_id} for c in chunks],
        )

        for c in chunks:
            self.chunks_by_id[c.chunk_id] = c

        tokenized = [_tokenize(t) for t in [c.text for c in self.chunks_by_id.values()]]
        self.bm25_ids = list(self.chunks_by_id.keys())
        self.bm25 = BM25Okapi(tokenized)

    def semantic_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        q_vec = embed_query(query)
        result = self.collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=min(top_k, max(len(self.chunks_by_id), 1)),
        )
        out = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns squared-L2 (or cosine) distance; convert to a
            # similarity-like score in [0, 1] for display purposes.
            score = 1.0 / (1.0 + dist)
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    page=meta["page"],
                    text=doc,
                    semantic_score=round(float(score), 4),
                    matched_by=["semantic"],
                )
            )
        return out

    def keyword_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self.bm25_ids, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        out = []
        for cid, score in ranked:
            if score <= 0:
                continue
            chunk = self.chunks_by_id[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    page=chunk.page,
                    text=chunk.text,
                    keyword_score=round(float(score), 4),
                    matched_by=["keyword"],
                )
            )
        return out

    def hybrid_search(self, query: str, top_k: int, rrf_k: int = 60) -> List[RetrievedChunk]:
        """Reciprocal Rank Fusion of semantic + keyword rankings."""
        sem = self.semantic_search(query, top_k=max(top_k, 8))
        kw = self.keyword_search(query, top_k=max(top_k, 8))

        fused: Dict[str, RetrievedChunk] = {}
        for rank, item in enumerate(sem):
            fused[item.chunk_id] = item
            fused[item.chunk_id].combined_score = 1.0 / (rrf_k + rank + 1)

        for rank, item in enumerate(kw):
            if item.chunk_id in fused:
                existing = fused[item.chunk_id]
                existing.keyword_score = item.keyword_score
                existing.matched_by = ["semantic", "keyword"]
                existing.combined_score += 1.0 / (rrf_k + rank + 1)
            else:
                item.combined_score = 1.0 / (rrf_k + rank + 1)
                fused[item.chunk_id] = item

        ranked = sorted(fused.values(), key=lambda x: x.combined_score, reverse=True)
        for r in ranked:
            r.combined_score = round(r.combined_score, 5)
        return ranked[:top_k]

    def search(self, query: str, top_k: int, mode: str) -> List[RetrievedChunk]:
        if mode == "semantic":
            return self.semantic_search(query, top_k)
        if mode == "keyword":
            return self.keyword_search(query, top_k)
        return self.hybrid_search(query, top_k)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class StoreRegistry:
    """Keeps one DocumentStore per uploaded document in memory, backed by a
    persistent Chroma client so embeddings survive a server restart."""

    def __init__(self, persist_dir: str = CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._stores: Dict[str, DocumentStore] = {}

    def get_or_create(self, doc_id: str) -> DocumentStore:
        if doc_id not in self._stores:
            self._stores[doc_id] = DocumentStore(doc_id, self.client)
        return self._stores[doc_id]

    def get(self, doc_id: str) -> DocumentStore | None:
        return self._stores.get(doc_id)
