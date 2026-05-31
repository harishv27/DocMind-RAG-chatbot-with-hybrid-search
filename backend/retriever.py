import os
import math
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

class HybridRetriever:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # In-memory store (replace with Qdrant for production)
        self._chunks: List[Dict] = []
        self._embeddings: List[List[float]] = []
        self._doc_registry: Dict[str, Dict] = {}

    async def index_chunks(self, chunks: List[Dict]):
        if not chunks:
            return
        # Register document
        first = chunks[0]
        self._doc_registry[first["doc_id"]] = {
            "doc_id": first["doc_id"],
            "filename": first["filename"],
            "chunk_count": len(chunks)
        }
        # Get embeddings in batches
        texts = [c["text"] for c in chunks]
        embeddings = await self._embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            self._chunks.append(chunk)
            self._embeddings.append(emb)

    async def _embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = await self.client.embeddings.create(
                model="text-embedding-3-large",
                input=batch
            )
            all_embeddings.extend([d.embedding for d in resp.data])
        return all_embeddings

    async def retrieve(self, query: str, doc_ids: Optional[List[str]] = None, top_k: int = 15) -> List[Dict]:
        if not self._chunks:
            return []

        # Filter by doc_ids if provided
        candidate_indices = [
            i for i, c in enumerate(self._chunks)
            if doc_ids is None or c["doc_id"] in doc_ids
        ]
        if not candidate_indices:
            return []

        candidates = [self._chunks[i] for i in candidate_indices]
        cand_embeddings = [self._embeddings[i] for i in candidate_indices]

        # Vector search
        query_emb = (await self._embed_batch([query]))[0]
        vector_scores = [self._cosine(query_emb, e) for e in cand_embeddings]

        # BM25 search
        tokenized_corpus = [c["text"].lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())

        # Normalize and combine
        v_max = max(vector_scores) if vector_scores else 1
        b_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
        norm_v = [s / v_max for s in vector_scores]
        norm_b = [s / b_max for s in bm25_scores]
        hybrid = [0.6 * v + 0.4 * b for v, b in zip(norm_v, norm_b)]

        # Sort and return top_k
        ranked = sorted(
            zip(hybrid, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        results = []
        for score, chunk in ranked[:top_k]:
            c = dict(chunk)
            c["score"] = score
            results.append(c)
        return results

    async def rerank(self, query: str, chunks: List[Dict], top_n: int = 20) -> List[Dict]:
        """Simple reranking using OpenAI embeddings similarity (replace with BGE reranker for production)"""
        if not chunks:
            return []
        query_emb = (await self._embed_batch([query]))[0]
        scored = []
        for chunk in chunks:
            # Find embedding
            idx = next((i for i, c in enumerate(self._chunks) if c["chunk_id"] == chunk["chunk_id"]), None)
            if idx is not None:
                score = self._cosine(query_emb, self._embeddings[idx])
                c = dict(chunk)
                c["rerank_score"] = score
                scored.append(c)
            else:
                chunk["rerank_score"] = chunk.get("score", 0)
                scored.append(chunk)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]

    async def expand_context(self, chunks: List[Dict]) -> List[Dict]:
        """For each chunk, also pull adjacent page chunks if available"""
        expanded_ids = set()
        expanded = []
        for chunk in chunks:
            if chunk["chunk_id"] in expanded_ids:
                continue
            expanded_ids.add(chunk["chunk_id"])
            expanded.append(chunk)
            # Get neighbors (same doc, page ±1)
            neighbors = [
                c for c in self._chunks
                if c["doc_id"] == chunk["doc_id"]
                and abs(c["page"] - chunk["page"]) <= 1
                and c["chunk_id"] != chunk["chunk_id"]
                and c["chunk_id"] not in expanded_ids
            ]
            for n in neighbors[:2]:  # at most 2 neighbors per chunk
                expanded_ids.add(n["chunk_id"])
                expanded.append(n)
        return expanded

    def get_indexed_documents(self) -> List[Dict]:
        return list(self._doc_registry.values())

    def delete_document(self, doc_id: str):
        self._chunks = [c for c in self._chunks if c["doc_id"] != doc_id]
        self._doc_registry.pop(doc_id, None)

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
