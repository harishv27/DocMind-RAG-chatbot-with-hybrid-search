import os
import sys
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional

# ── Windows event loop fix ──────────────────────────────────────────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import aiofiles

from document_processor import DocumentProcessor
from retriever import HybridRetriever
from llm import LLMHandler

app = FastAPI(title="DocMind RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

processor = DocumentProcessor()
retriever  = HybridRetriever()
llm        = LLMHandler()

# ── Serve frontend ──────────────────────────────────────────────────────────
FRONTEND = Path(__file__).parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND)), name="frontend")

@app.get("/")
async def root():
    index = FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "DocMind RAG API is running. Open frontend/index.html in your browser."}

# ── Schemas ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[dict]
    chunks_used: int

# ── Routes ───────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    fname = file.filename or ""
    if not fname.lower().endswith((".pdf", ".txt", ".docx")):
        raise HTTPException(400, "Only PDF, TXT, or DOCX files are supported.")

    doc_id    = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}_{fname}"

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(await file.read())

    chunks = await processor.process(str(save_path), doc_id, fname)
    await retriever.index_chunks(chunks)

    return {
        "doc_id":   doc_id,
        "filename": fname,
        "chunks":   len(chunks),
        "status":   "indexed",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. Generate sub-queries for multi-query retrieval
    sub_queries = await llm.generate_sub_queries(req.query)
    all_queries = [req.query] + sub_queries

    # 2. Hybrid retrieval across all queries
    all_chunks = []
    for q in all_queries:
        chunks = await retriever.retrieve(q, doc_ids=req.document_ids, top_k=15)
        all_chunks.extend(chunks)

    # 3. Deduplicate by chunk_id
    seen: set = set()
    unique: List[dict] = []
    for c in all_chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique.append(c)

    # 4. Rerank
    reranked = await retriever.rerank(req.query, unique, top_n=20)

    # 5. Expand context (adjacent pages)
    expanded = await retriever.expand_context(reranked)

    # 6. Generate answer with citations
    answer, citations = await llm.generate_answer(req.query, expanded)

    return ChatResponse(answer=answer, citations=citations, chunks_used=len(expanded))


@app.get("/documents")
async def list_documents():
    return retriever.get_indexed_documents()


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    retriever.delete_document(doc_id)
    # Also delete file from disk
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)
    return {"status": "deleted"}


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
