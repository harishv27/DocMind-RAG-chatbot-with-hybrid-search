# DocMind — RAG Chatbot

> A document-only RAG chatbot with hybrid search, multi-query retrieval, and page-level citations. Built with FastAPI + OpenAI + BM25.

---

## Architecture

```
User Query
    │
    ▼
Multi-Query Generation (GPT-4o-mini)
    │
    ▼
Hybrid Retrieval ──── Vector Search (OpenAI Embeddings)
    │            └─── BM25 Keyword Search
    ▼
Deduplication
    │
    ▼
Reranking (cosine similarity)
    │
    ▼
Context Expansion (±1 page)
    │
    ▼
Answer Generation + Citation Verification (GPT-4o-mini)
    │
    ▼
Final Answer with Page-Level Citations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-3-large |
| Keyword Search | BM25 (rank_bm25) |
| PDF Parsing | PyMuPDF |
| Frontend | Vanilla HTML / CSS / JS |

---

## Project Structure

```
DocMind/
├── backend/
│   ├── main.py                 ← FastAPI server & API routes
│   ├── document_processor.py   ← PDF/TXT page-aware chunking
│   ├── retriever.py            ← Hybrid search (Vector + BM25) + reranking
│   ├── llm.py                  ← Multi-query generation + answer + citations
│   ├── requirements.txt        ← Python dependencies
│   ├── .env.example            ← API key
│   └── uploads/                ← Uploaded files (git-ignored)
├── frontend/
│   └── index.html              ← Full chat UI (no framework)
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Miniconda or Anaconda
- OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/docmind-rag.git
cd docmind-rag
```

### 2. Create and activate conda environment

```bash
conda create -n env_name python=3.11 -y
conda activate env_name
```

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Configure API key

```bash
# Copy the template and add your key
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-your-key-here
```

### 5. Start the backend

```bash
conda activate env_name
cd backend
python main.py
```

Server starts at: `http://127.0.0.1:8000`

### 6. Open the frontend

**VS Code (recommended):**
- Install the **Live Server** extension
- Right-click `frontend/index.html` → **Open with Live Server**

**Direct:**
- Double-click `frontend/index.html` in File Explorer

---

## Features

- PDF, TXT, DOCX upload support
- Drag & drop file upload
- Hybrid search — semantic + keyword in one query
- Multi-query retrieval — auto-generates sub-queries for better recall
- Page-level citations — every answer cites exact filename + page number
- Context expansion — pulls adjacent pages for richer context
- 38+ chunks used per query (vs 3–5 in basic RAG)
- Clean dark UI — no framework, pure HTML/CSS/JS

---