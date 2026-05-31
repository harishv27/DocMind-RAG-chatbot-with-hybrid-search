import re
import asyncio
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def process(self, file_path: str, doc_id: str, filename: str) -> List[Dict]:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            pages = self._extract_pdf(file_path)
        elif ext == ".txt":
            pages = self._extract_txt(file_path)
        else:
            pages = self._extract_txt(file_path)  # fallback

        chunks = self._chunk_pages(pages, doc_id, filename)
        return chunks

    def _extract_pdf(self, path: str) -> List[Dict]:
        pages = []
        doc = fitz.open(path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "page": page_num + 1,
                    "text": text
                })
        doc.close()
        return pages

    def _extract_txt(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # Split into fake pages every ~2000 chars
        chunk_size = 2000
        pages = []
        for i, start in enumerate(range(0, len(text), chunk_size)):
            pages.append({
                "page": i + 1,
                "text": text[start:start + chunk_size]
            })
        return pages

    def _chunk_pages(self, pages: List[Dict], doc_id: str, filename: str) -> List[Dict]:
        chunks = []
        chunk_idx = 0
        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]
            # Split page into overlapping chunks
            words = text.split()
            for start in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[start:start + self.chunk_size]
                chunk_text = " ".join(chunk_words).strip()
                if len(chunk_text) < 30:
                    continue
                chunks.append({
                    "chunk_id": f"{doc_id}_p{page_num}_c{chunk_idx}",
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_num,
                    "text": chunk_text,
                    "chunk_index": chunk_idx
                })
                chunk_idx += 1
        return chunks
