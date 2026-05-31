import os
import json
import re
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMHandler:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"   # works on openai v1.x and v2.x

    async def generate_sub_queries(self, query: str) -> List[str]:
        """Generate alternative queries for multi-query retrieval."""
        prompt = (
            "Given this question, generate 3 alternative search queries to find relevant information.\n"
            "Return ONLY a JSON array of strings, no explanation, no markdown.\n\n"
            f"Question: {query}\n\n"
            'Example output: ["query1", "query2", "query3"]'
        )
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
            queries = json.loads(raw)
            return queries[:3] if isinstance(queries, list) else []
        except Exception:
            return []

    async def generate_answer(
        self, query: str, chunks: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """Generate answer with inline citations."""
        if not chunks:
            return (
                "I couldn't find relevant information in the uploaded documents.",
                [],
            )

        # Build numbered context blocks
        context_parts = []
        source_map: Dict[str, Dict] = {}
        for i, chunk in enumerate(chunks):
            marker = f"[SOURCE_{i + 1}]"
            source_map[marker] = {
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            }
            context_parts.append(f"{marker}\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a precise document analysis assistant.\n"
            "Answer questions ONLY using the provided document context.\n"
            "Rules:\n"
            "1. Cite sources using [SOURCE_N] markers inline in your answer.\n"
            "2. Cite all relevant sources if information spans multiple chunks.\n"
            "3. If the answer is not in the context, say so clearly.\n"
            "4. Be specific — include exact values, dates, and names when present.\n"
            "5. Use clear sections for multi-part answers."
        )

        user_prompt = (
            f"Document context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer (use [SOURCE_N] citations):"
        )

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.1,
        )

        answer: str = resp.choices[0].message.content.strip()

        # Collect cited sources
        citations = []
        seen: set = set()
        for marker, info in source_map.items():
            if marker in answer:
                key = f"{info['filename']}:p{info['page']}"
                if key not in seen:
                    seen.add(key)
                    citations.append(
                        {
                            "filename": info["filename"],
                            "page": info["page"],
                            "excerpt": (
                                info["text"][:200] + "..."
                                if len(info["text"]) > 200
                                else info["text"]
                            ),
                        }
                    )

        # Replace [SOURCE_N] markers with readable inline refs
        answer = self._replace_markers(answer, source_map)
        return answer, citations

    def _replace_markers(self, answer: str, source_map: Dict) -> str:
        """Turn [SOURCE_N] into [Filename, p.N] inline."""
        def replacer(m: re.Match) -> str:
            marker = m.group(0)
            info = source_map.get(marker, {})
            fname = info.get("filename", "Document")
            page = info.get("page", "?")
            return f"[{fname}, p.{page}]"

        return re.sub(r"\[SOURCE_\d+\]", replacer, answer)
