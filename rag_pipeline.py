"""
Movie RAG pipeline wrapped in a class for clarity.
Data -> embeddings (OpenAI) -> FAISS retrieval -> LLM answer -> structured JSON.
Environment: expects OPENAI_API_KEY from .env or shell.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def load_env_from_dotenv(path: str = ".env") -> None:
    """Lightweight .env loader to avoid extra deps."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Chunk:
    text: str
    title: str
    chunk_id: int


class MovieRAG:
    def __init__(
        self,
        csv_path: str,
        subset: int = 12000,
        chunk_size: int = 120,
        top_k: int = 3,
        cache_dir: str = ".rag_cache",
        use_cache: bool = True,
    ):
        load_env_from_dotenv()
        self.csv_path = csv_path
        self.subset = subset
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self._chunks: List[Chunk] = []
        self._index = None
        self._chunk_embeddings: np.ndarray | None = None

    # ---- Data loading and chunking ----
    def _load_movies(self) -> pd.DataFrame:
        # Sample broadly from the full CSV 
        df = pd.read_csv(self.csv_path, usecols=["Title", "Plot"])
        df = df.dropna(subset=["Title", "Plot"])
        df = df.sample(n=min(self.subset, len(df)), random_state=42)
        return df.reset_index(drop=True)

    def _chunk_plot(self, title: str, plot: str) -> List[Chunk]:
        words = plot.split()
        chunks: List[Chunk] = []
        for i in range(0, len(words), self.chunk_size):
            piece = " ".join(words[i : i + self.chunk_size])
            chunks.append(Chunk(text=f"{title} :: {piece}", title=title, chunk_id=len(chunks)))
        return chunks

    def _make_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for _, row in df.iterrows():
            all_chunks.extend(self._chunk_plot(row["Title"], row["Plot"]))
        return all_chunks

    # ---- Embeddings and FAISS ----
    def _embed_texts(self, texts: Sequence[str], model: str = "text-embedding-3-small", batch: int = 64) -> np.ndarray:
        from openai import OpenAI  #

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or your shell.")

        result_vectors: List[List[float]] = []
        for start in range(0, len(texts), batch):
            batch_texts = texts[start : start + batch]
            resp = client.embeddings.create(model=model, input=batch_texts)
            result_vectors.extend([item.embedding for item in resp.data])
        arr = np.array(result_vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        return arr / norms

    def _build_index(self, embeddings: np.ndarray):
        import faiss  # type: ignore

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    # ---- Caching ---------------------------------------------------------
    def _cache_paths(self) -> Tuple[Path, Path, Path]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        idx_path = self.cache_dir / "index.faiss"
        chunks_path = self.cache_dir / "chunks.json"
        meta_path = self.cache_dir / "meta.json"
        return idx_path, chunks_path, meta_path

    def _load_cache(self) -> bool:
        if not self.use_cache:
            return False
        idx_path, chunks_path, meta_path = self._cache_paths()
        if not (idx_path.exists() and chunks_path.exists() and meta_path.exists()):
            return False
        try:
            import faiss  # type: ignore
        except Exception:
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            
            csv_mtime = os.path.getmtime(self.csv_path)
            if (
                meta.get("subset") != self.subset
                or meta.get("chunk_size") != self.chunk_size
                or meta.get("csv_mtime") != csv_mtime
            ):
                return False
            index = faiss.read_index(str(idx_path))
            chunks_raw = json.loads(chunks_path.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**c) for c in chunks_raw]
            self._index = index
            return True
        except Exception:
            return False

    def _save_cache(self, index, chunks: List[Chunk]) -> None:
        if not self.use_cache:
            return
        try:
            import faiss  
        except Exception:
            return
        idx_path, chunks_path, meta_path = self._cache_paths()
        faiss.write_index(index, str(idx_path))
        chunks_path.write_text(json.dumps([asdict(c) for c in chunks]), encoding="utf-8")
        meta = {
            "subset": self.subset,
            "chunk_size": self.chunk_size,
            "csv_mtime": os.path.getmtime(self.csv_path),
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def prepare(self) -> None:
        """Load data, chunk plots, embed, and build FAISS index."""
        if self._load_cache():
            print("[ready] Loaded cached index and chunks. Ask away!")
            return
        print(f"[setup] Loading data (sample {self.subset} rows)...")
        df = self._load_movies()
        print("[setup] Chunking plots...")
        self._chunks = self._make_chunks(df)
        chunk_texts = [c.text for c in self._chunks]
        print(f"[setup] Embedding {len(chunk_texts)} chunks with OpenAI (this can take a moment)...")
        self._chunk_embeddings = self._embed_texts(chunk_texts)
        print("[setup] Building FAISS index...")
        self._index = self._build_index(self._chunk_embeddings)
        self._save_cache(self._index, self._chunks)
        print("[ready] Index ready. Ask away!")

    # ---- Retrieval and generation ----
    def _retrieve(self, query: str) -> List[Tuple[float, Chunk]]:
        if self._index is None:
            raise RuntimeError("Index not ready. Call prepare() first.")
        query_vec = self._embed_texts([query])
        pool_k = max(self.top_k * 3, self.top_k)
        scores, idxs = self._index.search(query_vec.astype(np.float32), pool_k)
        results: List[Tuple[float, Chunk, float]] = []
        q_terms = set(query.lower().split())
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            chk = self._chunks[int(i)]
            title_terms = set(chk.title.lower().split())
            overlap = len(q_terms & title_terms)
            boost = 0.1 * overlap
            results.append((float(score) + boost, chk, float(score)))
        results.sort(key=lambda x: x[0], reverse=True)
        trimmed = results[: self.top_k]
        return [(r[2], r[1]) for r in trimmed]

    def _call_llm(self, question: str, contexts: List[str]) -> str:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or your shell.")

        prompt = (
            "Answer the movie question using only the provided context. "
            "Include the movie title, a concise description, and key facts. "
            "If context is thin, say so.\n\n"
            "Context:\n- " + "\n- ".join(contexts) + "\n\n"
            f"Question: {question}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def ask(self, question: str) -> dict:
        """Run retrieval + generation, return structured JSON-ready dict."""
        retrieved = self._retrieve(question)
        contexts = [c.text for _, c in retrieved]
        answer = self._call_llm(question, contexts)
        titles = [c.title for _, c in retrieved]
        scores = [round(s, 3) for s, _ in retrieved]
        reasoning = (
            f"Embedded {len(self._chunks)} chunks; FAISS inner-product search on question embedding; "
            f"top titles: {titles} with scores {scores}."
        )
        return {"answer": answer, "contexts": contexts, "reasoning": reasoning}


def run_cli(csv_path: str = "wiki_movie_plots_deduped.csv") -> None:
    """Interactive CLI loop for multiple questions."""
    rag = MovieRAG(csv_path=csv_path)
    rag.prepare()
    print("Mini Movie RAG ready. Ask anything about movies (blank to exit).")
    while True:
        try:
            q = input("Your question: ").strip()
        except EOFError:
            break
        if not q:
            break
        result = rag.ask(q)
        print(json.dumps(result, indent=2))


__all__ = ["MovieRAG", "run_cli"]
