# Mini RAG System (Movie Plots)

Lightweight RAG pipeline over the Wikipedia movie plots CSV. Loads data, chunks plots, embeds with OpenAI, stores vectors in FAISS, retrieves top-k, and answers via OpenAI chat. Outputs structured JSON (`answer`, `contexts`, `reasoning`).

## Quick start

Download the CSV Dataset and store in root directory as it was a big file in size couldn't upload it
Link: https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots?resource=download

" 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 "
Add this in env file

1) Install deps (system/global or inside a venv—either works):
   ```
   pip install "numpy<2" "pandas<2.2" faiss-cpu openai
   ```
   - If your base Python already has conflicting NumPy binaries, use a venv to isolate. Otherwise, installing globally is fine.
   
2) Set your OpenAI key in `.env` (already present if you placed it):
   ```
   OPENAI_API_KEY=your_key_here
   ```
3) Run the CLI:
   ```
   python cli.py
   ```
   First run builds embeddings and a FAISS index (takes time). The cached FAISS index was ~265 MB, so it isn’t in git; a fresh clone must rebuild once. Subsequent runs load from `.rag_cache` and are instant.

## How it works
- Reads a random sample of the CSV (default 6,000 rows across all years for coverage).
- Chunks plots (~120 words).
- Embeds chunks with `text-embedding-3-small`.
- Builds an in-memory FAISS index.
- Embeds your question, retrieves top-k matches, and feeds the snippets to `gpt-4o-mini`.
- Returns JSON with the LLM answer, the retrieved snippets, and reasoning (titles + similarity scores).
