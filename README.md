# RAG-PT-QA — Minimal PyTorch RAG (Embeddings + Local Generator)

A minimal, **recruiter-friendly** Retrieval-Augmented Generation (RAG) demo that:
- uses **PyTorch** + `sentence-transformers` for **embeddings**,
- stores vectors locally with **FAISS**,
- answers questions with a small **local LLM** (`google/flan-t5-small`, via `transformers`).

No paid APIs required. Runs on CPU.

## Why this is good for your portfolio
- Shows **LLM app fundamentals** without relying on external APIs.
- Clear separation of **ingestion** and **query** steps.
- Easy to extend (rerankers, better models, Streamlit UI, Docker, etc.).

---

## Quickstart

```bash
# 1) Create & activate env (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Put documents in ./data (supports .txt, .pdf, .docx)

# 4) Build the vector index
python ingest.py

# 5) Ask questions
python query.py --q "What are the key points?"
```

### Example
```bash
python query.py --q "Summarize the main takeaways."
```

You should see:
- Model answer
- The top source snippets used

---

## Project Structure

```
rag-pt-qa/
├── README.md
├── requirements.txt
├── ingest.py        # loads & chunks docs, builds FAISS, saves index
├── query.py         # retrieves chunks and runs local generator for answers
├── utils.py         # loaders + chunking helpers
├── data/
│   └── sample.txt
└── index/           # generated after running ingest.py
    ├── faiss.index
    └── chunks.pkl
```

---

## Notes

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (fast, ~384-dim).
- Generator: `google/flan-t5-small` (great for CPU demo; upgrade to `base`/`large` later).
- FAISS index uses **inner product** with normalized vectors (cosine similarity).

## Ideas to Extend (for future commits)
- Add a **reranker** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- Swap generator to a better local model (e.g., `mistral-7b` w/ GGUF + llama.cpp bindings).
- Add a **Streamlit** or **Gradio** UI.
- Log runs & metrics, write tests, add pre-commit hooks.
- Package as a module & add CLI entrypoints.

---

**Author**: You 👋 — built to showcase end-to-end LLM/RAG ability.
