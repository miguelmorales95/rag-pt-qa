import os
import faiss
import pickle
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INDEX_DIR = "index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"

def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
        obj = pickle.load(f)
    return index, obj["chunks"], obj["sources"]

def embed(texts):
    model = SentenceTransformer(EMB_MODEL)
    return model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)

def retrieve(query, index, chunks, sources, k=5):
    q_emb = embed([query])[0].reshape(1, -1)
    D, I = index.search(q_emb, k)
    ctx = []
    used = set()
    for idx in I[0]:
        if idx == -1: 
            continue
        if idx not in used:
            used.add(idx)
            ctx.append((chunks[idx], sources[idx]))
    return ctx

def build_prompt(contexts, question, max_chars=3000):
    context_texts = []
    total = 0
    for c, _ in contexts:
        if total + len(c) > max_chars:
            break
        context_texts.append(c)
        total += len(c)
    context_block = "\n\n".join(f"- {c}" for c in context_texts)
    prompt = f"""You are a helpful assistant. Answer the question **only** using the context below. If the answer isn't in the context, say you don't know.

Context:
{context_block}

Question: {question}
Answer:"""
    return prompt

def generate_answer(prompt):
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    inputs = tok(prompt, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_new_tokens=256)
    return tok.decode(output[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Your question")
    ap.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    args = ap.parse_args()

    index, chunks, sources = load_index()
    ctx = retrieve(args.q, index, chunks, sources, k=args.k)
    if not ctx:
        print("No context found. Did you run ingest.py?")
        return
    prompt = build_prompt(ctx, args.q)
    answer = generate_answer(prompt)

    print("\n=== Answer ===")
    print(answer)
    print("\n=== Sources (top snippets) ===")
    for i, (c, s) in enumerate(ctx, 1):
        snippet = (c[:200] + "...") if len(c) > 200 else c
        print(f"[{i}] {s} :: {snippet}")

if __name__ == "__main__":
    main()
