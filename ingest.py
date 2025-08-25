import os
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import load_documents, chunk_text

DATA_DIR = "data"
INDEX_DIR = "index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(INDEX_DIR, exist_ok=True)

def embed_chunks(chunks):
    model = SentenceTransformer(EMB_MODEL)
    embs = model.encode(chunks, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs

def build_index():
    docs = load_documents(DATA_DIR)
    if not docs:
        raise SystemExit("No documents found in ./data. Add .txt/.pdf/.docx files and rerun.")
    sources = []
    chunks = []
    for src, txt in docs:
        for ch in chunk_text(txt):
            chunks.append(ch)
            sources.append(src)
    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")

    embs = embed_chunks(chunks)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine since we normalized
    index.add(embs)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources}, f)
    print("Saved index to ./index")

if __name__ == "__main__":
    build_index()
