import os
import re
import math
from typing import List, Tuple
from pathlib import Path

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def read_docx(path: str) -> str:
    import docx
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_documents(data_dir: str) -> List[Tuple[str, str]]:
    """Returns list of (source_path, text)."""
    paths = []
    p = Path(data_dir)
    for ext in ("*.txt", "*.pdf", "*.docx"):
        paths.extend(p.rglob(ext))
    docs = []
    for path in paths:
        path = str(path)
        try:
            if path.lower().endswith(".txt"):
                text = read_txt(path)
            elif path.lower().endswith(".pdf"):
                text = read_pdf(path)
            elif path.lower().endswith(".docx"):
                text = read_docx(path)
            else:
                continue
            if text and text.strip():
                docs.append((path, clean_text(text)))
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
    return docs

def clean_text(text: str) -> str:
    # Simple whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Naive chunker on sentence-ish boundaries."""
    # Split on punctuation to approximate sentences/paragraphs
    pieces = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) + 1 <= chunk_size:
            current = (current + " " + piece).strip()
        else:
            if current:
                chunks.append(current)
            # start new, but include small overlap from end of previous
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                current = (tail + " " + piece).strip()
            else:
                current = piece
    if current:
        chunks.append(current)
    return chunks
