# src/embed.py

from sentence_transformers import SentenceTransformer
import numpy as np


def paper_text(p):
    return f"{p['title']}\n\n{p['summary']}"


def embed(papers, model_name: str):
    model = SentenceTransformer(model_name)
    texts = [paper_text(p) for p in papers]
    embs = model.encode(texts, normalize_embeddings=True)
    return np.array(embs)
