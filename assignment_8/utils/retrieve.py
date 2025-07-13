# utils/retrieve.py

import numpy as np
import pickle
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_similar_chunks(query, texts, index, k=9):
    numeric_keywords = ["average", "mean", "how many", "percent", "rate", "median", "sum", "total", "%"]
    
    if any(word in query.lower() for word in numeric_keywords):
        # full dataset instead of selected chunks
        return "\n\n".join(texts)
    
    # Otherwise, retrieve top-k relevant chunks
    query_vec = embedder.encode([query])
    _, I = index.search(np.array(query_vec).astype("float32"), k)
    return "\n\n".join([texts[i] for i in I[0]])






