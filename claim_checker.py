import os
import faiss
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Setup paths and device
INDEX_PATH, DATA_PATH = "news_index.faiss", "news_data.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Index & Data
INDEX = faiss.read_index(INDEX_PATH)
DF = pickle.load(open(DATA_PATH, "rb"))

# Lightweight Models (<150MB total)
BI_ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)

def check_claim(user_text, top_k=15):
    if not user_text.strip():
        return {"confidence": 0, "evidence_headlines": [], "evidence_dates": []}

    # Step A: Vector Retrieval
    query_vec = BI_ENCODER.encode([user_text], normalize_embeddings=True).astype("float32")
    distances, indices = INDEX.search(query_vec, top_k)
    
    candidates = DF.iloc[indices[0]]
    headlines = candidates["headline_text"].tolist()
    
    # Step B: Semantic Reranking (Peak Accuracy Stage)
    pairs = [[user_text, h] for h in headlines]
    scores = RERANKER.predict(pairs)
    ranked_idx = np.argsort(scores)[::-1]
    
    # Final Selection for UI
    final_evidence = [headlines[i] for i in ranked_idx[:6]]
    final_dates = [str(candidates.iloc[i]["publish_date"]) for i in ranked_idx[:6]]
    
    # Sigmoid Normalization (0-100)
    max_s = float(np.max(scores))
    norm_sim = (1 / (1 + np.exp(-max_s))) * 100

    return {
        "confidence": round(norm_sim, 2),
        "evidence_headlines": final_evidence,
        "evidence_dates": final_dates
    }
