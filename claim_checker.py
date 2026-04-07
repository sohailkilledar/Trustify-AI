import os
import faiss
import torch
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download

# using updated v3 dataset for new news
# integrated the core news database 5gb with the latest_news database, new news data will be added regularly to improve the quality of Trustify AI !
REPO_ID = "sohailkilledar/Trustify-Data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

index_file = hf_hub_download(repo_id=REPO_ID, filename="news_index.faiss", repo_type="dataset")
data_file = hf_hub_download(repo_id=REPO_ID, filename="news_data.pkl", repo_type="dataset")
latest_index_file = hf_hub_download(repo_id=REPO_ID, filename="latest_news_data.faiss", repo_type="dataset")
latest_data_file = hf_hub_download(repo_id=REPO_ID, filename="latest_news_data.pkl", repo_type="dataset")

INDEX = faiss.read_index(index_file)
with open(data_file, "rb") as f:
    DF = pickle.load(f)

LATEST_INDEX = faiss.read_index(latest_index_file)
with open(latest_data_file, "rb") as f:
    LATEST_DF = pickle.load(f)

DF = DF.reset_index(drop=True)
LATEST_DF = LATEST_DF.reset_index(drop=True)

INDEX.merge_from(LATEST_INDEX)
DF = pd.concat([DF, LATEST_DF], ignore_index=True)

del LATEST_INDEX
del LATEST_DF

BI_ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)

def check_claim(user_text, top_k=15):
    if not user_text.strip():
        return {"confidence_score": "0%", "evidence_headlines": []}
    
    query_vec = BI_ENCODER.encode([user_text], normalize_embeddings=True).astype("float32")
    distances, indices = INDEX.search(query_vec, top_k)
    
    candidates = DF.iloc[indices[0]]
    headlines = candidates["headline_text"].tolist()
    
    pairs = [[user_text, h] for h in headlines]
    scores = RERANKER.predict(pairs)
    ranked_idx = np.argsort(scores)[::-1]
    
    final_evidence = [headlines[i] for i in ranked_idx[:6]]
    max_s = float(np.max(scores))
    norm_sim = (1 / (1 + np.exp(-max_s))) * 100
    
    return {
        "confidence_score": f"{round(norm_sim, 2)}%",
        "evidence_headlines": final_evidence
    }
