import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from claim_checker import check_claim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VERIFIED NLI MODEL
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)

def check_text(text):
    # 1. Fetch Evidence
    retrieval = check_claim(text)
    evidence = retrieval["evidence_headlines"]
    sim_score = retrieval["confidence"] / 100 

    contra_score, entail_score = 0, 0
    if evidence:
        # Check logic against the top news headline
        inputs = nli_tokenizer(evidence[0], text, return_tensors="pt", truncation=True).to(DEVICE)
        with torch.no_grad():
            logits = nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            # Mapping for this specific model: [0: contradiction, 1: entailment, 2: neutral]
            contra_score, entail_score = probs[0], probs[1]

    # 2. Fast Keyword Engines (No heavy model downloads)
    text_lower = text.lower()
    toxic_words = ["hate", "kill", "terror", "idiot", "violence"]
    tox_score = sum(1 for w in toxic_words if w in text_lower) / 10
    prop_count = sum(1 for w in ["conspiracy", "agenda", "fake", "betrayal"] if w in text_lower)

    # 3. Final Decision Logic (Targeting 95%+ Accuracy)
    news_type = "Unverified"
    
    # Logical Contradiction Check (If truth is 'Al-Qaeda' and claim is 'India')
    if contra_score > 0.45:
        news_type = "False / Fake News"
        final_conf = contra_score
    elif entail_score > 0.6 and sim_score > 0.6:
        news_type = "Real / Verified News"
        final_conf = (entail_score + sim_score) / 2
    elif sim_score < 0.2:
        news_type = "False / Fake News"
        final_conf = 0.85
    elif 0.3 <= sim_score <= 0.7:
        news_type = "Partially Supported"
        final_conf = sim_score
    else:
        news_type = "Unverified / Ambiguous"
        final_conf = 0.5

    return {
        "news_type": news_type,
        "confidence_score": f"{round(final_conf * 100, 2)}%",
        "evidence_headlines": evidence,
        "local_scores": {
            "fake_probability": round(float(contra_score if news_type == "False / Fake News" else (1-sim_score)), 4),
            "toxicity": min(tox_score, 1.0),
            "nli_consensus": "Contradiction" if contra_score > 0.45 else "Neutral",
            "propaganda_keywords": prop_count
        },
        "ai_reasoning": "Detected logical mismatch with verified news history." if news_type == "False / Fake News" else "Analysis complete."
    }
