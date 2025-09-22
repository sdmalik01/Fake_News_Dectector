# verifier.py
# Simple Hugging Face pipeline wrapper for a sequence-classification verifier.
# It uses CPU by default (device=-1). If you have a GPU, change device to 0.

from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

# Default model name (change in .env if you want)
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "microsoft/deberta-v3-small")

# Create the pipeline. device=-1 uses CPU. If you have CUDA, set device=0
try:
    verifier_pipeline = pipeline(
        "text-classification",
        model=VERIFIER_MODEL,
        return_all_scores=True,
        device=-1
    )
except Exception as e:
    # If model loading fails, create a dummy fallback pipeline-like object
    verifier_pipeline = None
    _load_error = e

def run_verifier(claim: str, passage: str):
    """
    Returns a dict: { "best_label": str, "best_score": float, "all_scores": [...] }
    If model failed to load, returns a safe fallback result.
    """
    if verifier_pipeline is None:
        # fallback: return placeholder neutral score
        return {"best_label": "ERROR_NO_MODEL", "best_score": 0.0, "all_scores": []}
    text = f"Claim: {claim}\nPassage: {passage}"
    try:
        res = verifier_pipeline(text, truncation=True, max_length=512)
        labels = res[0]
        best = max(labels, key=lambda x: x["score"])
        return {"best_label": best["label"], "best_score": float(best["score"]), "all_scores": labels}
    except Exception as e:
        # In case pipeline fails at runtime, return safe fallback
        return {"best_label": "ERROR_RUNTIME", "best_score": 0.0, "all_scores": []}
