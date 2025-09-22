# utils.py
import os
import json
import hashlib
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_get(key: str):
    p = CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def cache_set(key: str, obj):
    p = CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
