# retriever.py
import os, hashlib, math
from serpapi import GoogleSearch
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from utils import chunk_text, cache_get, cache_set
from dotenv import load_dotenv
load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
EMB_MODEL = os.getenv("EMB_MODEL", "all-mpnet-base-v2")
MAX_PAGES = int(os.getenv("MAX_PAGES", 12))

emb_model = SentenceTransformer(EMB_MODEL)

def serp_search(query, num=6):
    params = {"q": query, "engine": "google", "api_key": SERPAPI_KEY, "num": num}
    search = GoogleSearch(params)
    res = search.get_dict()
    items = res.get("organic_results", []) or []
    results = []
    for it in items:
        url = it.get("link")
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        results.append({"url": url, "title": title, "snippet": snippet})
    return results

def fetch_article(url):
    # caching per URL
    cached = cache_get(url)
    if cached:
        return cached.get("title"), cached.get("text")
    try:
        art = Article(url)
        art.download()
        art.parse()
        title = art.title or ""
        text = art.text or ""
        cache_set(url, {"title": title, "text": text})
        return title, text
    except Exception as e:
        return None, None

def build_index_from_queries(queries):
    # gather URLs
    urls = []
    for q in queries:
        hits = serp_search(q, num=6)
        for h in hits:
            u = h.get("url")
            if u and u not in urls:
                urls.append(u)
            if len(urls) >= MAX_PAGES:
                break
        if len(urls) >= MAX_PAGES:
            break

    passages = []
    metas = []
    for u in urls:
        title, text = fetch_article(u)
        if not text:
            continue
        chunks = chunk_text(text, chunk_size=900, overlap=150)
        for i, c in enumerate(chunks):
            passages.append(c)
            metas.append({"url": u, "title": title, "offset": i})

    if not passages:
        return None

    # embeddings and FAISS index
    emb = emb_model.encode(passages, show_progress_bar=False, convert_to_numpy=True)
    # normalize for cosine similarity
    faiss.normalize_L2(emb)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    return {"index": index, "embeddings": emb, "passages": passages, "metas": metas}

def retrieve_top_k(claim, index_struct, k=5):
    claim_emb = emb_model.encode([claim], convert_to_numpy=True)
    faiss.normalize_L2(claim_emb)
    D, I = index_struct["index"].search(claim_emb, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0: continue
        hits.append({
            "score": float(score),
            "passage": index_struct["passages"][idx],
            "meta": index_struct["metas"][idx]
        })
    return hits
