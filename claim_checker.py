#!/usr/bin/env python3
"""
claim_checker.py (integrated aggregator + improved queries)

Usage:
    python claim_checker.py "Your claim here"

Notes:
 - Requires an HF API key in env var HF_API_KEY to use HuggingFace Inference API.
 - Uses facebook/bart-large-mnli by default for zero-shot classification.
 - Conservative aggregation: domain trust, reporting penalty, recency, and min-reputable-docs required.
"""

import os
import re
import time
import json
import feedparser
import urllib.parse
import requests
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from urllib.parse import urlparse
from datetime import datetime, timezone

# ---------- CONFIG ----------
HF_API_KEY = os.getenv("HF_API_KEY", "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") or "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"  # set in environment
HF_MODEL = os.getenv("HF_MODEL", "facebook/bart-large-mnli")
HF_BASE = "https://api-inference.huggingface.co/models"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "8"))
PARAPHRASE_MODEL = os.getenv("PARAPHRASE_MODEL", "")  # optional
MAX_PARAPHRASES = int(os.getenv("MAX_PARAPHRASES", "2"))

CANDIDATE_LABELS = ["supports", "refutes", "not enough evidence"]
LABEL_TO_VERDICT = {
    "supports": "Supported",
    "refutes": "Refuted",
    "not enough evidence": "Not enough evidence"
}
FACTCHECK_DOMAINS = ["snopes.com", "politifact.com", "factcheck.org", "apnews.com"]

# Aggregator tuning (you can adjust)
DOMAIN_TRUST = {
    "snopes.com": 1.00,
    "politifact.com": 1.00,
    "factcheck.org": 1.00,
    "apnews.com": 0.95,
    "reuters.com": 0.95,
    "bbc.co.uk": 0.95,
    "nytimes.com": 0.90,
    "theguardian.com": 0.90,
    "timesofindia.com": 0.6,
    "pune-pulse.com": 0.4,
}
DEFAULT_TRUST = 0.4
REPORTING_REGEX = re.compile(
    r'\b(alleged|allegedly|allege|report(s)?|reports that|police said|police have said|according to police|according to authorities|claimed|claim|allegation|investigation under way|investigat|hoax)\b',
    flags=re.I
)
MIN_REPUTABLE_DOCS = 2
MIN_AGG_WEIGHT = 1.0
REPORTING_PENALTY = 0.4  # lower -> stronger downweight for reporting language (tune)

# ---------------- Helpers ----------------
def normalize_claim(claim: str) -> str:
    q = re.sub(r"[\n\r]+", " ", claim)
    q = re.sub(r"[^0-9A-Za-z\s\-\']+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def generate_queries_simple(claim: str) -> List[str]:
    q = claim.strip()
    return [
        q,
        f'{q} fact check',
        f'{q} hoax',
        f'{q} police statement',
        f'"{q}"',
        f'{q} site:snopes.com OR site:politifact.com OR site:factcheck.org OR site:apnews.com'
    ]

def hf_paraphrase_text(text: str, model: str, max_outputs: int = 2) -> List[str]:
    if not HF_API_KEY or not model:
        return []
    payload = {"inputs": text, "parameters": {"max_new_tokens": 64, "num_return_sequences": max_outputs}}
    try:
        r = requests.post(f"{HF_BASE}/{model}", headers=HEADERS, json=payload, timeout=30)
        if r.status_code != 200:
            return []
        j = r.json()
        outs = []
        if isinstance(j, dict) and "generated_text" in j:
            outs.append(j["generated_text"])
        elif isinstance(j, list):
            for item in j:
                if isinstance(item, dict) and "generated_text" in item:
                    outs.append(item["generated_text"])
                elif isinstance(item, str):
                    outs.append(item)
        uniq = []
        for o in outs:
            s = o.strip()
            if s and s not in uniq and s.lower() != text.lower():
                uniq.append(s)
        return uniq[:max_outputs]
    except Exception:
        return []

def generate_queries_with_paraphrases(claim: str) -> List[str]:
    qbase = generate_queries_simple(claim)
    queries = list(qbase)
    if PARAPHRASE_MODEL and HF_API_KEY:
        paras = hf_paraphrase_text(claim, PARAPHRASE_MODEL, max_outputs=MAX_PARAPHRASES)
        for p in paras:
            if p not in queries:
                queries.append(p)
                queries.append(f"{p} fact check")
    # unique preserve order
    seen = set(); out = []
    for q in queries:
        if q not in seen:
            seen.add(q); out.append(q)
    return out

def fetch_news(query: str, page_size: int = MAX_ARTICLES) -> List[Dict]:
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:page_size]:
        articles.append({
            "title": entry.title,
            "description": entry.get("summary", "") or entry.get("description", ""),
            "url": entry.link,
            "publishedAt": entry.get("published", None),
            "source": {"name": (entry.get("source", {}) or {}).get("title", "Google News")}
        })
    return articles

def get_articles_for_claim(claim: str, max_total: int = MAX_ARTICLES) -> List[Dict]:
    queries = generate_queries_with_paraphrases(claim)
    seen = set(); combined = []
    for q in queries:
        try:
            items = fetch_news(q, page_size=max_total)
        except Exception:
            items = []
        for a in items:
            url = a.get("url","")
            if not url or url in seen:
                continue
            seen.add(url); combined.append(a)
            if len(combined) >= max_total:
                break
        if len(combined) >= max_total:
            break
    # prioritize fact-check domains
    def score_for_sort(a):
        url = (a.get("url") or "").lower()
        for i,d in enumerate(FACTCHECK_DOMAINS):
            if d in url:
                return -10 + i
        return 0
    return sorted(combined, key=score_for_sort)[:max_total]

# ------------- HF helpers ---------------
def check_model_available(model: str) -> Tuple[bool, str]:
    if not HF_API_KEY:
        return (False, "HF_API_KEY not set")
    try:
        r = requests.get(f"{HF_BASE}/{model}", headers=HEADERS, timeout=8)
        return (r.status_code == 200, f"{r.status_code}: {r.text[:500]}")
    except Exception as e:
        return (False, f"exception: {e}")

def hf_zero_shot(article_text: str, claim: str, model: str = HF_MODEL):
    if not HF_API_KEY:
        return {"ok": False, "error": "No HF_API_KEY set"}
    payload = {
        "inputs": article_text,
        "parameters": {
            "candidate_labels": CANDIDATE_LABELS,
            "hypothesis_template": f"This article {{}} the claim: \"{claim}\""
        }
    }
    try:
        r = requests.post(f"{HF_BASE}/{model}", headers=HEADERS, json=payload, timeout=60)
    except Exception as e:
        return {"ok": False, "error": f"request exception: {e}"}
    if r.status_code != 200:
        return {"ok": False, "error": f"{r.status_code}: {r.text[:800]}"}
    try:
        j = r.json()
    except Exception as e:
        return {"ok": False, "error": f"invalid json: {e}; raw:{r.text[:500]}"}
    if isinstance(j, dict) and ("labels" in j and "scores" in j):
        return {"ok": True, "result": dict(zip(j["labels"], j["scores"]))}
    if isinstance(j, list) and len(j)>0 and isinstance(j[0], dict) and "labels" in j[0]:
        return {"ok": True, "result": dict(zip(j[0]["labels"], j[0]["scores"]))}
    return {"ok": False, "error": f"unexpected response shape: {j}"}

# ---------------- Conservative aggregator ----------------
def get_domain_trust(url: str) -> float:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."): host = host[4:]
    except Exception:
        return DEFAULT_TRUST
    for d,t in DOMAIN_TRUST.items():
        if d in host: return float(t)
    return float(DEFAULT_TRUST)

def recency_factor(publishedAt: str) -> float:
    if not publishedAt: return 1.0
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(publishedAt, fmt)
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            days = max(0.0, (datetime.now(timezone.utc) - dt).days)
            return 1.0 / (1.0 + 0.0015 * days)
        except Exception:
            continue
    return 1.0

def reporting_penalty(text: str) -> float:
    if not text: return 1.0
    return REPORTING_PENALTY if REPORTING_REGEX.search(text) else 1.0

def normalize_label(raw_label: str) -> str:
    lab = (raw_label or "").upper()
    if "SUPPORT" in lab or "ENTAIL" in lab or "TRUE" in lab: return "SUPPORT"
    if "REFUTE" in lab or "CONTRA" in lab or "FALSE" in lab: return "REFUTE"
    return "NEI"

def conservative_aggregate(per_article_results: List[Dict]):
    details = []
    s_sum = r_sum = n_sum = 0.0
    for r in per_article_results:
        url = r.get("url","")
        pub = r.get("publishedAt") or r.get("published") or None
        scores = r.get("scores", {})
        best_label, best_prob = None, 0.0
        for lab, sc in scores.items():
            if sc is None: continue
            if float(sc) > best_prob:
                best_prob = float(sc); best_label = lab
        norm = normalize_label(best_label)
        trust = get_domain_trust(url)
        rec = recency_factor(pub)
        rep = reporting_penalty((r.get("title","") or "") + " " + (r.get("description","") or ""))
        sim_equiv = 1.0
        weight = sim_equiv * best_prob * trust * rec * rep
        details.append({
            "url": url, "title": r.get("title",""), "norm": norm, "prob": best_prob,
            "weight": weight, "trust": trust, "recency": rec, "report_penalty": rep
        })
        if norm=="SUPPORT": s_sum += weight
        elif norm=="REFUTE": r_sum += weight
        else: n_sum += weight
    # fact-check short-circuit
    for d in details:
        if any(fd in (d["url"] or "").lower() for fd in FACTCHECK_DOMAINS) and d["norm"]=="REFUTE" and d["prob"]>0.35:
            return {"verdict":"Refuted", "reason":"Fact-check site refuted", "details":details}
    # trust override single
    for d in details:
        if d["trust"]>=0.99 and d["weight"]>0.6:
            if d["norm"]=="SUPPORT": return {"verdict":"Supported","reason":"High trust single support","details":details}
            if d["norm"]=="REFUTE": return {"verdict":"Refuted","reason":"High trust single refute","details":details}
    # require multiple reputable docs
    rep_s = sum(1 for d in details if d["norm"]=="SUPPORT" and d["trust"]>=0.8)
    rep_r = sum(1 for d in details if d["norm"]=="REFUTE" and d["trust"]>=0.8)
    if rep_s >= MIN_REPUTABLE_DOCS and s_sum > r_sum*1.25:
        return {"verdict":"Supported","reason":"Multiple reputable supports","details":details}
    if rep_r >= MIN_REPUTABLE_DOCS and r_sum > s_sum*1.25:
        return {"verdict":"Refuted","reason":"Multiple reputable refutes","details":details}
    # single strong aggregate
    if s_sum >= MIN_AGG_WEIGHT and s_sum > r_sum*1.5:
        return {"verdict":"Supported","reason":"Strong aggregate support","details":details}
    if r_sum >= MIN_AGG_WEIGHT and r_sum > s_sum*1.5:
        return {"verdict":"Refuted","reason":"Strong aggregate refute","details":details}
    return {"verdict":"Not enough evidence","reason":"Insufficient trustworthy corroboration","details":details}

# ---------------- Orchestrator ----------------
def check_claim(claim: str):
    norm = normalize_claim(claim)
    print("Normalized query:", norm)
    articles = get_articles_for_claim(norm, max_total=MAX_ARTICLES)
    print(f"Found {len(articles)} articles (after query expansion & dedupe).")
    if not articles:
        return {"verdict":"Not enough evidence","reason":"No articles retrieved."}
    hf_ok, hf_msg = check_model_available(HF_MODEL)
    print(f"[HF model availability] {HF_MODEL} -> {hf_ok} ; msg: {hf_msg[:300].replace(chr(10),' ')}")
    per_article_results = []
    if hf_ok:
        for i, art in enumerate(articles, start=1):
            text = art.get("title","") + "\n\n" + (art.get("description","") or "")
            print(f"[HF] classifying article #{i}: {art.get('title')[:120]}")
            resp = hf_zero_shot(text, claim, model=HF_MODEL)
            if resp.get("ok"):
                per_article_results.append({
                    "id": i, "title": art.get("title"), "scores": resp["result"],
                    "url": art.get("url"), "publishedAt": art.get("publishedAt") or art.get("published")
                })
            else:
                print(f"[HF error on article #{i}]: {resp.get('error')}")
        if per_article_results:
            agg = conservative_aggregate(per_article_results)
            out_json = {"verdict": agg["verdict"], "reasoning": agg.get("reason",""), "details": agg.get("details",[])}
            return {"source":"hf_nli","model":HF_MODEL,"articles":per_article_results,"json":out_json}
    # fallback
    print("[FALLBACK] Using keyword-overlap heuristic.")
    scores = evidence_score_by_overlap(claim, articles)
    hv = heuristic_verdict_from_scores(scores)
    return {"source":"heuristic","model":None,"articles":articles,"json":{"verdict":hv["verdict"],"reasoning":hv["reasoning"],"scores":scores}}

# --------------- CLI ---------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python claim_checker.py \"<claim>\"")
        sys.exit(1)
    claim = " ".join(sys.argv[1:])
    out = check_claim(claim)
    print(json.dumps(out.get("json") or {"error":"no json"}, indent=2))
    print("\n--- debug summary ---")
    print("source:", out.get("source"))
    print("model:", out.get("model"))
