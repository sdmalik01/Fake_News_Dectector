# aggregator.py
import re
from urllib.parse import urlparse
from datetime import datetime, timezone
import math

# Small trusted-domain list (tune to your needs)
DOMAIN_TRUST = {
    "snopes.com": 1.0,
    "politifact.com": 1.0,
    "factcheck.org": 1.0,
    "apnews.com": 0.95,
    "reuters.com": 0.95,
    "bbc.co.uk": 0.95,
    "nytimes.com": 0.9,
    "theguardian.com": 0.9,
    "timesofindia.com": 0.6,
    "pune-pulse.com": 0.4,
    # add domains you care about
}

DEFAULT_TRUST = 0.4

def get_domain(url: str):
    try:
        host = urlparse(url).netloc.lower()
        # strip www
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

def get_domain_trust(url: str) -> float:
    host = get_domain(url)
    for d, t in DOMAIN_TRUST.items():
        if d in host:
            return float(t)
    return float(DEFAULT_TRUST)

# Recency factor: prefers newer sources. Expect publish_date string ISO-ish or None.
def recency_factor(publish_date_str):
    if not publish_date_str:
        return 1.0
    try:
        # try common ISO formats
        dt = datetime.fromisoformat(publish_date_str.replace("Z", "+00:00"))
    except Exception:
        try:
            dt = datetime.strptime(publish_date_str.split("T")[0], "%Y-%m-%d")
        except Exception:
            return 1.0
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - dt).days)
    # decay: same-day = 1.0, 30 days -> 0.85, 365 days -> 0.5 (tunable)
    return 1.0 / (1.0 + 0.0015 * days)  # gentle decay

# Heuristic: downweight passages that are clearly "reporting" a claim (vs verifying)
REPORTING_PATTERNS = re.compile(
    r'\b(alleged|allegedly|allege|report(s)?|reports that|police said|police have said|according to police|according to authorities|claimed|claim|allegation|investigation under way|investigat)', 
    flags=re.I
)

def reporting_penalty(passage: str) -> float:
    if not passage:
        return 1.0
    if REPORTING_PATTERNS.search(passage):
        return 0.5  # halve weight for reporting language (tunable)
    return 1.0

# Map verifier model labels to canonical stance labels.
# You should inspect your model.config.id2label and adapt mapping if necessary.
def normalize_label(raw_label: str) -> str:
    lab = raw_label.upper()
    # common label names from NLI/FC models:
    if "ENTAIL" in lab or "SUPPORT" in lab or "TRUE" in lab:
        return "SUPPORT"
    if "CONTRA" in lab or "REFUTE" in lab or "FALSE" in lab:
        return "REFUTE"
    if "NEUTRAL" in lab or "NOT_ENOUGH" in lab or "NOTENOUGH" in lab or "LABEL_2" in lab:
        return "NEI"
    # fallback: treat ambiguous as NEI
    return "NEI"

def aggregate(verdicts):
    """
    verdicts: list of dict with keys:
       - source (url)
       - sim (similarity float 0..1)
       - vlabel (raw verifier label)
       - vscore (verifier confidence 0..1)
       - passage (text)
       - publish_date (optional string)
    Returns dict: {final, scores: {supports, refutes, nei}, details: [...]}
    """
    s_sum = 0.0
    r_sum = 0.0
    n_sum = 0.0
    s_docs = 0
    r_docs = 0
    details = []

    for v in verdicts:
        src = v.get("source", "")
        sim = float(v.get("sim", 0.0)) if v.get("sim") is not None else 0.0
        raw_label = v.get("vlabel", "")
        vprob = float(v.get("vscore", 0.0))
        passage = v.get("passage", "") or ""
        pubdate = v.get("publish_date", None)

        norm = normalize_label(raw_label)
        trust = get_domain_trust(src)
        rec = recency_factor(pubdate)
        rep_pen = reporting_penalty(passage)

        # final weight: similarity * model confidence * domain_trust * recency * reporting_penalty
        weight = sim * vprob * trust * rec * rep_pen

        details.append({
            "source": src, "norm": norm, "weight": weight,
            "sim": sim, "vprob": vprob, "trust": trust, "recency": rec, "report_penalty": rep_pen
        })

        if norm == "SUPPORT":
            s_sum += weight
            s_docs += 1
        elif norm == "REFUTE":
            r_sum += weight
            r_docs += 1
        else:
            n_sum += weight

    # Conservative decision rules:
    # 1) If there is a high-trust ClaimReview or domain trust == 1.0 with a decisive single doc -> use it
    for det in details:
        host = det["source"]
        host_trust = det["trust"]
        if host_trust >= 0.99 and det["weight"] > 0.6:
            # direct trust override
            if det["norm"] == "SUPPORT":
                return {"final": "SUPPORTED", "scores": {"supports": s_sum, "refutes": r_sum, "nei": n_sum}, "details": details}
            if det["norm"] == "REFUTE":
                return {"final": "REFUTED", "scores": {"supports": s_sum, "refutes": r_sum, "nei": n_sum}, "details": details}

    # 2) Require at least 2 reputable docs (trust >= 0.8) supporting/refuting OR a strong weighted sum
    reputable_support_docs = sum(1 for det in details if det["norm"]=="SUPPORT" and det["trust"]>=0.8)
    reputable_refute_docs = sum(1 for det in details if det["norm"]=="REFUTE" and det["trust"]>=0.8)

    # thresholds (tunable)
    MIN_REPUTABLE_DOCS = 2
    MIN_WEIGHT_FOR_SINGLE = 1.0  # if aggregated weight passes this, accept
    # decide
    if reputable_support_docs >= MIN_REPUTABLE_DOCS and s_sum > r_sum * 1.25:
        return {"final": "SUPPORTED", "scores": {"supports": s_sum, "refutes": r_sum, "nei": n_sum}, "details": details}
    if reputable_refute_docs >= MIN_REPUTABLE_DOCS and r_sum > s_sum * 1.25:
        return {"final": "REFUTED", "scores": {"supports": s_sum, "refutes": r_sum, "nei": n_sum}, "details": details}

    # accept single strong aggregate even if not from reputable docs
    if s_sum >= MIN_WEIGHT_FOR_SINGLE and s_sum > r_sum * 1.5:
        return {"final": "SUPPORTED", "scores": {"supports": s_sum, "refutes": r_sum, "nei": n_sum}, "details": details}
    if r_sum >= MIN_WEIGHT_FOR_SINGLE and r_sum > s_sum * 1.5:
        return {"final": "REFUTED", "scores": {"supports": s_sum, "refutes": r_sum, "nei": n_sum}, "details": details}

    # otherwise, be conservative
    return {"final": "UNVERIFIED", "scores": {"supports": s_sum, "refutes": r_sum, "nei": n_sum}, "details": details}
