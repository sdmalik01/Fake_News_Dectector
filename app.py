# app.py
import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from retriever import build_index_from_queries, retrieve_top_k
from verifier import run_verifier

def generate_queries(claim):
    # simple query generator: exact, fact-check, site queries
    q1 = claim
    q2 = claim + " fact check"
    q3 = claim + " hoax"
    q4 = claim + " site:snopes.com OR site:politifact.com OR site:apnews.com"
    return [q1, q2, q3, q4]

def pipeline(claim):
    if not claim or len(claim.strip()) < 3:
        return {"error": "Please enter a valid short headline/claim."}
    queries = generate_queries(claim)
    idx_struct = build_index_from_queries(queries)
    if not idx_struct:
        return {"error": "No documents found for this claim."}
    hits = retrieve_top_k(claim, idx_struct, k=6)
    verdicts = []
    support_count = 0
    refute_count = 0
    for h in hits:
        v = run_verifier(claim, h["passage"])
        # naive mapping — adapt after you check model labels
        label = v["best_label"]
        score = v["best_score"]
        if "SUPPORT" in label.upper() or "ENTAIL" in label.upper():
            support_count += h["score"] * score
        if "REFUTE" in label.upper() or "CONTRA" in label.upper():
            refute_count += h["score"] * score
        verdicts.append({
            "source": h["meta"]["url"],
            "title": h["meta"]["title"],
            "snippet": h["passage"][:400] + ("..." if len(h["passage"])>400 else ""),
            "sim": round(h["score"], 3),
            "vlabel": label,
            "vscore": round(score, 3),
        })
    # aggregate naive:
    if support_count > refute_count * 1.2:
        final = "LIKELY TRUE"
    elif refute_count > support_count * 1.2:
        final = "LIKELY FALSE"
    else:
        final = "UNVERIFIED / INSUFFICIENT"
    return {"final": final, "evidence": verdicts}

def ui_run(claim):
    res = pipeline(claim)
    if "error" in res:
        return res["error"], "", []
    # prepare evidence display as a list of strings
    evidence_list = []
    for e in res["evidence"]:
        s = f"{e['title']}\n{e['source']}\nSim:{e['sim']} Verifier:{e['vlabel']}({e['vscore']})\n{e['snippet']}\n"
        evidence_list.append(s)
    return f"Verdict: {res['final']}", "Top evidence (source, similarity, verifier label):", evidence_list

with gr.Blocks() as demo:
    gr.Markdown("# Live Web-Grounded Fake News Prototype")
    inp = gr.Textbox(lines=2, placeholder="Type a news headline or short claim here...", label="Claim / Headline")
    out_label = gr.Textbox(label="Verdict")
    out_header = gr.Textbox(label="")
    out_list = gr.Textbox(label="Evidence (each block is one passage + metadata)", lines=12)
    btn = gr.Button("Check Claim")
    btn.click(fn=ui_run, inputs=[inp], outputs=[out_label, out_header, out_list])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
