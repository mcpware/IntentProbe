#!/usr/bin/env python3
"""A: does the cross-source generalization moat (confirmed on PI) ALSO hold for tool-poisoning?
Leave-one-CORPUS-out across 3 tool-poison corpora (routeguard skill-text / MCPTox / minpairs),
with nested Edison layer selection (inner: leave-one-inner-corpus-out) and bootstrap CI on the
held-out corpus AUROC difference (probe vs TF-IDF). Same method as the PI Edison CI. Seed 42.
"""
from __future__ import annotations
import json, os, time, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from collections import Counter

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
OUTDIR = "research/_results_published"; EMBDIR = f"{OUTDIR}/_emb_alllayers"; os.makedirs(EMBDIR, exist_ok=True)
CAP = 350

def load_corpora():
    texts, y, corp = [], [], []
    rng = np.random.default_rng(SEED)
    def add(rows_lab, name):
        # rows_lab: list of (text, label)
        for lab in (0, 1):
            pool = [t for t, l in rows_lab if l == lab and t]
            if len(pool) > CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
            for t in pool: texts.append(t); y.append(lab); corp.append(name)
    # routeguard skill-text
    rg = [r for r in json.load(open("research/datasets/routeguard_external_v0.json")) if r.get("carrier") in ("SKILL.md", "skill_package_text") and r.get("text")]
    add([(r["text"], 1 if r["label"] == "poisoned" else 0) for r in rg], "routeguard")
    # mcptox
    mc = [(r["description"], 0) for r in json.load(open("research/datasets/mcptox_clean_descriptions_labeled.json")) if r.get("description")]
    mc += [(r["description"], 1) for r in json.load(open("research/datasets/mcptox_poisoned_descriptions_labeled.json")) if r.get("description")]
    add(mc, "mcptox")
    # minpairs
    mp = json.load(open("research/datasets/minpairs_v1.json"))
    add([(p["clean"], 0) for p in mp] + [(p["poison"], 1) for p in mp], "minpairs")
    return texts, np.array(y), np.array(corp)

texts, y, corp = load_corpora()
CORP = list(dict.fromkeys(corp.tolist()))
print("corpora:", {c: dict(Counter(y[corp == c].tolist())) for c in CORP}, "total", len(texts), flush=True)

def embed_all(model_name):
    cache = f"{EMBDIR}/{model_name.replace('/','_')}__tpcc__{hashlib.md5(('|'.join(texts)).encode()).hexdigest()[:10]}.npy"
    if os.path.exists(cache): print(f"  cache {model_name}", flush=True); return np.load(cache)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, output_hidden_states=True, dtype=torch.float32); mdl.eval()
    F, t0 = [], time.time()
    for i, t in enumerate(texts):
        ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad(): hs = mdl(**ids).hidden_states
        F.append(np.stack([h[0].float().mean(0).numpy() for h in hs]))
        if (i + 1) % 300 == 0: print(f"    {model_name} {i+1}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    arr = np.array(F, dtype=np.float32); del mdl; np.save(cache, arr); return arr

print("embedding tool-poison corpora (all layers)...", flush=True)
EMB = {m: embed_all(m) for m in MODELS}

def auroc(yt, ys): return roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")
def pipe(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))
def boot(yt, pa, ta, n=2000):
    yt = np.asarray(yt); rng = np.random.default_rng(SEED); idx = np.arange(len(yt)); d = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(yt[s])) < 2: continue
        d.append(auroc(yt[s], pa[s]) - auroc(yt[s], ta[s]))
    return [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])]

folds = []
for ts in CORP:
    te = corp == ts; trc = [c for c in CORP if c != ts]; tr = np.isin(corp, trc)
    cand = {}
    for m in MODELS:
        for l in range(EMB[m].shape[1]):
            vals = []
            for inner in trc:
                itr = tr & (corp != inner); ite = tr & (corp == inner)
                if len(set(y[itr])) < 2 or len(set(y[ite])) < 2: continue
                clf = pipe().fit(EMB[m][itr, l, :], y[itr])
                vals.append(auroc(y[ite], clf.predict_proba(EMB[m][ite, l, :])[:, 1]))
            if vals: cand[(m, l)] = float(np.nanmean(vals))
    bm, bl = max(cand, key=cand.get)
    clf = pipe().fit(EMB[bm][tr, bl, :], y[tr]); pa = clf.predict_proba(EMB[bm][te, bl, :])[:, 1]
    tf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=5000, random_state=SEED))
    tf.fit([texts[i] for i in np.where(tr)[0]], y[tr]); ta = tf.predict_proba([texts[i] for i in np.where(te)[0]])[:, 1]
    pA, tA = auroc(y[te], pa), auroc(y[te], ta); ci = boot(y[te], pa, ta)
    sig = "SIGNIFICANT win" if ci[0] > 0 else ("SIGNIFICANT loss" if ci[1] < 0 else "tie")
    folds.append({"held_out_corpus": ts, "n": int(te.sum()), "best_config": f"{bm} L{bl}",
                  "probe": round(pA, 3), "tfidf": round(tA, 3), "diff_ci95": ci, "verdict": sig})
    print(f"  held-out {ts:11s} best {bm} L{bl} | probe {pA:.3f} tfidf {tA:.3f} | diff CI{ci} -> {sig}", flush=True)

mp_ = float(np.nanmean([f["probe"] for f in folds])); mt_ = float(np.nanmean([f["tfidf"] for f in folds]))
out = {"summary": {"mean_probe": round(mp_, 3), "mean_tfidf": round(mt_, 3), "mean_diff": round(mp_ - mt_, 3),
                   "n_sig_wins": sum(1 for f in folds if f["verdict"] == "SIGNIFICANT win")}, "folds": folds}
json.dump(out, open(f"{OUTDIR}/toolpoison_crosscorpus.json", "w"), indent=2)
print(f"\n=== A: TOOL-POISON CROSS-CORPUS (does the moat extend?) ===", flush=True)
print(f"  mean probe {out['summary']['mean_probe']} vs tfidf {out['summary']['mean_tfidf']} | sig wins {out['summary']['n_sig_wins']}/{len(folds)}", flush=True)
