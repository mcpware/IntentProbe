#!/usr/bin/env python3
"""Recovery: the Monte Carlo embedding job was OOM-killed during 3B, but 0.5B + 1.5B all-layer
activations are cached. Run the nested-CV Edison search over the cached models (cheap, no embedding).
Same data/seed as exp_pi_montecarlo_search.py so the cached .npy match.
"""
from __future__ import annotations
import json, os, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from collections import Counter

SEED = 42; np.random.seed(SEED)
OUTDIR = "research/_results_published"; EMBDIR = f"{OUTDIR}/_emb_alllayers"
SOURCES = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json",
           "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
PER_CLASS_CAP = 350

def load_balanced():
    texts, y, src = [], [], []
    rng = np.random.default_rng(SEED)
    for name, fn in SOURCES.items():
        rows = [r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
        for lab in (0, 1):
            pool = [r for r in rows if int(r["label"]) == lab]
            if len(pool) > PER_CLASS_CAP:
                pool = [pool[i] for i in sorted(rng.choice(len(pool), PER_CLASS_CAP, replace=False))]
            for r in pool:
                texts.append(r["text"]); y.append(lab); src.append(name)
    return texts, np.array(y), np.array(src)

texts, y, src = load_balanced()
print("rows:", len(texts), {s: dict(Counter(y[src==s].tolist())) for s in SOURCES}, flush=True)
SRC_LIST = list(SOURCES.keys())

# load cached all-layer embeddings (whichever models are present)
EMB = {}
for npy in sorted(glob.glob(f"{EMBDIR}/*.npy")):
    name = os.path.basename(npy).split("__")[0].replace("Qwen_Qwen", "Qwen/Qwen")
    arr = np.load(npy)
    if arr.shape[0] == len(texts):
        EMB[name] = arr; print(f"  loaded {name}: {arr.shape}", flush=True)
    else:
        print(f"  SKIP {name}: shape {arr.shape} != n={len(texts)} (stale cache)", flush=True)
MODELS = list(EMB.keys())

def auroc(yt, ys):
    return roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")

folds = []
for test_src in SRC_LIST:
    te = src == test_src; trsrc = [s for s in SRC_LIST if s != test_src]; tr = np.isin(src, trsrc)
    cand = {}
    for m in MODELS:
        for l in range(EMB[m].shape[1]):
            vals = []
            for inner in trsrc:
                itr = tr & (src != inner); ite = tr & (src == inner)
                if len(set(y[itr])) < 2 or len(set(y[ite])) < 2: continue
                clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(EMB[m][itr, l, :], y[itr])
                vals.append(auroc(y[ite], clf.predict_proba(EMB[m][ite, l, :])[:, 1]))
            if vals: cand[(m, l)] = float(np.nanmean(vals))
    bm, bl = max(cand, key=cand.get)
    clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(EMB[bm][tr, bl, :], y[tr])
    pa = auroc(y[te], clf.predict_proba(EMB[bm][te, bl, :])[:, 1])
    tf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000, random_state=SEED))
    tf.fit([texts[i] for i in np.where(tr)[0]], y[tr])
    ta = auroc(y[te], tf.predict_proba([texts[i] for i in np.where(te)[0]])[:, 1])
    lo = LogisticRegression(max_iter=1000, random_state=SEED).fit(np.array([[len(texts[i])] for i in np.where(tr)[0]]), y[tr])
    la = auroc(y[te], lo.predict_proba(np.array([[len(texts[i])] for i in np.where(te)[0]]))[:, 1])
    folds.append({"held_out": test_src, "n_test": int(te.sum()), "best_config": f"{bm} L{bl}",
                  "inner_val_auroc": round(cand[(bm, bl)], 3), "probe": round(pa, 3), "tfidf": round(ta, 3), "length": round(la, 3)})
    print(f"  held-out {test_src:12s}: best {bm} L{bl} (inner {cand[(bm,bl)]:.3f}) -> probe {pa:.3f} | tfidf {ta:.3f} | length {la:.3f}", flush=True)

mp = float(np.nanmean([f["probe"] for f in folds])); mt = float(np.nanmean([f["tfidf"] for f in folds])); ml = float(np.nanmean([f["length"] for f in folds]))
summary = {"models_searched": MODELS, "mean_probe_heldout": round(mp, 3), "mean_tfidf_heldout": round(mt, 3),
           "mean_length_heldout": round(ml, 3), "probe_minus_tfidf": round(mp - mt, 3)}
json.dump({"summary": summary, "folds": folds}, open(f"{OUTDIR}/pi_montecarlo_analyzed.json", "w"), indent=2)
print("\n=== EDISON SEARCH (cached 0.5B+1.5B, nested-CV, held-out REAL source) ===", flush=True)
print(f"  probe (best per fold): {summary['mean_probe_heldout']}", flush=True)
print(f"  TF-IDF:                {summary['mean_tfidf_heldout']}", flush=True)
print(f"  length-only:           {summary['mean_length_heldout']}", flush=True)
print(f"  -> probe minus TF-IDF: {summary['probe_minus_tfidf']}", flush=True)
