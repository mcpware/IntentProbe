#!/usr/bin/env python3
"""Best layer-COMBO search (Nicole: not necessarily 3 or 4 layers, not necessarily contiguous).
Greedy FORWARD layer selection: start empty, repeatedly add the layer that most improves inner-CV
AUROC, stop when no gain (or cap). Subset can be any size, non-contiguous. Per model. Nested:
selection happens on the OUTER-TRAIN sources only (inner leave-one-source-out); the held-out source
is never touched during selection. Report held-out AUROC + bootstrap CI vs TF-IDF and vs the shipped
0.5B-L13-15 fixed config. Cached PI all-layer embeddings. Seed 42.
"""
from __future__ import annotations
import json, os, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

SEED = 42; np.random.seed(SEED)
OUTDIR = "research/_results_published"; EMBDIR = f"{OUTDIR}/_emb_alllayers"
SOURCES = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json",
           "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
CAP = 350; MAX_LAYERS = 6; MIN_GAIN = 0.002; FIRST_CAND_LAYER = 4

def load_balanced():
    texts, y, src = [], [], []; rng = np.random.default_rng(SEED)
    for name, fn in SOURCES.items():
        rows = [r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
        for lab in (0, 1):
            pool = [r for r in rows if int(r["label"]) == lab]
            if len(pool) > CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
            for r in pool: texts.append(r["text"]); y.append(lab); src.append(name)
    return texts, np.array(y), np.array(src)

texts, y, src = load_balanced(); SRC = list(SOURCES.keys())
EMB = {}
for npy in sorted(glob.glob(f"{EMBDIR}/*.npy")):
    if "__bd9cd4c746" not in os.path.basename(npy): continue
    name = os.path.basename(npy).split("__")[0].replace("Qwen_Qwen", "Qwen/Qwen"); arr = np.load(npy)
    if arr.shape[0] == len(texts): EMB[name] = arr
MODELS = list(EMB.keys()); print("models:", {m: EMB[m].shape for m in MODELS}, flush=True)

def auroc(yt, ys): return roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")
def feat(m, subset): return np.concatenate([EMB[m][:, l, :] for l in subset], axis=1)
def fit_auroc(Xtr, ytr, Xte, yte, mi=500):
    return auroc(yte, LogisticRegression(max_iter=mi, random_state=SEED).fit(Xtr, ytr).predict_proba(Xte)[:, 1])

def inner_score(m, subset, trsrc, trmask):
    vals = []
    X = feat(m, subset)
    for inner in trsrc:
        itr = trmask & (src != inner); ite = trmask & (src == inner)
        if len(set(y[itr])) < 2 or len(set(y[ite])) < 2: continue
        vals.append(fit_auroc(X[itr], y[itr], X[ite], y[ite]))
    return float(np.nanmean(vals)) if vals else float("nan")

def greedy(m, trsrc, trmask):
    L = EMB[m].shape[1]; cand = list(range(FIRST_CAND_LAYER, L)); sel = []; best = -1
    while cand and len(sel) < MAX_LAYERS:
        scored = [(inner_score(m, sel + [c], trsrc, trmask), c) for c in cand]
        s, c = max(scored, key=lambda x: x[0])
        if s <= best + MIN_GAIN: break
        sel.append(c); best = s; cand.remove(c)
    return sorted(sel), best

def boot(yt, pa, ta, n=2000):
    yt = np.asarray(yt); rng = np.random.default_rng(SEED); idx = np.arange(len(yt)); d = []
    for _ in range(n):
        ss = rng.choice(idx, len(idx), replace=True)
        if len(set(yt[ss])) < 2: continue
        d.append(auroc(yt[ss], pa[ss]) - auroc(yt[ss], ta[ss]))
    return [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])]

folds = []
for ts in SRC:
    te = src == ts; trsrc = [s for s in SRC if s != ts]; tr = np.isin(src, trsrc)
    # greedy per model, pick best by inner score
    cfgs = {m: greedy(m, trsrc, tr) for m in MODELS}
    bm = max(cfgs, key=lambda m: cfgs[m][1]); subset, inner = cfgs[bm]
    X = feat(bm, subset)
    pa = LogisticRegression(max_iter=2000, random_state=SEED).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    tf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000, random_state=SEED))
    tf.fit([texts[i] for i in np.where(tr)[0]], y[tr]); ta = tf.predict_proba([texts[i] for i in np.where(te)[0]])[:, 1]
    pA, tA = auroc(y[te], pa), auroc(y[te], ta); ci = boot(y[te], pa, ta)
    folds.append({"held_out": ts, "best_model": bm, "layer_combo": subset, "n_layers": len(subset),
                  "inner_auroc": round(inner, 3), "probe": round(pA, 3), "tfidf": round(tA, 3),
                  "diff_ci95": ci, "verdict": "SIG win" if ci[0] > 0 else ("SIG loss" if ci[1] < 0 else "tie")})
    print(f"  {ts:11s} {bm.split('-')[-1]} layers {subset} | probe {pA:.3f} tfidf {tA:.3f} CI{ci} {folds[-1]['verdict']}", flush=True)

mp = float(np.nanmean([f["probe"] for f in folds])); mt = float(np.nanmean([f["tfidf"] for f in folds]))
out = {"summary": {"mean_probe": round(mp, 3), "mean_tfidf": round(mt, 3), "mean_diff": round(mp - mt, 3),
                   "shipped_fixed_mean": 0.980, "n_sig": sum(1 for f in folds if f["verdict"] == "SIG win")}, "folds": folds}
json.dump(out, open(f"{OUTDIR}/pi_layercombo.json", "w"), indent=2)
print(f"\n=== BEST LAYER-COMBO (greedy, any size/non-contiguous, nested) ===", flush=True)
print(f"  mean probe {out['summary']['mean_probe']} vs tfidf {out['summary']['mean_tfidf']} (vs shipped-fixed 0.980) | sig {out['summary']['n_sig']}/4", flush=True)
