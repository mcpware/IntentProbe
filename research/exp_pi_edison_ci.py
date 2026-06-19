#!/usr/bin/env python3
"""Solidify the Edison reversal: same nested-CV held-out-real-source design, but
(a) StandardScaler + max_iter=5000 (fix convergence warnings),
(b) bootstrap 95% CI on the held-out AUROC DIFFERENCE (probe - TF-IDF) per source.
If the CI of the difference excludes 0, the probe's win is real (not noise). Cached embeddings. Seed 42.
"""
from __future__ import annotations
import json, os, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    texts, y, src = [], [], []; rng = np.random.default_rng(SEED)
    for name, fn in SOURCES.items():
        rows = [r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
        for lab in (0, 1):
            pool = [r for r in rows if int(r["label"]) == lab]
            if len(pool) > PER_CLASS_CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), PER_CLASS_CAP, replace=False))]
            for r in pool: texts.append(r["text"]); y.append(lab); src.append(name)
    return texts, np.array(y), np.array(src)

texts, y, src = load_balanced(); SRC = list(SOURCES.keys())
EMB = {}
for npy in sorted(glob.glob(f"{EMBDIR}/*.npy")):
    name = os.path.basename(npy).split("__")[0].replace("Qwen_Qwen", "Qwen/Qwen"); arr = np.load(npy)
    if arr.shape[0] == len(texts): EMB[name] = arr
MODELS = list(EMB.keys()); print("models:", MODELS, "rows:", len(texts), flush=True)

def auroc(yt, ys): return roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")
def probe_pipe(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))

def boot_diff(yt, pa, ta, n=2000):
    yt = np.asarray(yt); rng = np.random.default_rng(SEED); idx = np.arange(len(yt)); diffs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(yt[s])) < 2: continue
        diffs.append(auroc(yt[s], pa[s]) - auroc(yt[s], ta[s]))
    return [round(float(x), 3) for x in np.percentile(diffs, [2.5, 97.5])]

folds = []
for ts in SRC:
    te = src == ts; trsrc = [s for s in SRC if s != ts]; tr = np.isin(src, trsrc)
    cand = {}
    for m in MODELS:
        for l in range(EMB[m].shape[1]):
            vals = []
            for inner in trsrc:
                itr = tr & (src != inner); ite = tr & (src == inner)
                if len(set(y[itr])) < 2 or len(set(y[ite])) < 2: continue
                clf = probe_pipe().fit(EMB[m][itr, l, :], y[itr])
                vals.append(auroc(y[ite], clf.predict_proba(EMB[m][ite, l, :])[:, 1]))
            if vals: cand[(m, l)] = float(np.nanmean(vals))
    bm, bl = max(cand, key=cand.get)
    clf = probe_pipe().fit(EMB[bm][tr, bl, :], y[tr]); pa = clf.predict_proba(EMB[bm][te, bl, :])[:, 1]
    tf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=5000, random_state=SEED))
    tf.fit([texts[i] for i in np.where(tr)[0]], y[tr]); ta = tf.predict_proba([texts[i] for i in np.where(te)[0]])[:, 1]
    pA, tA = auroc(y[te], pa), auroc(y[te], ta)
    ci = boot_diff(y[te], pa, ta)
    sig = "SIGNIFICANT win" if ci[0] > 0 else ("SIGNIFICANT loss" if ci[1] < 0 else "tie (CI crosses 0)")
    folds.append({"held_out": ts, "best_config": f"{bm} L{bl}", "probe_auroc": round(pA, 3),
                  "tfidf_auroc": round(tA, 3), "diff": round(pA - tA, 3), "diff_ci95": ci, "verdict": sig})
    print(f"  {ts:12s} best {bm} L{bl} | probe {pA:.3f} tfidf {tA:.3f} | diff {pA-tA:+.3f} CI{ci} -> {sig}", flush=True)

mp = float(np.nanmean([f["probe_auroc"] for f in folds])); mt = float(np.nanmean([f["tfidf_auroc"] for f in folds]))
out = {"summary": {"mean_probe": round(mp, 3), "mean_tfidf": round(mt, 3), "mean_diff": round(mp - mt, 3),
                   "n_significant_wins": sum(1 for f in folds if f["verdict"].startswith("SIGNIFICANT win"))}, "folds": folds}
json.dump(out, open(f"{OUTDIR}/pi_edison_ci.json", "w"), indent=2)
print(f"\n=== EDISON + CI ===\n  mean probe {out['summary']['mean_probe']} vs tfidf {out['summary']['mean_tfidf']} (diff {out['summary']['mean_diff']})", flush=True)
print(f"  significant per-source wins: {out['summary']['n_significant_wins']}/{len(folds)}", flush=True)
