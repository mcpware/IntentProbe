#!/usr/bin/env python3
"""Honest DEPLOYABLE number: the 0.984 Edison result picks a different (model,layer) per fold (an
oracle envelope, not shippable). A real product commits to ONE fixed config. This measures several
fixed configs (including the SHIPPED 0.5B concat-L13-15) with leave-one-source-out: SAME config every
fold, no per-fold selection. Tells us (a) what the shipped config actually scores cross-source, and
(b) whether a single better config exists to ship. Cached PI all-layer embeddings. Seed 42.
"""
from __future__ import annotations
import json, os, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

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
    if arr.shape[0] == len(texts) and "__bd9cd4c746" in os.path.basename(npy): EMB[name] = arr
print("loaded:", {k: v.shape for k, v in EMB.items()}, flush=True)

def auroc(yt, ys): return roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")
def feat(cfg):
    m, layers = cfg
    return np.concatenate([EMB[m][:, l, :] for l in layers], axis=1)

# candidate FIXED configs (same for every fold)
CONFIGS = {
    "SHIPPED 0.5B concat L13-15": ("Qwen/Qwen2.5-0.5B", [13, 14, 15]),
    "0.5B L14":                   ("Qwen/Qwen2.5-0.5B", [14]),
    "1.5B L16":                   ("Qwen/Qwen2.5-1.5B", [16]),
    "1.5B L20":                   ("Qwen/Qwen2.5-1.5B", [20]),
    "1.5B concat L16-18":         ("Qwen/Qwen2.5-1.5B", [16, 17, 18]),
}
def pipe(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))

results = {}
for name, cfg in CONFIGS.items():
    if cfg[0] not in EMB: continue
    X = feat(cfg); per = {}
    for ts in SRC:
        te = src == ts; tr = ~te
        clf = pipe().fit(X[tr], y[tr])
        per[ts] = round(auroc(y[te], clf.predict_proba(X[te])[:, 1]), 3)
    results[name] = {"per_source": per, "mean": round(float(np.nanmean(list(per.values()))), 3)}
    print(f"  {name:26s} mean {results[name]['mean']}  per {per}", flush=True)

# TF-IDF baseline, same leave-one-source-out
tf_per = {}
for ts in SRC:
    te = src == ts; tr = ~te
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=5000, random_state=SEED))
    clf.fit([texts[i] for i in np.where(tr)[0]], y[tr])
    tf_per[ts] = round(auroc(y[te], clf.predict_proba([texts[i] for i in np.where(te)[0]])[:, 1]), 3)
tf_mean = round(float(np.nanmean(list(tf_per.values()))), 3)
results["TF-IDF (same data)"] = {"per_source": tf_per, "mean": tf_mean}
print(f"  {'TF-IDF (same data)':26s} mean {tf_mean}  per {tf_per}", flush=True)

json.dump(results, open(f"{OUTDIR}/pi_fixedconfig.json", "w"), indent=2)
best = max((k for k in results if k != "TF-IDF (same data)"), key=lambda k: results[k]["mean"])
print(f"\n=== DEPLOYABLE (single fixed config, no per-fold selection) ===", flush=True)
print(f"  SHIPPED (0.5B L13-15): {results.get('SHIPPED 0.5B concat L13-15',{}).get('mean')}", flush=True)
print(f"  best single config: {best} = {results[best]['mean']}", flush=True)
print(f"  TF-IDF: {tf_mean}  -> best-fixed minus TF-IDF: {round(results[best]['mean']-tf_mean,3)}", flush=True)
