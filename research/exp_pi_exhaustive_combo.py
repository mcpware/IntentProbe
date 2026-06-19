#!/usr/bin/env python3
"""Answer 'did greedy find the real best, or miss a combo?': EXHAUSTIVELY evaluate ALL single
layers + ALL pairs (per model, candidate layers 4..L-1), nested (select on inner LOO-source, report
on held-out source). If no pair beats the best single, greedy's local optimum was fine. Honest scope:
singles+pairs are exhaustive; triples+ are NOT (greedy explored that path and did not improve, and
2^L full enumeration is intractable). Cached PI all-layer embeddings. Seed 42.
"""
from __future__ import annotations
import json, os, glob, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 42; np.random.seed(SEED)
OUTDIR = "research/_results_published"; EMBDIR = f"{OUTDIR}/_emb_alllayers"
SOURCES = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json",
           "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
CAP = 350; FIRST = 4

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
MODELS = list(EMB.keys())
# candidate subsets per model: all singles + all pairs of layers FIRST..L-1
SUBSETS = {m: [[l] for l in range(FIRST, EMB[m].shape[1])] +
              [list(p) for p in itertools.combinations(range(FIRST, EMB[m].shape[1]), 2)] for m in MODELS}
print("models:", {m: EMB[m].shape for m in MODELS}, "subset counts:", {m: len(SUBSETS[m]) for m in MODELS}, flush=True)

def auroc(yt, ys): return roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")
def lr(): return LogisticRegression(max_iter=300, solver="liblinear", random_state=SEED)

folds = []
for ts in SRC:
    te = src == ts; trsrc = [s for s in SRC if s != ts]; tr = np.isin(src, trsrc)
    best = (-1, None, None)  # (inner_score, model, subset)
    for m in MODELS:
        for sub in SUBSETS[m]:
            X = np.concatenate([EMB[m][:, l, :] for l in sub], axis=1)
            vals = []
            for inner in trsrc:
                itr = tr & (src != inner); ite = tr & (src == inner)
                if len(set(y[itr])) < 2 or len(set(y[ite])) < 2: continue
                vals.append(auroc(y[ite], lr().fit(X[itr], y[itr]).predict_proba(X[ite])[:, 1]))
            sc = float(np.nanmean(vals)) if vals else -1
            if sc > best[0]: best = (sc, m, sub)
    _, bm, bsub = best
    X = np.concatenate([EMB[bm][:, l, :] for l in bsub], axis=1)
    pA = auroc(y[te], LogisticRegression(max_iter=2000, random_state=SEED).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
    folds.append({"held_out": ts, "best_model": bm, "best_subset": bsub, "n_layers": len(bsub),
                  "inner_auroc": round(best[0], 3), "heldout_auroc": round(pA, 3)})
    print(f"  {ts:11s} EXHAUSTIVE best: {bm.split('-')[-1]} layers {bsub} (inner {best[0]:.3f}) -> held-out {pA:.3f}", flush=True)

mp = float(np.nanmean([f["heldout_auroc"] for f in folds]))
out = {"summary": {"mean_heldout_auroc": round(mp, 3), "greedy_was": 0.984, "shipped_fixed": 0.980,
                   "note": "exhaustive over all singles+pairs (candidates 4..L-1); triples+ not enumerated"}, "folds": folds}
json.dump(out, open(f"{OUTDIR}/pi_exhaustive_combo.json", "w"), indent=2)
print(f"\n=== EXHAUSTIVE singles+pairs: mean held-out {out['summary']['mean_heldout_auroc']} (greedy 0.984, shipped 0.980) ===", flush=True)
print("  -> if ~same as greedy, greedy's local optimum was fine and the combo doesn't matter.", flush=True)
