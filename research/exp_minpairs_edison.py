#!/usr/bin/env python3
"""Re-test the matched-vocab thesis WITH fair layer selection (the earlier falsification used the
fixed tool-poison-tuned L13-15, which the PI Edison search showed was a handicap). Nested GroupKFold:
outer GroupKFold(5) by pair_id; inner GroupKFold(3) picks best (model, layer); OOF predictions.
Compare probe vs TF-IDF vs length on the SAME folds, with bootstrap CI on the OOF AUROC difference.
Focus: does proper layer selection let the probe beat TF-IDF on same-vocab minimal pairs? Seed 42.
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
OUTDIR = "research/_results_published"; EMBDIR = f"{OUTDIR}/_emb_alllayers"; os.makedirs(EMBDIR, exist_ok=True)
MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]

pairs = json.load(open("research/datasets/minpairs_v1.json"))
texts, y, grp, kwflag = [], [], [], []
for i, p in enumerate(pairs):
    for side, lab in (("clean", 0), ("poison", 1)):
        texts.append(p[side]); y.append(lab); grp.append(i); kwflag.append(bool(p.get("keyword_flaggable", False)))
y = np.array(y); grp = np.array(grp); kwflag = np.array(kwflag)
print(f"rows={len(texts)} pairs={len(pairs)} innocuous(non-kw)={int((~kwflag).sum())}", flush=True)

def embed_all(model_name):
    cache = f"{EMBDIR}/{model_name.replace('/','_')}__minpairs186__{hashlib.md5(('|'.join(texts)).encode()).hexdigest()[:10]}.npy"
    if os.path.exists(cache): print(f"  cache {model_name}", flush=True); return np.load(cache)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, output_hidden_states=True, dtype=torch.float32); mdl.eval()
    F = []
    for i, t in enumerate(texts):
        ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad(): hs = mdl(**ids).hidden_states
        F.append(np.stack([h[0].float().mean(0).numpy() for h in hs]))
        if (i+1) % 60 == 0: print(f"    {model_name} {i+1}/{len(texts)}", flush=True)
    arr = np.array(F, dtype=np.float32); del mdl; np.save(cache, arr); return arr

print("embedding minpairs (all layers)...", flush=True)
EMB = {m: embed_all(m) for m in MODELS}

def auroc(yt, ys): return roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")
def pipe(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))

outer = GroupKFold(5)
oof_probe = np.full(len(y), np.nan); oof_cfg = [None]*len(y)
for tr, te in outer.split(np.zeros(len(y)), y, grp):
    # inner selection of (model, layer)
    inner = GroupKFold(3); cand = {}
    for m in MODELS:
        for l in range(EMB[m].shape[1]):
            vals = []
            for i_tr, i_te in inner.split(np.zeros(len(tr)), y[tr], grp[tr]):
                a, b = tr[i_tr], tr[i_te]
                if len(set(y[a])) < 2 or len(set(y[b])) < 2: continue
                clf = pipe().fit(EMB[m][a, l, :], y[a])
                vals.append(auroc(y[b], clf.predict_proba(EMB[m][b, l, :])[:, 1]))
            if vals: cand[(m, l)] = float(np.nanmean(vals))
    bm, bl = max(cand, key=cand.get)
    clf = pipe().fit(EMB[bm][tr, bl, :], y[tr])
    oof_probe[te] = clf.predict_proba(EMB[bm][te, bl, :])[:, 1]
    for t in te: oof_cfg[t] = f"{bm} L{bl}"

# TF-IDF + length OOF on same outer folds
from sklearn.model_selection import cross_val_predict
oof_tfidf = cross_val_predict(make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1), LogisticRegression(max_iter=5000, random_state=SEED)),
                              texts, y, groups=grp, cv=outer, method="predict_proba")[:, 1]
oof_len = cross_val_predict(LogisticRegression(max_iter=1000, random_state=SEED),
                            np.array([[len(t)] for t in texts]), y, groups=grp, cv=outer, method="predict_proba")[:, 1]

def boot(mask, n=2000):
    yt = y[mask]; pp = oof_probe[mask]; tp = oof_tfidf[mask]
    rng = np.random.default_rng(SEED); idx = np.arange(len(yt)); diffs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(yt[s])) < 2: continue
        diffs.append(auroc(yt[s], pp[s]) - auroc(yt[s], tp[s]))
    return [round(float(x), 3) for x in np.percentile(diffs, [2.5, 97.5])]

def report(mask, name):
    ci = boot(mask)
    r = {"subset": name, "n": int(mask.sum()),
         "probe_auroc": round(auroc(y[mask], oof_probe[mask]), 3),
         "tfidf_auroc": round(auroc(y[mask], oof_tfidf[mask]), 3),
         "length_auroc": round(auroc(y[mask], oof_len[mask]), 3),
         "probe_minus_tfidf_ci95": ci,
         "verdict": "SIGNIFICANT win" if ci[0] > 0 else ("SIGNIFICANT loss" if ci[1] < 0 else "tie")}
    print(f"  {name:22s} n={r['n']:3d} probe {r['probe_auroc']} tfidf {r['tfidf_auroc']} length {r['length_auroc']} | diff CI{ci} -> {r['verdict']}", flush=True)
    return r

print("\n=== MINPAIRS with EDISON layer selection (nested GroupKFold by pair) ===", flush=True)
res = {"all": report(np.ones(len(y), bool), "ALL_pairs"),
       "innocuous": report(~kwflag, "innocuous_swap (strongest)"),
       "configs_picked": list(sorted(set(c for c in oof_cfg if c)))}
json.dump(res, open(f"{OUTDIR}/minpairs_edison.json", "w"), indent=2)
print("  configs picked across folds:", res["configs_picked"], flush=True)
