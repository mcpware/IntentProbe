#!/usr/bin/env python3
"""THE load-bearing test (audit's cheapest-decisive-next): on the matched-vocabulary
minimal pairs, does the activation probe beat EVERY surface-text baseline?

Pre-registered (REDO-SPEC §5): on the innocuous-swap subset (vocab near-identical),
text/keyword baselines MUST be ~chance. Thesis SUPPORTED iff the probe beats every text
baseline with NON-OVERLAPPING 95% CIs. If TF-IDF or length-only matches the probe, the
"reads intent, text cannot" differentiator collapses.

Grouped CV by pair_id (clean+poison twins never split across folds — the exact leak that
inflated the retracted 0.966). Baselines: TF-IDF+LogReg, length-only, framing-word counter,
DeBERTa zero-shot, static-keyword scanner. Sensor = shipped Qwen2.5-0.5B, layers 13-15. Seed 42.
"""
from __future__ import annotations
import json, re, os, time, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
LAYERS = (13, 14, 15); SENSOR = "Qwen/Qwen2.5-0.5B"
OUTDIR = "research/_results_published"; EMBDIR = f"{OUTDIR}/_emb"; os.makedirs(EMBDIR, exist_ok=True)
TELLS = ["does not","doesn't","do not","read-only","read only","without","never","purely","only",
         "also","additionally","automatically","in addition","as well","furthermore","moreover","just"]
SENS = ["private","secret","secrets"," key","keys","password","credential","credentials","token",
        "cookie","bearer","passphrase","exfiltrat","upload","external","remote","attacker"]

pairs = json.load(open("research/datasets/minpairs_v1.json"))
def flatten(ps):
    texts, y, grp, kwflag = [], [], [], []
    for i, p in enumerate(ps):
        for side, lab in (("clean", 0), ("poison", 1)):
            texts.append(p[side]); y.append(lab); grp.append(i); kwflag.append(p.get("keyword_flaggable", False))
    return texts, np.array(y), np.array(grp), np.array(kwflag)
texts, y, grp, kwflag = flatten(pairs)
print(f"rows={len(texts)} pairs={len(pairs)} pos={int(y.sum())}", flush=True)

# Qwen activations (cached)
tok = AutoTokenizer.from_pretrained(SENSOR)
mdl = AutoModel.from_pretrained(SENSOR, output_hidden_states=True, dtype=torch.float32); mdl.eval()
def embed(txts):
    key = f"{EMBDIR}/{SENSOR.replace('/','_')}__minpairs__{hashlib.md5(('|'.join(txts)).encode()).hexdigest()[:10]}.npy"
    if os.path.exists(key): return np.load(key)
    f = []
    for t in txts:
        ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad(): hs = mdl(**ids).hidden_states
        f.append(torch.cat([hs[l][0].mean(0) for l in LAYERS]).float().numpy())
    a = np.array(f, dtype=np.float32); np.save(key, a); return a
print("embedding minpairs ...", flush=True)
X = embed(texts)

# DeBERTa zero-shot scores (for AUROC) + preds
clf = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2",
               truncation=True, max_length=512, device=-1, top_k=None)
def deberta_scores(txts):
    out = []
    for r in clf([t or " " for t in txts], batch_size=16):
        d = {str(x["label"]).upper(): x["score"] for x in r}
        out.append(d.get("INJECTION", d.get("LABEL_1", 0.0)))
    return np.array(out)
deb = deberta_scores(texts)

cv = GroupKFold(n_splits=5)
def oof_proba(estimator, feats):
    return cross_val_predict(estimator, feats, y, groups=grp, cv=cv, method="predict_proba")[:, 1]

len_feat = np.array([[len(t)] for t in texts], dtype=float)
fram_feat = np.array([[sum(t.lower().count(w) for w in TELLS)] for t in texts], dtype=float)
kw_pred = np.array([1 if any(w in t.lower() for w in SENS) else 0 for t in texts])

models = {
    "probe_qwen0.5b": ("oof", LogisticRegression(max_iter=2000, random_state=SEED), X),
    "tfidf_logreg":   ("oof", make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1), LogisticRegression(max_iter=2000, random_state=SEED)), texts),
    "length_only":    ("oof", LogisticRegression(max_iter=2000, random_state=SEED), len_feat),
    "framing_counter":("oof", LogisticRegression(max_iter=2000, random_state=SEED), fram_feat),
}

def bootf1(yy, pp, n=1000):
    yy, pp = np.asarray(yy), np.asarray(pp); rng = np.random.default_rng(SEED); idx = np.arange(len(yy)); v = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(yy[s])) < 2: continue
        v.append(f1_score(yy[s], pp[s], zero_division=0))
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]

def evalset(mask, name):
    yy = y[mask]; res = {}
    # CV models: get OOF proba on full set, then slice mask
    for mname, (kind, est, feats) in models.items():
        proba_full = oof_proba(est, feats if not isinstance(feats, list) else feats) if not isinstance(feats, list) else None
        if isinstance(feats, list):  # text list for tfidf
            proba_full = cross_val_predict(est, feats, y, groups=grp, cv=cv, method="predict_proba")[:, 1]
        pf = proba_full[mask]; pred = (pf >= 0.5).astype(int)
        res[mname] = {"f1": round(f1_score(yy, pred, zero_division=0), 3), "f1_ci": bootf1(yy, pred),
                      "auroc": round(roc_auc_score(yy, pf), 3) if len(set(yy)) > 1 else None}
    # zero-shot: DeBERTa, keyword
    dp = (deb[mask] >= 0.5).astype(int)
    res["deberta_zeroshot"] = {"f1": round(f1_score(yy, dp, zero_division=0), 3), "f1_ci": bootf1(yy, dp),
                                "auroc": round(roc_auc_score(yy, deb[mask]), 3) if len(set(yy)) > 1 else None}
    kp = kw_pred[mask]
    res["static_keyword"] = {"f1": round(f1_score(yy, kp, zero_division=0), 3), "f1_ci": bootf1(yy, kp), "auroc": None}
    return {"n_rows": int(mask.sum()), "n_pos": int(yy.sum()), "scores": res}

results = {
    "meta": {"sensor": SENSOR, "layers": list(LAYERS), "seed": SEED, "cv": "GroupKFold(5) by pair_id",
             "pairs_total": len(pairs)},
    "ALL_pairs": evalset(np.ones(len(texts), bool), "all"),
    "innocuous_swap_subset_STRONGEST": evalset(~kwflag, "innocuous"),
    "keyword_flaggable_subset": evalset(kwflag, "kwflag"),
}
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
path = f"{OUTDIR}/minpairs_falsification_{ts}.json"
json.dump(results, open(path, "w"), indent=2)
print(f"\nsaved -> {path}", flush=True)
for k in ["ALL_pairs", "innocuous_swap_subset_STRONGEST", "keyword_flaggable_subset"]:
    s = results[k]
    print(f"\n== {k}  (rows={s['n_rows']}, pos={s['n_pos']}) ==", flush=True)
    for m, v in s["scores"].items():
        print(f"  {m:18s} F1={v['f1']} CI{v['f1_ci']} AUROC={v['auroc']}", flush=True)
