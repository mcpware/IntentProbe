#!/usr/bin/env python3
"""Edison search WITH the nested-CV guardrail: sweep many (model, layer) configs to find the
best PI detector, but select on INNER data and report on a HELD-OUT REAL SOURCE the search never
touched — so "found the filament" is distinguished from "looked golden on the bench it was found on".

Design:
- Real-ish PI sources (both classes): deepset, safeguard, SPML, jayavibhav. (gandalf = injection-only,
  used as extra positives in training only.) Each capped + class-balanced.
- Candidate sensors: Qwen2.5-0.5B / 1.5B / 3B. For each, ALL layers extracted in ONE forward pass
  (mean-pooled) and cached -> sweeping layers is then cheap.
- LEAVE-ONE-SOURCE-OUT (outer): test source never seen in training OR selection.
  Inner (on the training sources only): leave-one-inner-source-out to pick best (model, layer) by val AUROC.
  Refit best config on all training sources -> evaluate on the held-out source. Rotate.
- Baselines on the SAME outer folds: TF-IDF+LogReg, length-only (fair, same train/test).
Pre-registered: the headline number is the held-out-source AUROC of the config picked WITHOUT seeing it.
Seed 42.
"""
from __future__ import annotations
import json, os, time, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score
from collections import Counter

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
OUTDIR = "research/_results_published"; EMBDIR = f"{OUTDIR}/_emb_alllayers"; os.makedirs(EMBDIR, exist_ok=True)
MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
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
print("sources:", {s: dict(Counter(y[src == s].tolist())) for s in SOURCES}, flush=True)
SRC_LIST = list(SOURCES.keys())

def embed_all_layers(model_name):
    cache = f"{EMBDIR}/{model_name.replace('/','_')}__{hashlib.md5(('|'.join(texts)).encode()).hexdigest()[:10]}.npy"
    if os.path.exists(cache):
        print(f"  cache hit {model_name}", flush=True); return np.load(cache)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, output_hidden_states=True, dtype=torch.float32); mdl.eval()
    t0 = time.time(); feats = []
    for i, t in enumerate(texts):
        ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad(): hs = mdl(**ids).hidden_states  # (L+1) x [1,seq,H]
        feats.append(np.stack([h[0].float().mean(0).numpy() for h in hs]))  # [L+1, H]
        if (i + 1) % 300 == 0: print(f"    {model_name} {i+1}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    arr = np.array(feats, dtype=np.float32); del mdl; np.save(cache, arr); return arr  # [N, L+1, H]

print("extracting activations (all layers, all models)...", flush=True)
EMB = {m: embed_all_layers(m) for m in MODELS}
for m in MODELS: print(f"  {m}: shape {EMB[m].shape}", flush=True)

def auroc(yt, ys):
    return roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")

# leave-one-source-out (outer), nested inner selection of (model, layer)
fold_results = []
for test_src in SRC_LIST:
    te = src == test_src; trsrc = [s for s in SRC_LIST if s != test_src]
    tr = np.isin(src, trsrc)
    # inner: leave-one-inner-source-out to score each (model, layer)
    cand_scores = {}
    for m in MODELS:
        L = EMB[m].shape[1]
        for l in range(L):
            vals = []
            for inner_test in trsrc:
                itr = tr & (src != inner_test); ite = tr & (src == inner_test)
                if len(set(y[itr])) < 2 or len(set(y[ite])) < 2: continue
                clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(EMB[m][itr, l, :], y[itr])
                vals.append(auroc(y[ite], clf.predict_proba(EMB[m][ite, l, :])[:, 1]))
            if vals: cand_scores[(m, l)] = float(np.nanmean(vals))
    best = max(cand_scores, key=cand_scores.get)
    bm, bl = best
    # refit best config on all training sources, eval on held-out test source
    clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(EMB[bm][tr, bl, :], y[tr])
    probe_auroc = auroc(y[te], clf.predict_proba(EMB[bm][te, bl, :])[:, 1])
    # baselines on same outer fold
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000, random_state=SEED))
    tfidf.fit([texts[i] for i in np.where(tr)[0]], y[tr])
    tfidf_auroc = auroc(y[te], tfidf.predict_proba([texts[i] for i in np.where(te)[0]])[:, 1])
    lo = LogisticRegression(max_iter=1000, random_state=SEED).fit(np.array([[len(texts[i])] for i in np.where(tr)[0]]), y[tr])
    len_auroc = auroc(y[te], lo.predict_proba(np.array([[len(texts[i])] for i in np.where(te)[0]]))[:, 1])
    fold_results.append({"held_out_source": test_src, "n_test": int(te.sum()),
                         "best_config": f"{bm} L{bl}", "best_inner_val_auroc": round(cand_scores[best], 3),
                         "probe_heldout_auroc": round(probe_auroc, 3),
                         "tfidf_heldout_auroc": round(tfidf_auroc, 3),
                         "length_only_heldout_auroc": round(len_auroc, 3)})
    print(f"  held-out {test_src}: best {bm} L{bl} (innerval {cand_scores[best]:.3f}) -> "
          f"probe {probe_auroc:.3f} | tfidf {tfidf_auroc:.3f} | length {len_auroc:.3f}", flush=True)

mp = float(np.nanmean([f["probe_heldout_auroc"] for f in fold_results]))
mt = float(np.nanmean([f["tfidf_heldout_auroc"] for f in fold_results]))
ml = float(np.nanmean([f["length_only_heldout_auroc"] for f in fold_results]))
summary = {"mean_probe_heldout_auroc": round(mp, 3), "mean_tfidf_heldout_auroc": round(mt, 3),
           "mean_length_heldout_auroc": round(ml, 3), "probe_minus_tfidf": round(mp - mt, 3)}
out = {"meta": {"models": MODELS, "sources": SRC_LIST, "per_class_cap": PER_CLASS_CAP, "seed": SEED,
                "design": "leave-one-source-out outer; nested leave-one-inner-source-out (model,layer) selection"},
       "summary": summary, "folds": fold_results}
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
json.dump(out, open(f"{OUTDIR}/pi_montecarlo_{ts}.json", "w"), indent=2)
print("\n=== EDISON SEARCH SUMMARY (held-out REAL source, config picked WITHOUT seeing it) ===", flush=True)
print(f"  probe (best per fold) mean held-out AUROC: {summary['mean_probe_heldout_auroc']}", flush=True)
print(f"  TF-IDF mean held-out AUROC:                {summary['mean_tfidf_heldout_auroc']}", flush=True)
print(f"  length-only mean held-out AUROC:           {summary['mean_length_heldout_auroc']}", flush=True)
print(f"  -> probe minus TF-IDF (real moat): {summary['probe_minus_tfidf']}", flush=True)
