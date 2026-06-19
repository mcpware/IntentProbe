#!/usr/bin/env python3
"""THE moat test: on NOVEL attack families (vocab the text model never trained on),
does the activation probe beat a surface-text baseline? This is the one place text/keyword
methods fundamentally cannot win — they can only flag vocabulary they have seen.

Leave-one-family-out on real third-party routeguard data, filtered to genuine skill text
(carrier SKILL.md + skill_package_text; drops BIPIA external-context QA and MASB metadata-only
weak labels per the fairness audit). For each held-out family: train probe(Qwen) + TF-IDF on the
OTHER families, test on the held-out one. Apple-to-apple (same train data, same eval). Seed 42.

Headline: mean(probe AUROC - TF-IDF AUROC) across held-out novel families. Positive & large => moat.
"""
from __future__ import annotations
import json, os, time, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, roc_auc_score
from collections import Counter

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
SENSOR = "Qwen/Qwen2.5-0.5B"; LAYERS = (13, 14, 15)
OUTDIR = "research/_results_published"; EMBDIR = f"{OUTDIR}/_emb"; os.makedirs(EMBDIR, exist_ok=True)
KEEP_CARRIER = {"SKILL.md", "skill_package_text"}

d = json.load(open("research/datasets/routeguard_external_v0.json"))
rows = [r for r in d if r.get("carrier") in KEEP_CARRIER and r.get("text")]
for r in rows:
    r["y"] = 1 if r["label"] == "poisoned" else 0
texts = [r["text"] for r in rows]
y = np.array([r["y"] for r in rows])
fam = np.array([r["family"] for r in rows])
print("filtered rows:", len(rows), "label:", dict(Counter(y.tolist())))
print("family x label:")
for f in sorted(set(fam)):
    m = fam == f
    print(f"  {f:30s} n={m.sum():4d}  clean={int((y[m]==0).sum()):4d} poison={int((y[m]==1).sum()):4d}")

# embed Qwen activations (cached)
tok = AutoTokenizer.from_pretrained(SENSOR)
mdl = AutoModel.from_pretrained(SENSOR, output_hidden_states=True, dtype=torch.float32); mdl.eval()
key = f"{EMBDIR}/{SENSOR.replace('/','_')}__routeguard_skill__{hashlib.md5(('|'.join(texts)).encode()).hexdigest()[:10]}.npy"
if os.path.exists(key):
    X = np.load(key); print("cache hit activations")
else:
    print("embedding", len(texts), "rows ...", flush=True); t0 = time.time(); F = []
    for i, t in enumerate(texts):
        ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad(): hs = mdl(**ids).hidden_states
        F.append(torch.cat([hs[l][0].mean(0) for l in LAYERS]).float().numpy())
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(F, dtype=np.float32); np.save(key, X)

# DeBERTa zero-shot scores
clf = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2",
               truncation=True, max_length=512, device=-1, top_k=None)
deb = []
for r in clf([t or " " for t in texts], batch_size=16):
    dd = {str(x["label"]).upper(): x["score"] for x in r}
    deb.append(dd.get("INJECTION", dd.get("LABEL_1", 0.0)))
deb = np.array(deb)

def metrics(yy, proba):
    pred = (proba >= 0.5).astype(int)
    return {"f1": round(f1_score(yy, pred, zero_division=0), 3),
            "auroc": round(roc_auc_score(yy, proba), 3) if len(set(yy)) > 1 else None}

perfam = {}
fams = [f for f in sorted(set(fam)) if (fam == f).sum() >= 20 and len(set(y[fam == f])) > 1]
print("\nleave-one-family-out over:", fams, flush=True)
for hf in fams:
    te = fam == hf; tr = ~te
    probe = LogisticRegression(max_iter=2000, random_state=SEED).fit(X[tr], y[tr])
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                          LogisticRegression(max_iter=2000, random_state=SEED)).fit([texts[i] for i in np.where(tr)[0]], y[tr])
    pp = probe.predict_proba(X[te])[:, 1]
    tp = tfidf.predict_proba([texts[i] for i in np.where(te)[0]])[:, 1]
    perfam[hf] = {"n_test": int(te.sum()), "pos": int(y[te].sum()),
                  "probe": metrics(y[te], pp), "tfidf": metrics(y[te], tp), "deberta_zs": metrics(y[te], deb[te])}
    print(f"  {hf:28s} probe {perfam[hf]['probe']}  tfidf {perfam[hf]['tfidf']}  deberta {perfam[hf]['deberta_zs']}", flush=True)

def mean_auroc(k):
    vs = [perfam[f][k]["auroc"] for f in perfam if perfam[f][k]["auroc"] is not None]
    return round(float(np.mean(vs)), 3) if vs else None
summary = {"sensor": SENSOR, "n_families": len(fams),
           "mean_auroc_probe": mean_auroc("probe"), "mean_auroc_tfidf": mean_auroc("tfidf"), "mean_auroc_deberta": mean_auroc("deberta_zs"),
           "mean_f1_probe": round(float(np.mean([perfam[f]["probe"]["f1"] for f in perfam])), 3),
           "mean_f1_tfidf": round(float(np.mean([perfam[f]["tfidf"]["f1"] for f in perfam])), 3)}
out = {"meta": {"sensor": SENSOR, "test": "leave-one-family-out, routeguard skill-text", "seed": SEED}, "summary": summary, "per_family": perfam}
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
json.dump(out, open(f"{OUTDIR}/moat_crossfamily_{ts}.json", "w"), indent=2)
print("\n=== MOAT SUMMARY (mean over held-out novel families) ===", flush=True)
print(f"  probe   mean AUROC {summary['mean_auroc_probe']}  F1 {summary['mean_f1_probe']}", flush=True)
print(f"  TF-IDF  mean AUROC {summary['mean_auroc_tfidf']}  F1 {summary['mean_f1_tfidf']}", flush=True)
print(f"  DeBERTa mean AUROC {summary['mean_auroc_deberta']}", flush=True)
print(f"  -> probe beats TF-IDF on novel families by AUROC: {None if summary['mean_auroc_probe'] is None else round(summary['mean_auroc_probe']-summary['mean_auroc_tfidf'],3)}", flush=True)
