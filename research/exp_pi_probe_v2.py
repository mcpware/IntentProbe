#!/usr/bin/env python3
"""PI exp v2 — diagnose the Exp-1-vs-Exp-2 gap: does the probe's signal TRANSFER
to a different PI corpus (just needs threshold recalibration) or genuinely fail (OOD)?

Adds over v1:
- AUROC on both eval sets from probe.predict_proba (rank-separation, threshold-free).
- FAIR threshold recalibration on safeguard: stratified 50/50 calib/test split; pick the
  threshold that maximizes F1 on the calib half; report metrics on the held-out test half.
  (recalibrated >> out-of-box => signal transfers, threshold was the problem.)
- Caches Qwen activations to disk so cross-sensor reruns are instant.
Same sensor (Qwen2.5-0.5B, layers 13-15 mean-pool), seed 42.
"""
from __future__ import annotations
import json, time, os, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
LAYERS = (13, 14, 15)
SENSOR = "Qwen/Qwen2.5-0.5B"
DDIR, OUTDIR = "research/datasets", "research/_results_published"
EMBDIR = f"{OUTDIR}/_emb"; os.makedirs(EMBDIR, exist_ok=True)

load = lambda n: json.load(open(f"{DDIR}/{n}"))
rows_of = lambda rows, s: [r for r in rows if r.get("split") == s and r.get("text")]

tok = AutoTokenizer.from_pretrained(SENSOR)
model = AutoModel.from_pretrained(SENSOR, output_hidden_states=True, dtype=torch.float32); model.eval()
sslug = SENSOR.replace("/", "_")

def embed(texts, tag):
    key = f"{EMBDIR}/{sslug}__{tag}__{hashlib.md5(('|'.join(texts)).encode()).hexdigest()[:10]}.npy"
    if os.path.exists(key):
        print(f"    cache hit {tag}"); return np.load(key)
    feats, t0 = [], time.time()
    for i, t in enumerate(texts):
        ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            hs = model(**ids).hidden_states
        feats.append(torch.cat([hs[l][0].mean(0) for l in LAYERS]).float().numpy())
        if (i + 1) % 300 == 0: print(f"    {tag} {i+1}/{len(texts)} ({time.time()-t0:.0f}s)")
    arr = np.array(feats, dtype=np.float32); np.save(key, arr); return arr

def boot_ci(y, p, metric, n=1000):
    y, p = np.asarray(y), np.asarray(p); rng = np.random.default_rng(SEED); idx = np.arange(len(y)); v = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(y[s])) < 2: continue
        v.append(metric(y[s], p[s], zero_division=0))
    return [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]

def sc(y, p):
    return {"n": len(y), "pos": int(sum(y)), "acc": round(accuracy_score(y, p), 4),
            "precision": round(precision_score(y, p, zero_division=0), 4),
            "recall": round(recall_score(y, p, zero_division=0), 4),
            "f1": round(f1_score(y, p, zero_division=0), 4),
            "f1_ci95": boot_ci(y, p, f1_score)}

# data
deep = load("deepset_prompt_injections.json")
dtr, dte = rows_of(deep, "train"), rows_of(deep, "test")
sg = load("pi_safeguard.json"); sgall = rows_of(sg, "test") or rows_of(sg, "train")
rng = np.random.default_rng(SEED)
if len(sgall) > 1500: sgall = [sgall[i] for i in sorted(rng.choice(len(sgall), 1500, replace=False))]

Xtr = embed([r["text"] for r in dtr], "deepset_train"); ytr = np.array([r["label"] for r in dtr])
Xte = embed([r["text"] for r in dte], "deepset_test"); yte = np.array([r["label"] for r in dte])
Xsg = embed([r["text"] for r in sgall], "safeguard"); ysg = np.array([r["label"] for r in sgall])

probe = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
p_te = probe.predict_proba(Xte)[:, 1]
p_sg = probe.predict_proba(Xsg)[:, 1]

# safeguard fair recalibration: split, tune threshold on calib, report on test
Xc_i, Xt_i = train_test_split(np.arange(len(ysg)), test_size=0.5, stratify=ysg, random_state=SEED)
cal_p, cal_y = p_sg[Xc_i], ysg[Xc_i]
best_t, best_f1 = 0.5, -1
for t in np.linspace(0.05, 0.95, 91):
    f = f1_score(cal_y, (cal_p >= t).astype(int), zero_division=0)
    if f > best_f1: best_f1, best_t = f, t
test_y, test_p = ysg[Xt_i], p_sg[Xt_i]

results = {
    "meta": {"sensor": SENSOR, "layers": list(LAYERS), "seed": SEED,
             "deberta_note": "see v1 for DeBERTa zero-shot (deepset 0.54 / safeguard 0.91)"},
    "exp1_deepset_test": {"auroc": round(roc_auc_score(yte, p_te), 4),
                          "probe_thr0.5": sc(yte, (p_te >= 0.5).astype(int))},
    "exp2_safeguard_heldout": {
        "auroc_full": round(roc_auc_score(ysg, p_sg), 4),
        "out_of_box_thr0.5": sc(ysg, (p_sg >= 0.5).astype(int)),
        "recalibrated": {"calib_best_threshold": round(float(best_t), 3),
                         "metrics_on_heldout_half": sc(test_y, (test_p >= best_t).astype(int)),
                         "auroc_heldout_half": round(roc_auc_score(test_y, test_p), 4)},
    },
    "interpretation_hint": "AUROC>=0.85 => signal transfers, threshold was the issue (recalibrated F1 recovers). AUROC~0.5-0.65 => probe genuinely does not generalize to this corpus.",
}
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
path = f"{OUTDIR}/pi_probe_v2_{ts}.json"
json.dump(results, open(path, "w"), indent=2)
print(f"\nsaved -> {path}\n" + json.dumps(results, indent=2))
