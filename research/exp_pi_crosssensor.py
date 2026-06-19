#!/usr/bin/env python3
"""PI Exp 3 — does a BIGGER/stronger sensor fix the cross-dataset generalization
failure found in v2? (Nicole's hypothesis: stronger model understands intent better.)

Sweep sensors Qwen2.5 {0.5B, 1.5B, 3B}, matched RELATIVE depth (~58% layers, mean-pool 3).
For each: train a logistic probe on deepset-train, then measure
  - in-distribution: deepset-test AUROC + F1
  - HELD-OUT generalization: safeguard AUROC + F1 (the number that was 0.578 / 0.53 on 0.5B)
The headline = safeguard AUROC vs sensor size. Climbing toward >=0.85 => hypothesis holds, PI revives.
Flat ~0.58 => bigger sensor does not fix it; PI is not where the probe wins.
Reuses cached 0.5B embeddings from v2 (same texts/seed). Seed 42.
"""
from __future__ import annotations
import json, time, os, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
DDIR, OUTDIR = "research/datasets", "research/_results_published"
EMBDIR = f"{OUTDIR}/_emb"; os.makedirs(EMBDIR, exist_ok=True)
# (sensor, layers) — matched to ~0.58 relative depth (0.5B:24, 1.5B:28, 3B:36 layers)
SENSORS = [("Qwen/Qwen2.5-0.5B", (13, 14, 15)),
           ("Qwen/Qwen2.5-1.5B", (15, 16, 17)),
           ("Qwen/Qwen2.5-3B",   (20, 21, 22))]

load = lambda n: json.load(open(f"{DDIR}/{n}"))
rows_of = lambda rows, s: [r for r in rows if r.get("split") == s and r.get("text")]

def f1ci(y, p, n=1000):
    y, p = np.asarray(y), np.asarray(p); rng = np.random.default_rng(SEED); idx = np.arange(len(y)); v = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(y[s])) < 2: continue
        v.append(f1_score(y[s], p[s], zero_division=0))
    return [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]

# data (identical subset to v2 so 0.5B cache hits + eval set is comparable)
deep = load("deepset_prompt_injections.json")
dtr, dte = rows_of(deep, "train"), rows_of(deep, "test")
sg = load("pi_safeguard.json"); sgall = rows_of(sg, "test") or rows_of(sg, "train")
rng = np.random.default_rng(SEED)
if len(sgall) > 1500: sgall = [sgall[i] for i in sorted(rng.choice(len(sgall), 1500, replace=False))]
TR = [r["text"] for r in dtr]; yTR = np.array([r["label"] for r in dtr])
TE = [r["text"] for r in dte]; yTE = np.array([r["label"] for r in dte])
SG = [r["text"] for r in sgall]; ySG = np.array([r["label"] for r in sgall])

def embed(model, tok, layers, texts, sslug, tag):
    key = f"{EMBDIR}/{sslug}__{tag}__{hashlib.md5(('|'.join(texts)).encode()).hexdigest()[:10]}.npy"
    if os.path.exists(key):
        print(f"    cache hit {sslug} {tag}", flush=True); return np.load(key)
    feats, t0 = [], time.time()
    for i, t in enumerate(texts):
        ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            hs = model(**ids).hidden_states
        feats.append(torch.cat([hs[l][0].mean(0) for l in layers]).float().numpy())
        if (i + 1) % 400 == 0: print(f"    {sslug} {tag} {i+1}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    arr = np.array(feats, dtype=np.float32); np.save(key, arr); return arr

results = {"meta": {"seed": SEED, "eval_heldout": "xTRam1/safe-guard-prompt-injection (n=%d)" % len(SG),
                    "train": "deepset/prompt-injections train (n=%d)" % len(TR)}, "by_sensor": []}
for sensor, layers in SENSORS:
    sslug = sensor.replace("/", "_")
    print(f"\n[{time.strftime('%H:%M:%S')}] === {sensor} layers={layers} ===", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(sensor)
        model = AutoModel.from_pretrained(sensor, output_hidden_states=True, dtype=torch.float32); model.eval()
        Xtr = embed(model, tok, layers, TR, sslug, "deepset_train")
        Xte = embed(model, tok, layers, TE, sslug, "deepset_test")
        Xsg = embed(model, tok, layers, SG, sslug, "safeguard")
        del model
        probe = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, yTR)
        pte = probe.predict_proba(Xte)[:, 1]; psg = probe.predict_proba(Xsg)[:, 1]
        # recalibrate safeguard threshold fairly (50/50 split)
        ci, ti = train_test_split(np.arange(len(ySG)), test_size=0.5, stratify=ySG, random_state=SEED)
        bt, bf = 0.5, -1
        for t in np.linspace(0.05, 0.95, 91):
            f = f1_score(ySG[ci], (psg[ci] >= t).astype(int), zero_division=0)
            if f > bf: bf, bt = f, t
        rec_f1 = f1_score(ySG[ti], (psg[ti] >= bt).astype(int), zero_division=0)
        row = {
            "sensor": sensor, "params_B": float(sensor.split("-")[-1][:-1]), "layers": list(layers),
            "probe_dim": int(Xtr.shape[1]),
            "deepset_test_indist": {"auroc": round(roc_auc_score(yTE, pte), 4),
                                     "f1@0.5": round(f1_score(yTE, (pte >= .5).astype(int), zero_division=0), 4)},
            "safeguard_heldout": {"auroc": round(roc_auc_score(ySG, psg), 4),
                                   "f1@0.5": round(f1_score(ySG, (psg >= .5).astype(int), zero_division=0), 4),
                                   "f1_recalibrated": round(rec_f1, 4),
                                   "auroc_ci_note": "point est; safeguard n=%d" % len(SG)},
        }
        print("   ", json.dumps(row["safeguard_heldout"]), flush=True)
        results["by_sensor"].append(row)
    except Exception as e:
        print(f"    SENSOR FAILED {sensor}: {e}", flush=True)
        results["by_sensor"].append({"sensor": sensor, "error": str(e)})

ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
path = f"{OUTDIR}/pi_crosssensor_{ts}.json"
json.dump(results, open(path, "w"), indent=2)
print(f"\nsaved -> {path}", flush=True)
print("HEADLINE — safeguard held-out AUROC by sensor size:", flush=True)
for r in results["by_sensor"]:
    if "error" not in r:
        print(f"  {r['sensor']:22s} ({r['params_B']}B): in-dist AUROC {r['deepset_test_indist']['auroc']}"
              f"  |  HELD-OUT AUROC {r['safeguard_heldout']['auroc']}  F1rec {r['safeguard_heldout']['f1_recalibrated']}", flush=True)
