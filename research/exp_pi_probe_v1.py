#!/usr/bin/env python3
"""PI track exp 1+2: does a 22 KB linear probe on frozen Qwen2.5-0.5B (layers 13-15,
mean-pooled — same as the shipped artifact) detect prompt injection, and does it
generalize to a DIFFERENT real PI corpus it never trained on?

Honest design (REDO-SPEC methodology):
- Train probe ONLY on deepset/prompt-injections TRAIN split.
- Exp 1 (in-distribution): eval on deepset TEST split (the paper's n=116 0.967 claim).
- Exp 2 (cross-dataset HELD-OUT): eval the SAME deepset-trained probe on
  xTRam1/safe-guard-prompt-injection (different source) -> the honest generalization number.
- Baseline both: protectai DeBERTa zero-shot (the incumbent PI text classifier).
- Every headline number gets a 1000x bootstrap 95% CI. Fixed seed 42.
- NOTE: DeBERTa here is ZERO-SHOT (out of its training task is PI, so this is its home turf,
  not OOD). The paper's "ties at F1 0.957" is a FINE-TUNED DeBERTa (CPU-prohibitive to redo here);
  we report zero-shot now and flag fine-tuned as a follow-up. No overclaiming.
"""
from __future__ import annotations
import json, time, os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
LAYERS = (13, 14, 15)
SENSOR = "Qwen/Qwen2.5-0.5B"
DEBERTA = "protectai/deberta-v3-base-prompt-injection-v2"
DDIR = "research/datasets"
OUTDIR = "research/_results_published"
os.makedirs(OUTDIR, exist_ok=True)


def load(name):
    return json.load(open(f"{DDIR}/{name}"))


def split_rows(rows, split):
    return [r for r in rows if r.get("split") == split and r.get("text")]


print(f"[{time.strftime('%H:%M:%S')}] loading sensor {SENSOR} ...")
tok = AutoTokenizer.from_pretrained(SENSOR)
model = AutoModel.from_pretrained(SENSOR, output_hidden_states=True, torch_dtype=torch.float32)
model.eval()


def embed(texts, bs_log=200):
    feats = []
    t0 = time.time()
    for i, t in enumerate(texts):
        ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            hs = model(**ids).hidden_states
        pooled = torch.cat([hs[l][0].mean(0) for l in LAYERS])
        feats.append(pooled.numpy())
        if (i + 1) % bs_log == 0:
            print(f"    embedded {i+1}/{len(texts)} ({time.time()-t0:.0f}s)")
    return np.array(feats, dtype=np.float32)


def boot_ci(y_true, y_pred, metric, n=1000):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y_true)); vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(y_true[s])) < 2:
            continue
        vals.append(metric(y_true[s], y_pred[s], zero_division=0))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def scores(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "pos": int(sum(y_true)),
        "acc": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "f1_ci95": boot_ci(y_true, y_pred, f1_score),
        "recall_ci95": boot_ci(y_true, y_pred, recall_score),
    }


# ---- data ----
deep = load("deepset_prompt_injections.json")
dtr, dte = split_rows(deep, "train"), split_rows(deep, "test")
sg = load("pi_safeguard.json")
sg_eval = split_rows(sg, "test") or split_rows(sg, "train")
# cap held-out eval for CPU time, keep label balance
rng = np.random.default_rng(SEED)
if len(sg_eval) > 1500:
    sg_eval = [sg_eval[i] for i in sorted(rng.choice(len(sg_eval), 1500, replace=False))]
print(f"deepset train={len(dtr)} test={len(dte)} ; safeguard held-out eval={len(sg_eval)}")

# ---- probe features ----
print(f"[{time.strftime('%H:%M:%S')}] embedding deepset-train ...")
Xtr = embed([r["text"] for r in dtr]); ytr = np.array([r["label"] for r in dtr])
print(f"[{time.strftime('%H:%M:%S')}] embedding deepset-test ...")
Xte = embed([r["text"] for r in dte]); yte = np.array([r["label"] for r in dte])
print(f"[{time.strftime('%H:%M:%S')}] embedding safeguard held-out ...")
Xsg = embed([r["text"] for r in sg_eval]); ysg = np.array([r["label"] for r in sg_eval])

probe = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED).fit(Xtr, ytr)
probe_dte = probe.predict(Xte)
probe_sg = probe.predict(Xsg)

# ---- DeBERTa zero-shot baseline ----
print(f"[{time.strftime('%H:%M:%S')}] DeBERTa zero-shot ...")
clf = pipeline("text-classification", model=DEBERTA, truncation=True, max_length=512, device=-1)
def deb_pred(texts):
    out = []
    for r in clf([t or " " for t in texts], batch_size=16):
        lab = str(r["label"]).upper()
        out.append(1 if ("INJECT" in lab or lab.endswith("1")) else 0)
    return np.array(out)
deb_dte = deb_pred([r["text"] for r in dte])
deb_sg = deb_pred([r["text"] for r in sg_eval])

results = {
    "meta": {
        "sensor": SENSOR, "layers": list(LAYERS), "seed": SEED,
        "probe_dim": int(Xtr.shape[1]), "probe_params_bytes": int(probe.coef_.size * 4 + 4),
        "deberta": DEBERTA, "deberta_mode": "zero-shot (PI is its home task; not OOD)",
        "train": {"dataset": "deepset/prompt-injections", "split": "train", "n": len(dtr)},
        "note": "Paper's 0.957 fine-tuned-DeBERTa tie NOT reproduced here (CPU-prohibitive); zero-shot only.",
    },
    "exp1_in_distribution_deepset_test": {
        "probe_qwen0.5b": scores(yte, probe_dte),
        "deberta_zeroshot": scores(yte, deb_dte),
    },
    "exp2_heldout_crossdataset_safeguard": {
        "eval_source": "xTRam1/safe-guard-prompt-injection",
        "probe_qwen0.5b": scores(ysg, probe_sg),
        "deberta_zeroshot": scores(ysg, deb_sg),
    },
}
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
path = f"{OUTDIR}/pi_probe_v1_{ts}.json"
json.dump(results, open(path, "w"), indent=2)
print(f"\n[{time.strftime('%H:%M:%S')}] saved -> {path}\n")
print(json.dumps(results, indent=2))
