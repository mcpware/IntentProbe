#!/usr/bin/env python3
"""Cleanest apples-to-apples: our probe = frozen generic LM (Qwen2.5-0.5B) activations + LogReg on our
small data. Fair text counterpart = frozen generic LM (bare microsoft/deberta-v3-base, MLM-pretrained,
ZERO PI training) features + LogReg on the SAME small data. Both = frozen-generic-LM-features + linear
head + identical closed-book data. Mean-pool DeBERTa's last hidden state (forward pass only, no backprop
-> CPU-feasible, unlike full fine-tuning which OOM'd). Leave-one-source-out AUROC + HackAPrompt
recall@clean-FPR, head-to-head with the probe and TF-IDF. Seed 42."""
from __future__ import annotations
import json, subprocess, urllib.request, urllib.parse, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
SRC = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json", "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
PROBE = {"deepset": 0.933, "safeguard": 0.999, "spml": 0.990, "jayavibhav": 1.000}
TFIDF = {"deepset": 0.732, "safeguard": 0.993, "spml": 0.935, "jayavibhav": 0.997}
CAP = 350; rng = np.random.default_rng(SEED)
def load(name):
    rows = [r for r in json.load(open(f"research/datasets/{SRC[name]}")) if r.get("text")]
    t, y = [], []
    for lab in (0, 1):
        pool = [r for r in rows if int(r["label"]) == lab]
        if len(pool) > CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
        for r in pool: t.append(r["text"]); y.append(lab)
    return t, np.array(y)
data = {n: load(n) for n in SRC}

tok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")  # same deberta-v3-base vocab; avoids the spm/tiktoken bug
mdl = AutoModel.from_pretrained("microsoft/deberta-v3-base"); mdl.eval()
@torch.no_grad()
def feats(texts):
    out = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i+16], return_tensors="pt", truncation=True, max_length=256, padding=True)
        h = mdl(**enc).last_hidden_state                       # (B,T,H)
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)         # masked mean-pool
        out.append(pooled.numpy())
        if (i+16) % 320 == 0: print(f"    feats {i+16}/{len(texts)}", flush=True)
    return np.concatenate(out, 0)
print("extracting DeBERTa features (frozen, forward-only)...", flush=True)
F = {n: feats(data[n][0]) for n in SRC}
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))

print("\n=== LOSO cross-source AUROC: probe vs TF-IDF vs frozen-DeBERTa-features (all same data) ===", flush=True)
res = {}
for held in SRC:
    Xtr = np.concatenate([F[s] for s in SRC if s != held]); ytr = np.concatenate([data[s][1] for s in SRC if s != held])
    clf = lr().fit(Xtr, ytr); s = clf.predict_proba(F[held])[:, 1]
    a = roc_auc_score(data[held][1], s); res[held] = round(float(a), 3)
    print(f"  {held:11s} probe {PROBE[held]:.3f} | tfidf {TFIDF[held]:.3f} | frozen-DeBERTa {a:.3f}", flush=True)
mean = round(float(np.mean(list(res.values()))), 3)
print(f"  {'mean':11s} probe 0.980 | tfidf 0.914 | frozen-DeBERTa {mean}", flush=True)

# HackAPrompt: train LogReg on all-4-source DeBERTa features, score HackAPrompt, recall@clean-FPR
tok_hf = subprocess.check_output(["gopass","show","-o","dev/env/huggingface/HF_TOKEN"]).decode().strip()
Hh = {"Authorization": f"Bearer {tok_hf}", "User-Agent": "Mozilla/5.0"}
offs = sorted(set(int(x) for x in np.random.default_rng(SEED).integers(0, 601757-100, 40))); hp = []
for off in offs:
    try:
        req = urllib.request.Request("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {"dataset":"hackaprompt/hackaprompt-dataset","config":"default","split":"train","offset":off,"length":100}), headers=Hh)
        for it in json.load(urllib.request.urlopen(req, timeout=60))["rows"]:
            t = (it["row"].get("user_input") or it["row"].get("prompt") or "").strip()
            if t and len(t) > 5: hp.append(t[:2000])
    except Exception: pass
    time.sleep(0.1)
Xall = np.concatenate([F[s] for s in SRC]); yall = np.concatenate([data[s][1] for s in SRC])
clf = lr().fit(Xall, yall)
Fhp = feats(hp); sc_hp = clf.predict_proba(Fhp)[:, 1]
sc_clean = clf.predict_proba(Xall[yall == 0])[:, 1]
hpr = {f"@{int(f*100)}": round(float((sc_hp >= np.quantile(sc_clean, 1-f)).mean()), 3) for f in (0.05, 0.01)}
print(f"\n=== HackAPrompt recall@clean-FPR (n={len(hp)}) ===", flush=True)
print(f"  probe          90.3% @5 | 88.3% @1", flush=True)
print(f"  TF-IDF         52.8% @5 | 30.3% @1", flush=True)
print(f"  frozen-DeBERTa {hpr['@5']*100:.1f}% @5 | {hpr['@1']*100:.1f}% @1", flush=True)
json.dump({"loso": {k: {"probe": PROBE[k], "tfidf": TFIDF[k], "frozen_deberta": res[k]} for k in res},
           "loso_mean": {"probe": 0.980, "tfidf": 0.914, "frozen_deberta": mean},
           "hackaprompt": {"probe": {"@5":0.903,"@1":0.883}, "tfidf": {"@5":0.528,"@1":0.303}, "frozen_deberta": hpr}},
          open("research/_results_published/deberta_frozen_features.json", "w"), indent=2)
print("\nsaved research/_results_published/deberta_frozen_features.json", flush=True)
