#!/usr/bin/env python3
"""TRUE cross-MECHANISM test (what Nicole described): train each detector ONLY on prompt-injection
(the 4 PI sources), then zero-shot test on MCPTox tool-poisoning — a different attack mechanism the
detector never trained on. Does the activation probe carry an 'attack intent' representation that
transfers PI -> tool-poison, while text representations (TF-IDF, frozen-DeBERTa-features) do not?
probe = LogReg on frozen Qwen2.5-0.5B activations (L13-15); DeBERTa = LogReg on frozen bare
deberta-v3-base features; TF-IDF = LogReg on n-grams. All trained on PI, tested on MCPTox. Seed 42."""
from __future__ import annotations
import json, os, glob, hashlib
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
EMBDIR = "research/_results_published/_emb_alllayers"; CAP = 350
SRC = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json", "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
# PI training set (matches the cached bd9cd4c746 order)
pi_t, pi_y = [], []
for fn in SRC.values():
    rows = [r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
    for lab in (0, 1):
        pool = [r for r in rows if int(r["label"]) == lab]
        if len(pool) > CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
        for r in pool: pi_t.append(r["text"]); pi_y.append(lab)
pi_y = np.array(pi_y)
# MCPTox test set (tool-poison; different mechanism)
mc_t = [r["description"] for r in json.load(open("research/datasets/mcptox_clean_descriptions_labeled.json")) if r.get("description")]
mc_y = [0]*len(mc_t)
pois = [r["description"] for r in json.load(open("research/datasets/mcptox_poisoned_descriptions_labeled.json")) if r.get("description")]
mc_t += pois; mc_y += [1]*len(pois); mc_y = np.array(mc_y)
print(f"PI train: {len(pi_t)} | MCPTox test: {len(mc_t)} ({int((mc_y==1).sum())} poison)", flush=True)
def lr(): return make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, random_state=SEED))

# ---- 1) Qwen activation probe ----
h = hashlib.md5(("|".join(pi_t)).encode()).hexdigest()[:10]
EMBpi = np.load(glob.glob(f"{EMBDIR}/Qwen_Qwen2.5-0.5B__bd9cd4c746*.npy")[0]); assert EMBpi.shape[0] == len(pi_t), (EMBpi.shape, len(pi_t), h)
qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"); qm = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B", output_hidden_states=True, dtype=torch.float32); qm.eval()
@torch.no_grad()
def qwen_feats(texts):
    F = []
    for t in texts:
        ids = qtok(t or " ", return_tensors="pt", truncation=True, max_length=256)
        hs = qm(**ids).hidden_states
        F.append(np.concatenate([hs[l][0].float().mean(0).numpy() for l in (13, 14, 15)]))
    return np.array(F, dtype=np.float32)
def pi_feat_qwen(): return np.concatenate([EMBpi[:, l, :] for l in (13, 14, 15)], axis=1)
Xpi_q = pi_feat_qwen(); Xmc_q = qwen_feats(mc_t)
auroc_probe = roc_auc_score(mc_y, lr().fit(Xpi_q, pi_y).predict_proba(Xmc_q)[:, 1])

# ---- 2) frozen DeBERTa features ----
dtok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2"); dm = AutoModel.from_pretrained("microsoft/deberta-v3-base"); dm.eval()
@torch.no_grad()
def deb_feats(texts):
    out = []
    for i in range(0, len(texts), 16):
        enc = dtok(texts[i:i+16], return_tensors="pt", truncation=True, max_length=256, padding=True)
        hh = dm(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hh*m).sum(1)/m.sum(1).clamp(min=1)).numpy())
    return np.concatenate(out, 0)
Xpi_d = deb_feats(pi_t); Xmc_d = deb_feats(mc_t)
auroc_deb = roc_auc_score(mc_y, lr().fit(Xpi_d, pi_y).predict_proba(Xmc_d)[:, 1])

# ---- 3) TF-IDF ----
tf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000, random_state=SEED)).fit(pi_t, pi_y)
auroc_tf = roc_auc_score(mc_y, tf.predict_proba(mc_t)[:, 1])

print("\n=== TRAIN on PI -> ZERO-SHOT test on MCPTox tool-poison (cross-MECHANISM) ===", flush=True)
print(f"  Qwen activation probe   {auroc_probe:.3f}", flush=True)
print(f"  frozen-DeBERTa features {auroc_deb:.3f}", flush=True)
print(f"  TF-IDF                  {auroc_tf:.3f}", flush=True)
print(f"\n  -> {'ACTIVATION transfers PI->tool-poison, text does NOT (cross-mechanism thesis supported)' if auroc_probe - max(auroc_deb, auroc_tf) > 0.08 else 'no clear activation-specific cross-mechanism transfer'}", flush=True)
json.dump({"train": "4 PI sources", "test": "MCPTox tool-poison", "n_test": len(mc_t),
           "qwen_probe": round(float(auroc_probe), 3), "frozen_deberta": round(float(auroc_deb), 3), "tfidf": round(float(auroc_tf), 3)},
          open("research/_results_published/pi_to_mcptox_crossmechanism.json", "w"), indent=2)
print("\nsaved research/_results_published/pi_to_mcptox_crossmechanism.json", flush=True)
