#!/usr/bin/env python3
"""Fill the gap: how does the REAL opponent product — off-the-shelf ProtectAI DeBERTa
(protectai/deberta-v3-base-prompt-injection-v2, ~100x training data) — score ZERO-SHOT on MCPTox
tool-poison (n=827)? We already have it at 0.56 on synthetic minpairs but never on MCPTox itself.
Head-to-head with: our PI-trained Qwen probe (0.954, from pi_to_mcptox_crossmechanism), same-data
frozen-DeBERTa-features (0.876), TF-IDF (0.826). Off-the-shelf DeBERTa = SequenceClassification,
injection-prob = softmax[:,1]. Seed 42."""
from __future__ import annotations
import json
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
mc_t = [r["description"] for r in json.load(open("research/datasets/mcptox_clean_descriptions_labeled.json")) if r.get("description")]
mc_y = [0]*len(mc_t)
pois = [r["description"] for r in json.load(open("research/datasets/mcptox_poisoned_descriptions_labeled.json")) if r.get("description")]
mc_t += pois; mc_y += [1]*len(pois); mc_y = np.array(mc_y)
print(f"MCPTox: {len(mc_t)} ({int((mc_y==1).sum())} poison / {int((mc_y==0).sum())} clean)", flush=True)

MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL); mdl.eval()
# label 1 = INJECTION in protectai v2
print("id2label:", mdl.config.id2label, flush=True)
inj_idx = [i for i, l in mdl.config.id2label.items() if str(l).upper() in ("INJECTION", "LABEL_1")]
inj_idx = inj_idx[0] if inj_idx else 1
@torch.no_grad()
def inj_prob(texts):
    s = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i+16], return_tensors="pt", truncation=True, max_length=256, padding=True)
        p = torch.softmax(mdl(**enc).logits, -1)[:, inj_idx]
        s.extend(p.tolist())
    return np.array(s)
sc = inj_prob(mc_t)
auroc = roc_auc_score(mc_y, sc)
print("\n=== ZERO-SHOT on MCPTox tool-poison — vs the REAL product opponent ===", flush=True)
print(f"  our Qwen probe (trained on PI)        0.954", flush=True)
print(f"  frozen-DeBERTa-features (same data)   0.876", flush=True)
print(f"  TF-IDF (same data)                    0.826", flush=True)
print(f"  ProtectAI DeBERTa OFF-THE-SHELF (product, ~100x data, zero-shot)  {auroc:.3f}  <- the gap", flush=True)
verdict = "probe still highest vs the real product" if 0.954 - auroc > 0.02 else ("tie" if abs(0.954-auroc) <= 0.02 else "off-the-shelf product wins")
print(f"\n  -> {verdict}", flush=True)
json.dump({"test": "MCPTox tool-poison zero-shot", "n": len(mc_t), "n_poison": int((mc_y==1).sum()),
           "our_probe_PItrained": 0.954, "frozen_deberta_features_samedata": 0.876, "tfidf_samedata": 0.826,
           "protectai_deberta_offshelf_zeroshot": round(float(auroc), 3), "id2label": mdl.config.id2label, "verdict": verdict},
          open("research/_results_published/deberta_offshelf_mcptox.json", "w"), indent=2)
print("\nsaved research/_results_published/deberta_offshelf_mcptox.json", flush=True)
