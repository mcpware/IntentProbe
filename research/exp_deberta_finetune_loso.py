#!/usr/bin/env python3
"""TRUE apples-to-apples: fine-tune DeBERTa-v3-base on the SAME training data + protocol as the probe
(leave-one-source-out: train on 3 of {deepset,safeguard,spml,jayavibhav}, test on held-out 4th), so
both models see identical data. Removes the probe's sibling-family training advantage that the
off-the-shelf comparison had. Start with the deepset fold (largest probe edge / most discriminating).
If same-data fine-tuned DeBERTa reaches/beats the probe's 0.933 on held-out deepset, the probe's edge
was the sibling advantage; if it stays below, activation generalizes better even on identical data.
CPU. Seed 42."""
from __future__ import annotations
import json, random
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score

SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SRC = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json", "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
CAP = 350; HELD = "deepset"
rng = np.random.default_rng(SEED)
def load(name):
    rows = [r for r in json.load(open(f"research/datasets/{SRC[name]}")) if r.get("text")]
    t, y = [], []
    for lab in (0, 1):
        pool = [r for r in rows if int(r["label"]) == lab]
        if len(pool) > CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
        for r in pool: t.append(r["text"]); y.append(lab)
    return t, y
tr_t, tr_y = [], []
for s in SRC:
    if s == HELD: continue
    t, y = load(s); tr_t += t; tr_y += y
te_t, te_y = load(HELD)
print(f"train (3 sources, no {HELD}): {len(tr_t)} | test {HELD}: {len(te_t)}", flush=True)

MODEL = "microsoft/deberta-v3-base"
# microsoft/deberta-v3-base ships only spm.model and this transformers version mis-converts it as
# tiktoken; protectai's repo ships a working tokenizer.json with the SAME deberta-v3-base vocab.
tok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
def enc(texts):
    e = tok(texts, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
    return e["input_ids"], e["attention_mask"]
Xi, Xm = enc(tr_t); yt = torch.tensor(tr_y)
dl = DataLoader(TensorDataset(Xi, Xm, yt), batch_size=16, shuffle=True)
opt = torch.optim.AdamW(mdl.parameters(), lr=2e-5)
mdl.train()
EPOCHS = 3
for ep in range(EPOCHS):
    tot = 0.0
    for ids, mask, lab in dl:
        opt.zero_grad()
        out = mdl(input_ids=ids, attention_mask=mask, labels=lab)
        out.loss.backward(); opt.step(); tot += out.loss.item()
    print(f"  epoch {ep+1}/{EPOCHS} loss {tot/len(dl):.4f}", flush=True)
mdl.eval()
Ti, Tm = enc(te_t)
scores = []
with torch.no_grad():
    for i in range(0, len(te_t), 16):
        out = mdl(input_ids=Ti[i:i+16], attention_mask=Tm[i:i+16]).logits
        scores.extend(torch.softmax(out, -1)[:, 1].tolist())
auroc = roc_auc_score(te_y, scores)
print(f"\n=== held-out {HELD} AUROC (SAME-DATA, apples-to-apples) ===", flush=True)
print(f"  probe (0.5B activation)      0.933", flush=True)
print(f"  TF-IDF (same data)           0.732", flush=True)
print(f"  DeBERTa off-the-shelf        0.876", flush=True)
print(f"  DeBERTa FINE-TUNED same data {auroc:.3f}  <- the fair number", flush=True)
json.dump({"held_out": HELD, "probe": 0.933, "tfidf": 0.732, "deberta_offshelf": 0.876,
           "deberta_finetuned_samedata": round(float(auroc), 3), "n_train": len(tr_t), "n_test": len(te_t)},
          open("research/_results_published/deberta_finetune_loso_deepset.json", "w"), indent=2)
print("\nsaved research/_results_published/deberta_finetune_loso_deepset.json", flush=True)
