#!/usr/bin/env python3
"""命門 test: is the probe's cross-source generalization edge real against a REAL text opponent
(ProtectAI DeBERTa, the incumbent PI classifier used by Snyk/Lakera/promptfoo), not just our weak
same-data TF-IDF baseline? Score off-the-shelf protectai/deberta-v3-base-prompt-injection-v2 on the
SAME on-disk splits — the 4 curated PI sources (per-source AUROC) and the matched-vocab minimal pairs —
and tabulate vs the probe's known shipped-0.5B leave-one-source-out numbers and TF-IDF. Honest framing:
DeBERTa is evaluated zero-shot per source (its own training, the realistic deployment of the incumbent),
the probe is leave-one-source-out on our 3 sibling sources. Read-only. Seed 42."""
from __future__ import annotations
import json, os, time
import numpy as np
from sklearn.metrics import roc_auc_score

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
SOURCES = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json",
           "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
# known probe (shipped 0.5B L13-15, leave-one-source-out) + TF-IDF, from pi_fixedconfig.json
PROBE = {"deepset": 0.933, "safeguard": 0.999, "spml": 0.990, "jayavibhav": 1.000}
TFIDF = {"deepset": 0.732, "safeguard": 0.993, "spml": 0.935, "jayavibhav": 0.997}
CAP = 350

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
print(f"loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL); mdl.eval()
# find which label index = INJECTION
id2label = mdl.config.id2label
inj_idx = next((i for i, l in id2label.items() if "INJECT" in str(l).upper()), 1)
print("id2label:", id2label, "| injection idx:", inj_idx, flush=True)

def deberta_scores(texts):
    out = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, inj_idx].tolist()
        out.extend(probs)
        if (i+16) % 320 == 0: print(f"  scored {i+16}/{len(texts)}", flush=True)
    return np.array(out)

rng = np.random.default_rng(42)
print("\n=== per-source AUROC: probe (shipped 0.5B LOSO) vs TF-IDF vs DeBERTa (off-the-shelf) ===", flush=True)
results = {}
for name, fn in SOURCES.items():
    rows = [r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
    texts, y = [], []
    for lab in (0, 1):
        pool = [r for r in rows if int(r["label"]) == lab]
        if len(pool) > CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
        for r in pool: texts.append(r["text"]); y.append(lab)
    t0 = time.time(); s = deberta_scores(texts)
    auroc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    results[name] = round(float(auroc), 3)
    print(f"  {name:11s} probe {PROBE[name]:.3f} | tfidf {TFIDF[name]:.3f} | DeBERTa {auroc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

# minpairs (matched-vocab, within-distribution)
mp = json.load(open("research/datasets/minpairs_v1.json"))
mtexts = [p["clean"] for p in mp if p.get("clean")] + [p["poison"] for p in mp if p.get("poison")]
my = [0]*sum(1 for p in mp if p.get("clean")) + [1]*sum(1 for p in mp if p.get("poison"))
ms = deberta_scores(mtexts)
mp_auroc = round(float(roc_auc_score(my, ms)), 3)
print(f"\n  minpairs (matched-vocab)  probe ~0.737 | tfidf ~0.821 | DeBERTa {mp_auroc}", flush=True)

mean_deberta = round(float(np.mean(list(results.values()))), 3)
out = {"per_source": {k: {"probe": PROBE[k], "tfidf": TFIDF[k], "deberta": results[k]} for k in results},
       "mean": {"probe": 0.980, "tfidf": 0.914, "deberta": mean_deberta},
       "minpairs": {"probe": 0.737, "tfidf": 0.821, "deberta": mp_auroc},
       "model": MODEL, "note": "DeBERTa off-the-shelf zero-shot per source; probe leave-one-source-out shipped 0.5B"}
json.dump(out, open("research/_results_published/probe_vs_deberta.json", "w"), indent=2)
print(f"\n=== MEAN cross-source AUROC: probe 0.980 | tfidf 0.914 | DeBERTa {mean_deberta} ===", flush=True)
print("saved research/_results_published/probe_vs_deberta.json", flush=True)
