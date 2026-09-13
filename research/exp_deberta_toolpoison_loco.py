#!/usr/bin/env python3
"""Decisive test of the OOD thesis: on tool-poisoning (a DIFFERENT mechanism than the PI the wording-
novelty tests used), does a frozen TEXT representation generalize, or does only the activation probe?
Same leave-one-CORPUS-out protocol as exp_toolpoison_crosscorpus.py (train on 2 of {routeguard, MCPTox,
minpairs}, test held-out), but with frozen-bare-DeBERTa-features + LogReg, head-to-head with the probe
(known) and TF-IDF (known). If frozen-DeBERTa-features fails like TF-IDF (~chance) while the probe holds
on MCPTox (0.738), activation generalizes OOD where text doesn't -> thesis vindicated. If DeBERTa-features
matches the probe, the generalization is not activation-specific. Seed 42."""
from __future__ import annotations
import json
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
CAP = 350
# known numbers from toolpoison_crosscorpus.json
PROBE = {"routeguard": 0.640, "mcptox": 0.738, "minpairs": 0.494}
TFIDF_KNOWN = {"routeguard": 0.582, "mcptox": 0.545, "minpairs": 0.498}

def load_corpora():
    texts, y, corp = [], [], []
    def add(rows_lab, name):
        for lab in (0, 1):
            pool = [t for t, l in rows_lab if l == lab and t]
            if len(pool) > CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
            for t in pool: texts.append(t); y.append(lab); corp.append(name)
    rg = [r for r in json.load(open("research/datasets/routeguard_external_v0.json")) if r.get("carrier") in ("SKILL.md", "skill_package_text") and r.get("text")]
    add([(r["text"], 1 if r["label"] == "poisoned" else 0) for r in rg], "routeguard")
    mc = [(r["description"], 0) for r in json.load(open("research/datasets/mcptox_clean_descriptions_labeled.json")) if r.get("description")]
    mc += [(r["description"], 1) for r in json.load(open("research/datasets/mcptox_poisoned_descriptions_labeled.json")) if r.get("description")]
    add(mc, "mcptox")
    mp = json.load(open("research/datasets/minpairs_v1.json"))
    add([(p["clean"], 0) for p in mp] + [(p["poison"], 1) for p in mp], "minpairs")
    return texts, np.array(y), np.array(corp)

texts, y, corp = load_corpora()
CORP = ["routeguard", "mcptox", "minpairs"]
from collections import Counter
print("corpora:", {c: dict(Counter(y[corp == c].tolist())) for c in CORP}, "total", len(texts), flush=True)

tok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
mdl = AutoModel.from_pretrained("microsoft/deberta-v3-base"); mdl.eval()
@torch.no_grad()
def feats(idx):
    out = []
    for i in range(0, len(idx), 16):
        batch = [texts[j] for j in idx[i:i+16]]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
        h = mdl(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).numpy())
    return np.concatenate(out, 0)
print("extracting DeBERTa features for all tool-poison texts...", flush=True)
ALLF = feats(list(range(len(texts))))
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))

print("\n=== tool-poison LEAVE-ONE-CORPUS-OUT AUROC: probe vs TF-IDF vs frozen-DeBERTa (same data) ===", flush=True)
res = {}
for held in CORP:
    tr = corp != held; te = corp == held
    if len(set(y[te])) < 2: res[held] = float("nan"); continue
    clf = lr().fit(ALLF[tr], y[tr]); s = clf.predict_proba(ALLF[te])[:, 1]
    a = roc_auc_score(y[te], s); res[held] = round(float(a), 3)
    print(f"  {held:11s} probe {PROBE[held]:.3f} | tfidf {TFIDF_KNOWN[held]:.3f} | frozen-DeBERTa {a:.3f}", flush=True)
print("\n  KEY: on MCPTox, probe held at 0.738 while TF-IDF cratered to 0.545.", flush=True)
print(f"       frozen-DeBERTa-features on MCPTox = {res.get('mcptox')}  -> "
      f"{'fails like text (activation thesis VINDICATED on OOD)' if (res.get('mcptox') or 1) < 0.62 else 'also generalizes (not activation-specific)'}", flush=True)
json.dump({"loco": {k: {"probe": PROBE[k], "tfidf": TFIDF_KNOWN[k], "frozen_deberta": res[k]} for k in res}},
          open("research/_results_published/deberta_toolpoison_loco.json", "w"), indent=2)
print("\nsaved research/_results_published/deberta_toolpoison_loco.json", flush=True)
