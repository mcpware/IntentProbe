#!/usr/bin/env python3
"""Decision-A path: re-run the Edison/greedy layer search HONESTLY to see if a 0.5B config selected
WITHOUT looking at HackAPrompt beats the shipped L13-15's 88.3% recall@1%FPR (uniform sample), while
keeping the gap vs TF-IDF big.

INTEGRITY RULE (this is the whole point): the model/layer config is selected purely by leave-one-
SOURCE-out inner CV on the 4 curated PI TRAINING sources. HackAPrompt is evaluated ONCE on the
selected config. We NEVER pick the config by its HackAPrompt score — doing that is exactly how the
order-biased 97% became fake (selecting on the test set).

For transparency we ALSO compute the "p-hack ceiling" = the combo that WOULD maximize HackAPrompt
recall, and label it NOT-USED, so the gap between honest-selection and cherry-pick is visible.

Uses only cached 0.5B all-layer embeddings (HackAPrompt is positive-only, so the probe needs only the
cached uniform embeddings — no re-embed, no network). Seed 42.
"""
from __future__ import annotations
import json, os, glob, hashlib, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

SEED = 42; np.random.seed(SEED)
OUTDIR = "research/_results_published"; EMBDIR = f"{OUTDIR}/_emb_alllayers"
SRC = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json",
       "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
CAP = 350; FIRST = 4
TFIDF_FIXED = {"recall@fpr1": 0.303, "recall@fpr5": 0.528, "recall@fpr10": 0.631}  # from pi_hackaprompt_uniform.json


def load_balanced():
    texts, y, src = [], [], []
    rng = np.random.default_rng(SEED)
    for name, fn in SRC.items():
        rows = [r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
        for lab in (0, 1):
            pool = [r for r in rows if int(r["label"]) == lab]
            if len(pool) > CAP:
                pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
            for r in pool:
                texts.append(r["text"]); y.append(lab); src.append(name)
    return texts, np.array(y), np.array(src)


texts, y, src = load_balanced()
h = hashlib.md5(("|".join(texts)).encode()).hexdigest()[:10]
print(f"train texts md5={h} n={len(texts)} (expect bd9cd4c746)", flush=True)
EMBtr = np.load(glob.glob(f"{EMBDIR}/Qwen_Qwen2.5-0.5B__bd9cd4c746*.npy")[0])
EMBhp = np.load(glob.glob(f"{EMBDIR}/Qwen_Qwen2.5-0.5B__hackaprompt_uniform__*.npy")[0])
assert EMBtr.shape[0] == len(texts), (EMBtr.shape, len(texts))
print(f"train emb {EMBtr.shape}  hp emb {EMBhp.shape}", flush=True)
NL = EMBtr.shape[1]; SRCS = list(SRC.keys()); clean = (y == 0)


def feat(E, combo): return np.concatenate([E[:, l, :] for l in combo], axis=1)
def pipe(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def inner_loso_auroc(combo):
    """mean leave-one-SOURCE-out AUROC on the 4 training sources — BLIND to HackAPrompt."""
    X = feat(EMBtr, combo); vals = []
    for held in SRCS:
        tr = src != held; te = src == held
        if len(set(y[te])) < 2:
            continue
        s = pipe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        vals.append(roc_auc_score(y[te], s))
    return float(np.mean(vals))


def hp_recall(combo):
    X = feat(EMBtr, combo); clf = pipe().fit(X, y)
    sc = clf.predict_proba(X[clean])[:, 1]; sh = clf.predict_proba(feat(EMBhp, combo))[:, 1]
    return {f"recall@fpr{int(f*100)}": round(float((sh >= np.quantile(sc, 1 - f)).mean()), 3) for f in (0.01, 0.05, 0.10)}


cand = [[l] for l in range(FIRST, NL)] + [list(p) for p in itertools.combinations(range(FIRST, NL), 2)] + [[13, 14, 15]]
print(f"candidate combos: {len(cand)} (singles {NL-FIRST} + pairs + shipped triple)", flush=True)

# 1) HONEST selection: best mean inner-LOSO AUROC, never touching HackAPrompt
inner = [(inner_loso_auroc(c), c) for c in cand]
inner.sort(key=lambda z: -z[0])
best_inner_auroc, best_combo = inner[0]

# 2) compute HackAPrompt recall for the honest-selected combo + shipped + the p-hack ceiling
legit = hp_recall(best_combo)
shipped = hp_recall([13, 14, 15])
all_hp = [(hp_recall(c), c) for c in cand]
phack = max(all_hp, key=lambda z: z[0]["recall@fpr1"])

out = {
    "integrity": "config selected by inner leave-one-SOURCE-out AUROC on 4 PI training sources; HackAPrompt evaluated once; p-hack ceiling reported but NOT used",
    "tfidf_fixed": TFIDF_FIXED,
    "honest_selected": {"combo": best_combo, "inner_loso_auroc": round(best_inner_auroc, 4), "hackaprompt_recall": legit},
    "shipped_L13_15": {"combo": [13, 14, 15], "hackaprompt_recall": shipped},
    "phack_ceiling_NOT_USED": {"combo": phack[1], "hackaprompt_recall": phack[0]},
    "top5_by_inner": [{"combo": c, "inner_auroc": round(a, 4)} for a, c in inner[:5]],
}
json.dump(out, open(f"{OUTDIR}/pi_hackaprompt_config_search.json", "w"), indent=2)

print("\n=== HONEST config search (selection BLIND to HackAPrompt) ===", flush=True)
print(f"  honest-selected combo {best_combo} (inner LOSO AUROC {best_inner_auroc:.3f})  -> HackAPrompt {legit}", flush=True)
print(f"  shipped L13-15                                          -> HackAPrompt {shipped}", flush=True)
print(f"  TF-IDF (fixed)                                          -> HackAPrompt {TFIDF_FIXED}", flush=True)
print(f"  [p-hack ceiling, NOT used] combo {phack[1]}             -> HackAPrompt {phack[0]}", flush=True)
print("  top-5 by inner AUROC:", [(c, round(a, 3)) for a, c in inner[:5]], flush=True)
