#!/usr/bin/env python3
"""Why is tool-poison-LOCO (0.738) WORSE than PI-only (0.954) at detecting MCPTox? Decompose: train the
Qwen activation probe on EACH source ALONE (PI / routeguard / minpairs / routeguard+minpairs) and test
on MCPTox. If PI-only >> routeguard-only/minpairs-only, the lower LOCO is a training-data-quality +
signal-mismatch effect (those corpora teach the wrong attack style), not the probe being weak on tool-
poison. Plus: cosine distance of the MCPTox-poison activation centroid to each training-attack centroid,
to confirm 'MCPTox poison is semantically PI-like (injected instructions)'. Seed 42."""
from __future__ import annotations
import json, glob
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
EMBDIR = "research/_results_published/_emb_alllayers"; CAP = 350
SRC = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json", "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
def cap(rows_lab):
    t, y = [], []
    for lab in (0, 1):
        pool = [x for x, l in rows_lab if l == lab and x]
        if len(pool) > CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
        for x in pool: t.append(x); y.append(lab)
    return t, np.array(y)
# sources
pi_t = []; pi_y = []
for fn in SRC.values():
    rows = [r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
    t, y = cap([(r["text"], int(r["label"])) for r in rows]); pi_t += t; pi_y = np.concatenate([pi_y, y]) if len(pi_y) else y
rg = [r for r in json.load(open("research/datasets/routeguard_external_v0.json")) if r.get("carrier") in ("SKILL.md","skill_package_text") and r.get("text")]
rg_t, rg_y = cap([(r["text"], 1 if r["label"]=="poisoned" else 0) for r in rg])
mp = json.load(open("research/datasets/minpairs_v1.json"))
mp_t, mp_y = cap([(p["clean"],0) for p in mp]+[(p["poison"],1) for p in mp])
mc_t = [r["description"] for r in json.load(open("research/datasets/mcptox_clean_descriptions_labeled.json")) if r.get("description")]
mc_y = [0]*len(mc_t); pois=[r["description"] for r in json.load(open("research/datasets/mcptox_poisoned_descriptions_labeled.json")) if r.get("description")]
mc_t += pois; mc_y += [1]*len(pois); mc_y = np.array(mc_y)
print(f"PI {len(pi_t)} | routeguard {len(rg_t)} | minpairs {len(mp_t)} | MCPTox {len(mc_t)}", flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"); mdl = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B", output_hidden_states=True, dtype=torch.float32); mdl.eval()
@torch.no_grad()
def feats(texts):
    F = []
    for t in texts:
        hs = mdl(**tok(t or " ", return_tensors="pt", truncation=True, max_length=256)).hidden_states
        F.append(np.concatenate([hs[l][0].float().mean(0).numpy() for l in (13,14,15)]))
    return np.array(F, dtype=np.float32)
EMBpi = np.load(glob.glob(f"{EMBDIR}/Qwen_Qwen2.5-0.5B__bd9cd4c746*.npy")[0])
Xpi = np.concatenate([EMBpi[:, l, :] for l in (13,14,15)], axis=1) if EMBpi.shape[0]==len(pi_t) else feats(pi_t)
print("extracting routeguard/minpairs/MCPTox activations...", flush=True)
Xrg, Xmp, Xmc = feats(rg_t), feats(mp_t), feats(mc_t)
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))

print("\n=== train each source ALONE -> test MCPTox (Qwen probe) ===", flush=True)
out = {}
for name, X, Y in [("PI-only", Xpi, pi_y), ("routeguard-only", Xrg, rg_y), ("minpairs-only", Xmp, mp_y)]:
    a = roc_auc_score(mc_y, lr().fit(X, Y).predict_proba(Xmc)[:, 1]); out[name] = round(float(a), 3)
    print(f"  {name:18s} -> MCPTox AUROC {a:.3f}", flush=True)
# combined routeguard+minpairs (the LOCO training)
Xc = np.concatenate([Xrg, Xmp]); Yc = np.concatenate([rg_y, mp_y])
a = roc_auc_score(mc_y, lr().fit(Xc, Yc).predict_proba(Xmc)[:, 1]); out["routeguard+minpairs (LOCO)"] = round(float(a), 3)
print(f"  {'rg+minpairs (LOCO)':18s} -> MCPTox AUROC {a:.3f}", flush=True)

print("\n=== centroid cosine: MCPTox-poison nearest which training-attack set? ===", flush=True)
def unit(v): return v / (np.linalg.norm(v) + 1e-9)
cen = {"PI-attack": Xpi[pi_y==1].mean(0), "routeguard-attack": Xrg[rg_y==1].mean(0), "minpairs-attack": Xmp[mp_y==1].mean(0)}
mc_pois_cen = Xmc[mc_y==1].mean(0)
for k, v in cen.items():
    cos = float(np.dot(unit(mc_pois_cen), unit(v)))
    print(f"  cos(MCPTox-poison, {k:18s}) = {cos:.3f}", flush=True)
json.dump({"train_alone_to_mcptox": out}, open("research/_results_published/mcptox_teacher_decomp.json","w"), indent=2)
print("\nsaved research/_results_published/mcptox_teacher_decomp.json", flush=True)
