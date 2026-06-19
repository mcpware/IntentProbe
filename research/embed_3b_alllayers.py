#!/usr/bin/env python3
"""Standalone 3B all-layer embedding (OOM-safe: only the 3B model + its growing feature list in
RAM, not the other models' cached arrays). Same data/seed as the Edison search so the cache joins
the existing 0.5B/1.5B caches and exp_pi_edison_ci.py picks it up automatically.
"""
from __future__ import annotations
import json, os, time, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
EMBDIR = "research/_results_published/_emb_alllayers"; os.makedirs(EMBDIR, exist_ok=True)
MODEL = "Qwen/Qwen2.5-3B"
SOURCES = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json",
           "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
PER_CLASS_CAP = 350

texts, y, src = [], [], []
rng = np.random.default_rng(SEED)
for name, fn in SOURCES.items():
    rows = [r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
    for lab in (0, 1):
        pool = [r for r in rows if int(r["label"]) == lab]
        if len(pool) > PER_CLASS_CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), PER_CLASS_CAP, replace=False))]
        for r in pool: texts.append(r["text"]); y.append(lab); src.append(name)
print(f"rows={len(texts)} (must match Edison: 2550)", flush=True)

cache = f"{EMBDIR}/{MODEL.replace('/','_')}__{hashlib.md5(('|'.join(texts)).encode()).hexdigest()[:10]}.npy"
if os.path.exists(cache):
    print("already cached:", cache); raise SystemExit(0)
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL, output_hidden_states=True, dtype=torch.float32); mdl.eval()
t0 = time.time(); F = []
for i, t in enumerate(texts):
    ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad(): hs = mdl(**ids).hidden_states
    F.append(np.stack([h[0].float().mean(0).numpy() for h in hs]).astype(np.float16))  # fp16 store, halve disk/RAM
    if (i + 1) % 200 == 0: print(f"  {i+1}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
arr = np.array(F, dtype=np.float32)  # save as fp32 for sklearn
np.save(cache, arr)
print(f"saved {cache} shape {arr.shape} ({time.time()-t0:.0f}s)", flush=True)
