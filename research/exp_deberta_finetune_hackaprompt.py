#!/usr/bin/env python3
"""APPLES-TO-APPLES headline test: fine-tune DeBERTa-v3-base on the SAME 4 PI sources the probe used,
then test on HackAPrompt with the SAME protocol (recall at a clean-FPR threshold set on the training
clean). This removes the confound that off-the-shelf DeBERTa (98.3%) has ~100x more PI training data
that likely already covers HackAPrompt-style attacks. If same-data DeBERTa <= probe's 90.3%, the
off-the-shelf win was a data-quantity artifact, not a text-vs-activation generalization win. CPU. Seed 42."""
from __future__ import annotations
import json, subprocess, urllib.request, urllib.parse, time, random
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SRC = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json", "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
CAP = 350; rng = np.random.default_rng(SEED)
tr_t, tr_y = [], []
for fn in SRC.values():
    rows = [r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
    for lab in (0, 1):
        pool = [r for r in rows if int(r["label"]) == lab]
        if len(pool) > CAP: pool = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))]
        for r in pool: tr_t.append(r["text"]); tr_y.append(lab)
# HackAPrompt uniform (same seed/offsets as the probe's 90.3 number)
tok_hf = subprocess.check_output(["gopass","show","-o","dev/env/huggingface/HF_TOKEN"]).decode().strip()
Hh = {"Authorization": f"Bearer {tok_hf}", "User-Agent": "Mozilla/5.0"}
offs = sorted(set(int(x) for x in np.random.default_rng(SEED).integers(0, 601757-100, 40)))
hp = []
for off in offs:
    try:
        req = urllib.request.Request("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {"dataset":"hackaprompt/hackaprompt-dataset","config":"default","split":"train","offset":off,"length":100}), headers=Hh)
        for it in json.load(urllib.request.urlopen(req, timeout=60))["rows"]:
            t = (it["row"].get("user_input") or it["row"].get("prompt") or "").strip()
            if t and len(t) > 5: hp.append(t[:2000])
    except Exception: pass
    time.sleep(0.1)
clean_ref = [tr_t[i] for i in range(len(tr_t)) if tr_y[i] == 0]
print(f"train: {len(tr_t)} (4 sources) | HackAPrompt: {len(hp)} | clean ref: {len(clean_ref)}", flush=True)

tok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
mdl = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)
def enc(texts):
    e = tok(texts, return_tensors="pt", truncation=True, max_length=256, padding="max_length"); return e["input_ids"], e["attention_mask"]
Xi, Xm = enc(tr_t); yt = torch.tensor(tr_y)
dl = DataLoader(TensorDataset(Xi, Xm, yt), batch_size=16, shuffle=True)
opt = torch.optim.AdamW(mdl.parameters(), lr=2e-5); mdl.train()
for ep in range(3):
    tot = 0.0
    for ids, mask, lab in dl:
        opt.zero_grad(); out = mdl(input_ids=ids, attention_mask=mask, labels=lab); out.loss.backward(); opt.step(); tot += out.loss.item()
    print(f"  epoch {ep+1}/3 loss {tot/len(dl):.4f}", flush=True)
mdl.eval()
def score(texts):
    s = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            ii, mm = enc(texts[i:i+16]); s.extend(torch.softmax(mdl(input_ids=ii, attention_mask=mm).logits, -1)[:,1].tolist())
    return np.array(s)
sc_clean = score(clean_ref); sc_hp = score(hp)
print("\n=== HackAPrompt recall @ matched clean-FPR — SAME-DATA fair test ===", flush=True)
print(f"  probe (0.5B, ours)            90.3% @5 | 88.3% @1", flush=True)
print(f"  DeBERTa off-the-shelf (~100x data, confounded)  98.3% @5 | 96.2% @1", flush=True)
row = {}
for fpr in (0.05, 0.01):
    thr = np.quantile(sc_clean, 1 - fpr); row[f"@{int(fpr*100)}"] = round(float((sc_hp >= thr).mean()), 3)
print(f"  DeBERTa SAME-DATA (4 sources only)  {row['@5']*100:.1f}% @5 | {row['@1']*100:.1f}% @1  <- the fair number", flush=True)
json.dump({"n_hp": len(hp), "deberta_samedata": row, "probe": {"@5":0.903,"@1":0.883}, "deberta_offshelf": {"@5":0.983,"@1":0.962}},
          open("research/_results_published/deberta_samedata_hackaprompt.json","w"), indent=2)
print("\nsaved research/_results_published/deberta_samedata_hackaprompt.json", flush=True)
