#!/usr/bin/env python3
"""Complete the 命門: how does the REAL opponent (ProtectAI DeBERTa) do on the HackAPrompt headline,
where the probe gets 90.3%/88.3% recall @5%/1% clean-FPR vs TF-IDF 52.8%/30.3%? Re-fetch the SAME
uniform-random HackAPrompt sample (seed 42, same offsets as exp_pi_hackaprompt_uniform.py), score
DeBERTa, set its threshold on the SAME training-clean reference (the 4 PI sources' benign), and report
DeBERTa recall at the matched clean-FPR. If DeBERTa also recalls ~90% the headline gap is vs a weak
baseline only; if DeBERTa drops on novel/disguised wording the probe's headline holds vs the real
opponent. Read-only. Seed 42."""
from __future__ import annotations
import json, subprocess, urllib.request, urllib.parse, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42; rng = np.random.default_rng(SEED); TOTAL = 601757
tok_hf = subprocess.check_output(["gopass", "show", "-o", "dev/env/huggingface/HF_TOKEN"]).decode().strip()
H = {"Authorization": f"Bearer {tok_hf}", "User-Agent": "Mozilla/5.0"}
offs = sorted(set(int(x) for x in rng.integers(0, TOTAL - 100, 40)))
hp = []
for off in offs:
    try:
        req = urllib.request.Request("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {"dataset": "hackaprompt/hackaprompt-dataset", "config": "default", "split": "train", "offset": off, "length": 100}), headers=H)
        for it in json.load(urllib.request.urlopen(req, timeout=60))["rows"]:
            r = it["row"]; t = (r.get("user_input") or r.get("prompt") or "").strip()
            if t and len(t) > 5: hp.append(t[:2000])
    except Exception:
        pass
    time.sleep(0.1)
print(f"uniform HackAPrompt attacks: {len(hp)}", flush=True)

# clean reference = the 4 PI sources' benign texts (same set the probe's FPR threshold was conceptually set on)
SRC = {"deepset": "deepset_prompt_injections.json", "safeguard": "pi_safeguard.json", "spml": "pi_spml.json", "jayavibhav": "pi_jayavibhav.json"}
clean = []
for fn in SRC.values():
    clean += [r["text"] for r in json.load(open(f"research/datasets/{fn}")) if r.get("text") and int(r["label"]) == 0]
print(f"clean reference: {len(clean)}", flush=True)

MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
tk = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL); mdl.eval()
inj = next((i for i, l in mdl.config.id2label.items() if "INJECT" in str(l).upper()), 1)
def score(texts):
    out = []
    for i in range(0, len(texts), 16):
        enc = tk(texts[i:i+16], return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad(): out.extend(torch.softmax(mdl(**enc).logits, -1)[:, inj].tolist())
    return np.array(out)
sc_clean = score(clean); sc_hp = score(hp)
print("\n=== HackAPrompt recall @ matched clean-FPR — probe vs TF-IDF vs DeBERTa ===", flush=True)
print(f"  {'':22s} @5%FPR   @1%FPR", flush=True)
print(f"  {'probe (0.5B, ours)':22s} 90.3%    88.3%", flush=True)
print(f"  {'TF-IDF (same data)':22s} 52.8%    30.3%", flush=True)
row = {}
for fpr in (0.05, 0.01):
    thr = np.quantile(sc_clean, 1 - fpr); rec = float((sc_hp >= thr).mean()); row[f"@{int(fpr*100)}"] = round(rec, 3)
print(f"  {'DeBERTa (real opp.)':22s} {row['@5']*100:.1f}%    {row['@1']*100:.1f}%", flush=True)
json.dump({"n_hackaprompt": len(hp), "deberta_recall": row,
           "probe": {"@5": 0.903, "@1": 0.883}, "tfidf": {"@5": 0.528, "@1": 0.303}},
          open("research/_results_published/deberta_hackaprompt.json", "w"), indent=2)
print("\nsaved research/_results_published/deberta_hackaprompt.json", flush=True)
