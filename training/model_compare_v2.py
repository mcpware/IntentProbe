"""intentprobe — model comparison v2: GENERALIZATION, not in-distribution.

v1's 5-fold on hard_v3 + best-of-all-layers is optimistic (overfit risk). v2:
  - TRAIN the probe on ONE set (hard_v3)
  - pick the layer by INNER 5-fold CV on the TRAIN set ONLY (no test leakage)
  - TEST on INDEPENDENT held-out sets the probe never saw (hard, hard_v2, neutral, matched)
  - report in-dist CV (mean +/- std) AND cross-set generalization
Model selection metric = cross-set generalization, NOT the inflated in-dist number.
"""
import json, time, gc
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
D = Path.home() / "MyGithub/claude-code-organizer/research/datasets"
load = lambda f: json.load(open(D / f))
descs = lambda d: [(x["description"] if isinstance(x, dict) else x) for x in d]
def pair(cf, pf):
    c, p = descs(load(cf)), descs(load(pf))
    return c + p, np.array([0]*len(c) + [1]*len(p))

TRX, TRy = pair("hard_v3_clean.json", "hard_v3_poisoned.json")   # train set
HELD = {                                                          # independent held-out sets
    "hard_v1": pair("hard_clean.json", "hard_poisoned.json"),
    "hard_v2": pair("hard_v2_clean.json", "hard_v2_poisoned.json"),
    "neutral": pair("neutral_clean.json", "neutral_poisoned.json"),
    "matched": pair("hard_v3_matched_clean.json", "hard_v3_matched_poisoned.json"),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
clf = lambda: LogisticRegression(max_iter=2000, C=0.1)
print(f"train hard_v3 n={len(TRX)} | held-out: " + ", ".join(f"{k}({len(v[1])})" for k, v in HELD.items()) + "\n", flush=True)

MODELS = [
    ("gpt2-124m",    "openai-community/gpt2"),
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M"),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M"),
    ("gemma3-270m",  "unsloth/gemma-3-270m"),
    ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B"),
    ("gemma3-1b",    "unsloth/gemma-3-1b-pt"),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B"),
    ("gemma3n-e2b",  "unsloth/gemma-3n-E2B-it"),
]

def load_model(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    try: m = AutoModel.from_pretrained(repo, dtype=torch.float32, trust_remote_code=True)
    except Exception: m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, trust_remote_code=True)
    m.eval(); return tok, m

@torch.no_grad()
def emb(tok, m, ts):
    per = None
    for t in ts:
        t = (t or "").strip() or "n/a"
        ids = tok(t, return_tensors="pt", truncation=True, max_length=160)
        hs = m(**ids, output_hidden_states=True).hidden_states
        if per is None: per = [[] for _ in hs]
        for L, h in enumerate(hs): per[L].append(h[0, -1, :].float().numpy())
    return [np.array(v) for v in per]

rows = []
for name, repo in MODELS:
    try:
        print(f"--- {name} ---", flush=True)
        tok, m = load_model(repo)
        nparam = sum(p.numel() for p in m.parameters()) / 1e6
        EMtr = emb(tok, m, TRX)
        # inner CV on TRAIN ONLY -> pick layer
        scored = [(L, cross_val_score(clf(), EMtr[L], TRy, cv=cv)) for L in range(len(EMtr))]
        L, best_cv = max(scored, key=lambda s: s[1].mean())
        indist_m, indist_s = best_cv.mean(), best_cv.std()
        probe = clf().fit(EMtr[L], TRy)              # final probe at chosen layer
        # test on held-out sets the probe never saw, SAME layer L
        cross = {}
        for hn, (HX, Hy) in HELD.items():
            cross[hn] = probe.score(emb(tok, m, HX)[L], Hy)
        cm = float(np.mean(list(cross.values())))
        print(f"  params={nparam:.0f}M  layer L{L}/{len(EMtr)-1}  in-dist={indist_m:.3f}±{indist_s:.3f}  "
              f"cross-set={cm:.3f}  [" + " ".join(f"{k}:{v:.2f}" for k, v in cross.items()) + "]", flush=True)
        rows.append((name, nparam, L, indist_m, indist_s, cm, cross))
        del m, tok, EMtr; gc.collect()
    except Exception as e:
        print(f"  SKIP {name}: {type(e).__name__}: {str(e)[:140]}", flush=True)
        rows.append((name, None, None, None, None, None, None))

print("\n==== GENERALIZATION leaderboard (selection = cross-set, not in-dist) ====", flush=True)
print(f"{'model':15}{'params':>8}{'layer':>7}{'in-dist':>14}{'cross-set':>11}", flush=True)
for name, p, L, im, isd, cm, cr in sorted(rows, key=lambda r: (r[5] is None, -(r[5] or 0))):
    if p is None: print(f"{name:15}{'FAIL':>8}", flush=True); continue
    print(f"{name:15}{p:>7.0f}M{('L'+str(L)):>7}{im:>8.3f}±{isd:.3f}{cm:>11.3f}", flush=True)
print("\nNote: cross-set = mean accuracy on hard_v1/hard_v2/neutral/matched, never seen in training.", flush=True)
print("DONE", flush=True)
