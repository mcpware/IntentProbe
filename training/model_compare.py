"""intentprobe — model 性價比擂台.

Compare modern sub-2B models as the activation-probe backbone, using the EXACT
methodology from the original GPT-2 paper (reproduce-experiments.ipynb):
  - last-token residual-stream activation (not mean-pool)
  - LogisticRegression(max_iter=2000, C=0.1), no StandardScaler
  - StratifiedKFold(5, shuffle, seed=42)
  - sweep layers, report best
Metric: hard_v3 (same-vocab) accuracy vs params vs CPU embed speed = 性價比.
"""
import json, time, gc
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
D = Path.home() / "MyGithub/claude-code-organizer/research/datasets"
load = lambda f: json.load(open(D / f))
descs = lambda d: [(x["description"] if isinstance(x, dict) else x) for x in d]

hc, hp = descs(load("hard_v3_clean.json")), descs(load("hard_v3_poisoned.json"))
H, hy = hc + hp, np.array([0]*len(hc) + [1]*len(hp))
nc, npo = descs(load("neutral_clean.json")), descs(load("neutral_poisoned.json"))
Nn, ny = nc + npo, np.array([0]*len(nc) + [1]*len(npo))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
clf = lambda: LogisticRegression(max_iter=2000, C=0.1)  # paper: C=0.1, no scaler

MODELS = [
    ("gpt2-124m",     "openai-community/gpt2"),          # paper baseline
    ("smollm2-135m",  "HuggingFaceTB/SmolLM2-135M"),
    ("smollm2-360m",  "HuggingFaceTB/SmolLM2-360M"),
    ("gemma3-270m",   "unsloth/gemma-3-270m"),
    ("qwen2.5-0.5b",  "Qwen/Qwen2.5-0.5B"),
    ("gemma3-1b",     "unsloth/gemma-3-1b-pt"),
    ("qwen2.5-1.5b",  "Qwen/Qwen2.5-1.5B"),
    ("gemma3n-e2b",   "unsloth/gemma-3n-E2B-it"),         # Nicole's "Gemma 4 E2B"; may need special class
]

def load_model(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    try:
        m = AutoModel.from_pretrained(repo, dtype=torch.float32, trust_remote_code=True)
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, trust_remote_code=True)
    m.eval()
    return tok, m

@torch.no_grad()
def embed(tok, m, ts):
    per, t0 = None, time.time()
    for t in ts:
        t = (t or "").strip() or "n/a"
        ids = tok(t, return_tensors="pt", truncation=True, max_length=160)
        hs = m(**ids, output_hidden_states=True).hidden_states
        if per is None: per = [[] for _ in hs]
        for L, h in enumerate(hs):
            per[L].append(h[0, -1, :].float().numpy())  # last-token (paper method)
    return [np.array(v) for v in per], (time.time() - t0) / len(ts) * 1000

def best_layer(EM, y):
    best = (0, 0.0)
    for L in range(len(EM)):
        a = cross_val_score(clf(), EM[L], y, cv=cv).mean()
        if a > best[1]: best = (L, a)
    return best

tf = make_pipeline(TfidfVectorizer(max_features=1000), LogisticRegression(max_iter=2000, C=0.1))
print(f"[text baseline] TF-IDF hard_v3: {cross_val_score(tf, H, hy, cv=cv).mean():.3f}\n", flush=True)

results = []
for name, repo in MODELS:
    try:
        print(f"--- {name} ({repo}) ---", flush=True)
        tok, m = load_model(repo)
        nparam = sum(p.numel() for p in m.parameters()) / 1e6
        EMh, ms = embed(tok, m, H); Lh, ah = best_layer(EMh, hy)
        EMn, _ = embed(tok, m, Nn); Ln, an = best_layer(EMn, ny)
        nL = len(EMh) - 1
        print(f"  params={nparam:.0f}M  embed={ms:.0f}ms/each  hard_v3={ah:.3f}(L{Lh}/{nL})  neutral={an:.3f}(L{Ln}/{nL})", flush=True)
        results.append((name, nparam, ms, ah, an, f"L{Lh}/{nL}"))
        del m, tok, EMh, EMn; gc.collect()
    except Exception as e:
        print(f"  SKIP {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        results.append((name, None, None, None, None, "FAIL"))

print("\n==== 性價比擂台 (last-token, paper method) ====", flush=True)
print(f"{'model':15}{'params':>9}{'ms/ea':>8}{'hard_v3':>9}{'neutral':>9}", flush=True)
for name, p, ms, ah, an, L in results:
    if p is None:
        print(f"{name:15}{'—':>9}{'—':>8}{'FAIL':>9}{'':>9}", flush=True)
    else:
        print(f"{name:15}{p:>8.0f}M{ms:>8.0f}{ah:>9.3f}{an:>9.3f}", flush=True)
print("\nDONE", flush=True)
