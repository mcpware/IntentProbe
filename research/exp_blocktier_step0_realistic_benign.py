#!/usr/bin/env python3
"""STEP 0: is the block-tier keyword gate a leftover from the leaky n=989 era? Measure the SHIPPED
probe's probe-ALONE false-block rate (activation >= block_threshold 0.85) on a REALISTIC deployment
benign distribution — not the narrow npm-only sample. Sources: the official MCP Registry (the exact
genre the scanner faces at install time) + broad npm (general + SENSITIVE-TOPIC keywords like
ssh/credential/backup/aws/upload/secret, to surface REAL benign tools that legitimately deal with
sensitive things = the realistic version of the matched-vocab worst case). Report overall + sensitive-
subset false-block rate with a Clopper-Pearson 95% upper bound. Rate-limit-aware. Read-only; does NOT
touch core.py or the artifact."""
from __future__ import annotations
import json, time, urllib.request, urllib.parse
import numpy as np
from intentprobe.scanner.core import scan_texts
from intentprobe.scanner.cli import DEFAULT_ARTIFACT

UA = {"User-Agent": "Mozilla/5.0 intentprobe-research"}
def get(url, timeout=30):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout))

benign = {}   # desc -> tag (general/sensitive/mcp)
# --- MCP Registry (exact deployment genre) ---
try:
    cursor = ""; pulled = 0
    for _ in range(15):
        url = "https://registry.modelcontextprotocol.io/v0/servers?limit=100" + (f"&cursor={cursor}" if cursor else "")
        d = get(url)
        for s in d.get("servers", []):
            desc = (s.get("description") or "").strip()
            if desc and len(desc) > 20: benign.setdefault(desc[:500], "mcp")
        cursor = (d.get("metadata") or {}).get("nextCursor") or ""
        pulled += len(d.get("servers", []))
        if not cursor: break
        time.sleep(0.5)
    print(f"MCP registry: {pulled} servers, {sum(1 for v in benign.values() if v=='mcp')} usable desc", flush=True)
except Exception as e:
    print("MCP registry err:", e, flush=True)

# --- npm: general + sensitive-topic ---
GENERAL = ["cli","api","server","database","file","http","parser","logger","config","queue","cache",
           "email","image","pdf","csv","markdown","test","websocket","scheduler","router","template",
           "validation","date","math","stream","crypto","compress","graphql","orm","ui"]
SENSITIVE = ["ssh","credential","password","secret","token","oauth","aws","s3","backup","env",
             "keychain","private-key","upload","exfil","dotenv","vault","gcp","azure-storage","ftp","scp"]
def npm(kw, tag, size=50):
    for attempt in range(3):
        try:
            d = get("https://registry.npmjs.org/-/v1/search?" + urllib.parse.urlencode({"text": kw, "size": size}))
            n = 0
            for o in d.get("objects", []):
                desc = ((o.get("package") or {}).get("description") or "").strip()
                if desc and len(desc) > 20 and desc not in benign:
                    benign[desc[:500]] = tag; n += 1
            return n
        except Exception as e:
            if "429" in str(e): time.sleep(5 * (attempt+1)); continue
            return 0
    return 0
for kw in GENERAL:
    npm(kw, "general"); time.sleep(1.5)
for kw in SENSITIVE:
    npm(kw, "sensitive"); time.sleep(1.5)
print(f"total benign: {len(benign)}  (mcp={sum(v=='mcp' for v in benign.values())}, "
      f"sensitive={sum(v=='sensitive' for v in benign.values())}, general={sum(v=='general' for v in benign.values())})", flush=True)

texts = list(benign.keys()); tags = np.array([benign[t] for t in texts])
res = scan_texts(texts, DEFAULT_ARTIFACT, local_files_only=True)
act = np.array([float(r.get("activation_score") or 0) for r in res])
dec = [r.get("decision") for r in res]

def clopper_upper(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0: return float("nan")
    return 1.0 if k == n else float(beta.ppf(1 - alpha/2, k+1, n-k))

def report(mask, label):
    a = act[mask]; n = len(a)
    if n == 0: print(f"  {label}: n=0"); return
    k85 = int((a >= 0.85).sum())
    print(f"  {label:28s} n={n:5d} | probe>=0.85 (probe-alone false-block): {k85/n*100:5.1f}% "
          f"({k85}/{n}), Clopper-Pearson 95% upper {clopper_upper(k85,n)*100:.1f}% | "
          f">=0.50: {(a>=0.5).mean()*100:.1f}% | median {np.median(a):.3f}", flush=True)

print("\n=== probe-ALONE false-block rate on REALISTIC benign (block_threshold=0.85) ===", flush=True)
report(np.ones(len(act), bool), "ALL realistic benign")
report(tags == "mcp", "MCP-registry (deploy genre)")
report(tags == "sensitive", "sensitive-topic tools")
report(tags == "general", "general tools")
import collections
print("\n  actual decisions (with keyword gate):", dict(collections.Counter(dec)), flush=True)
k85_all = int((act >= 0.85).sum())
out = {"n": len(act), "by_tag": {t: {"n": int((tags==t).sum()),
        "fp_block_0.85": round(float((act[tags==t]>=0.85).mean()),4) if (tags==t).any() else None}
        for t in ("mcp","sensitive","general")},
       "overall_fp_block_0.85": round(k85_all/len(act),4),
       "overall_clopper95_upper": round(clopper_upper(k85_all,len(act)),4)}
json.dump(out, open("research/_results_published/blocktier_step0_realistic.json","w"), indent=2)
print("\nsaved research/_results_published/blocktier_step0_realistic.json", flush=True)
