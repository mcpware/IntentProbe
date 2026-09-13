#!/usr/bin/env python3
"""Quantify the block-tier headroom: how noisy is the SHIPPED probe on DIVERSE real benign that it
never trained on? Scrape ~500 npm package descriptions (benign, tool-description-like, not in the
probe's MCPTox-heavy training set), score them with the shipped scanner, and report how many would be
FALSE-BLOCKED if we blocked on probe activation alone (>= block_threshold 0.85), vs warn/allow. This
measures the real cost behind the docstring claim "activation-only high scores are too noisy for
automatic block". Read-only: does NOT touch the artifact. Seed n/a (descriptive)."""
from __future__ import annotations
import json, urllib.request, urllib.parse, collections
import numpy as np
from intentprobe.scanner.core import scan_texts
from intentprobe.scanner.cli import DEFAULT_ARTIFACT

KWS = ["cli", "api", "server", "database", "file", "auth", "http", "parser", "logger",
       "config", "queue", "cache", "email", "image", "pdf", "csv", "markdown", "test",
       "websocket", "scheduler"]
benign, seen = [], set()
for kw in KWS:
    try:
        url = "https://registry.npmjs.org/-/v1/search?" + urllib.parse.urlencode({"text": kw, "size": 40})
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for o in d.get("objects", []):
            desc = ((o.get("package") or {}).get("description") or "").strip()
            if desc and len(desc) > 20 and desc not in seen:
                seen.add(desc); benign.append(desc[:500])
    except Exception as e:
        print(f"  kw={kw} err={e}", flush=True)
print(f"diverse benign npm descriptions: {len(benign)}", flush=True)

res = scan_texts(benign, DEFAULT_ARTIFACT, local_files_only=True)
print("RESULT[0] structure:", json.dumps(res[0], default=str)[:600], flush=True)

def activation_of(r):
    for k in ("activation_score", "activation", "raw_activation_score", "probe_score"):
        if isinstance(r.get(k), (int, float)): return float(r[k])
    sub = r.get("risk") or {}
    for k in ("activation_score", "activation", "raw_activation_score", "probe_score", "score"):
        if isinstance(sub.get(k), (int, float)): return float(sub[k])
    return None

acts = np.array([a for a in (activation_of(r) for r in res) if a is not None], dtype=float)
risks = np.array([float(r.get("risk_score") or 0) for r in res], dtype=float)
decs = collections.Counter(r.get("decision") for r in res)
print(f"\n=== DIVERSE BENIGN (n={len(res)}) — shipped scanner ===", flush=True)
print("actual decisions (with real static-corroboration logic):", dict(decs), flush=True)
print(f"activation extracted for {len(acts)}/{len(res)} items", flush=True)
if len(acts):
    for thr in (0.30, 0.50, 0.85):
        print(f"  activation >= {thr}: {(acts>=thr).mean()*100:5.1f}%  ({int((acts>=thr).sum())}/{len(acts)})  "
              f"<- would-be FALSE-BLOCK on probe-alone at {thr}" if thr == 0.85 else
              f"  activation >= {thr}: {(acts>=thr).mean()*100:5.1f}%  ({int((acts>=thr).sum())}/{len(acts)})", flush=True)
    print("  activation percentiles:", {p: round(float(np.percentile(acts, p)), 3) for p in (50, 75, 90, 95, 99)}, flush=True)
print(f"\nrisk_score (combined) >= 0.85: {(risks>=0.85).mean()*100:.1f}%  >= 0.30 (warn): {(risks>=0.30).mean()*100:.1f}%", flush=True)
out = {"n_benign": len(res), "decisions": dict(decs),
       "activation_ge": {str(t): round(float((acts>=t).mean()), 4) for t in (0.30, 0.50, 0.85)} if len(acts) else None,
       "activation_pctiles": {str(p): round(float(np.percentile(acts, p)), 3) for p in (50, 75, 90, 95, 99)} if len(acts) else None}
json.dump(out, open("research/_results_published/blocktier_fp_benign.json", "w"), indent=2)
print("\nsaved research/_results_published/blocktier_fp_benign.json", flush=True)
