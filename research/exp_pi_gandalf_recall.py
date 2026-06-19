#!/usr/bin/env python3
"""Definitive REAL-HUMAN test: train probe + TF-IDF on the 4 curated PI sources, then measure RECALL
on 2,483 REAL human attacks (Gandalf) — a source neither detector trained on — at MATCHED
clean-FPR. Gandalf is positive-only (all real attacks), so we compare recall at fixed FPR set on
the training clean data. If the probe catches real human attacks at materially higher recall than
TF-IDF at the same FPR, the cross-source moat holds on genuinely real adversarial human input.
Shipped config (0.5B concat L13-15). Cached training embeddings; embed Gandalf fresh. Seed 42.
"""
from __future__ import annotations
import json, os, glob, hashlib, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,(os.cpu_count() or 4)-1))
SENSOR="Qwen/Qwen2.5-0.5B"; OUTDIR="research/_results_published"; EMBDIR=f"{OUTDIR}/_emb_alllayers"
SOURCES={"deepset":"deepset_prompt_injections.json","safeguard":"pi_safeguard.json","spml":"pi_spml.json","jayavibhav":"pi_jayavibhav.json"}
CAP=350

def load_balanced():
    texts,y=[],[]; rng=np.random.default_rng(SEED)
    for fn in SOURCES.values():
        rows=[r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
        for lab in (0,1):
            pool=[r for r in rows if int(r["label"])==lab]
            if len(pool)>CAP: pool=[pool[i] for i in sorted(rng.choice(len(pool),CAP,replace=False))]
            for r in pool: texts.append(r["text"]); y.append(lab)
    return texts, np.array(y)

texts,y=load_balanced()
# cached 0.5B all-layer training embeddings (2550 rows)
EMBtr=None
for npy in glob.glob(f"{EMBDIR}/*.npy"):
    b=os.path.basename(npy)
    if b.startswith("Qwen_Qwen2.5-0.5B__bd9cd4c746"):
        a=np.load(npy)
        if a.shape[0]==len(texts): EMBtr=a
assert EMBtr is not None, "no cached 0.5B training emb"
print("train emb:", EMBtr.shape, flush=True)

# Gandalf real attacks
hp=[r["text"] for r in json.load(open("research/datasets/pi_gandalf.json")) if r.get("text")]
print("hackaprompt attacks:", len(hp), flush=True)

tok=AutoTokenizer.from_pretrained(SENSOR)
mdl=AutoModel.from_pretrained(SENSOR,output_hidden_states=True,dtype=torch.float32); mdl.eval()
def embed_all(txts,tag):
    key=f"{EMBDIR}/{SENSOR.replace('/','_')}__{tag}__{hashlib.md5(('|'.join(txts)).encode()).hexdigest()[:10]}.npy"
    if os.path.exists(key): print("cache",tag,flush=True); return np.load(key)
    F,t0=[],time.time()
    for i,t in enumerate(txts):
        ids=tok(t or " ",return_tensors="pt",truncation=True,max_length=256)
        with torch.no_grad(): hs=mdl(**ids).hidden_states
        F.append(np.stack([h[0].float().mean(0).numpy() for h in hs]))
        if (i+1)%300==0: print(f"  hp {i+1}/{len(txts)} ({time.time()-t0:.0f}s)",flush=True)
    a=np.array(F,dtype=np.float32); np.save(key,a); return a
EMBhp=embed_all(hp,"gandalf1000")
print("hp emb:",EMBhp.shape,flush=True)

def feat(E,layers): return np.concatenate([E[:,l,:] for l in layers],axis=1)
def pipe(): return make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,random_state=SEED))

CONFIGS={"shipped 0.5B L13-15":[13,14,15],"0.5B L14":[14]}
clean_mask=(y==0)
res={"n_hackaprompt":len(hp),"by_config":{}}
for name,layers in CONFIGS.items():
    Xtr=feat(EMBtr,layers); Xhp=feat(EMBhp,layers)
    clf=pipe().fit(Xtr,y)
    s_clean=clf.predict_proba(Xtr[clean_mask])[:,1]; s_hp=clf.predict_proba(Xhp)[:,1]
    row={}
    for fpr in (0.01,0.05,0.10):
        thr=np.quantile(s_clean,1-fpr); row[f"recall@fpr{int(fpr*100)}"]=round(float((s_hp>=thr).mean()),3)
    res["by_config"][name]=row
    print(f"  PROBE {name}: {row}",flush=True)
# TF-IDF
tf=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2),LogisticRegression(max_iter=5000,random_state=SEED)).fit(texts,y)
sc=tf.predict_proba([texts[i] for i in np.where(clean_mask)[0]])[:,1]; sh=tf.predict_proba(hp)[:,1]
tfrow={}
for fpr in (0.01,0.05,0.10):
    thr=np.quantile(sc,1-fpr); tfrow[f"recall@fpr{int(fpr*100)}"]=round(float((sh>=thr).mean()),3)
res["by_config"]["TF-IDF (text)"]=tfrow
print(f"  TFIDF: {tfrow}",flush=True)
json.dump(res,open(f"{OUTDIR}/pi_gandalf_recall.json","w"),indent=2)
print("\n=== REAL-HUMAN recall on Gandalf (trained on other sources) ===",flush=True)
for k,v in res["by_config"].items(): print(f"  {k:22s} {v}",flush=True)
