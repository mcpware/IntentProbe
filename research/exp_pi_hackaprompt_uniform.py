#!/usr/bin/env python3
"""Address the sampling-bias concern: re-run the HackAPrompt real-human recall test on a UNIFORM-RANDOM
sample (random offsets across all 601,757 rows) instead of the first 2,483. If recall holds, the moat
result is not a sample-order artifact. Shipped 0.5B L13-15 vs TF-IDF, matched clean-FPR. Seed 42."""
import json, os, glob, hashlib, time, urllib.request, urllib.parse, subprocess
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,(os.cpu_count() or 4)-1))
SENSOR="Qwen/Qwen2.5-0.5B"; OUTDIR="research/_results_published"; EMBDIR=f"{OUTDIR}/_emb_alllayers"
# uniform-random HackAPrompt sample via random offsets
tok_hf=subprocess.check_output(["gopass","show","-o","dev/env/huggingface/HF_TOKEN"]).decode().strip()
H={"Authorization":f"Bearer {tok_hf}","User-Agent":"Mozilla/5.0"}
TOTAL=601757; rng=np.random.default_rng(SEED)
offs=sorted(set(int(x) for x in rng.integers(0,TOTAL-100,40)))
hp=[]
for off in offs:
    try:
        req=urllib.request.Request("https://datasets-server.huggingface.co/rows?"+urllib.parse.urlencode({"dataset":"hackaprompt/hackaprompt-dataset","config":"default","split":"train","offset":off,"length":100}),headers=H)
        d=json.load(urllib.request.urlopen(req,timeout=60))
        for it in d["rows"]:
            r=it["row"]; t=(r.get("user_input") or r.get("prompt") or "").strip()
            if t and len(t)>5: hp.append(t[:2000])
    except Exception as e: pass
    time.sleep(0.1)
print(f"uniform-random hackaprompt sample: {len(hp)} from {len(offs)} random windows",flush=True)
# training: 4 curated sources (random subsample, same as before)
SRC={"deepset":"deepset_prompt_injections.json","safeguard":"pi_safeguard.json","spml":"pi_spml.json","jayavibhav":"pi_jayavibhav.json"}
texts,y=[],[]; r2=np.random.default_rng(SEED)
for fn in SRC.values():
    rows=[r for r in json.load(open(f"research/datasets/{fn}")) if r.get("text")]
    for lab in (0,1):
        pool=[r for r in rows if int(r["label"])==lab]
        if len(pool)>350: pool=[pool[i] for i in sorted(r2.choice(len(pool),350,replace=False))]
        for r in pool: texts.append(r["text"]); y.append(lab)
y=np.array(y)
EMBtr=None
for npy in glob.glob(f"{EMBDIR}/*.npy"):
    if os.path.basename(npy).startswith("Qwen_Qwen2.5-0.5B__bd9cd4c746"):
        a=np.load(npy)
        if a.shape[0]==len(texts): EMBtr=a
mdl=AutoModel.from_pretrained(SENSOR,output_hidden_states=True,dtype=torch.float32); mdl.eval()
tk=AutoTokenizer.from_pretrained(SENSOR)
def emb(txts,tag):
    key=f"{EMBDIR}/{SENSOR.replace('/','_')}__{tag}__{hashlib.md5(('|'.join(txts)).encode()).hexdigest()[:10]}.npy"
    if os.path.exists(key): return np.load(key)
    F,t0=[],time.time()
    for i,t in enumerate(txts):
        ids=tk(t or " ",return_tensors="pt",truncation=True,max_length=256)
        with torch.no_grad(): hs=mdl(**ids).hidden_states
        F.append(np.stack([h[0].float().mean(0).numpy() for h in hs]))
        if (i+1)%300==0: print(f"  {i+1}/{len(txts)} ({time.time()-t0:.0f}s)",flush=True)
    a=np.array(F,dtype=np.float32); np.save(key,a); return a
EMBhp=emb(hp,"hackaprompt_uniform")
def feat(E,L): return np.concatenate([E[:,l,:] for l in L],axis=1)
def pipe(): return make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,random_state=SEED))
L=[13,14,15]; cm=(y==0)
clf=pipe().fit(feat(EMBtr,L),y); sc=clf.predict_proba(feat(EMBtr,L)[cm])[:,1]; sh=clf.predict_proba(feat(EMBhp,L))[:,1]
tf=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2),LogisticRegression(max_iter=5000,random_state=SEED)).fit(texts,y)
tc=tf.predict_proba([texts[i] for i in np.where(cm)[0]])[:,1]; th=tf.predict_proba(hp)[:,1]
out={"n":len(hp),"sampling":"uniform-random offsets","probe_0.5B_L13-15":{},"tfidf":{}}
for fpr in (0.01,0.05,0.10):
    out["probe_0.5B_L13-15"][f"recall@fpr{int(fpr*100)}"]=round(float((sh>=np.quantile(sc,1-fpr)).mean()),3)
    out["tfidf"][f"recall@fpr{int(fpr*100)}"]=round(float((th>=np.quantile(tc,1-fpr)).mean()),3)
json.dump(out,open(f"{OUTDIR}/pi_hackaprompt_uniform.json","w"),indent=2)
print("=== UNIFORM-RANDOM HackAPrompt recall (vs first-N: probe 0.971/0.977, tfidf 0.499/0.702) ===",flush=True)
print("  probe:",out["probe_0.5B_L13-15"],flush=True); print("  tfidf:",out["tfidf"],flush=True)
