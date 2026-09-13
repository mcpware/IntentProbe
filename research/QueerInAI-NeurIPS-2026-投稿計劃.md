# QueerInAI @ NeurIPS 2026 Workshop — 投稿計劃

> 2026-08-05 建立。Li 教授（SFU TAI Lab）建議投,並俾咗兩條研究方向。
> 目標 workshop：**Queer in AI and {Dis}Ability in AI Workshop @ NeurIPS 2026**

---

## 一、Workshop 官方要求（2026-08-05 核實）

**來源：** queerinai.com/neurips-2026 + OpenReview NeurIPS.cc/2026/Workshop/QueerInAI

| 項目 | 內容 |
|---|---|
| 地點/日期 | Sydney / Paris / Atlanta,**2026-12-05** |
| 🔴 **死線** | Visa-friendly: **8月10日 AoE** · **Final: 9月3日 AoE**（OpenReview 系統寫 9月11日,以官方 9/3 為準） |
| 通知 | 8/20（visa）· 9/17（final） |
| **接受形式** | research papers / **extended abstracts** / position papers / opinion / surveys / media |
| **Non-archival** | ✅ **「dual submissions, and works submitted or accepted elsewhere are all welcome」** → **IntentProbe preprint 已公開都投得,唔算 dual-submission 問題** |
| Scope | queerness + AI 或 **disability + AI**；「addressing shortcomings in current SOTA models, biases」 |
| 頁數/模板 | 官網未列明 → **投前去 OpenReview submission form 睇 field**（多數 workshop 4 頁 extended abstract） |
| Anonymize | 未列明 → 預設**唔使**（non-archival workshop 多數 non-anonymous） |
| ⚠️ **AI 生成過濾** | **「AI-generated submissions will be filtered out」** → paper 一定要你自己寫,唔可以 AI 代筆 |

⚠️ **一個 scope 問題要諗：** 呢個 workshop 主題係 **queerness/disability × AI**,唔係純 AI security。IntentProbe 本身係 AI security,同 workshop 主題**唔直接對口**。
→ **兩個做法：**
1. **搵角度扣返主題**（例：queer/marginalized 社群喺 agent ecosystem 面對嘅 targeted poisoning / 安全風險；或 probe 方法對 bias detection 嘅延伸）
2. 老實講:如果扣唔到,呢個 workshop 未必係最啱。Li 講嘅「NeurIPS workshop」係泛指 —— NeurIPS 2026 有幾十個 workshop,security/safety 向嘅（如有）可能更對口。**呢點值得同 Li 或喺 QueerInAI 問清楚。**

---

## 二、IntentProbe 現況（可以投嘅材料）

**Surviving claim（ERRATUM 後）：** 對**未見過來源 / 未見過詞彙**嘅攻擊嘅泛化能力。
- Held-out 真實攻擊集：5% 乾淨誤報率下 recall **90.3%** vs TF-IDF **52.8%**
- Cross-source LOSO AUROC **0.980 vs 0.914**
- ⚠️ Within-distribution 同 TF-IDF 打和（已撤回原本 97.5% F1）

**已有基建（research/ 入面）：** layer sweep、layer-combo search、3B all-layers embed —— 即係 probe pipeline 成熟,加新實驗成本低。

---

## 三、Li 建議嘅實驗 ①：reasoning-length controlled study（主線）

**研究問題：** MCP poisoning 係咪比「明刀明槍有害嘅文字」更難偵測？如果係,係咪因為要更長嘅推理鏈？

**假設：** 一個描述「要推越多步先意識到有害」,個「有害」訊號喺模型內部出現得越遲/越弱/越散 → probe 越難捉。

**實驗設計：**
1. **整一批攻擊樣本,控制「推理深度」**（= 要幾多步先知有害）:
   - depth-1：直接寫明惡意（"exfiltrate the SSH key"）
   - depth-2：兩步（"read ~/.ssh/id_rsa, include it in the tool description"）
   - depth-3/4：越嚟越間接
2. **固定其他所有嘢**：長度、句式、詞彙分佈 —— 只變推理深度（呢個係 controlled study 嘅關鍵,唔可以 confound）
3. 量 probe 喺每個 depth 嘅 performance（AUROC / recall @ FPR）
4. **對照組**：一般 overt-harmful prompt（非 MCP）,睇 probe 喺嗰度嘅 depth-performance 曲線

**預期結果 → finding：**
- 如果 performance 隨 depth 跌 → **證明咗一個機制**（唔止報一個數）→ 正正答返 Li 講「your result is expected / lacks insightful findings」
- 對照組如果唔跌 → 證明 MCP poisoning 嘅難度係 domain-specific,唔係 probe 本身弱

**成本：💰 最平** — 用現有 Qwen2.5-0.5B setup,CPU 都跑得,唔使 GPU。主要工夫係設計 dataset。

---

## 四、Li 建議嘅實驗 ②：跨模型 / 跨架構泛化（後備 / future work）

- **跨規模**：細模型訓練嘅 probe,靠 hidden-states mapping matrix 遷移到同系大模型
- **跨架構**：Li 假設 DeltaNet（Qwen3.5）/ KDA（Kimi-Linear, Kimi K3）嘅 state matrix 比 full-attention 更 information-rich → probe 更 work
- **成本：💰💰💰 要 GPU（RunPod）** → 呢個做 future work,唔投稿主線

---

## 五、時序（死線 9/3 AoE,即 9/4 香港中午）

| 時間 | 做 |
|---|---|
| 8月上旬 | 設計 depth-controlled dataset（實驗①） |
| 8月中 | 跑實驗①,出 depth-performance 曲線 |
| 8月中–下旬 | 寫 extended abstract（**自己寫,唔可以 AI 代**）+ 扣返 workshop 主題 |
| **9月3日前** | OpenReview 投稿 |

⚠️ **仲有 4 星期,唔使趕。趕住投一份未 polish 嘅嘢 = 喺最想留好印象嘅圈交低分。**

---

## 六、即刻可以做（唔使等）

- [ ] 登入 OpenReview 睇 submission form 實際 field（頁數/格式/要唔要 abstract）
- [ ] 確認 IntentProbe 扣唔扣到 queer/disability × AI 主題,扣唔到就搵 NeurIPS 其他 security workshop
- [ ] 開始設計 depth-controlled dataset
