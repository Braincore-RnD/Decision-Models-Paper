---
language:
  - id
  - en
  - multilingual
license: mit
tags:
  - decision-model
  - system-one
  - classification
  - calibrated-probabilities
  - xlm-roberta
  - multilingual
  - routing
base_model: FacebookAI/xlm-roberta-base
pipeline_tag: text-classification
---

# xlm-roberta-systemone

A **Multilingual System One Decision Model** built on top of [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) (278M parameters).

Instead of generating text autoregressively (System 2, $O(T)$ latency, prone to JSON parsing errors), this model reads a contextual state and scores candidate options of typed questions in a **single forward pass** ($O(1)$ decoding, `usage.output_tokens = 0`), returning mathematically calibrated probabilities that automated software pipelines, guardrails, and multi-agent routers can act on directly.

- **Backbone:** [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) (278M, bidirectional cross-attention, 100+ languages)
- **Architecture:** Semantic Option-Marker (`<mask_k>`) Head + State (`<s>`) Latent Projection (256-d) with Cosine Similarity Matching & Learned Scale ($\approx 10.02$)
- **Calibrated Temperature:** $T = 1.453$ (optimized post-hoc via L-BFGS on holdout validation NLL)
- **Training Mixture:** 5,526 samples across 3 sources (`LocalLLaMA/typed-decisions`, `BANKING77`, `CLINC150`), trained for 5 epochs on Tesla T4

---

## Benchmarks

Evaluated on official held-out test sets across 3 benchmark distributions:

| Benchmark Dataset | Accuracy | ECE (Expected Calibration Error) | Test Samples | Task Domain |
|---|:---:|:---:|:---:|---|
| **CLINC150** | **95.40%** | **0.0298** | 500 | Multi-domain conversational intent routing |
| **BANKING77** | **94.80%** | **0.0366** | 500 | Fine-grained banking & financial service routing |
| **Typed Decisions** | **50.26%** | **0.1414** | 382 | Software agent workflow & policy decisions |

*Note: ECE < 0.037 indicates probabilities closely reflect empirical truth (e.g., $P \ge 0.90$ means $\approx 90\%$ true accuracy).*

---

## Live Multilingual Test Cases

Inference examples demonstrating zero-shot cross-lingual transfer and high-confidence routing:

### 1. Bahasa Indonesia (Kasus Perbankan & ATM)
* **State:** *"Kartu debit saya tertelan di mesin ATM dan saldo terpotong dua kali tanpa uang keluar."*
* **Question:** *"Tim mana yang harus menangani tiket ini?"*
* **Candidate Options:** `billing_dispute`, `hardware_atm`, `sales_marketing`
* **Output:**
  * `hardware_atm`: **94.55%** (Selected)
  * `billing_dispute`: 3.78%
  * `sales_marketing`: 1.67%

### 2. English (SRE / Cloud Engineering Incident Routing)
* **State:** *"API returning 504 Gateway Timeout during peak hours on user login microservice."*
* **Question:** *"Which engineering squad should be paged?"*
* **Candidate Options:** `infrastructure`, `database_admin`, `frontend_ui`
* **Output:**
  * `infrastructure`: **95.16%** (Selected)
  * `database_admin`: 2.52%
  * `frontend_ui`: 2.32%

### 3. Banking Logistics Intent
* **State:** *"How long does it take for my new replacement Mastercard to arrive in the mail?"*
* **Question:** *"Which intent does this banking customer inquiry express?"*
* **Candidate Options:** `card_arrival`, `change_pin`, `lost_stolen_card`
* **Output:**
  * `card_arrival`: **99.95%** (Selected)
  * `change_pin`: 0.03%
  * `lost_stolen_card`: 0.03%

---

## How to Use (Inference in Python)

You can load and run this model in any Python environment using `transformers` and `huggingface_hub`:

```python
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

# 1. Definisi Arsitektur Semantic Head
class SemanticDecisionModel(nn.Module):
    def __init__(self, backbone: PreTrainedModel, hidden_size: int = 768, proj_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.state_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.opt_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.log_temp = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=0.05, max=10.0)

    def forward(self, input_ids, attention_mask, mask_positions, mask_valid):
        H = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B, max_opts = mask_positions.shape
        D = H.size(-1)

        state_vec = F.normalize(self.state_proj(H[:, 0, :]), p=2, dim=-1)
        idx_exp = mask_positions.unsqueeze(-1).expand(B, max_opts, D)
        mask_reps = torch.gather(H, dim=1, index=idx_exp)
        opt_vec = F.normalize(self.opt_proj(mask_reps), p=2, dim=-1)

        cos_sim = torch.bmm(opt_vec, state_vec.unsqueeze(-1)).squeeze(-1)
        logits = (cos_sim * self.scale).masked_fill(~mask_valid, -1e4)
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)

# 2. Load Model & Tokenizer dari Hub
REPO_ID = "itsdevice/xlm-roberta-systemone"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
backbone = AutoModel.from_pretrained(REPO_ID)
model = SemanticDecisionModel(backbone, hidden_size=backbone.config.hidden_size).to(device)

head_file = hf_hub_download(repo_id=REPO_ID, filename="decision_head.pt")
head_data = torch.load(head_file, map_location=device)
model.state_proj.load_state_dict(head_data["state_proj"])
model.opt_proj.load_state_dict(head_data["opt_proj"])
model.scale.data.fill_(head_data["scale"])
model.log_temp.data.fill_(math.log(head_data["temperature"]))
model.eval()

# 3. Fungsi Predict Single-Pass
def predict(state: str, question: str, criteria: dict):
    mask_token = tokenizer.mask_token or "<mask"
    options = list(criteria.keys())
    lines = [f"State: {state.strip()}", f"Question: {question.strip()}", "Options:"]
    for opt, desc in criteria.items():
        lines.append(f"{mask_token} {opt}: {desc}")

    enc = tokenizer("\n".join(lines), max_length=512, truncation=True, return_tensors="pt").to(device)
    mask_indices = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].tolist()
    
    mask_pos = torch.tensor([mask_indices], device=device)
    mask_val = torch.ones((1, len(mask_indices)), dtype=torch.bool, device=device)

    with torch.no_grad():
        probs = model(enc["input_ids"], enc["attention_mask"], mask_pos, mask_val)[0].cpu().tolist()

    best_idx = int(torch.tensor(probs).argmax().item())
    return {
        "choice": options[best_idx],
        "confidence": round(probs[best_idx], 4),
        "probabilities": {opt: round(p, 4) for opt, p in zip(options, probs)}
    }

# 4. Contoh Pemanggilan
result = predict(
    state="Kartu ATM saya tertelan di mesin dan saldo terpotong.",
    question="Pilih kategori penanganan:",
    criteria={
        "hardware_atm": "Masalah kartu tertelan dan mesin ATM fisik rusak",
        "billing_dispute": "Klaim sengketa dana atau salah tagihan",
        "general_faq": "Informasi umum pembukaan rekening"
    }
)
print("Hasil:", result)
```

---

## Architectural Comparison

| Dimension | Standard Generative LLM (e.g. LLaMA/Qwen) | This System One Model (`xlm-roberta-systemone`) |
|---|---|---|
| **Forward Passes** | $O(T)$ ($50\text{--}200$ sequential tokens) | **$O(1)$ (Exact single forward pass)** |
| **Output Token Cost** | High ($50\text{--}200$ generated tokens) | **Zero ($0$ output tokens)** |
| **Inference Latency** | $1,000\text{--}3,500\text{ ms}$ | **$20\text{--}35\text{ ms}$ on GPU T4** |
| **JSON Parse Safety** | Can fail (hallucinated keys, malformed syntax) | **100% Type-Safe (Constrained to input criteria)** |
| **Multilingual Support** | Varies by prompt | **Native 100+ Languages (XLM-RoBERTa)** |

---

## Files in Repository

* `model.safetensors`, `config.json` : XLM-RoBERTa base encoder backbone
* `decision_head.pt` : State & Option projection matrices + learned scale & temperature
* `decision_meta.json` : Benchmark evaluation metrics, calibration temperature, and metadata
* `tokenizer.json`, `tokenizer_config.json` : XLM-RoBERTa SentencePiece tokenizer files

---

## Citation & Base Attribution

This model adapts Google/Meta's XLM-RoBERTa encoder. If using this decision model, please cite the base backbone:

```bibtex
@article{conneau2019unsupervised,
  title={Unsupervised Cross-lingual Representation Learning at Scale},
  author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1911.02116},
  year={2019}
}
```
