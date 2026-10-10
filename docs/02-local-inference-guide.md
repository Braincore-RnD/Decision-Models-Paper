---
title: "Inference Decision Model di Local: predict(), Decision API & Serving"
date: 2026-10-08
tags:
  - decision-models
  - inference
  - unsloth
  - systemone-api
  - serving
aliases:
  - Decision Local Inference
  - SystemOne API Guide
---

# Inference Decision Model di Local: `predict()`, Decision API & Serving

> [!abstract] Definisi
> Inference decision model = **satu prefill atas (state + skema) → skor semua opsi sekaligus → softmax per pertanyaan pada temperatur terkalibrasi**. Di local ada dua jalan: (a) **Python** via `FastDecisionModel.predict()` untuk batch/skrip, dan (b) **Decision API** (`POST /v1/systemone`, kompatibel Jev) via Unsloth Studio untuk diakses aplikasi apa pun. Tidak ada jalur GGUF/llama.cpp/vLLM yang didukung untuk perilaku decision head.

Catatan terkait: [[00-finetune-process-and-algorithm|Pipeline & Algoritma]], [[03-typesafe-jev-architecture|Arsitektur Jev]], [[04-laya-decision-engine|Laya]].

---

## 1. Intuisi: kenapa inference-nya murah

LLM autoregresif membayar $T_{\text{autoregressive}} = \sum_t t_{\text{step}}(t)$ (satu pass per token output). Decision model membayar **satu prefill** saja:

$$T_{\text{decision}} = t_{\text{prefill}}(S, Q) + t_{\text{head}}(M) \approx 70\text{–}500\text{ ms}$$

dengan $t_{\text{head}}$ kecil (head = LayerNorm + 2 routing layer + 4 decoder layer atas $M$ field). `usage` selalu melaporkan `output_tokens: 0` — tidak ada token yang di-generate.

---

## 2. Minimal example: `predict()` di Python

```python
FastDecisionModel.for_inference(model)  # switch ke mode eval/optimized
answers = FastDecisionModel.predict(
    model, tokenizer,
    "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
    {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, errors",
                "sales": "pricing, new plans",
            },
        },
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    },
)
print(answers["team"]["answer"])        # "billing"
print(answers["team"]["probabilities"]) # {"billing": ..., "sales": ..., "technical": ...}
```

Bentuk jawaban per tipe (`systemone_answer`):

- `choice` → `{type, choice, confidence, probabilities}` (`confidence` = probabilitas opsi menang).
- `score` → `{type, score, confidence, legend, probabilities}` (`score` = ekspektasi level).
- `noul` → `{type, noul}` (probabilitas `true`, tanpa confidence).

Perilaku penting `predict()`: membaca s.d. **`CLEF_SERVE_MAX_LEN` (16.384 token) apapun panjang train** — state panjang yang saat train terpotong tetap dibaca penuh saat serve ("cutting a state drops evidence"). Temperatur yang dipakai = hasil `calibrate()` (`head_temperature` × per-type), bukan 1.0 mentah.

---

## 3. Simpan & muat ulang (format file)

```python
model.save_pretrained("qwen-decisions")              # LoRA adapters + head
model.save_pretrained_merged("qwen-decisions-merged")  # model full 16-bit
# model.push_to_hub("hf_username/qwen-decisions", token="hf_...")

# muat ulang seperti base model:
model, tokenizer = FastDecisionModel.from_pretrained("qwen-decisions", load_in_4bit=True)
```

Di Hub, artefak decision model = safetensors backbone (+ adapter) + `joint_head.safetensors` + `joint_head_config.json` (pola yang sama terlihat pada [Cloudflare/clef](https://huggingface.co/Cloudflare/clef): backbone sharded + head terpisah + `joint_schema_model.py` sebagai custom code). Karena head adalah **custom `torch.nn.Module`** di luar arsitektur transformers standar, load/serve di luar ekosistem Unsloth/Clef membutuhkan kode tersebut — inilah akar keterbatasan di §5.

---

## 4. Serve sebagai API: Decision API (kompatibel Jev)

Dua cara menghidupkan server yang sama:

| Cara | Perintah / langkah |
|------|-------------------|
| Studio GUI (no-code) | Tab Train → **Use in Decision API**; request dengan `model: laya/default/jev-latest` otomatis mengarah ke modelmu; atur di Settings → API |
| Model hasil train via code | `UNSLOTH_SYSTEMONE_MODEL=/path/to/qwen-decisions-merged unsloth studio -H 0.0.0.0 -p 8888`, lalu nyalakan **Decision API** di Settings → API |

Migrasi dari Jev tanpa ubah kode (`pip install typesafe-sdk`):

```bash
export TYPESAFE_BASE_URL=http://localhost:8888
export TYPESAFE_API_KEY=sk-unsloth-YOUR_KEY
# model "jev-latest" kini dipetakan ke model lokal pilihanmu
```

Contoh request/response (dari [dok Decision API](https://unsloth.ai/docs/models/decision-laya)):

```bash
curl http://localhost:8888/v1/systemone \
  -H "Authorization: Bearer sk-unsloth-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "laya",
       "state": "Hi, I was charged twice for my March invoice (#4411). Please refund the duplicate today.",
       "questions": {
         "team": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors",
                               "sales": "pricing, new plans", "other": "everything else"}},
         "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
         "urgency": {"type": "score", "instructions": "How urgent is this?",
                     "criteria": ["not urgent", "soon", "today"]}}}'
```

```json
{
  "model": "laya-multilingual",
  "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 1.0,
             "probabilities": {"billing": 1.0, "technical": 0.0, "sales": 0.0, "other": 0.0}},
    "refund": {"type": "noul", "noul": 0.9794},
    "urgency": {"type": "score", "score": 1.9005, "confidence": 0.698,
                "legend": {"0": "not urgent", "1": "soon", "2": "today"},
                "probabilities": {"0": 0.004, "1": 0.0915, "2": 0.9045}}
  },
  "usage": {"input_tokens": 172, "output_tokens": 0}
}
```

Catatan operasional:

- First request me-load model ke **GPU** (10–20 dtk s.d. beberapa menit); setelah itu <1 dtk di kebanyakan CPU/GPU. Selama GPU dipakai training, Decision API menunggu dan pulih otomatis.
- `state` boleh string atau JSON apa pun. Batas: **64 pertanyaan/request**, 255 opsi/choice, 10 level/score.
- **Threshold aksi pakai `probabilities`, bukan `confidence`** — formula confidence Laya/Unsloth berbeda dari Jev, threshold Jev tidak transferable.

---

## 5. Status dukungan HF & llama.cpp (ringkas, diverifikasi)

- **Hugging Face**: training + hosting didukung (`push_to_hub`, HF Jobs). Inference decision **tidak first-class**: [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) berjalan via custom `joint_schema_model.py` (`load_release_model` → `encode_record` → `systemone()`), tab vLLM/SGLang hanya mengekspos `/v1/chat/completions`, dan model "isn't deployed by any Inference Provider".
- **llama.cpp / GGUF**: didukung untuk backbone generatif (`save_pretrained_gguf`, cth. `unsloth/LFM2-350M-GGUF`), tetapi **tidak ada jalur yang didukung untuk mengekspor decision head ke GGUF/llama.cpp/vLLM dengan perilaku decision yang utuh** (konfirmasi Q&A docs Unsloth; repo memang memuat `decision_gguf.py` eksperimental — perlakukan sebagai unsupported). Bukti independen: kuantisasi clef-flash mencatat `imatrix: no (not possible for a decision model)`; proyek [colibri](https://github.com/JustVugg/colibri/blob/main/docs/clef.md) harus menulis engine C + converter khusus (`joint_head.safetensors` terpisah) untuk serve `/v1/systemone`.
