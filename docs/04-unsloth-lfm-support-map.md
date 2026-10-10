---
title: "Unsloth x LFM: peta dukungan & arsip model (SFT vs Decision Head vs GGUF)"
date: 2026-10-08
tags:
  - decision-models
  - unsloth
  - lfm
  - support-matrix
aliases:
  - Unsloth LFM Support Map
---

# Unsloth x LFM: peta dukungan & arsip model

> [!abstract] Definisi
> Tiga hal berbeda yang sering tertukar: (1) **Unsloth SFT/GRPO untuk LFM2.5** — didukung resmi; (2) **Unsloth `FastDecisionModel` di atas LFM** — tidak terkonfirmasi; (3) **artefak `unsloth/LFM2-*`** — repack/kuant inference, bukan bukti training. Proyek `lfm-2.5-350m-decs1` memakai (1) bila perlu dan head manual untuk decision, bukan (2).

Catatan terkait: [[00-finetune-process-and-algorithm|Pipeline & Algoritma]], [[03-eval-benchmark-guide|Eval Guide]].

---

## 1. `unsloth/LFM2-350M` itu apa (dicek 2026-10-08)

Bukan LFM2.5. Ini repack Unsloth atas **`LiquidAI/LFM2-350M`** — generasi lama:

| Properti | `unsloth/LFM2-350M` (= LFM2) | `LiquidAI/LFM2.5-350M` (backbone kita) |
|---|---|---|
| Generasi | LFM2 (10T token, tanpa RL) | LFM2.5 (28T token + RL) |
| Load | `trust_remote_code=True`, transformers git-main (era rilis) | native `transformers>=5.2`, tanpa remote code |
| Format | safetensors BF16 (+ varian GGUF) | safetensors BF16 (+ GGUF/MLX/ONNX) |
| Binar | 354.483.968 param, 16 layer (10 conv + 6 attn), ctx 32K, vocab 65.536 | 350M, ctx 32K |
| Lisensi | LFM Open License v1.0 | lfm1.0 |

> [!quote] Model card unsloth/LFM2-350M
> "Base model: LiquidAI/LFM2-350M … LFM2 is a new generation of hybrid models … Hybrid model with multiplicative gates and short convolutions: 10 double-gated short-range LIV convolution blocks and 6 grouped query attention (GQA) blocks."
> — [unsloth/LFM2-350M](https://huggingface.co/unsloth/LFM2-350M)

Implikasi: memakai `unsloth/LFM2-350M` sebagai backbone = **downgrade generasi** (10T vs 28T + RL). Tetap `LiquidAI/LFM2.5-350M`, kecuali butuh repack yang "lebih jinak" di-load lib Unsloth — hipotesis yang belum teruji juga.

---

## 2. Matriks dukungan (terverifikasi dari docs, bukan tebakan)

| Jalur | Status | Sumber |
|---|---|---|
| Unsloth SFT/GRPO/CPT di LFM2.5 (`FastLanguageModel`, LoRA r=16) | ✅ Resmi didukung ("fully supported"), ada notebook Colab | [Liquid Unsloth guide](https://docs.liquid.ai/lfm/fine-tuning/unsloth.md), [Unsloth LFM2.5 tutorial](https://unsloth.ai/docs/models/tutorials/lfm2.5) |
| Unsloth `FastDecisionModel` di LFM / hybrid | ❓ Tidak terkonfirmasi (klaim umum "should work", nol bukti) | Q&A [decision guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) |
| GGUF (`unsloth/LFM2-350M-GGUF`, `LiquidAI/*-GGUF`) | ✅ Inference-only (llama.cpp), tidak bisa training | HF model pages |
| Head manual kita (port Clef, transformers murni) | ✅ Wiring terbukti dgn bobot `LFM2.5-350M` asli | `code/` di folder ini + uji forward §3 |
| LEAP Finetune / TRL (SFT, LoRA, DPO, GRPO) | ✅ Alternatif resmi Liquid bila Unsloth bermasalah | [Liquid fine-tuning overview](https://docs.liquid.ai/lfm/fine-tuning/overview) |

---

## 3. Bug eval yang ditemukan & diperbaiki (2026-10-08, dari source Unsloth)

Gejala: `build_dataset` me-skip 500/500 baris BANKING77+CLINC150 pada model fresh, tapi lolos pada model final. Sebab (dari `unsloth/models/decision.py` + `decision_from_lm.py`):

- `build_dataset` memakai `model.decision_config["max_len"]` sebagai budget `encode_record`.
- Model fresh: `max_len = max_seq_length` (script kita: 2048). Skema 77/150 opsi melebihi itu → skip `"does not fit"`.
- Model final: `max_len = CLEF_MAX_LEN` (4096) karena load tanpa `max_seq_length`.

Bukan data rusak — murni budget konteks. Fix di `code/unsloth_eval_dec.py`: model before di-load dengan `max_seq_length=4096`, sama dengan final. Pelajaran: **samakan `max_len` saat membandingkan before vs after**, atau kolom before akan bias (nan).

---

## 4. Hasil run Kaggle (2026-10-08, T4)

| Model | typed-decisions | BANKING77 | CLINC150 | Holdout acc |
|---|---|---|---|---|
| v1 (typed saja, 2ep) | 31% → **52%** | 2% → **3%** | 17% → **1%** | 56% |
| v2 (mixture 13 sumber, 3ep) | 31% → **57%** | 2% → **68%** | 17% → **68%** | 72% |
| v3 (typed 2×, 2ep) | 31% → **57%** | 2% → **70%** | 17% → **61%** | 62%* |
| v4 (30k no-augment + lunak, 3ep) | 31% → **60%** | 2% → **26%** ⬇ | 17% → **29%** ⬇ | 63%* |
| v5 (rebalance intent 32%, 3ep) | 31% → **52%** | 2% → **36%** | 17% → **33%** | 55%* |
| v6 (resep v5 + AUGMENT=1) | 31% → **57%** | 2% → **83%** | 17% → **73%** | 59%* |
| Unsloth Qwen3.5-0.8B (acuan) | 36% → **73%** | 7% → **74%** | 19% → **76%** | 78% |

v2 membuktikan hipotesis mixture: transfer ke taksonomi tak terlihat pulih total
(3%→68%, 1%→68%). Gap tersisa vs Qwen-0.8B ada di typed-decisions (57 vs 73:
record 5-pertanyaan + label lunak, paling sulit) — wajar untuk 350M.
Sinyal overfit: eval_loss ep1→2→3 = 0.786→0.772→0.964; sweet spot ≈ epoch 2.
\* Holdout v3 (62%) dihitung eval di holdout typed-only (rekonstruksi split),
bukan holdout mixture (69.5%) — beda distribusi, jangan dibandingkan langsung.

## 5. Rencana bertahap yang legitimate (bukan spekulatif)

1. **Head manual di atas `LFM2.5-350M` post-trained** — kode `code/train_dec.py`, full Kaggle. Satu-satunya yang wiring-nya sudah dibuktikan.
2. **(Opsional, bila backbone kurang domain) SFT dulu via Unsloth** (`FastLanguageModel`, didukung resmi, 500–5.000 contoh per anjuran Liquid), lalu head training di atas hasilnya. Mulai dari `LFM2.5-350M-Base` untuk tahap ini.
3. **(Eksperimental) `code/unsloth_kaggle_dec.py`** — 1 script format tutorial Unsloth + ckpt/resume. Ekspektasi jujur: titik gagal paling mungkin = `FastDecisionModel.from_pretrained/get_peft_model` pada arsitektur hybrid (kernel LoRA Unsloth mem-patch tipe layer transformer). Sesi smoke (`MAX_STEPS=60`) dirancang untuk menemukan ini dalam hitungan menit, bukan jam.
