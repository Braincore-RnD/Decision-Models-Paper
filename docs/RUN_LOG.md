---
title: "LFM-2.5-350M Decision Models: run log v1–v3 (README)"
date: 2026-10-08
tags:
  - decision-models
  - lfm
  - unsloth
  - run-log
  - kaggle
aliases:
  - decs1 run log
  - from scratch README
---

# LFM-2.5-350M Decision Models (`lfm-2.5-350m-decs1`): run log

> [!abstract] Ringkasan
> Fine-tune `LiquidAI/LFM2.5-350M` menjadi System One decision model via Unsloth
> `FastDecisionModel`, full-Kaggle. v1 membuktikan wiring + belajar in-distribution;
> v2 membuktikan mixture memulihkan transfer; v3 mengunci kalibrasi tanpa overfit.
> Model rilis: [itsdevice/lfm-2.5-350m-decs1](https://huggingface.co/itsdevice/lfm-2.5-350m-decs1) (v2).

Catatan terkait: [[00-index-rlcd-and-system-one|Hub RLCD]], [[03-typesafe-jev-architecture|Arsitektur Jev]], [[07-evaluation-methodology-and-benchmarks|Metodologi Evaluasi]].
Detail proses: `train_decision_models/00-finetune-process-and-algorithm.md`.
Peta dukungan & bug: `train_decision_models/04-unsloth-lfm-support-map.md`.

---

## 1. Hasil akhir

| Model | typed-decisions | BANKING77 | CLINC150 | Holdout acc |
|---|---|---|---|---|
| v1 (typed saja, 2ep) | 31% → **52%** | 2% → **3%** | 17% → **1%** ⬇ | 56% |
| v2 (mixture equal, 3ep, augment) | 31% → **57%** | 2% → **68%** | 17% → **68%** | 72% |
| v3 (typed 2×, 2ep, augment) | 31% → **57%** | 2% → **70%** | 17% → **61%** | 62%* |
| v4 (30k no-augment + lunak, 3ep) | 31% → **60%** | 2% → **26%** ⬇ | 17% → **29%** ⬇ | 63%* |
| v5 (rebalance intent, no-augment, 3ep) | 31% → **52%** | 2% → **36%** | 17% → **33%** | 55%* |
| v6 (resep v5 + augment) | 31% → **57%** | 2% → **83%** | 17% → **73%** | 59%* |
| Unsloth Qwen3.5-0.8B (acuan) | 36% → **73%** | 7% → **74%** | 19% → **76%** | 78% |
| Unsloth Qwen3.5-0.8B (acuan) | 36% → **73%** | 7% → **74%** | 19% → **76%** | 78% |

\* Holdout v3 dihitung di holdout typed-only (rekonstruksi split), bukan holdout
mixture (69.5%) — beda distribusi, jangan dibandingkan langsung.

**Verdict:** rilis = **v3** (kalibrasi terbaik: ECE 0.052, tanpa overfit, transfer
seimbang). v2 = angka mentah tertinggi. Gap typed ~57 vs 73 = plafon kapasitas
350M (record 5-pertanyaan + label lunak), bukan resep.

---

## 2. Kronologi run

| Run | Kernel Kaggle | Data | Epoch | Kunci temuan |
|---|---|---|---|---|
| v1 train | `amalanfadilgaib/lfm-2-5-350m-decs1-unsloth` | typed `all` 1.2k baris / 6k decisions | 2 | `Fast Lfm2 patching` — Unsloth cocok dengan hybrid LFM (hipotesis tolak terbantahkan). Loss 1.18→1.06, ECE 0.068. |
| v1 eval | `…-decs1-eval` v1–v3 | 400 + 500 + 500 canonical | – | Bug `max_len` (before nan) → fix samakan 4096. Negative transfer CLINC 17%→1%. |
| v2 train | `…-decs1-v2` | mixture 13 sumber, 12k baris / 20.252 decisions, dekontaminasi | 3 | Holdout 72%. Overfit masuk ep-3 (eval_loss 0.77→0.96). |
| v2 eval | `…-decs1-eval` v4 | sama | – | Transfer pulih: 68/68. |
| v3 train | `…-decs1-v3` | mixture, typed bobot 2×, 23.065 decisions | 2 | Flat, ECE terbaik. Bobot menggeser (banking +2, clinc −7), bukan mengangkat. |
| v3 eval | `…-decs1-eval` v5 | sama | – | Tabel final §1. |
| HF push | `…-decs1-push` | – | – | [itsdevice/lfm-2.5-350m-decs1](https://huggingface.co/itsdevice/lfm-2.5-350m-decs1) (isi = v2). |

> Catatan penamaan kernel: train v2 memakai slug `…-decs1-v2`; train v3 memakai
> slug `…-decs1-v3`. Eval selalu di slug `…-decs1-eval` (versi 1–5).

---

## 3. Kode (`train_decision_models/code/`)

| File | Peran | Status |
|---|---|---|
| `unsloth_kaggle_dec.py` | Living (v3): format tutorial + ckpt/resume + mixture + dekontaminasi | Aktif |
| `unsloth_eval_dec.py` | Eval before→after 3 benchmark + holdout, tabel markdown | Aktif |
| `archive/train_v1_typed-only-2ep.py` | Arsip setia run v1 | Beku |
| `archive/train_v2_mixture-3ep.py` | Snapshot persis run v2 | Beku |
| `head.py`, `data_dec.py`, `train_dec.py`, `kaggle_run_dec.py` | Jalur manual (port Clef, transformers murni, tanpa Unsloth) | Cadangan terverifikasi wiring |
| `README_HF.md` | Model card yang ter-push ke HF | Arsip |

Konvensi: file living berevolusi; setiap run dibekukan ke `archive/` dengan nama
`train_vN_<data>-<ep>ep.py`. Kernel Kaggle dipush dari file living.

---

---

## 4. Contoh data: mentah → format decision (2 sampel per benchmark)

Pola umum: baris mentah → converter (`convert_intent` / passthrough typed) →
train: `augment_row` (parafrase, rename opsi, derived noul) · eval: `canonical_row`
→ `build_dataset` → items (`input_ids`, `markers`, `target`, `label`).

### typed-decisions (sudah format decision sejak lahir)

Mentah = final (kolom `state`/`questions`/`gold` berupa JSON string):
`state={"agent": {"autonomy": "checkpointed", …}, "task": "Delete personal data…",
"trace_summary": {"steps": 7, "constraint_violations": 0, …}}`,
`questions={"action": {"criteria": {"continue": …, "stop": …}, "instructions": …},
"needs_review": {…}, "outcome": {…}, "risk": {…}, "urgency": {…}}`,
`gold={"action": {"label": "human_review", "probabilities": {"continue": 0.28,
"human_review": 0.43, …}}, …}` — label KERAS + distribusi LUNAK konsensus.

### BANKING77 (`legacy-datasets/banking77`, 77 intent)

Mentah: `{"text": "I still have not received my new card, I ordered over a week ago.", "label": 11}`
→ decision: `state={"message": "I still have not…"} stagnasi`,
`questions={"intent": {"type": "choice",
"instructions": "Which intent does the customer's message express?",
"criteria": {"activate_my_card": "activate my card", …, "card_arrival": "card arrival", … (77)}}}`,
`gold={"intent": "card_arrival"}`. Sampel 2: `"How do I locate my card?"` → gold sama (`card_arrival`).

### CLINC150 (`clinc/clinc_oos`, config `plus`, 150 intent + `oos`)

Mentah: `{"text": "how would you say fly in italian", "intent": 61}`
→ decision: `state={"message": …}`,
`questions={"intent": {"type": "choice", "instructions": "Which intent…?",
"criteria": {"restaurant_reviews": "restaurant reviews", …, "translate": "translate",
…, "oos": "None of the other intents applies." (151)}}}`,
`gold={"intent": "translate"}`. Sampel 2: `"what's the spanish word for pasta"` → gold sama.

Alur lengkap + teori label lunak: `train_decision_models/01-dataset-format-guide.md`.

---

## 5. Keputusan desain yang terkunci

1. **Backbone `LiquidAI/LFM2.5-Base`? Tidak — post-trained.** Head-only butuh backbone instruction-following; `-Base` hanya bila SFT-berat dulu (tak dilakukan).
2. **`unsloth/LFM2-350M`? Tidak.** Itu repack generasi LFM2 lama (10T token), bukan 2.5 (28T + RL).
3. **GGUF/llama.cpp untuk head? Tidak didukung.** Hanya backbone generatif; perilaku decision tidak survive export GGUF.
4. **Threshold aksi di `probabilities`, bukan `confidence`.** Formula confidence beda Jev vs Clef-style.
5. **Satu variabel per run** (aturan TDD repo): v1=data dasar, v2=+mixture+epoch, v3=bobot+epoch fix.

---

## 6. Next step (terkunci 2026-10-08)

Bukan tambah sumber keras lagi — mixture 13 sudah jenuh (bukti: bobot 2× hanya
menggeser). Yang kurang: **data berlabel lunak konsensus** di domain target
(multi-anotator atau distribusi teacher), satu-satunya bahan kalibrasi yang
belum dimiliki selain typed-decisions. Kandidat setelah itu: full-finetune /
rank naik (v4) bila plafon 350M mau didobrak.

### 6.1 Rekomendasi lanjutan (pasca-v4, 2026-10-08)

1. **Verifikasi urutan `conv_abuse`.** Jika eval-nya di bawah chance (~<50%),
   urutan terbalik → flip indeks + retrain. Blast radius kecil (5% data).
2. **Pakai split bawaan Open-Jev (`test`/`ood`) sebagai eval.** Lebih bersih
   dari holdout rekonstruksi; `calibration` split bisa gantikan holdout
   fit-temperatur.
3. **Early stopping, bukan epoch fixed.** Sinyal sudah ada (eval_loss spike
   ep-3 v2): berhenti saat eval_loss naik 2 eval berturut, hemat kuota.
4. **v5 kapasitas** (bila gap typed 57→73 masih dikejar): full-finetune 350M
   (muat T4) atau LoRA rank 32 — satu variabel, setelah data terkunci.
5. **Eksperimen paralel Qwen3.5-0.8B** (acuan 73–78%): pisahkan plafon arsitektur
   vs resep; backbone ganti = run terpisah, bukan vN.
6. **Rilis HF per versi mayor** (repo `-v2`, `-v3` … atau revisi) + rotasi token
   tiap push dari kernel (token pernah tertulis di riwayat kode).
7. **Fallback resmi bila Unsloth pecah:** LEAP Finetune / TRL SFT (didukung
   Liquid), lalu head manual `code/train_dec.py`.
