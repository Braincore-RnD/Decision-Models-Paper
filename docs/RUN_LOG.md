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

---

## 7. Run v7 (2026-10-11–12, 7 train + 6 eval)

Protokol v7 (`code/train_decision_models/unsloth_kaggle_dec.py` living):
`MAX_SEQ_LEN` 4096, 3 epoch + best-`eval_loss` (early stop patience 2),
split train/cal/test (`SEED`, `SEED+5`), `FORCE_NEW=1`, warmup fallback.
Overfit ep-3 konsisten di semua run (best selalu ep-2).

| # | Kernel / job | Env kunci | Hasil kunci |
|---|---|---|---|
| 1 | `amalanfadilgaib/lfm-decs1-v7-seed3407-full` | `SEED=3407` | typed-ext 55.9; 43171 decisions |
| 2 | `…-seed41-full` | `SEED=41` | typed-ext 53.3; best ep-2 (overfit 0.72→0.90) |
| 3 | `…-seed42-full` | `SEED=42` | typed-ext 55.5; best ep-2 (0.62→1.01) |
| 4 | `…-loto-banking-full` | `HOLDOUT_TAXONOMY=banking77` | BANKING77 transfer **44% [38–47%]** (eval #9) |
| 5 | `…-loto-clinc-full` | `HOLDOUT_TAXONOMY=clinc150` | CLINC150 transfer **55.6%** (eval #12, Colab) |

## 8. Run #7 — A-seed41/42 (2026-10-18/19, AUG=none + EVAL_BATCH=2)

A-seed3407 sudah ada (`ablasi-none-full`: BANK 86/CLINC 84 in-dist, ext 0.5925).
`A-seed41`: ext 0.58, int 0.7376, eval_loss 0.767→0.676→0.8685 (best ep-2).
`A-seed42`: ext 0.5165, int 0.7071, eval_loss 0.7177→0.6126→0.756 (best ep-2).
Catatan: `external_test` = typed-test in-dist (bukan bench LOTO);
MAX_DECISIONS tak aktif (rows 20400 < 31754) → confound +25% tetap ada, disclose.
Ablasi in-dist 3-seed (typed-test ext): A 0.563±0.033 vs E 0.549±0.011 —
selisih 1.4pt < spread A sendiri (7.6pt). Klaim "canonical wins" TAK TAHAN (#7).

## 9. Run #15/#16 — LFM2-700M recipe-E (2026-10-19, ~4h GPU)

`lfm-decs1-v7-700m`: MODEL_ID 700M, lainnya identik E-seed3407.
ext 0.583 / int 0.7644 / ECE 0.065; eval_loss 0.577→0.596→0.718 (best ep-1!).
vs E-350M 0.549±0.011: +3.4pt dari 2× parameter — capacity bantu sedikit,
tapi gap ke Qwen-0.8B vendor (73%) tetap 15pt → klaim "architectural capacity,
not recipe" TAK TAHAN penuh (#16); recipe+scale tak tutup gap.

## 10. Vonis #6 — A-transfer (2026-10-20, eval Kaggle GPU ~0.5h)

Bobot A-LOTO di-push ke HF via kernel CPU (`kernel_sources` attach output train):
`itsdevice/lfm-decs1-A-loto-banking77`, `itsdevice/lfm-decs1-A-loto-clinc150`.
Eval `predict_dump` 500 rows/bench, skor `score_preds --boot 2000`
(`docs/eval/kaggle/A-loto/`):

| | A (none) | E (full) |
|---|---|---|
| BANK | 35.4% [31–39] | 44% [38–47] |
| CLINC | 54.8% [50–59] | 55.6% |

BANK +8.6pt (CI nyaris tak overlap) → augmentasi BENAR membeli transfer di
banking. CLINC imbang → klaim headline TAHAN PARSIAL, tulis per-bench di naskah.

## 11. #13 full-test + #19 cal-1k (2026-10-20, eval Kaggle GPU ~1h)

E-seed3407 (`itsdevice/lfm-decs1-E-seed3407`), `docs/eval/kaggle/E-full/`.
Full-test: BANK 75.6% [74.1–77.0] n=3080, CLINC 67.6% [66.3–68.9] n=5500 —
TURUN ~5pt dari subsample-500 (81.3/72.9): sampling bias terkonfirmasi,
angka full yang jujur.
Cal-1k (500 bank + 500 clinc train rows): TS T=1.72 → ECE bank 0.119→0.039,
transfer ke clinc 0.091→0.048. #19 TERPENUHI (cal≥1k + CI + transfer).
OOS GAGAL (bug: 1000 iterasi pertama tak mengandung oos) → rerun dengan filter.
| 6 | `…-ablasi-none-full` | `AUG_MODE=none`, 28000 rows (`BATCH=4`, OOM batch 8) | BANK **86** / CLINC **84** (eval #10) |
| 7 | `…-ablasi-full-full` | `AUG_MODE=full`, 28000 rows | BANK **79** / CLINC **70** (eval #11) |

Eval: #9 LOTO-banking (`N_BOOT=30`), #10/#11 A vs E (`N_BOOT=30`),
#12 LOTO-clinc (Colab T4, `N_BOOT=0`), #13–15 tiga seed lean (`N_BOOT=0`).
In-dist 3-seed: typed 54.9, BANK 81.3, CLINC 72.9 (test identik).
RQ3: `predict_dump.py` → `docs/eval/colab/*/items_pred.json` →
`score_preds.py` → `metrics*.json` (ECE transfer 0.026→0.366, AUROC bertahan).

Artefak per run: final+merged+ckpt+`run_meta.json` di output kernel
masing-masing; hasil ringan di `docs/eval/`; tracker di
`docs/RECOMMENDATION.md` §6.4. Takeaway: `RECOMMENDATION.md` §7
(transfer 44/55.6, confound truncasi, kalibrasi tak transfer).

## 8. Baseline encoder ModernBERT (2026-10-14, Colab T4, #5a)

Notebook `code/train_decision_models/colab_train_modernbert_base_decision.ipynb`
(SEED=3407, batch 4xaccum4, 5 epoch, best-val ep5 77.10%, T=1.600).
Bobot: [itsdevice/modernbert-systemone](https://huggingface.co/itsdevice/modernbert-systemone);
dump: `docs/eval/encoder-modernbert/` (items di HF, metrik di repo).

| Bench | Full acc [CI] | ECE | AUROC |
|---|---|---|---|
| typed | 48.8 [43.8-54.0] (400) | 0.194 | 0.704 |
| BANKING77 | 77.0 [75.5-78.5] (3080) | 0.158 | 0.845 |
| CLINC150 | 72.8 [71.6-74.0] (5500) | 0.215 | 0.832 |
| OOS recall | 23.3 (1000) | — | — |

Takeaway: encoder menang CLINC (72.8 vs 69.4), kalah BANK (77.0 vs 80.6)
dan typed (48.8 vs 58.8); kalibrasi jauh lebih buruk; OOS sama-sama jebol
(23.3 vs 23.5). Sisa #5: baseline generatif (LFM2.5-350M JSON).

## 9. Baseline generatif backbone-sama (2026-10-14, Colab T4, #5b)

`code/train_decision_models/eval_generative.py`, MODEL=LiquidAI/LFM2.5-350M
(bobot identik), prompt->JSON zero-shot + fewshot-1, 500 banking intents.
Dump: `docs/eval/generative/`.

| Setting | acc | violation | ms/row |
|---|---|---|---|
| zero-shot | 0.0 | ~100% | 183 |
| fewshot-1 (n=100) | 0.0 | ~100% | — |

Takeaway: base backbone tak bisa ikut skema JSON sama sekali (output sampah
HTML-ish) — head-lah yang membuat backbone usable. Tak ada varian instruct
350M vendor; klaim latensi-vs-generatif = kompleksitas, bukan pengukuran.
