---
title: "Eval & Benchmark Fine-Tune Decision Model: cara mengukur performa"
date: 2026-10-08
tags:
  - decision-models
  - evaluation
  - benchmarks
  - calibration
  - ece
  - brier-score
aliases:
  - Decision Eval Guide
  - Decision Benchmark Guide
---

# Eval & Benchmark Fine-Tune: cara mengukur performa

> [!abstract] Definisi
> Mengukur fine-tune decision model = **tiga lapis**: (1) akurasi holdout before→after saat train, (2) metrik kalibrasi `calibrate()` (accuracy, ECE, loss) di holdout, (3) `evaluate()` pada **test set terpisah yang didekontaminasi** — dilengkapi benchmark eksternal (Decision Index, Workflow Evals) dan metrik operasional (coverage pada risiko selektif). Akurasi tanpa kalibrasi tidak cukup: model 80% yang selalu klaim 100% adalah model berbahaya.

Catatan terkait: [[00-finetune-process-and-algorithm|Pipeline & Algoritma]], [[01-dataset-format-guide|Format Dataset]], [[07-evaluation-methodology-and-benchmarks|Metodologi Evaluasi & Benchmark]] (formalisme ECE/Brier/selektif).

---

## 1. Intuisi: tiga lapis pengukuran (dan kenapa harus tiga)

| Lapis | Kapan | Fungsi | Bias |
|-------|-------|--------|------|
| **A. Holdout train-loop** (`eval_strategy="epoch"`) | Selama train | Sinyal belajar / overfitting, akurasi before→after | Optimistis (dipakai pilih checkpoint) |
| **B. `calibrate()`** | Setelah train, di holdout `split_holdout` | accuracy + **ECE** + loss; sekaligus fit temperatur | Optimistis ringan (temperatur di-fit di sini) |
| **C. `evaluate()`** | Terakhir, di **test berlabel terpisah** | Vonis jujur kualitas | Minimal, jika test didekontaminasi |

> [!quote] Unsloth tentang peran masing-masing
> "calibrate returns held-out accuracy, ECE, and loss. evaluate checks accuracy on labeled test items… Use held-out items for calibration, then evaluate on a separate labeled test set for a less biased quality check."
> — Q&A [Train your own Decision Model](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)

---

## 2. Lapis A — Akurasi before → after (efek fine-tune)

Unsloth melaporkan akurasi keputusan holdout **sebelum dan sesudah** training. Titik acuan publik:

| Model | typed-decisions | BANKING77 | CLINC150 | Holdout acc |
|---|---|---|---|---|
| Qwen3.5-0.8B | 36% → **73%** | 7% → **74%** | 19% → **76%** | 78% |
| Qwen3.5-2B | 33% → **78%** | 1% → 58% | 1% → 62% | 81% |

Untuk run sendiri, catat tiga angka ini sebagai **baseline → hasil** (aturan compute-routing repo: *no baseline = no claim*): akurasi head-random/init (~chance level, 30–37% pada tugas multi-opsi), akurasi backbone tanpa head, dan akurasi pasca-train. Jika holdout acc tidak bergerak dari chance level, masalahnya hampir pasti di **render/format data** (cek `report['skipped']`), bukan di model.

---

## 3. Lapis B — `calibrate()`: accuracy, ECE, loss

```python
metrics = FastDecisionModel.calibrate(model, tokenizer, eval_items)
print(metrics)  # held-out accuracy, calibration error (ece) and loss
```

Cara membaca:

- **Accuracy** — frekuensi argmax benar. Naik = head belajar; stagnan = cek data.
- **ECE (Expected Calibration Error)** — seberapa cocok confidence dengan kenyataan (formalisme: [[07-evaluation-methodology-and-benchmarks|note 07]], §2.1). Target operasional: prediksi ber-confidence ~80% harus benar ~80% dari waktu. **Lower is better**; tidak ada threshold universal — bandingkan antar run pada setup eval yang identik.
- **Loss** — nilai objektif kalibrasi; **hanya komparatif antar run** ("don't treat an absolute value as a universal quality threshold").

Ingat mekanisme di baliknya (dari sumber kode): temperatur di-fit dengan LBFGS pada **hard label** agar confidence dibaca sebagai "peluang jawaban benar" — kombinasi `head_temperature` global + per-tipe. Setelah kalibrasi, **threshold aksi selalu pada `probabilities`, bukan `confidence`** (formula confidence Unsloth/Laya ≠ Jev).

---

## 4. Lapis C — `evaluate()` pada test terpisah

```python
test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model)
print(FastDecisionModel.evaluate(model, tokenizer, test_items))
```

Syarat test yang sah: **didekontaminasi** dari train (Unsloth: test 3.000 baris terpisah — 2.000 typed-decisions + 500 BANKING77 + 500 CLINC150). Untuk data sendiri: pisahkan berdasarkan waktu/sumber, bukan sekadar shuffle, agar tidak mengukur hafalan.

---

## 5. Benchmark eksternal: Decision Index & Workflow Evals

Untuk perbandingan lintas model (termasuk vs Jev/Laya/Clef), dua leaderboard menjadi acuan:

**Decision Index** ([clef-evals.workers-ai-mle.workers.dev](https://clef-evals.workers-ai-mle.workers.dev), suite 0.2.1, ±40 benchmark): mengukur akurasi keputusan per tugas (BANKING77 macro-F1, CLINC150+OOS, BFCL, ToolRet, ContractNLI…), **Brier score** (ForecastBench — lower better, satu-satunya metrik kalibrasi eksplisit di papan), dan **latensi** (median & p95 ms). Contoh snapshot: Clef 94.2 BANKING77 / 209.3ms median; Clef-flash 38.8ms; Jev 524.1ms; Laya 5.8ms (CPU encoder). Cara pakai: petakan tugasmu ke 2–3 benchmark terdekat (klasifikasi intent → BANKING77/CLINC150; routing tool → BFCL/ToolRet), lalu ukur modelmu pada slice yang sama sebelum–sesudah fine-tune.

**Workflow Evals** ([evals.typesafe.ai](https://evals.typesafe.ai/)): empat alur bisnis end-to-end (invoice processing, customer service, security incidents, agent trace observability) dengan **label referensi konsensus**, metrik *exact actions* / *primary action*. Snapshot: Clef unggul invoice exact actions 64.7 vs Jev 61.8; customer service imbang ~76–77. Cara pakai: tiru polanya — bangun 50–200 kasus dari alurmu sendiri dengan label konsensus (≥2 penilai + aturan resolusi), ukur exact-action match, bukan sekadar akurasi per pertanyaan.

---

## 6. Metrik operasional penentu: selective coverage

Untuk keputusan go/no-go otomatisasi, metrik penentu bukan akurasi mentah melainkan **coverage pada risiko selektif** (formalisme: [[07-evaluation-methodology-and-benchmarks|note 07]], §2.3):

$$\text{Coverage}_{@\text{Risk} \le 5\%} = \max\{\Phi(\tau) : \text{Risk}(\tau) \le 0.05\}$$

Prosedur: (1) kumpulkan probabilitas + benar/salah pada test; (2) sapu threshold $\tau$ atas probabilitas opsi menang; (3) laporkan coverage (% kasus di atas $\tau$) vs selective risk. Model terkalibrasi baik memberi coverage tinggi pada risk ≤5% — inilah angka yang diterjemahkan ke "berapa % tiket bisa auto-proses, sisanya eskalasi".

---

## 7. Resep run eval lengkap (salin-tempel alur)

```python
# 0. baseline: backbone + head acak → catat acc_0
# 1. train (trainer di [[00-finetune-process-and-algorithm|note pipeline]])
trainer.train()                                   # pantau holdout acc per epoch (Lapis A)
# 2. kalibrasi di holdout (Lapis B)
metrics = FastDecisionModel.calibrate(model, tokenizer, eval_items)
# 3. vonis di test terpisah (Lapis C)
test_items, _ = FastDecisionModel.build_dataset(test_split, tokenizer, model)
print(FastDecisionModel.evaluate(model, tokenizer, test_items))
# 4. arsipkan: delta acc, ECE before/after, loss, coverage@risk≤5% + keputusan lanjut/stop
```

Setiap run wajib mencatat **delta + keputusan** di md run-nya (baseline → train → eval harness identik → arsip `data/eval/`), sesuai aturan repo. Mode kegagalan umum: ECE turun tapi akurasi stagnan (temperatur menutupi head yang tak belajar — cek Lapis A); akurasi holdout tinggi tapi test jatuh (kebocoran kontaminasi — perketat dekontaminasi); confidence jauh di atas akurasi (kalibrasi dilewati — jangan serve sebelum `calibrate()`).
