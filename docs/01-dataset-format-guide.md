---
title: "Format Dataset Ideal untuk Decision Model (state, questions, gold)"
date: 2026-10-08
tags:
  - decision-models
  - dataset
  - unsloth
  - typed-decisions
  - calibration
aliases:
  - Decision Dataset Format
  - typed-decisions Guide
---

# Format Dataset Ideal untuk Decision Model

> [!abstract] Definisi
> Satu contoh latih decision model = **tiga bagian**: `state` (situasi: string atau JSON apa pun), `questions` (skema keputusan bertipe: `choice`/`score`/`noul`), dan **gold answers** (label benar, idealnya plus distribusi konsensus lunak). Kualitas dataset ditentukan oleh **deskripsi kriteria yang tak ambigu**, **label konsensus** (bukan satu anotator), dan **test set yang didekontaminasi** dari data latih.

Catatan terkait: [[00-finetune-process-and-algorithm|Pipeline & Algoritma]], [[00-index-rlcd-and-system-one|Hub RLCD]].

---

## 1. Anatomi satu baris (format train code)

```python
state = "Hi, I was charged twice for invoice #4411. Please refund the duplicate today."

questions = {
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
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["not urgent", "soon", "today"]},  # level 0..L-1
}
# gold: choice → key opsi ("billing"); noul → True/False; score → nomor level
```

Spesifikasi tiga tipe:

| Tipe | `criteria` | Gold | Jawaban inference |
|------|-----------|------|-------------------|
| `choice` | mapping `option_id → deskripsi`, $K \in [2,255]$ | key opsi benar | `choice` + `probabilities` per opsi + `confidence` |
| `score` | list deskripsi terurut, $L \in [2,10]$ | indeks level | `score` $= \mathbb{E}[v(L)] = \sum_j p_j v(l_j)$ + `legend` + `probabilities` |
| `noul` | opsional (default true/false generik) | boolean | `noul` $= P(\text{True})\in[0,1]$ |

Aturan render yang memengaruhi penulisan data: opsi `choice` **diurut alfabetis** saat render (jangan mengandalkan urutan tulis); `instructions` boleh kosong (fallback = ID pertanyaan); deskripsi tiap opsi berbagi **satu budget token** — lewat ~20 opsi berdeskripsi, label mulai terpotong.

---

## 2. Referensi emas: `LocalLLaMA/typed-decisions`

> [!quote] Dataset referensi Unsloth
> "Training uses the same dataset format as Laya and Clef… For typed-decisions, set Subset to `all`."
> — [Train your own Decision Model with Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)

Dataset: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0, ±1.6k baris, subset `all`; 4 workflow + `agent_trace_observability`, tiap subset 400 baris: train 300 / test 100).

Kolom per baris (dari Dataset Viewer):

| Kolom | Isi | Contoh |
|-------|-----|--------|
| `id`, `workflow`, `split` | identitas (`tr_agent_trace_observability_000000`, …), nama workflow, train/test | — |
| `state` | **JSON** situasi (agen, constraint, task, trace_summary: steps, duration, violations…) | `{"agent": {"autonomy": "checkpointed", …}, "task": "Delete personal data…", …}` |
| `questions` | **JSON string** skema 5 pertanyaan (`action` choice, `needs_review` noul, `outcome` choice, `risk` score, `urgency` score) | `{"action": {"criteria": {"continue": …, "stop": …}, "instructions": …}, …}` |
| `gold` | **JSON string** label + distribusi per pertanyaan | `{"action": {"label": "human_review", "probabilities": {…}, "confidence": 0.24, "type": "choice"}, …}` |
| `factors` | ringkasan faktor penentu (auditabilitas label) | `{"autonomy": …, "violated_constraint": true, …}` |
| `n_questions` | selalu 5 di dataset ini | `5` |
| `label_agreement` | statistik konsensus anotator (`argmax_agree`, `argmax_majority`, `total_variation`) | — |
| `<qid>__label/__confidence/__probabilities` (+ `__score`, `__probability_true`) | kolom datar per pertanyaan: label keras, confidence, **distribusi lunak** | `outcome__probabilities = {"failure": 0.08, "partial": 0.49, "success": 0.42, …}` |

Poin penting: gold **bukan satu label**, melainkan **distribusi konsensus** (rata-rata banyak anotator/vote) + label argmax. Inilah yang memungkinkan (a) loss lunak saat train, dan (b) kalibrasi yang jujur — model belajar bahwa kasus ambigu memang punya probabilitas tersebar, bukan dipaksa 100% pada satu jawaban. `total_variation` / `argmax_agree=false` menandai baris-baris ambigu tersebut.

---

## 3. Campuran data untuk generalisasi (resep benchmark Unsloth)

Model acuan Unsloth tidak hanya latih di typed-decisions, melainkan **12 sumber + typed-decisions**: `ag_news, arc, banking77, boolq, clinc150, commonsense_qa, mmlu, mnli, prompt_injections, snli, sst5, wanli`. Test set **3.000 baris** (2.000 typed-decisions + 500 BANKING77 + 500 CLINC150) dan **didekontaminasi** terhadap data latih. Hasilnya: Qwen3.5-0.8B naik 36%→73% (typed-decisions), 7%→74% (BANKING77), 19%→76% (CLINC150) — bukti head belajar pola keputusan umum, bukan menghafal workflow.

---

## 4. Checklist dataset ideal (untuk data sendiri)

1. **Sempit dulu.** Terutama untuk backbone ≤1B (mis. LFM-2.5-350M): satu domain, 3–8 pertanyaan stabil, kriteria dideskripsikan operasional ("invoices, payments, refunds" bukan "urusan uang").
2. **State kaya bukti.** Sertakan field yang benar-benar menentukan jawaban (angka, status, constraint) — head hanya bisa me-routing bukti yang ada di state; state yang dipotong dari ekor saat kepanjangan, jadi taruh fakta kunci di awal.
3. **Gold ganda.** Simpan label keras (untuk akurasi) + distribusi lunak/konsensus (untuk kalibrasi). Minimal 2–3 vote per baris ambigu; catat `total_variation`- ala typed-decisions untuk menandai ambiguitas.
4. **Dekontaminasi.** Test set terpisah baris dan (ideal) terpisah periode/sumber dari train; Unsloth memisahkan holdout via `split_holdout(fraction=0.1, max_items=400)` untuk kalibrasi — siapkan **satu split test lagi** khusus evaluasi akhir.
5. **Validasi otomatis.** Manfaatkan `validate` di `build_dataset` + perhatikan `report['skipped']`: kriteria kosong pada choice/score, tipe tak dikenal, dan skema melebihi batas token adalah penyebab skip/error umum.
6. **Volume.** Titik acuan: ±1.6k baris typed-decisions cukup untuk 0.8–4B (2 epoch); data lebih kecil → naikkan ke 3–4 epoch dan awasi overfitting via holdout.

```python
from datasets import load_dataset
dataset = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions")
train_items, eval_items = FastDecisionModel.split_holdout(items, seed=3407)
```
