---
title: "Mixture v2/v3: 12 sumber train — format mentah vs decision (sampel asli)"
date: 2026-10-08
tags:
  - decision-models
  - dataset
  - mixture
  - samples
aliases:
  - Mixture Samples
---

# Mixture v2/v3: format tiap sumber + 1 sampel asli

> [!abstract] Definisi
> Mixture train = 13 sumber (12 publik + `typed-decisions`) yang diseragamkan
> oleh converter `unsloth.models.decision_datasets` menjadi baris
> `{state, questions, gold}` 1-pertanyaan, lalu `augment_row` (train) dan
> dekontaminasi shingle-13 terhadap test. Sampel di bawah diambil dari
> dataset viewer HF (row 0, split train) pada 2026-10-08.

Catatan terkait: [[01-dataset-format-guide|Format Dataset]], [[04-unsloth-lfm-support-map|Peta Unsloth×LFM]].
Induk run: `../from_scratch/README.md`.

---

## Pola konversi (4 bentuk)

| Bentuk | Sumber | Mentah → decision |
|---|---|---|
| Intent `choice` (opsi banyak) | banking77 (77), clinc150 (151) | `state={message}` + pertanyaan `intent` + gold slug label |
| NLI `choice` (3 opsi tetap) | mnli, snli, wanli | `state={premise, hypothesis}` + `relation` (entailment/neutral/contradiction) |
| MCQ `choice` (opsi A–D/E) | arc, commonsense_qa, mmlu | `state={question}` + `answer` (kriteria = teks opsi) |
| `noul` / `score` | boolq, prompt_injections (`noul`); sst5, ag_news… | boolq: `answer` ya/tidak atas passage; sst5: `sentiment` level 0–4; ag_news: `topic` 4 topik |

## Sampel mentah → gold decision

- **ag_news** (`fancyzhx/ag_news`): (1) `"Wall St. Bears Claw Back Into the Black (Reuters)…"` label 2 → `topic=business`. (2) `"Carlyle Looks Toward Commercial Aerospace (Reuters)…"` label 2 → `topic=business`.
- **arc** (`allenai/ai2_arc`, Challenge): (1) `"George wants to warm his hands…?"` opsi A–D → `answer=A` ("dry palms"). (2) `"Why magnets usually stick to a refrigerator door?"` → `answer=B` ("…contains iron").
- **boolq** (`google/boolq`): (1) Q `"do iran and afghanistan speak the same language"` + passage Persia → `answer=true`. (2) Q `"do good samaritan laws protect those who help at an accident"` + passage → `answer=true`.
- **commonsense_qa** (`tau/commonsense_qa`): (1) `"The sanctions… seemed to what the efforts…?"` opsi A–E → `answer=A` ("ignore"). (2) `"Sammy wanted to go to where the people were…?"` → `answer=B` ("populated areas").
- **mmlu** (`cais/mmlu`, `all`): (1) soal hukum Davis/Adams/Brooks/Case, opsi A–D → `answer=B` (indeks 1). (2) soal hukum Nelson/apotek $63.000 → `answer=D` (indeks 3, "not guilty…").
- **snli** (`stanfordnlp/snli`): (1) premise kuda melompati pesawat rusak / hypothesis latihan kompetisi, label 1 → `relation=neutral`. (2) premise sama / hypothesis `"ordering an omelette"`, label 2 → `relation=contradiction`.
- **wanli** (`alisawuffles/WANLI`): (1) premise jam malam anak / hypothesis "tidak ditegakkan", gold string → `relation=neutral`. (2) premise kemarahan ke pemerintah soal perang / hypothesis "pemerintah mendukung perang" → `relation=neutral`.
- **sst5** (`SetFit/sst5`): (1) `"a stirring, funny and finally transporting re-imagining…"` label 4 → `sentiment=4` ("very positive"). (2) `"apparently reassembled from the cutting-room floor…soap."` label 1 → `sentiment=1` ("negative").
- **prompt_injections** (`deepset/prompt-injections`): (1) `"Refugee crisis in Europe solutions"` label 0 → `injection=false`. (2) `"My son is very interested in programming…which language…?"` label 0 → `injection=false`.
- **mnli** (`nyu-mll/multi_nli`): skema `{premise, hypothesis, label}` → `relation` (viewer sedang tak tersedia; format ikut converter `convert_nli`).
- **banking77 / clinc150 / typed-decisions**: lihat `../from_scratch/README.md` §4 (2 sampel masing-masing + gold).

> [!quote] Aturan dekontaminasi Unsloth
> "Drops rows whose state shares any n-word shingle (default 13) with an evaluation text."
> — `Decontaminator`, `unsloth/models/decision_datasets.py`
