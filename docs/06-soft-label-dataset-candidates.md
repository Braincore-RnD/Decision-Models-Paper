---
title: "Soft-Label Dataset Candidates for the Decision-Model Mixture"
date: 2026-10-08
tags:
  - decision-models
  - dataset
  - soft-labels
  - mixture
  - calibration
aliases:
  - Soft Label Candidates
---

# Soft-Label Dataset Candidates (primary-source verified)

> [!abstract] Goal
> The current mixture is 13 hard-labeled sources + `typed-decisions` as the only
> soft-labeled one. This note lists additional **soft/consensus-labeled** HF datasets
> convertible to `{state, questions, gold-with-probabilities}` rows. A candidate counts
> **only** if its real row schema was seen via the Hub dataset viewer API.
> Format reference: [[01-dataset-format-guide|Format Dataset]]; current mixture samples: [[05-mixture-sources-samples|Mixture Samples]].

Verification method: `GET https://datasets-server.huggingface.co/splits?dataset=<id>` then
`GET .../first-rows?dataset=<id>&config=<cfg>&split=<split>`, all fetched 2026-10-08.
Repo facts re-verified (not copied): `typed-decisions`/`all`/`train` row 0 exposes exactly
`id, workflow, split, state, questions, gold, factors, label_agreement, n_questions`
(matching the [[01-dataset-format-guide|guide]] table), and the mixture is 12 public sources
(`ag_news, arc, banking77, boolq, clinc150, commonsense_qa, mmlu, mnli, prompt_injections, snli, sst5, wanli`)
+ `typed-decisions` = 13 (matching [[05-mixture-sources-samples|mixture samples]]).

Soft gold = a distribution $p = (p_1, \dots, p_K)$, $\sum_k p_k = 1$, from $N$ annotator votes
$p_k = n_k / N$, usable directly as `probabilities` in `gold` and for soft-loss training
$L = -\sum_k p_k \log q_k$.

## Ranked recommendation (top 3)

| Rank | Candidate | Why |
|------|-----------|-----|
| 1 | `earino/chaosnli` (ChaosNLI) | 100 annotators/NLI pair $\to$ true `label_dist` over entailment/neutral/contradiction; drops straight into the existing `convert_nli` pattern as `choice` with probabilities. Paper: [Nie et al. 2020](https://arxiv.org/abs/2004.14339). |
| 2 | `tasksource/lewidi` (LeWiDi-2025, 7 tasks) | Native `soft_label` probability vectors on every row (verified on `hs_brexit`, `conv_abuse`, `csc`); several tasks map to `choice`/`score` with a thin converter. Overview: [LeWiDi shared task](https://codalab.lisn.upsaclay.fr/competitions/6146). |
| 3 | `google-research-datasets/go_emotions` (`raw` config) | Per-rater binary rows (`rater_id` + 28 emotion columns) aggregatable to per-emotion score distributions; Apache-2.0, large (~58k comments). Paper: [Demszky et al. 2020](https://arxiv.org/abs/2005.00547). Needs a new aggregation converter. |

## Included candidates (verified schemas)

### 1. `earino/chaosnli` — INCLUDE (top pick, NLI `choice`)
- Page: <https://huggingface.co/datasets/earino/chaosnli> · Paper: <https://arxiv.org/abs/2004.14339>
- Configs/splits (viewer): `snli`/`mnli_m`/`alphanli` × `validation`. Size: 4,645 rows total (1,514 SNLI + 1,599 MNLI-m + 1,532 $\alpha$NLI per the paper, 100 annotations each) — viewer `/size` confirms 4,645.
- License: `cc-by-nc-4.0` (non-commercial — check policy before training).
- Real row (`snli`/`validation`, trimmed) via <https://datasets-server.huggingface.co/first-rows?dataset=earino/chaosnli&config=snli&split=validation>:
  `uid="2407214681.jpg#0r1n"`, `label_counter={e:30, n:70, c:null}`,
  `label_dist=[0.3, 0.7, 0.0]`, `entropy=0.88`, `majority_label="n"`,
  `old_labels=["neutral","neutral","neutral","entailment","entailment"]`.
- Maps to: `choice` (relation entailment/neutral/contradiction), `gold.probabilities = label_dist`.
- Converter: fits existing `convert_nli` pattern; new option to emit soft probabilities.
- Verdict: **include** — the cleanest drop-in soft NLI source.

### 2. `tasksource/lewidi` — INCLUDE (native soft labels, `choice`/`score`)
- Page: <https://huggingface.co/datasets/tasksource/lewidi> · 7 configs verified via splits API:
  `armis, conv_abuse, csc, hs_brexit, md_agreement, mp, varierrnli` × train/validation/test. Total 43,168 rows (viewer `/size`).
- License: `other` (check per-task terms before training).
- Real rows (trimmed):
  - `hs_brexit`/`train` (<https://datasets-server.huggingface.co/first-rows?dataset=tasksource/lewidi&config=hs_brexit&split=train>): `annotations=6`, `soft_label=[1.0, 0.0]` on a Brexit hate-speech item.
  - `conv_abuse`/`train`: `annotations=3`, `soft_label=[1.0, 0.0]` (dialogue abuse).
  - `csc`/`train`: `annotations=6`, `soft_label=[0.0, 0.33, 0.33, 0.0, 0.17, 0.17]` (sarcasm, 6-way distribution — genuinely soft).
- Maps to: `hs_brexit`/`conv_abuse`/`armis` $\to$ `noul`/`choice` (abuse yes/no with $P(\text{true})=$ `soft_label[0]`); `csc`/`mp` $\to$ `choice` or `score` over the soft vector.
- Converter: thin new converter reading `soft_label` directly into `probabilities`.
- Verdict: **include** (`csc`, `mp`, `varierrnli`, `hs_brexit` first; skip configs whose soft vectors turn out degenerate).

### 3. `google-research-datasets/go_emotions` — INCLUDE (aggregate to `score` distributions)
- Page: <https://huggingface.co/datasets/google-research-datasets/go_emotions> · Paper: <https://arxiv.org/abs/2005.00547>
- Configs: `simplified` (train/validation/test) + `raw` (train). License: `apache-2.0`. ~58k Reddit comments (viewer `/size` total 265,488 = per-rating rows in `raw`).
- Real row (`raw`/`train`, trimmed) via <https://datasets-server.huggingface.co/first-rows?dataset=google-research-datasets/go_emotions&config=raw&split=train>:
  `text="That game hurt."`, `rater_id=1`, 28 binary columns (`sadness=1`, rest `0`).
  `simplified`/`train` row 0: `text="My favourite food is..."`, `labels=[27]`.
- Maps to: per-emotion `score` questions — aggregate rater rows per comment into
  $p_e = n_e / N$ per emotion $e$ (multi-`score`, one level = intensity/fraction).
- Converter: **new** aggregation converter (group by `id`, average rater columns); `simplified` alone is hard-label, use `raw`.
- Verdict: **include** via `raw` aggregation — biggest permissively-licensed soft source.

### 4. `mteb/summeval` — MAYBE (expert `score` distributions, tiny)
- Page: <https://huggingface.co/datasets/mteb/summeval> · Config `default` × `test`, 100 rows (viewer `/size`). License: `mit`.
- Real row (trimmed) via <https://datasets-server.huggingface.co/first-rows?dataset=mteb/summeval&config=default&split=test>:
  `relevance=[1.67, 1.67, 2.33, 3.33]`, `coherence=[1.33, 3.0, 1.0, 2.67]`,
  `fluency=[1.0, 4.67, 4.33, 4.33]`, `consistency=[1.0, 2.33, 4.67, 5.0]`
  (expert means on 1–5 Likert per summary).
- Maps to: four `score` questions (relevance/coherence/fluency/consistency); normalize each 1–5 mean to a level distribution or keep mean as soft target.
- Converter: new `convert_score_ratings` (mean $\to$ neighboring-level split, e.g. $3.33 \to \{3: 0.67, 4: 0.33\}$).
- Verdict: **maybe** — clean semantics but only 100 rows; use as eval/calibration slice, not train bulk.

### 5. `RicardoRei/wmt-mqm-human-evaluation` — MAYBE (mean MQM `score`, annotator counts)
- Page: <https://huggingface.co/datasets/RicardoRei/wmt-mqm-human-evaluation> · Config `default` × `train`, 150,347 rows. License: `apache-2.0`.
- Real row (trimmed) via <https://datasets-server.huggingface.co/first-rows?dataset=RicardoRei/wmt-mqm-human-evaluation&config=default&split=train>:
  `lp="en-de"`, `score=-2.333333`, `annotators=3`, `system="eTranslation.737"`, `year=2020`
  (score = mean over annotators; negative = error-weighted MQM).
- Maps to: translation-quality `score` question; rows sharing one `mt` from different annotators could rebuild a distribution, otherwise mean-as-soft-target.
- Converter: new; needs grouping by segment to recover spread.
- Verdict: **maybe** — large and permissive, but soft signal is a mean, not a vote vector; verify multi-annotator grouping before including.

### 6. `McGill-NLP/FaithDial` — MAYBE (weak soft: multi-label BEGIN/VRM)
- Page: <https://huggingface.co/datasets/McGill-NLP/FaithDial> · Config `plain_text`, 32,269 rows total. License: `mit`. Paper: [FaithDial](https://arxiv.org/abs/2112.07366).
- Real rows: train row 0 `BEGIN=["Hallucination"]`, `VRM=["Disclosure", "Ack."]`;
  validation row 0 `BEGIN=["Hallucination", "Entailment"]` (two annotators disagreeing = soft signal) via <https://datasets-server.huggingface.co/first-rows?dataset=McGill-NLP/FaithDial&config=plain_text&split=validation>.
- Maps to: faithfulness `choice` (BEGIN taxonomy) with vote fractions where annotators disagree.
- Converter: new, small.
- Verdict: **maybe** — disagreement exists but is sparse/single-taxonomy; include only if a quick count shows enough multi-label rows.

## Rejected (verified, with reason)

| Candidate | Real schema seen | Reason |
|-----------|------------------|--------|
| `stanfordnlp/snli` (`plain_text`) | `premise, hypothesis, label` only (row 0 verified) | **reject as soft** — this HF mirror dropped the 5-`annotator_labels`; soft votes live only in the original release, not on the Hub. |
| `nyu-mll/multi_nli` (`default`) | `promptID, pairID, premise, ..., genre, label` (no annotator columns) | **reject as soft** — same as SNLI: no vote fields in the Hub copy. |
| `facebook/anli` (`plain_text`) | `uid, premise, hypothesis, label, reason` (dev_r1 row 0 verified) | **reject as soft** — single label + free-text reason, no vote distribution. |
| `kiddothe2b/contract-nli` (`contractnli_a`) | `premise, hypothesis, label` (row 0 verified) | **reject as soft** — hard 3-way mirror; no annotator metadata. |
| `flowaicom/RAGTruth_test` (`qa`) | `id, response, score, task_type, ...` with `score=1` (row 0 verified) | **reject as soft** — single binary hallucination label, no rater spread. |
| `wandb/RAGTruth-processed` | `..., quality, hallucination_labels, hallucination_labels_processed={evident_conflict:0, baseless_info:0}` (row 0 verified) | **reject as soft** — structure exists but labels are single-source hard flags. |
| `RuyuanWan/Dynasent_Disagreement` | `text, binary_disagreement, disagreement_rate=0.667` (row 0 verified) | **reject as soft labels** — agreement *metadata*, not a label distribution; no license tag. (`dynabench/dynasent` itself is viewer-disabled, `aps/dynasent` is empty — both unverifiable.) |
| `google/jigsaw_unintended_bias`, `google/jigsaw_toxicity_pred` | viewer error: "runs arbitrary python code" (both verified) | **unverified, do not include** — original `target` is a rater fraction, but no Hub copy with viewable soft columns was found (`james-burton/...100K` mirror shows binarized `target=0/1` + null identity columns). |
| `slpkl/soft_labels_dataset` et al. | search hit only, schema not inspected | **unverified, do not include** — unknown provenance, 0 likes; model-generated "soft labels" need provenance checks. |

## Suggested converters

| Existing pattern reused | New converter needed |
|-------------------------|----------------------|
| `convert_nli` $\to$ ChaosNLI (`label_dist` $\to$ `probabilities`) | `convert_soft_vector` (LeWiDi `soft_label` $\to$ `probabilities`) |
| — | `convert_rater_table` (GoEmotions `raw` group-by-`id` $\to$ per-emotion $p_e$) |
| — | `convert_score_ratings` (SummEval/WMT means $\to$ `score` soft targets) |

> [!warning] Biggest caveat
> ChaosNLI's pairs are sampled from SNLI/MNLI **validation** sets already adjacent to the hard mixture —
> including it without the shingle-13 decontamination in [[05-mixture-sources-samples|mixture samples]]
> risks train/test leakage, and its `cc-by-nc-4.0` license is the most restrictive of the three picks.

## Addendum 2026-10-08: `ZefanCai/Open-Jev` — verdict: STRONG INCLUDE

> First-party source: [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev)
> (87 likes, `cc0-1.0` = public domain, tags: `open-jev, typed-decisions, probability-estimation`).

Ini bukan sekadar kandidat — ini **adik kandung typed-decisions dalam skala industri**:
sintetik, format decision native (`choice`/`score`/`noul`), target berupa **distribusi
lunak** (teramati: `[1,0,0]`, `[0,0.5,0.5]`, bahkan `[0.33,0.33,0.33]` uniform),
dan — yang tidak dimiliki dataset lain — **split bawaan
`train/calibration/validation/test/ood`**.

| Fakta | Nilai |
|---|---|
| Subset | 12 (`release-v2-redistributable` 114k, `mailroom-control-v1` 115k, `browser-drone-expansion-v1` 161k, + 9 control task 4k–20k) |
| Kolom | `id, group_id, split, source, kind, question, options (list), target (list), state_json, metadata_json, record_json` |
| Split contoh | `customer-control-v1`: train 79.1k / calibration 4.67k / validation 3.72k / test 10.4k / ood 15.7k |
| Lisensi | `cc0-1.0` — tanpa restriksi (lebih bebas dari ChaosNLI `nc`) |
| Converter | trivial: `options` (list, deskripsi inline `"account: Login, …"`) → `criteria` dict; `target` (list) → `probabilities`; `kind` → `type`; `state_json` → `state` |

Contoh baris nyata (`customer-control-v1:42:2:bug_severity`, `score`):
instruksi rubrik + kriteria `["Cosmetic…", "Broken…workaround exists", "Blocking…"]`
+ target `[0, 0.5, 0.5]` — ambiguitas terekam eksplisit sebagai probabilitas,
tepat bahan kalibrasi yang dicari §6 `../from_scratch/README.md`.

Rekomendasi integrasi (v4, sudah diimplementasikan di `code/unsloth_kaggle_dec.py`):
`release-v2` bobot 3 + `customer-control-v1` bobot 1 (split `train` saja);
split bawaan `test`/`ood` dicadangkan untuk eval.

Keputusan converter v4 (terverifikasi primer 2026-10-08):
- LeWiDi dipakai 2 config Inggris: `conv_abuse` → `noul` (urutan `[non_abusive, abusive]`
  dideduksi dari baris unanimous benign → indeks 0; card menulis "in labels order"),
  `csc` → `score` 6 level ordinal (aman by construction). `armis` (Arab) dan
  config lain di-skip; `varierrnli` format non-distribusi, ditunda.
- GoEmotions `raw`: group-by `id` → fraksi 28 emosi; baris all-nol/`very_unclear` di-drop.
- Semua gold dibungkus `{label, probabilities}` ala typed-decisions (terbukti
  dimakan `build_dataset` dengan skipped 0).
