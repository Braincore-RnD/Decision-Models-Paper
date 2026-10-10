---
title: "Paper outline: What Makes Small Decision Models Transfer?"
date: 2026-10-10
tags:
  - paper
  - decision-models
  - system-one
  - outline
  - related-work
aliases:
  - Paper Outline
  - decs1 paper
---

# Paper outline: *What Makes Small Decision Models Transfer?*

> [!abstract] Working title
> *What Makes Small Decision Models Transfer? Schema Augmentation and Mixture
> Composition for 350M System One Decision Models.* Empirical/systems paper:
> no new architecture (head = Cloudflare's Clef design) — the claims are about
> **training recipe**: schema augmentation drives zero-shot transfer to unseen
> taxonomies (26%→83% BANKING77), mixture composition trades off under fixed
> 350M capacity, and a hybrid non-transformer backbone (LFM2.5-350M) works as a
> decision backbone. Artifact: [itsdevice/lfm-2.5-350m-decs1](https://huggingface.co/itsdevice/lfm-2.5-350m-decs1).

Catatan terkait: [[00-index-rlcd-and-system-one|Hub RLCD]], [[01-rlcd-theory-and-formalism|Teori RLCD]], [[07-evaluation-methodology-and-benchmarks|Metodologi Evaluasi]], `train_decision_models/`, `from_scratch/README.md`.

---

## 1. Claims (what we can already defend with numbers)

| # | Claim | Evidence (ours) |
|---|---|---|
| C1 | Schema augmentation (option rename/paraphrase/derived questions) drives transfer to unseen taxonomies | v5→v6, identical recipe ± augment: BANKING77 36%→**83%**, CLINC150 33%→**73%** (status: cross-split, pending LOTO #4–5 + token-matched A vs E #6–7, tracker di RECOMMENDATION §6.4) |
| C2 | Under fixed 350M capacity, mixture composition is zero-sum across distributions | v2→v3→v4→v5 seesaw (intent share vs typed); §6 from_scratch |
| C3 | Hybrid conv-attention backbone (LFM2.5-350M) works as decision backbone | Wiring + training logs; first outside Qwen/Llama/Gemma family |
| C4 | Untrained-head baselines sit at chance; fine-tuning lifts all columns | before rows 31%/2%/17% (matches Unsloth's 36%/7%/19% pattern) |
| C5 | Soft/consensus labels are the missing calibration ingredient | Only typed-decisions has them; ECE 0.052–0.134 range; Open-Jev-type data proposed |

---

## 2. Related work (all primary-source verified, Oct 2026)

### 2.1 The artifact lineage (what we build on — cite as technical reports/blogs, not papers)

- TypeSafe Jev launch (blog, 15 Sep 2026): API shape `POST /v1/systemone`
  (`choice` ≤255 opts / `score` 2–10 levels / `noul`), 70–500 ms, $0.042/MTok in.
  RLCD = objective name only: *"answers with epistemically honest probabilities
  on System One tasks"* — **no paper, no architecture, no weights, closed API**.
  https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Cloudflare Clef (blog, 1 Oct 2026): frozen Qwen3.8-27B + rank-256 LoRA +
  label-smoothed CE + Brier + RLCD secondary (partial ordinal credit + reference
  penalty); open weights Apache-2.0; `joint_schema_model.py` + `joint_head.safetensors`
  (256 MB) + head config `{hidden 5120, width 1024, routing 2, layers 4, heads 16, ff 4096}`.
  https://blog.cloudflare.com/clef-decision-models · https://huggingface.co/Cloudflare/clef
- Unsloth decision guide: FastDecisionModel recipe (LoRA r=16, lr 2e-4, head LR 1e-4,
  2 epochs), mixture of 12 sources + typed-decisions, decontaminated 3.000-row test,
  table Qwen3.5-0.8B 36→73 / 7→74 / 19→76.
  https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth
- LFM2.5-350M: 350M, 16 layers (10 double-gated conv + 6 GQA), ctx 32K, vocab 65.536,
  28T tokens + RL, license `lfm1.0`. https://huggingface.co/LiquidAI/LFM2.5-350M
- Laya (Convai, Apache-2.0): ModernBERT-large 421M + 2-layer head, RLCD reward =
  strictly proper rules (log + spherical, RPS ordinal), GRPO-style REINFORCE.
  https://huggingface.co/convaiinnovations/laya
- Datasets: `LocalLLaMA/typed-decisions` (Apache-2.0, 4×400, soft consensus labels);
  `ZefanCai/Open-Jev` (CC0-1.0, 12 configs, native soft targets, train/cal/val/test/ood splits).

### 2.2 Zero-shot System One literature (position against: they don't train)

- **TypedBench** (Sharma et al., arXiv:2610.11392) — MUST-CITE. Calibration *relative
  to finite-sample noise floor*, paraphrase median+range, selective prediction, ordinal
  proper scoring, asymmetric-cost utility. Borrow its protocol for our eval upgrade.
  https://arxiv.org/abs/2610.11392
- **System Switch** (Bailo, arXiv:2610.09683) — MUST-CITE for calibration-vs-accuracy:
  0.15B–9B zero-shot actors, AUROC (sensitivity) distinct from accuracy/ECE, deferral
  gain ∝ AUROC (ρ=0.87). https://arxiv.org/abs/2610.09683
- **Jev benchmarking** (Li et al., arXiv:2610.11978) — Jev vs 19 LLMs: wins knowledge,
  loses math (−17.7 MathQA). Existence proof small-decision > small-generative.
  https://arxiv.org/abs/2610.11978
- **Moderation benchmark** (Mazzoni & Failla, arXiv:2610.07953) — Jev vs Laya,
  confidence Brier/ECE/AUC + selective-review coverage. https://arxiv.org/abs/2610.07953
- **Distillation** (Sadoun et al., arXiv:2610.09188) — Jev-labels → compact evaluator,
  multi-teacher +9.6 Elo. Precedent for training small from System-One labels.
  https://arxiv.org/abs/2610.09188
- **BOTTLED** (Sonthalia et al., arXiv:2610.08775) — SmolLM2-360M finetune baseline
  under matched token budgets; the small-model bar to beat. https://arxiv.org/abs/2610.08775
- MetaEncoder (2610.11316), JevKT (2610.11135), S1-MAS (2610.08155): one-line cites.

### 2.3 Foundations (calibration + selective + small-model evidence)

- Guo et al. 2017 (ICML): ECE/MCE, temperature scaling. https://arxiv.org/abs/1706.04599
- Brier 1950; Murphy 1973 decomposition ($BS=REL-RES+UNC$); Gneiting & Raftery 2007
  (strictly proper rules: Brier/log/spherical). DOI:10.1198/016214506000001437
- Geifman & El-Yaniv 2017 (NeurIPS, SGR) & 2019 (ICML, SelectiveNet):
  coverage $\Phi(g)$, selective risk $R(f,g)$. https://arxiv.org/abs/1705.08500
- Kamath et al. 2020 (ACL, selective QA under shift); Xin et al. 2021 (ACL, abstention).
- Desai & Durrett 2020 (EMNLP, transformer calibration, label smoothing OOD);
  Tian et al. 2023 (EMNLP, verbalized confidence halves ECE); Zhu et al. 2023
  (Findings EMNLP, instruction tuning hurts calibration, LoRA mitigates).
- Hinton et al. 2015 (distillation); DistilBERT 2019; Hsieh et al. 2023 Findings ACL
  (220–770M T5 beats 540B PaLM on classification via rationales).
- Pezeshkpour & Hruschka 2024 (Findings NAACL, option-order sensitivity 13–85%);
  Zong et al. 2023 (arXiv:2310.01651, permutation fragility) — analogy citations
  for why schema augmentation should matter (no direct prior: SAY SO explicitly).

### 2.4 Mandatory disambiguation (paste into paper)

> *We use RLCD to mean Reinforcement Learning **for** Calibrated Decisions
> (TypeSafe/Cloudflare: RL with a proper-scoring-rule reward). Unrelated to
> Yang et al. (2024), Reinforcement Learning **from** Contrastive Distillation
> (arXiv:2307.12950, ICLR 2024): human-free $(p^+,p^-)$→$(o^+,o^-)$ preference
> generation for alignment, no calibration objective. Shared acronym only.*

---

## 3. Method → our runs (mapping)

- Backbone+head: Clef-design head on LFM2.5-350M, LoRA r=16, head LR 1e-4
  (`train_decision_models/00-*`, `code/unsloth_kaggle_dec.py` + `archive/`).
- Data: v1 typed-only → v2/v3 13-source mixture (equal/typed-2×, augmented) →
  v4/v5/v6 30k-row + soft sources (Open-Jev, LeWiDi, GoEmotions), canonical vs
  augmented ablation (`07-mixture-compositions.md`, `05-mixture-samples.md`).
- Eval: Unsloth 3-column protocol + holdout (`03-eval-*.md`); v7: holdout dari
  artefak train (cal/test .pt, bukan rekonstruksi), grid parafrase 3 varian,
  bootstrap CI, latensi terukur, `eval_results.json`. Sisa upgrade path:
  TypedBench noise-floor ECE, selective/coverage, cost.

## 4. Threats & gaps (must close before submit)

1. **Own baselines**: rerun Qwen3.5-0.8B in identical protocol (no cited-numbers comparison). Status: eval v7 swap MODEL_ID+FINAL, run #8 belum.
2. **Seeds**: all runs seed-3407 single; need ≥3 + intervals. Status: train v7 SEED + eval v7 N_BOOT, run #1–3 belum.
3. **Augment ablations**: rename vs paraphrase vs derived, separately. Status: A vs E token-matched dulu (#6–7 belum); B/C/D label-only, batas lib.
4. **Broader eval**: Decision Index slices (BFCL/ToolRet) + latency + TypedBench grid. Status: latency + para grid jalan di eval v7; BFCL/ToolRet = tahap dua.
5. **Vendor-claim hygiene**: mark Unsloth/TypeSafe numbers as vendor-reported;
   open inconsistencies found (Unsloth r64/1ep vs r16/2ep; Clef 3.8 vs 3.5 code).
6. **License check**: `lfm1.0` full text unread; Open-Jev third-party notices apply.

## 5. Venue strategy

arXiv tech report first (speed: 9 papers/3 days in this field), then workshop
(efficient-ML / small-model venues) or Findings ACL/EMNLP. Top-tier conference
needs a novel method or much wider eval — not current scope.
