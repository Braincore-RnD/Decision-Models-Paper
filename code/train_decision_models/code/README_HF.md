---
language: en
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-350M
tags:
  - decision-model
  - system-one
  - classification
  - calibrated-probabilities
  - lfm2.5
  - unsloth
base_model: LiquidAI/LFM2.5-350M
---

# lfm-2.5-350m-decs1

A **System One decision model**: instead of generating text, it reads a state and scores
every option of every typed question in a single forward pass, returning calibrated
probabilities your software can act on. Compatible with the Jev / SystemOne API shape
(`POST /v1/systemone`).

- **Backbone:** [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) (350M, hybrid conv-attention, 4-bit QLoRA + fresh Clef-style decision head)
- **Trained with:** [Unsloth `FastDecisionModel`](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) (`DecisionTrainer`, LoRA r=16, lr 2e-4, cosine, 3 epochs) on Kaggle 2×T4
- **Data:** 12,000-row mixture of 13 sources (`ag_news`, `arc`, `banking77`, `boolq`, `clinc150`, `commonsense_qa`, `mmlu`, `mnli`, `prompt_injections`, `snli`, `sst5`, `wanli`, `typed-decisions`), decontaminated against the test sets

## Benchmarks

Accuracy before → after fine-tuning (same protocol as the [Unsloth decision guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth); test: 2,000 typed-decisions + 500 BANKING77 + 500 CLINC150):

| Model | typed-decisions | BANKING77 | CLINC150 | Holdout acc |
|---|---|---|---|---|
| lfm-2.5-350m-decs1 (this) | 31% → **57%** | 2% → **68%** | 17% → **68%** | 72% |
| Unsloth Qwen3.5-0.8B (reference) | 36% → **73%** | 7% → **74%** | 19% → **76%** | 78% |

Calibration on holdout: ECE 0.083. Act on `probabilities`, not `confidence` (see below).

## Use

### A. Unsloth (recommended)

```python
from unsloth import FastDecisionModel

model, tokenizer = FastDecisionModel.from_pretrained(
    "itsdevice/lfm-2.5-350m-decs1", load_in_4bit=True)
FastDecisionModel.for_inference(model)
answers = FastDecisionModel.predict(
    model, tokenizer,
    "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
    {"team": {"type": "choice",
              "instructions": "Which team should handle this?",
              "criteria": {"billing": "invoices, payments, refunds",
                           "technical": "bugs, outages, errors",
                           "sales": "pricing, new plans"}},
     "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}},
)
print(answers["team"]["answer"], answers["team"]["probabilities"])
```

### B. Decision API (Unsloth Studio / any Jev-compatible server)

```bash
curl http://localhost:8888/v1/systemone \
  -H "Authorization: Bearer sk-unsloth-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "laya",
       "state": "Hi, I was charged twice for invoice #4411. Please refund.",
       "questions": {
         "team": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"billing": "invoices, refunds",
                               "technical": "bugs, errors"}},
         "refund": {"type": "noul", "instructions": "Ask refund?"}}}'
```

Response shape per question type: `choice` → `{choice, confidence, probabilities}`;
`score` → `{score, confidence, legend, probabilities}`; `noul` → `{noul: P(true)}`.
`usage.output_tokens` is always 0 — nothing is generated.

> Threshold on `probabilities` (e.g. act only if P ≥ 0.8), not on `confidence`:
> confidence rescales differ between Jev and Laya/Clef-style heads.

## Files

| File | Purpose |
|---|---|
| `model.safetensors`, `config.json` | Backbone (merged 16-bit) |
| `joint_head.safetensors`, `joint_head_config.json` | Decision head (scores every option jointly) |
| `unsloth_decision_config.json` | Calibrated temperatures + eval config |
| `tokenizer.*`, `chat_template.jinja` | LFM2.5 tokenizer |

Limits: up to 64 questions per request, 255 options per `choice`, 10 levels per
`score`. Training context 2048 tokens (serving reads up to 16384); long states
are cut from the end, schemas never silently truncated.

## Limitations

- 350M-class model: strong on intent/tool routing, not suited for knowledge-heavy,
  math, or code decisions (same guidance as the base model card).
- English-centric evaluation; multilingual quality follows the LFM2.5 backbone.
- Probabilities are calibrated on the training mixture — re-calibrate
  (`FastDecisionModel.calibrate`) after further fine-tuning.

## License & citation

Weights under [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE)
(inherited from the base model). If you use this model, please cite the base:

```bibtex
@article{liquidai2025lfm2,
  title={LFM2 Technical Report},
  author={Liquid AI},
  year={2025}
}
```
