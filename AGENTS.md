# AGENTS.md

Repository for research, experimentation, and publication of the Decision Models paper (`paper/`).

## Core Objective

Produce an empirical scientific paper on System One Decision Models and RLCD. Every theoretical and architectural claim in `paper/` requires experimental validation; unverified claims must not be written.

## Compute Routing

Heavy compute runs remotely in the cloud. Local workstation is reserved for paper drafting, LaTeX compilation, data processing, and light configuration.

| Workload | Platform | Tooling / Skill | Destination Artifact |
|---|---|---|---|
| Model training & fine-tuning | Kaggle / Colab | `kaggle-cli`, `colab-operator` | Hugging Face Hub / Kaggle Output |
| Distributed or serverless workloads | Modal | `modal` | Modal Volume / HF Hub |
| Weights & checkpoint storage | Hugging Face | `hf-cli` | HF Models / Datasets |
| Local analysis & paper drafting | Local workstation | `uv run python paper/build.py` | `paper/main.pdf`, `docs/RUN_LOG.md` |

### Cloud Execution Protocol

1. **Dispatch**: Package scripts from `code/` and submit to cloud runners via CLI (`kaggle kernels push`, `modal run`, Colab notebook automation).
2. **Monitor**: Stream or inspect job logs via the platform's CLI tool.
3. **Capture**: Extract metrics for training (loss, learning rate, tokens/sec), evaluation (task accuracy, holdout accuracy, ECE calibration), and inference (latency, TTFT, throughput).
4. **Log**: Record every run in `docs/RUN_LOG.md` with:
   - Run identifier, kernel/job link, and commit hash
   - Dataset mixture and hyperparameters
   - Baseline vs treatment metric deltas
   - Empirical takeaway and paper readiness
5. **Feed to Paper**: Sync verified numbers, tables, and figures from `docs/RUN_LOG.md` into `paper/sections/` and `paper/PAPER.md`, then recompile via `uv run python paper/build.py`.

## Quality Criteria

- **No baseline, no claim**: Every metric or comparative statement in the methodology and results sections must trace directly to an entry in `docs/RUN_LOG.md`.
- **Reproducible artifacts**: Every checkpoint cited in the paper must exist in a remote repository (Hugging Face / cloud bucket) with full training recipes in `code/`.

## Paper Drafting Pipeline

The paper source lives in `paper/` using modular IEEEtran LaTeX:
- **Build runner**: Run `uv run python paper/build.py` to execute the full compilation cycle (`pdflatex` $\to$ `bibtex` $\to$ `pdflatex` $\to$ `pdflatex`) with PDF lock detection, outputting `paper/main.pdf`.
- **Root document**: `paper/main.tex` coordinates package configurations, metadata, section inputs, and `references.bib`.
- **Modular sections**: Edit chapter content in `paper/sections/` (`abstract.tex`, `introduction.tex`, `related_works.tex`, `dataset.tex`, `methodology.tex`, `experiment.tex`, `results.tex`, `conclusion.tex`).
- **Bibliography**: Place references in `paper/references.bib`.
- **Markdown sync**: Use `paper/PAPER.md` as the fast markdown drafting buffer before translating final text/tables to LaTeX.

## Agent skills

### Issue tracker

Issues and specs are tracked in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical five-role triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (`GLOSSARY.md` and `docs/adr/` at the repo root). See `docs/agents/domain.md`.
