"""Eval before->after ala tabel Unsloth (typed-decisions / BANKING77 / CLINC150 + holdout).

Jalankan sebagai kernel Kaggle TERPISAH dengan kernel_sources = kernel training
sebelumnya (ambil out-u/final), atau FINAL=<HF id/path>.

Resep disalin dari Unsloth: test = canonical rows (tanpa augmentasi) dari
  banking77 (legacy-datasets/banking77:test, 500),
  clinc150 (clinc/clinc_oos:plus:test, 500),
  typed_decisions (LocalLLaMA/typed-decisions:all:test, full).
BEFORE = backbone base + head baru (random, tanpa training) — setara angka
chance-level di tabel Unsloth. AFTER = model final latih.
Holdout = split_holdout(seed=3407) yang sama lalu evaluate final.

Catatan jujur: model ini latih HANYA di typed-decisions (bukan campuran 12
sumber ala Unsloth), jadi BANKING77/CLINC150 mengukur transfer zero-shot —
ekspektasi di bawah angka Unsloth yang in-distribution.

Env:
  MODEL_ID  default LiquidAI/LFM2.5-350M (before)
  FINAL     default auto: /kaggle/input/**/final, lalu HF_REPO_DEC
  HF_REPO_DEC default itsdevice/lfm-2.5-350m-decs1
  EVAL_N    default 500 (cap banking77/clinc150; typed_decisions full)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

IN = Path("/kaggle/input")
MODEL_ID = os.environ.get("MODEL_ID", "LiquidAI/LFM2.5-350M")
FINAL = os.environ.get("FINAL", "")
HF_REPO_DEC = os.environ.get("HF_REPO_DEC", "itsdevice/lfm-2.5-350m-decs1")
EVAL_N = int(os.environ.get("EVAL_N", "500"))


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


sh([sys.executable, "-m", "pip", "install", "-q", "unsloth", "datasets"])

from unsloth import FastDecisionModel  # noqa: E402

try:
    from unsloth.models.decision_datasets import canonical_row, load_source
except ImportError:  # fallback: unsloth versi lama tanpa modul ini
    from decision_datasets import canonical_row, load_source  # noqa: E402

BENCHES = [("typed_decisions", None), ("banking77", EVAL_N), ("clinc150", EVAL_N)]


def build_bench(tokenizer, model, name, limit):
    rows = load_source(name, "test", limit=limit, seed=3407)
    rows = [canonical_row(r) for r in rows]
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    print(f"{name}: {len(items)} items (skipped {report['skipped']})", flush=True)
    return items


def load_fresh():
    # max_seq_length HARUS >= model final (4096): kalau tidak, skema 77/150-opsi
    # melebihi budget dan build_dataset skip semua baris ("does not fit").
    # Ini bug eval v1/v2 (before nan), bukan data rusak.
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=4096, load_in_4bit=True)
    model = FastDecisionModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        use_gradient_checkpointing="unsloth", random_state=3407)
    return model, tokenizer


def find_final():
    if FINAL:
        return FINAL
    cands = list(IN.glob("**/final")) if IN.exists() else []
    if cands:
        print(f"final dari kernel input: {cands[0]}", flush=True)
        return str(cands[0])
    print(f"final dari Hub: {HF_REPO_DEC}", flush=True)
    return HF_REPO_DEC


print("== BUILD (sekali, dipakai before+after) ==", flush=True)
model0, tok0 = load_fresh()
bench_items = {}
for name, limit in BENCHES:
    bench_items[name] = build_bench(tok0, model0, name, limit)

print("== BEFORE (base + head random) ==", flush=True)
before = {}
for name, _ in BENCHES:
    items = bench_items[name]
    if not items:
        before[name] = (float("nan"), 0)
        continue
    m = FastDecisionModel.evaluate(model0, tok0, items)
    before[name] = (m["accuracy"] if isinstance(m, dict) else m, len(items))
    print(f"before {name}: {before[name]}", flush=True)

print("== AFTER (final latih) ==", flush=True)
model1, tok1 = FastDecisionModel.from_pretrained(find_final(), load_in_4bit=True)
after = {}
for name, _ in BENCHES:
    items = bench_items[name]
    if not items:
        after[name] = (float("nan"), 0)
        continue
    m = FastDecisionModel.evaluate(model1, tok1, items)
    after[name] = (m["accuracy"] if isinstance(m, dict) else m, len(items))
    print(f"after {name}: {after[name]}", flush=True)

# holdout: reproduksi split yang sama (seed sama + data sama = split sama)
from datasets import load_dataset  # noqa: E402

train = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
items, _ = FastDecisionModel.build_dataset(train, tok1, model1)
_, hold = FastDecisionModel.split_holdout(items, seed=3407)
hm = FastDecisionModel.evaluate(model1, tok1, hold)
hacc = hm["accuracy"] if isinstance(hm, dict) else hm

print("\n| Model | typed-decisions | BANKING77 | CLINC150 | Holdout acc |", flush=True)
b, a = before, after
row = (f"| LFM2.5-350M-decs1 | {b['typed_decisions'][0]:.0%} -> **{a['typed_decisions'][0]:.0%}** "
       f"| {b['banking77'][0]:.0%} -> **{a['banking77'][0]:.0%}** "
       f"| {b['clinc150'][0]:.0%} -> **{a['clinc150'][0]:.0%}** | {hacc:.0%} |")
print(row, flush=True)
print("done", flush=True)
