"""Dump prediksi per-item -> items_pred.json (RQ3: Brier/NLL/AUROC/coverage dihitung lokal).

Dipakai di Colab GPU (Unsloth butuh akselerator). Per baris: predict(state, questions)
-> {gold per question, answer + probabilities}. Gagal render -> catat error, lanjut.

Env:
  FINAL    wajib (dir final Unsloth: adapters+head)
  BENCH    default all (koma: banking77,clinc150,typed_decisions)
  EVAL_N   default 500 (cap banking77/clinc150; typed full)
  SEED     default 3407 (sampling test SAMA antar model)
  OUT      default items_pred.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

FINAL = os.environ["FINAL"]
BENCH = os.environ.get("BENCH", "all")
EVAL_N = int(os.environ.get("EVAL_N", "500"))
SEED = int(os.environ.get("SEED", "3407"))
OUT = os.environ.get("OUT", "items_pred.json")


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


sh([sys.executable, "-m", "pip", "install", "-q", "unsloth", "datasets"])

from unsloth import FastDecisionModel  # noqa: E402

try:
    from unsloth.models.decision_datasets import canonical_row, load_source
except ImportError:
    from decision_datasets import canonical_row, load_source  # noqa: E402

model, tok = FastDecisionModel.from_pretrained(FINAL, load_in_4bit=True)

benches = [b.strip() for b in BENCH.split(",")] if BENCH != "all" else ["typed_decisions", "banking77", "clinc150"]
limits = {"typed_decisions": None, "banking77": EVAL_N, "clinc150": EVAL_N}
dump, n_err = [], 0
for name in benches:
    for i, r in enumerate(load_source(name, "test", limit=limits[name], seed=SEED)):
        crow = canonical_row(r)
        try:
            ans = FastDecisionModel.predict(model, tok, crow["state"], crow["questions"])
            dump.append({"bench": name, "gold": crow.get("gold"), "answers": ans})
        except Exception as e:  # render gagal (skema kepanjangan dsb): catat, lanjut
            n_err += 1
            dump.append({"bench": name, "gold": crow.get("gold"), "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 50 == 0:
            print(f"{name}: {i + 1} rows, err {n_err}", flush=True)

with open(OUT, "w") as f:
    json.dump(dump, f, default=str)
print(f"saved {OUT}: {len(dump)} rows, err {n_err}", flush=True)
print("done", flush=True)
