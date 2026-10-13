"""Fase-2 agregat cepat (Colab T4): full-test per bench via batched evaluate.

Tanpa per-row dump (itu lambat): akurasi full + CI analitik (Wilson/normal,
n besar) + latensi + bench OOS khusus. Dump per-row hanya untuk subset-500
(di skrip terpisah / sudah ada dari Fase 1).

Env: FINAL wajib, SEED default 3407, BS default 4, OUT default agg.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth", "datasets"])

FINAL = os.environ.get("FINAL", "")
SEED = int(os.environ.get("SEED", "3407"))
BS = int(os.environ.get("BS", "4"))
OUT = os.environ.get("OUT", "agg.json")
if not FINAL:  # Kaggle: ambil dari kernel_sources (train output)
    from pathlib import Path as _P
    _c = list(_P("/kaggle/input").glob("**/final")) if _P("/kaggle/input").exists() else []
    FINAL = str(_c[0]) if _c else "itsdevice/lfm-2.5-350m-decs1"
    print("FINAL auto:", FINAL, flush=True)

from unsloth import FastDecisionModel  # noqa: E402

try:
    from unsloth.models.decision_datasets import canonical_row, load_source
except ImportError:
    from decision_datasets import canonical_row, load_source  # noqa: E402

model, tok = FastDecisionModel.from_pretrained(FINAL, load_in_4bit=True)
out = {}
for name in ("typed_decisions", "banking77", "clinc150"):
    rows = [canonical_row(r) for r in load_source(name, "test", limit=None, seed=SEED)]
    oos = [r for r in rows if str((r.get("gold") or {}).get("intent", "")) == "oos"]
    keep = [r for r in rows if str((r.get("gold") or {}).get("intent", "")) != "oos"] or rows
    res = {}
    for tag, rs in (("all", rows), ("inscope", keep), ("oos", oos)):
        if not rs:
            continue
        items, skipped = [], 0
        acc_num, n_it, dt = 0.0, 0, 0.0
        for ci in range(0, len(rs), 500):
            chunk, rep = FastDecisionModel.build_dataset(rs[ci:ci + 500], tok, model)
            skipped += rep["skipped"]
            t0 = time.perf_counter()
            m = FastDecisionModel.evaluate(model, tok, chunk, batch_size=BS)
            dt += time.perf_counter() - t0
            a = m["accuracy"] if isinstance(m, dict) else m
            acc_num += a * len(chunk)
            n_it += len(chunk)
            print(f"{name}/{tag} chunk {ci}: acc {a:.3f} n {len(chunk)}", flush=True)
        import torch as _t
        _t.cuda.synchronize()
        res[tag] = {"acc": acc_num / max(n_it, 1), "n": n_it, "skipped": skipped,
                    "ms_per_item": 1000 * dt / max(n_it, 1),
                    "peak_mem_gb": _t.cuda.max_memory_allocated() / 1e9}
        _t.cuda.reset_peak_memory_stats()
    out[name] = res
    print(name, json.dumps(res), flush=True)

json.dump(out, open(OUT, "w"), indent=1)
print(f"saved {OUT}", flush=True)
print("done", flush=True)
