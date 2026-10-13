"""Fase-2 full-test dump (Colab T4): canonical FULL + para-subset -> per-row answers.

Satu model per run (setup unsloth + load sekali). Semua metrik (acc/CI/ECE)
dihitung OFFLINE dari dump via score_preds.py --boot (nol GPU tambahan).

Env:
  FINAL    wajib (dir final Unsloth: adapters+head)
  SEED     default 3407 (sampling subset SAMA antar model)
  PARA_N   default 500 (cap subset parafrase; varian 0/1/2)
  OUT      default items_full.json (per-row dump) + summary_full.json
"""
from __future__ import annotations

import copy
import json
import os
import random as _r
import sys
import time

FINAL = os.environ["FINAL"]
SEED = int(os.environ.get("SEED", "3407"))
PARA_N = int(os.environ.get("PARA_N", "500"))
OUT = os.environ.get("OUT", "items_full.json")
BENCH = [b.strip() for b in os.environ.get("BENCH", "all").split(",")]

from unsloth import FastDecisionModel  # noqa: E402

try:
    from unsloth.models.decision_datasets import canonical_row, load_source
except ImportError:
    from decision_datasets import canonical_row, load_source  # noqa: E402

_INTENT_ALTS = ["Which intent does the customer's message express?",
                "What is the customer's intent?",
                "Classify the customer message into one of the intents."]


def _variant(crow, variant, seed):
    if variant == 0:
        return crow
    crow = copy.deepcopy(crow)
    rng = _r.Random(seed + variant)
    for _q in crow.get("questions", {}).values():
        if not isinstance(_q, dict):
            continue
        if "intent" in str(_q.get("instructions", "")).lower():
            _q["instructions"] = _INTENT_ALTS[variant % len(_INTENT_ALTS)]
        _crit = _q.get("criteria")
        if isinstance(_crit, dict) and len(_crit) > 1:
            _keys = list(_crit.keys())
            rng.shuffle(_keys)
            _q["criteria"] = {k: _crit[k] for k in _keys}
    return crow


model, tok = FastDecisionModel.from_pretrained(FINAL, load_in_4bit=True)


def save():
    with open(OUT, "w") as f:
        json.dump(dump, f, default=str)
    with open(OUT.replace("items_", "summary_"), "w") as f:
        json.dump(summary, f, indent=1)


dump, summary = [], {}
if os.environ.get("RESUME", "0") == "1" and os.path.exists(OUT):
    dump = json.load(open(OUT))
    print(f"resume: {len(dump)} rows loaded", flush=True)
done = {}
for r in dump:
    done[(r.get("bench"), r.get("variant", 0))] = done.get((r.get("bench"), r.get("variant", 0)), 0) + 1
CHUNK = 500  # build per chunk: full 5.5k x 4k ctx OOM kalau sekaligus
WANT = [("typed_decisions", True), ("banking77", True), ("clinc150", True)]
if BENCH != ["all"]:
    WANT = [w for w in WANT if w[0] in BENCH]
for name, full in WANT:
    rows = load_source(name, "test", limit=None, seed=SEED)
    t0, n_ok, n = time.perf_counter(), 0, 0
    skip = done.get((name, 0), 0)
    for ci in range(0, len(rows), CHUNK):
        for r in rows[ci:ci + CHUNK]:
            if skip > 0:
                skip -= 1
                continue
            crow = canonical_row(r)
            try:
                ans = FastDecisionModel.predict(model, tok, crow["state"], crow["questions"])
                dump.append({"bench": name, "variant": 0, "gold": crow.get("gold"), "answers": ans})
                n += 1
            except Exception as e:
                dump.append({"bench": name, "variant": 0, "gold": crow.get("gold"),
                             "error": f"{type(e).__name__}: {e}"})
        print(f"{name} chunk {ci}-{ci + CHUNK}: {n} rows", flush=True)
        save()  # tahan banting: tiap chunk tersimpan, VM mati pun bisa RESUME=1
    dt = time.perf_counter() - t0
    summary[name] = {"n_rows": n, "ms_per_row": 1000 * dt / max(n, 1)}
    print(f"{name} full: {n} rows, {summary[name]['ms_per_row']:.0f}ms/row", flush=True)
    sub = rows[:PARA_N]
    for v in (1, 2):
        t0, n2 = time.perf_counter(), 0
        skip = done.get((name, v), 0)
        for r in sub:
            if skip > 0:
                skip -= 1
                continue
            crow = _variant(canonical_row(r), v, SEED)
            try:
                ans = FastDecisionModel.predict(model, tok, crow["state"], crow["questions"])
                dump.append({"bench": name, "variant": v, "gold": crow.get("gold"), "answers": ans})
                n2 += 1
            except Exception as e:
                dump.append({"bench": name, "variant": v, "gold": crow.get("gold"),
                             "error": f"{type(e).__name__}: {e}"})
        print(f"{name} para v{v}: {n2} rows, {1000*(time.perf_counter()-t0)/max(n2,1):.0f}ms/row", flush=True)
        save()

save()
print(f"saved {OUT}: {len(dump)} rows", flush=True)
print("done", flush=True)
