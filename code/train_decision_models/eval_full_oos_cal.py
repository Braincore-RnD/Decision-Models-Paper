"""Eval #13 (full-test + OOS) + #19 (cal-1k dump) untuk E-seed3407.

Env: FINAL (repo HF), OUT13, OUT19.
OOS: clinc_oos/plus test, label 'oos' -> metrik abstain (prediksi harus ESCALATE
atau distribusi flat; diukur: abstain-rate + ECE in-scope).
Cal-1k: 1000 train rows (seed lain) -> dump probs buat fit TS offline.
"""
from __future__ import annotations

import json
import os

FINAL = os.environ["FINAL"]
OUT13 = os.environ.get("OUT13", "full13.json")
OUT19 = os.environ.get("OUT19", "cal1k.json")

import subprocess
import sys

sh = lambda c: subprocess.check_call(c)  # noqa: E731
sh([sys.executable, "-m", "pip", "install", "-q", "unsloth", "datasets"])

from datasets import load_dataset  # noqa: E402

from unsloth import FastDecisionModel  # noqa: E402

try:
    from unsloth.models.decision_datasets import canonical_row, load_source  # noqa: E402
except ImportError:
    from decision_datasets import canonical_row, load_source  # noqa: E402

model, tok = FastDecisionModel.from_pretrained(FINAL, load_in_4bit=True)

dump13, n_err = [], 0

# #13a: full BANK (3080) + full CLINC in-scope (4500)
for name, lim in (("banking77", None), ("clinc150", None)):
    for i, r in enumerate(load_source(name, "test", limit=lim, seed=3407)):
        crow = canonical_row(r)
        try:
            ans = FastDecisionModel.predict(model, tok, crow["state"], crow["questions"])
            dump13.append({"bench": name, "gold": crow.get("gold"), "answers": ans})
        except Exception as e:
            n_err += 1
            dump13.append({"bench": name, "gold": crow.get("gold"), "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 500 == 0:
            print(f"{name}: {i + 1} rows, err {n_err}", flush=True)

# #13b: OOS slice (1000 dari clinc_oos/plus test) — gold='oos', ukur abstain
oos = load_dataset("clinc/clinc_oos", "plus", split="test")
names = oos.features["intent"].names
n_oos, n_abs = 0, 0
for i in range(min(1000, len(oos))):
    r = oos[i]
    if names[r["intent"]] != "oos":
        continue
    n_oos += 1
    crow = canonical_row({"text": r["text"], "label": 0})  # render via banking skema? fallback:
    try:
        # pakai skema clinc penuh: OOS diukur dari max-prob (abstain bila < tau)
        from unsloth.models.decision_datasets import load_source as _ls  # noqa
        base = canonical_row(next(iter(_ls("clinc150", "test", limit=1, seed=0))))
        base["state"] = {"text": r["text"]}
        ans = FastDecisionModel.predict(model, tok, base["state"], base["questions"])
        probs = list(ans.values())[0].get("probabilities", {}) if isinstance(list(ans.values())[0], dict) else {}
        mx = max(probs.values()) if probs else 0.0
        dump13.append({"bench": "clinc_oos", "gold": "oos", "max_prob": mx,
                       "abstain_tau05": mx < 0.5, "answers": ans})
        n_abs += mx < 0.5
    except Exception as e:
        dump13.append({"bench": "clinc_oos", "gold": "oos", "error": f"{type(e).__name__}: {e}"})
print(f"oos: {n_oos} rows, abstain@0.5 = {n_abs}/{max(n_oos,1)}", flush=True)

json.dump(dump13, open(OUT13, "w"), default=str)
print(f"saved {OUT13}: {len(dump13)} rows, err {n_err}", flush=True)

# #19: cal-1k dump (1000 train rows, seed 99) — probs buat TS offline
dump19 = []
for i, r in enumerate(load_source("banking77", "train", limit=500, seed=99)):
    crow = canonical_row(r)
    try:
        ans = FastDecisionModel.predict(model, tok, crow["state"], crow["questions"])
        dump19.append({"bench": "cal_bank", "gold": crow.get("gold"), "answers": ans})
    except Exception as e:
        dump19.append({"bench": "cal_bank", "gold": crow.get("gold"), "error": f"{type(e).__name__}: {e}"})
for i, r in enumerate(load_source("clinc150", "train", limit=500, seed=99)):
    crow = canonical_row(r)
    try:
        ans = FastDecisionModel.predict(model, tok, crow["state"], crow["questions"])
        dump19.append({"bench": "cal_clinc", "gold": crow.get("gold"), "answers": ans})
    except Exception as e:
        dump19.append({"bench": "cal_clinc", "gold": crow.get("gold"), "error": f"{type(e).__name__}: {e}"})
    if (i + 1) % 250 == 0:
        print(f"cal: {len(dump19)} rows", flush=True)
json.dump(dump19, open(OUT19, "w"), default=str)
print(f"saved {OUT19}: {len(dump19)} rows", flush=True)
print("done", flush=True)
