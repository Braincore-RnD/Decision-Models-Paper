"""Parse kernel stdout logs -> loss_series.json (train loss per logging step + eval_loss per epoch).

Sumber: `kaggle kernels output` (file *.log), disimpan di /tmp/k7-keep/<slug>/.
Keluaran (commit ke repo): docs/eval/kaggle/<run>/loss_series.json + run_meta.json.

Jalan: python3 code/train_decision_models/parse_loss.py
"""
import glob
import json
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KEEP = "/tmp/k7-keep"
LEGACY = "/tmp/k7-seed3407"
OUTROOT = os.path.join(BASE, "docs/eval/kaggle")

NAMES = {"seed3407": "seed3407", "seed41-full": "seed41", "seed42-full": "seed42",
         "loto-banking-full": "loto-banking", "loto-clinc-full": "loto-clinc",
         "ablasi-none-full": "ablasi-A", "ablasi-full-full": "ablasi-E"}


def parse(logf):
    txt = open(logf, encoding="utf-8", errors="replace").read()
    train = [(float(m.group(2)), float(m.group(1)))
             for m in re.finditer(r"'loss': '([0-9.]+)'.*?epoch': '([0-9.]+)", txt)]
    ev = [(float(m.group(2)), float(m.group(1)))
          for m in re.finditer(r"'eval_loss': '([0-9.]+)'.*?epoch': '([0-9.]+)", txt)]
    return {"train": train, "eval": ev}


jobs = []
for f in glob.glob(os.path.join(KEEP, "*", "*.log")):
    jobs.append((NAMES.get(os.path.basename(os.path.dirname(f)), os.path.basename(os.path.dirname(f))), f,
                 os.path.join(KEEP, os.path.basename(os.path.dirname(f)), "run_meta.json")))
for f in glob.glob(os.path.join(LEGACY, "*.log")):
    jobs.append(("seed3407", f, os.path.join(LEGACY, "out-u/run_meta.json")))

for name, logf, metaf in jobs:
    outd = os.path.join(OUTROOT, name)
    os.makedirs(outd, exist_ok=True)
    s = parse(logf)
    json.dump(s, open(os.path.join(outd, "loss_series.json"), "w"))
    if os.path.exists(metaf):
        json.dump(json.load(open(metaf)), open(os.path.join(outd, "run_meta.json"), "w"), indent=1)
    print(name, "train_steps:", len(s["train"]), "eval:", s["eval"])
