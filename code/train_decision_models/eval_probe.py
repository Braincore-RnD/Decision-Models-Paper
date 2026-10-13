"""Fase-2 probe (Colab T4): label-shuffle + latency-vs-options. Model E/LOTO.

- Shuffle: kriteria intent diacak nama<->deskripsi (derangement, seed fix) pada
  subset 500 banking77/clinc150 -> acc drop = uji semantik sungguhan (#23).
- Latency: kriteria dipotong ke k opsi (8/24/77/150) -> ms/row + peak mem (#21).

Env: FINAL wajib, SEED default 3407, N default 500, OUT default probe.json.
"""
from __future__ import annotations

import copy
import json
import os
import random as _r
import time

FINAL = os.environ["FINAL"]
SEED = int(os.environ.get("SEED", "3407"))
N = int(os.environ.get("N", "500"))
OUT = os.environ.get("OUT", "probe.json")
BENCH = [b.strip() for b in os.environ.get("BENCH", "all").split(",")]
MODE = os.environ.get("MODE", "all")  # all|canon|shuffle|topk (job <20 mnt)
KS = [int(x) for x in os.environ.get("KS", "").split(",") if x.strip()] or None

import torch  # noqa: E402

from unsloth import FastDecisionModel  # noqa: E402

try:
    from unsloth.models.decision_datasets import canonical_row, load_source
except ImportError:
    from decision_datasets import canonical_row, load_source  # noqa: E402

model, tok = FastDecisionModel.from_pretrained(FINAL, load_in_4bit=True)
out = {}


def run(rows, tag):
    ok, n, t0 = 0, 0, time.perf_counter()
    for r in rows:
        try:
            ans = FastDecisionModel.predict(model, tok, r["state"], r["questions"])
            for qid, g in (r.get("gold") or {}).items():
                a = (ans or {}).get(qid, {})
                probs = (a.get("probabilities") or {}) if isinstance(a, dict) else {}
                gl = g.get("label", g.get("answer")) if isinstance(g, dict) else g
                if probs and gl is not None and max(probs, key=probs.__getitem__) == str(gl):
                    ok += 1
                n += 1
        except Exception:
            pass
    torch.cuda.synchronize()
    return {"acc": ok / max(n, 1), "n": n, "ms_per_row": 1000 * (time.perf_counter() - t0) / max(len(rows), 1),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}


WANT = ["banking77", "clinc150"] if BENCH == ["all"] else BENCH
for name in WANT:
    rows = [canonical_row(r) for r in load_source(name, "test", limit=N, seed=SEED)]
    out[name] = {}
    if MODE in ("all", "canon"):
        out[name]["canonical"] = run(rows, "canon")
    rng = _r.Random(SEED)
    shuf = []
    for crow in rows:
        c = copy.deepcopy(crow)
        for _q in c.get("questions", {}).values():
            if not isinstance(_q, dict):
                continue
            crit = _q.get("criteria")
            if isinstance(crit, dict) and len(crit) > 1:
                names = list(crit.keys())
                perm = names[:]
                while all(a == b for a, b in zip(perm, names)):
                    rng.shuffle(perm)
                _q["criteria"] = {p: crit[o] for p, o in zip(perm, names)}
        shuf.append(c)
    if MODE in ("all", "shuffle"):
        out[name]["label_shuffled"] = run(shuf, "shuf")
    if MODE in ("all", "topk"):
        _ks = KS or ((8, 24, 77) if name == "banking77" else (8, 24, 77, 150))
        for k in _ks:
            trunc = []
            for crow in rows:
                c = copy.deepcopy(crow)
                for _q in c.get("questions", {}).values():
                    if not isinstance(_q, dict):
                        continue
                    crit = _q.get("criteria")
                    if isinstance(crit, dict) and len(crit) > k:
                        _q["criteria"] = dict(list(crit.items())[:k])
                trunc.append(c)
            torch.cuda.reset_peak_memory_stats()
            out[name][f"top{k}"] = run(trunc, f"k{k}")
    print(name, json.dumps(out[name])[:200], flush=True)
    json.dump(out, open(OUT, "w"), indent=1)  # simpan per bench (tahan mati VM)

json.dump(out, open(OUT, "w"), indent=1)
print(f"saved {OUT}", flush=True)
print("done", flush=True)
