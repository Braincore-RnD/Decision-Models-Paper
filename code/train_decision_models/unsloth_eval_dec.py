"""Eval v7 before->after + paper-readiness (typed-decisions / BANKING77 / CLINC150).
Arsip: archive/eval_v6_3col.py.
v7 vs v6: SEED env; holdout dari artefak train (cal_items.pt/test_items.pt,
  fallback rekonstruksi typed-only + WARN incomparable); calibrate full dict;
  grid parafrase 3 varian (wording + urutan opsi); bootstrap CI; latensi T4;
  eval_results.json buat RUN_LOG.

Jalankan sebagai kernel Kaggle TERPISAH dengan kernel_sources = kernel training
sebelumnya (ambil out-u/final + .pt + run_meta.json), atau FINAL=<HF id/path>.

Resep disalin dari Unsloth: test = canonical rows (tanpa augmentasi) dari
  banking77 (legacy-datasets/banking77:test, 500),
  clinc150 (clinc/clinc_oos:plus:test, 500),
  typed_decisions (LocalLLaMA/typed-decisions:all:test, full).
BEFORE = backbone base + head baru (random, tanpa training) — setara angka
chance-level di tabel Unsloth. AFTER = model final latih.

Qwen rerun protokol identik: MODEL_ID=<base qwen> + FINAL=<qwen latih>.
Angka vendor Unsloth (73/74/76%) = konteks, bukan baseline.

Env:
  MODEL_ID  default LiquidAI/LFM2.5-350M (before)
  FINAL     default auto: /kaggle/input/**/final, lalu HF_REPO_DEC
  HF_REPO_DEC default itsdevice/lfm-2.5-350m-decs1
  EVAL_N    default 500 (cap banking77/clinc150; typed_decisions full)
  SEED      default 3407 (kunci sampling test; samakan dengan SEED train)
  TRAIN_OUT default "" (dir out-u train berisi cal/test .pt + run_meta.json)
  N_BOOT    default 100 (bootstrap CI akurasi AFTER; 0=skip. Biaya ≈ N_BOOT x eval)
  PARA      default 1 ("0" = skip grid parafrase)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
IN = Path("/kaggle/input")
MODEL_ID = os.environ.get("MODEL_ID", "LiquidAI/LFM2.5-350M")
FINAL = os.environ.get("FINAL", "")
HF_REPO_DEC = os.environ.get("HF_REPO_DEC", "itsdevice/lfm-2.5-350m-decs1")
EVAL_N = int(os.environ.get("EVAL_N", "500"))
SEED = int(os.environ.get("SEED", "3407"))
TRAIN_OUT = os.environ.get("TRAIN_OUT", "")
N_BOOT = int(os.environ.get("N_BOOT", "100"))
PARA = os.environ.get("PARA", "1") == "1"


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
_INTENT_ALTS = ["Which intent does the customer's message express?",
                "What is the customer's intent?",
                "Classify the customer message into one of the intents."]


def _variant(crow, variant, seed):
    """v7: varian wording + urutan opsi (robustness ala TypedBench, murah)."""
    if variant == 0:
        return crow
    import copy
    import random as _r

    crow = copy.deepcopy(crow)
    rng = _r.Random(seed + variant)
    try:
        for _q in crow.get("questions", {}).values():
            if not isinstance(_q, dict):
                continue
            _inst = str(_q.get("instructions", ""))
            if "intent" in _inst.lower():
                _q["instructions"] = _INTENT_ALTS[variant % len(_INTENT_ALTS)]
            _crit = _q.get("criteria")
            if isinstance(_crit, dict) and len(_crit) > 1:
                _keys = list(_crit.keys())
                rng.shuffle(_keys)
                _q["criteria"] = {k: _crit[k] for k in _keys}
    except Exception:
        pass
    return crow


def build_bench(tokenizer, model, name, limit, variant=0):
    rows = load_source(name, "test", limit=limit, seed=SEED)
    rows = [_variant(canonical_row(r), variant, SEED) for r in rows]
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    print(f"{name} v{variant}: {len(items)} items (skipped {report['skipped']})", flush=True)
    return items


def _acc(m):
    return m["accuracy"] if isinstance(m, dict) else m


def _timed_evaluate(model, tok, items):
    t0 = time.perf_counter()
    m = FastDecisionModel.evaluate(model, tok, items)
    dt = time.perf_counter() - t0
    return m, dt


def _boot_ci(items, model, tok, n_boot, seed):
    """v7: bootstrap CI 95% akurasi (percentile, tanpa numpy). AFTER saja (hemat)."""
    import random as _r

    if n_boot <= 0 or not items:
        return (float("nan"), float("nan"))
    rng = _r.Random(seed)
    n = len(items)
    accs = []
    for i in range(n_boot):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        accs.append(float(_acc(FastDecisionModel.evaluate(model, tok, sample))))
        if (i + 1) % 25 == 0:
            print(f"  boot {i + 1}/{n_boot}", flush=True)
    accs.sort()
    lo = accs[int(0.025 * (n_boot - 1))]
    hi = accs[int(0.975 * (n_boot - 1))]
    return (lo, hi)


def load_fresh():
    # max_seq_length HARUS >= model final (4096): kalau tidak, skema 77/150-opsi
    # melebihi budget dan build_dataset skip semua baris ("does not fit").
    # Ini bug eval v1/v2 (before nan), bukan data rusak.
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=4096, load_in_4bit=True)
    model = FastDecisionModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        use_gradient_checkpointing="unsloth", random_state=SEED)
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


def find_train_out():
    if TRAIN_OUT:
        return Path(TRAIN_OUT)
    if IN.exists():
        for f in ("cal_items.pt", "run_meta.json"):
            cands = list(IN.glob(f"**/{f}"))
            if cands:
                print(f"train_out dari kernel input: {cands[0].parent}", flush=True)
                return cands[0].parent
    return None


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
    before[name] = (float(_acc(m)), len(items))
    print(f"before {name}: {before[name]}", flush=True)

print("== AFTER (final latih) ==", flush=True)
model1, tok1 = FastDecisionModel.from_pretrained(find_final(), load_in_4bit=True)
after, latency, para, ci = {}, {}, {}, {}
for name, _ in BENCHES:
    items = bench_items[name]
    if not items:
        after[name] = (float("nan"), 0)
        latency[name] = float("nan")
        continue
    m, dt = _timed_evaluate(model1, tok1, items)
    after[name] = (float(_acc(m)), len(items))
    latency[name] = 1000 * dt / len(items)  # v7: ms/item terukur (ganti klaim sub-45ms)
    print(f"after {name}: {after[name]} {latency[name]:.1f}ms/item", flush=True)
    if PARA:  # v7: grid parafrase, lapor median+range (2 eval tambahan per bench)
        accs = [after[name][0]]
        for v in (1, 2):
            _lim = dict(BENCHES)[name]
            pv = build_bench(tok1, model1, name, _lim, v)
            accs.append(float(_acc(FastDecisionModel.evaluate(model1, tok1, pv))))
        accs.sort()
        para[name] = {"median": accs[1], "range": [accs[0], accs[2]]}
        print(f"para {name}: median {accs[1]:.1%} range [{accs[0]:.1%}, {accs[2]:.1%}]", flush=True)

print("== CI (bootstrap AFTER) ==", flush=True)
for name, _ in BENCHES:
    ci[name] = _boot_ci(bench_items[name], model1, tok1, N_BOOT, SEED + 7)
    print(f"ci {name}: [{ci[name][0]:.1%}, {ci[name][1]:.1%}]", flush=True)

# holdout + kalibrasi: dari artefak train bila ada (distribusi identik)
tout = find_train_out()
hold_label, hold_acc, cal_full = "fallback-typed-only", float("nan"), None
if tout is not None and (tout / "test_items.pt").exists():
    try:
        import torch as _torch

        _test = _torch.load(str(tout / "test_items.pt"), map_location="cpu", weights_only=False)
        _cal = _torch.load(str(tout / "cal_items.pt"), map_location="cpu", weights_only=False)
        hold_acc = float(_acc(FastDecisionModel.evaluate(model1, tok1, _test)))
        cal_full = FastDecisionModel.calibrate(model1, tok1, _cal)  # v7: dict penuh (ECE/loss)
        print({"holdout_internal": hold_acc, "n": len(_test)}, flush=True)
        print({"calibrate_full": cal_full, "n": len(_cal)}, flush=True)
        try:
            _meta = json.loads((tout / "run_meta.json").read_text())
            hold_label = (f"train-split seed={_meta.get('seed')} aug={_meta.get('aug_mode')} "
                          f"loto={_meta.get('holdout_taxonomy')} "
                          f"train/cal/test={_meta.get('n_train')}/{_meta.get('n_cal')}/{_meta.get('n_test')}")
        except Exception:
            hold_label = "train-split (meta tak terbaca)"
    except Exception as e:
        print(f"load artefak train gagal, fallback: {e}", flush=True)
if hold_acc != hold_acc:  # NaN = artefak tak ada -> rekonstruksi lama
    from datasets import load_dataset  # noqa: E402

    print("WARN holdout: rekonstruksi typed-only, JANGAN disandingkan dgn mixture holdout", flush=True)
    train = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    items, _ = FastDecisionModel.build_dataset(train, tok1, model1)
    _, hold = FastDecisionModel.split_holdout(items, seed=SEED)
    hold_acc = float(_acc(FastDecisionModel.evaluate(model1, tok1, hold)))

print("\n| Model | typed-decisions | BANKING77 | CLINC150 | Holdout acc |", flush=True)
b, a = before, after
row = (f"| LFM2.5-350M-decs1 | {b['typed_decisions'][0]:.0%} -> **{a['typed_decisions'][0]:.0%}** "
       f"[{ci['typed_decisions'][0]:.0%}, {ci['typed_decisions'][1]:.0%}] "
       f"| {b['banking77'][0]:.0%} -> **{a['banking77'][0]:.0%}** "
       f"[{ci['banking77'][0]:.0%}, {ci['banking77'][1]:.0%}] "
       f"| {b['clinc150'][0]:.0%} -> **{a['clinc150'][0]:.0%}** "
       f"[{ci['clinc150'][0]:.0%}, {ci['clinc150'][1]:.0%}] | {hold_acc:.0%} ({hold_label}) |")
print(row, flush=True)

results = {"model_id": MODEL_ID, "final": find_final(), "seed": SEED, "eval_n": EVAL_N,
           "before": {k: v[0] for k, v in before.items()},
           "after": {k: v[0] for k, v in after.items()},
           "ci95": {k: list(v) for k, v in ci.items()}, "paraphrase": para,
           "latency_ms_per_item": latency,
           "holdout": {"acc": hold_acc, "dist": hold_label}, "calibrate": cal_full}
(WORKING / "eval_results.json").write_text(json.dumps(results, default=str, indent=2))
print("saved eval_results.json — tempel ke docs/RUN_LOG.md", flush=True)
print("done", flush=True)
