"""lfm-2.5-350m-decs1 v7 (living) — 1 script Kaggle, format tutorial Unsloth + ckpt/resume.
Arsip: archive/train_v6_rebalance-aug-3ep.py (+ v1..v5).
v7 = resep v6 + 5 perbaikan paper-readiness (docs/RECOMMENDATION.md §6):
  1. MAX_SEQ_LEN default 4096 (samakan train+eval; bug before-nan v1).
  2. SEED env (multi-seed 3407/41/42); seed dekontaminasi dikunci 3407.
  3. AUG_MODE A-E (none/rename/paraphrase/derived/full) + HOLDOUT_TAXONOMY LOTO
     (none/banking77/clinc150) + MAX_DECISIONS (kontrol token-matched).
  4. Split 3-way train/cal/test; calibrate di cal, vonis di test.
  5. Early stopping (patience 2, best eval_loss) + run_meta.json buat RUN_LOG.
Batasan jujur: lib augmentasi bool -> rename/paraphrase/derived dicatat sbg full;
split pra-augment penuh butuh dukungan lib (dekon 13-gram tetap jalan).

Mengikuti https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth
(from_pretrained -> get_peft_model -> build_dataset -> split_holdout ->
DecisionTrainer -> calibrate -> save), ditambah:
  - MODE otomatis: fresh bila tak ada ckpt, resume bila ada (atau FORCE_NEW=1).
  - Checkpoint Trainer tiap epoch (save_total_limit=1) + resume_from_checkpoint.
  - Simpan adapters+head, merged 16-bit, push HF, backup dataset Kaggle.

Jalankan sebagai Kaggle kernel script (GPU T4 x2 disarankan; T4 x1 bisa).
Env:
  MODEL_ID   default LiquidAI/LFM2.5-350M
  DATA       default LocalLLaMA/typed-decisions | SUBSET default all
  SEED       default 3407 | MAX_SEQ_LEN default 4096
  EPOCHS     default 3 | MAX_STEPS default 0 (0=pakai epoch; 60=smoke ala Unsloth)
  EARLY_STOP default 1 (0=off, wajib saat smoke MAX_STEPS>0)
  AUG_MODE   default full (none|rename|paraphrase|derived|full; kompat AUGMENT=0/1)
  HOLDOUT_TAXONOMY default none (none|banking77|clinc150 = LOTO)
  MAX_DECISIONS default 0 (0=off; >0 subsample rows)
  LEWIDI_FLIP default 0 ("1" = balik urutan soft_label conv_abuse)
  FORCE_NEW  default 0 ("1" = abaikan ckpt, mulai fresh)
  HF_REPO    default itsdevice/lfm-2.5-350m-decs1 (push bila HF_TOKEN ada)
  CKPT_DS    default amalanfadilgaib/lfm-2-5-350m-decs1-ckpt (backup via KAGGLE_API_TOKEN)

Contoh run minimal paper (7 kernel inti + 1 Qwen, tracker di docs/RECOMMENDATION.md §6.4):
v6-resep x3 seed (SEED=3407/41/42), LOTO x2 (HOLDOUT_TAXONOMY=banking77/clinc150),
ablasi A vs E token-matched (AUG_MODE=none/full + MAX_DECISIONS sama),
Qwen rerun terpisah (MODEL_ID ganti, bukan vN).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
OUT = WORKING / "out-u"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("MODEL_ID", "LiquidAI/LFM2.5-350M")
DATA = os.environ.get("DATA", "LocalLLaMA/typed-decisions")
SUBSET = os.environ.get("SUBSET", "all")
SEED = int(os.environ.get("SEED", "3407"))  # v7: multi-seed (3407/41/42); semua rng turun dari sini
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))  # v7: samakan train+eval (bug before-nan v1)
EPOCHS = float(os.environ.get("EPOCHS", "3"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
MIXTURE = os.environ.get("MIXTURE", "1") == "1"  # "0" = typed-decisions saja (v1)
N_ROWS = int(os.environ.get("N_ROWS", "20400"))  # v5: standar (intent w3 di dlm)
# v7: AUG_MODE A-E (none/rename/paraphrase/derived/full). Lib hanya dukung bool,
# jadi rename/paraphrase/derived jalan sbg full + dicatat di meta. Kompat AUGMENT lama.
_AUG_MODE_ENV = os.environ.get("AUG_MODE", "")
if _AUG_MODE_ENV:
    AUG_MODE = _AUG_MODE_ENV
else:
    AUG_MODE = "full" if os.environ.get("AUGMENT", "1") == "1" else "none"
AUGMENT = AUG_MODE != "none"
_AUG_NOTE = "" if AUG_MODE in ("none", "full") else "label-only: lib bool, jalan sbg full"
HOLDOUT_TAXONOMY = os.environ.get("HOLDOUT_TAXONOMY", "none")  # v7 LOTO: none|banking77|clinc150
MAX_DECISIONS = int(os.environ.get("MAX_DECISIONS", "0"))  # v7: 0=off; >0 subsample rows (kontrol token-matched)
BATCH = int(os.environ.get("BATCH", "8"))  # v7: turunkan bila OOM skema kanonis full (AUG_MODE=none)
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))  # v7: jaga BATCH*GRAD_ACCUM=32 (efektif tetap)
LEWIDI_FLIP = os.environ.get("LEWIDI_FLIP", "0") == "1"  # v7: 1=balik urutan soft_label conv_abuse
MIX_SOURCES = ["ag_news", "arc", "banking77", "boolq", "clinc150", "commonsense_qa",
               "mmlu", "mnli", "prompt_injections", "snli", "sst5", "wanli"]
if HOLDOUT_TAXONOMY in MIX_SOURCES:  # v7 LOTO: taksonomi target keluar total dari train
    MIX_SOURCES = [s for s in MIX_SOURCES if s != HOLDOUT_TAXONOMY]
# v5: banking/clinc w3 (pulihkan porsi intent >=30%); typed normal.
MIX_WEIGHTS = {n: 1.0 for n in MIX_SOURCES} | {"typed_decisions": 1.0}
if "banking77" in MIX_SOURCES:
    MIX_WEIGHTS["banking77"] = 3.0
if "clinc150" in MIX_SOURCES:
    MIX_WEIGHTS["clinc150"] = 3.0
# v5 customs: (subset, n, seed). mailroom = triase email, sepupu intent (73k train).
OJ_SUBSETS = [("release-v2-redistributable", 3600, SEED + 1),
              ("mailroom-control-v1", 2400, SEED + 2)]
N_LEWIDI, N_GOEMO = 1200, 1200
FORCE_NEW = os.environ.get("FORCE_NEW", "0") == "1"
HF_REPO = os.environ.get("HF_REPO", "itsdevice/lfm-2.5-350m-decs1")
CKPT_DS = os.environ.get("CKPT_DS", "amalanfadilgaib/lfm-2-5-350m-decs1-ckpt")
IN = Path("/kaggle/input")


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


sh([sys.executable, "-m", "pip", "install", "-q", "unsloth", "datasets"])

from datasets import load_dataset  # noqa: E402
from transformers import EarlyStoppingCallback, TrainingArguments  # noqa: E402

from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported  # noqa: E402

print(f"model={MODEL_ID} data={DATA}:{SUBSET} epochs={EPOCHS} max_steps={MAX_STEPS} seed={SEED} aug={AUG_MODE} loto={HOLDOUT_TAXONOMY}", flush=True)
if _AUG_NOTE:
    print(f"WARN augment: {AUG_MODE} {_AUG_NOTE}", flush=True)

# 1. model + LoRA (format tutorial)
model, tokenizer = FastDecisionModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
model = FastDecisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

# 2. data (format tutorial; v4 = mixture standar + customs lunak, dekontaminasi)
if MIXTURE:
    import random as _random

    from unsloth.models.decision_datasets import (
        Decontaminator,
        build_decision_mixture,
        load_source,
    )

    print(f"mixture {MIX_SOURCES} n={N_ROWS} augment={AUGMENT}({AUG_MODE}) loto={HOLDOUT_TAXONOMY}", flush=True)
    eval_texts = []
    # seed dekontaminasi dikunci 3407: referensi tetap antar seed treatment
    for _name, _lim in (("typed_decisions", None), ("banking77", 500), ("clinc150", 500)):
        eval_texts += [r["state"] for r in load_source(_name, "test", limit=_lim, seed=3407)]
    rows = build_decision_mixture(
        MIX_WEIGHTS, N_ROWS, seed=SEED, split="train", augment=AUGMENT,
        decontaminate_against=eval_texts)

    def _soft_gold(keys, dist, qtype):
        s = sum(dist) or 1.0
        probs = {k: float(v) / s for k, v in zip(keys, dist)}
        label = max(probs, key=probs.__getitem__)
        if qtype == "score":
            label = keys.index(label)
        if qtype == "noul":
            label = "true" if label == "true" else "false"
        return {"label": label, "probabilities": probs}

    def _take(pool, n, seed):
        rng = _random.Random(seed)
        return rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]

    customs = []
    # Open-Jev generik per subset: options list ("id: deskripsi") -> criteria;
    # target list -> probs. Subset valid: release-v2-…, mailroom-… (12 nama, cek
    # BuilderConfig dulu — "customer-control-v1" BUKAN subset, pelajaran v4).
    for _subset, _n, _seed in OJ_SUBSETS:
        _oj = load_dataset("ZefanCai/Open-Jev", _subset, split="train")
        _rows = []
        for r in _oj:
            crit, keys = {}, []
            for i, o in enumerate(r["options"]):
                o = str(o)
                kid, _, desc = o.partition(":")
                kid, desc = kid.strip() or f"opt{i}", (desc.strip() or kid.strip())
                crit[kid], keys = {**crit, kid: desc}, keys + [kid]
            qtype = {"choice": "choice", "score": "score", "noul": "noul"}[r["kind"]]
            if qtype == "noul":
                crit = {"true": "yes", "false": "no"}
                keys = ["true", "false"]
            _rows.append({"source": f"openjev-{_subset}", "state": r["state_json"],
                          "questions": {"q": {"type": qtype, "instructions": r["question"],
                                             "criteria": crit}},
                          "gold": {"q": _soft_gold(keys, list(r["target"]), qtype)}})
        customs += _take(_rows, _n, _seed)
        print(f"openjev/{_subset}: pool {len(_rows)} -> take {_n}", flush=True)
    # LeWiDi: soft_label = share anotator urut label. conv_abuse dipetakan ke
    # true/false (lib noul menolak key lain — smoke v1: 1200 baris ke-skip);
    # dist [v0, v1] = [non_abusive, abusive] -> P(false)=v0, P(true)=v1.
    # csc = level sarkasme 1-6 ordinal.
    for _cfg, _n, _qt, _keys, _instr in (
        ("conv_abuse", N_LEWIDI, "noul", ["false", "true"],
         "Is this conversation abusive?"),
        ("csc", N_LEWIDI, "score", [f"level {i}" for i in range(1, 7)],
         "How sarcastic is this response?"),
    ):
        _ds = load_dataset("tasksource/lewidi", _cfg, split="train")
        _pool = []
        for r in _ds:
            dist = [float(v) for v in r["soft_label"]]
            if len(dist) != len(_keys):
                continue
            if _cfg == "conv_abuse" and LEWIDI_FLIP:
                dist = dist[::-1]  # v7: knob verifikasi urutan (RUN_LOG §6.1)
            _txt = r.get("user") or r.get("text") or r.get("response") or ""
            _ctx = r.get("context") or r.get("prev_user") or ""
            _pool.append({"source": f"lewidi-{_cfg}",
                          "state": {"context": _ctx, "text": _txt} if _ctx else {"text": _txt},
                            "questions": {"q": {"type": _qt, "instructions": _instr,
                                              "criteria": (_keys if _qt == "score"
                                                           else ({"false": "non abusive", "true": "abusive"}
                                                                 if _cfg == "conv_abuse"
                                                                  else {k: k.replace("_", " ") for k in _keys}))}},
                          "gold": {"q": _soft_gold(_keys, dist, _qt)}})
        print(f"lewidi/{_cfg}: order={_keys} flip={LEWIDI_FLIP} pool={len(_pool)}", flush=True)
        customs += _take(_pool, _n, SEED + 3)
    # GoEmotions raw: group by id -> fraksi emosi; drop baris all-nol/tak jelas.
    _emo_ds = load_dataset("google-research-datasets/go_emotions", "raw", split="train")
    _emos = ["admiration", "amusement", "anger", "annoyance", "approval", "caring",
             "confusion", "curiosity", "desire", "disappointment", "disapproval",
             "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
             "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
             "remorse", "sadness", "surprise", "neutral"]
    _by_id = {}
    for r in _emo_ds:
        if r.get("example_very_unclear"):
            continue
        _by_id.setdefault(r["id"], []).append(r)
    _pool = []
    for _iid, _rs in _by_id.items():
        _frac = [sum(x[e] for x in _rs) / len(_rs) for e in _emos]
        if sum(_frac) <= 0:
            continue
        _pool.append({"source": "goemotions", "state": {"message": _rs[0]["text"]},
                      "questions": {"q": {"type": "choice",
                                          "instructions": "Which emotion does the writer express?",
                                          "criteria": {e: e.replace("_", " ") for e in _emos}}},
                      "gold": {"q": _soft_gold(_emos, _frac, "choice")}})
    customs += _take(_pool, N_GOEMO, SEED + 4)
    cleaner = Decontaminator(eval_texts)
    customs = [r for r in customs if not cleaner.contaminated(r)]
    _rng = _random.Random(SEED)
    _rng.shuffle(customs)
    rows = rows + customs
    _rng.shuffle(rows)
    if MAX_DECISIONS > 0 and len(rows) > MAX_DECISIONS:  # v7: kontrol token-matched A vs E
        rows = _rng.sample(rows, MAX_DECISIONS)
        print(f"subsample rows -> {len(rows)} (MAX_DECISIONS)", flush=True)
    print(f"customs {len(customs)} (+openjev/lewidi/goemotions)", flush=True)
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
else:
    dataset = load_dataset(DATA, SUBSET, split="train")
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions", flush=True)
if report.get("reason"):
    print(f"skip reason: {report['reason']}", flush=True)
# v7: 3-way — train / cal (fit temperatur) / test (vonis jujur). Cal ≠ test.
# Holdout lib kecil (~1%) -> belah 50/50 dgn shuffle SEED; ukuran dicatat jujur di meta.
train_items, _holdout = FastDecisionModel.split_holdout(items, seed=SEED)
_hold = list(_holdout)
random.Random(SEED + 5).shuffle(_hold)
n_cal = max(1, len(_hold) // 2)
cal_items, test_items = _hold[:n_cal], _hold[n_cal:]
if not test_items:  # holdout terlalu kecil: vonis andalkan test eksternal + eval kernel
    test_items = cal_items
    print("WARN holdout terlalu kecil, test=cal (vonis andalkan test eksternal)", flush=True)
print(f"train {len(train_items)} cal {len(cal_items)} test {len(test_items)}", flush=True)

# 3. resume/new: cari checkpoint Trainer (dataset input -> OUT -> HF), kecuali FORCE_NEW
resume = None
if not FORCE_NEW:
    cands = sorted(IN.glob("**/checkpoint-*")) if IN.exists() else []
    cands += sorted(OUT.glob("checkpoint-*"))
    if cands:
        cp = cands[-1]
        dest = OUT / cp.name
        if cp.parent != OUT:
            shutil.copytree(cp, dest, dirs_exist_ok=True)
        resume = str(dest)
        print(f"resume {resume}", flush=True)
    elif os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import snapshot_download

            snap = snapshot_download(HF_REPO, token=os.environ["HF_TOKEN"],
                                     allow_patterns=["checkpoint-*", "adapter_*", "*.safetensors"])
            cps = sorted(Path(snap).glob("checkpoint-*"))
            if cps:
                shutil.copytree(cps[-1], OUT / cps[-1].name, dirs_exist_ok=True)
                resume = str(OUT / cps[-1].name)
                print(f"resume HF {resume}", flush=True)
        except Exception as e:
            print(f"HF resume miss: {e}", flush=True)
else:
    print("FORCE_NEW: mulai fresh", flush=True)

args = dict(
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=10,
    output_dir=str(OUT),
    report_to="none",
    seed=SEED,
)
if MAX_STEPS > 0:
    args["max_steps"] = MAX_STEPS
_callbacks = []
if os.environ.get("EARLY_STOP", "1") == "1":  # v7: off saat smoke (60 step tak simpan ckpt best)
    args.update(load_best_model_at_end=True, metric_for_best_model="eval_loss",
                greater_is_better=False)  # v7: best eval_loss (overfit ep-3 v2)
    _callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]  # v7: hemat kuota

try:  # v7: transformers lama tak kenal warmup_ratio -> fallback warmup_steps
    _targs = TrainingArguments(**args)
except TypeError as e:
    if "warmup" not in str(e):
        raise
    args.pop("warmup_ratio", None)
    args["warmup_steps"] = 10
    print("WARN transformers lama: warmup_ratio -> warmup_steps=10", flush=True)
    _targs = TrainingArguments(**args)

trainer = DecisionTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_items,
    eval_dataset=cal_items,
    args=_targs,
    callbacks=_callbacks,
)
trainer.train(resume_from_checkpoint=resume)

# 4. calibrate di CAL (format tutorial). Jangan fit temperatur di test.
metrics = FastDecisionModel.calibrate(model, tokenizer, cal_items)
print(metrics, flush=True)  # held-out accuracy, calibration error (ece) and loss

# 5. vonis di TEST internal + test eksternal terpisah (format tutorial)
test_eval = FastDecisionModel.evaluate(model, tokenizer, test_items)
print({"internal_test": test_eval, "n": len(test_items)}, flush=True)
ext_eval = None
try:
    test = load_dataset(DATA, SUBSET, split="test")
    _ext_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model)
    ext_eval = FastDecisionModel.evaluate(model, tokenizer, _ext_items)
    print({"external_test": ext_eval, "n": len(_ext_items)}, flush=True)
except Exception as e:
    print(f"test split tak ada, skip evaluate: {e}", flush=True)

# 5b. artefak buat kernel eval + RUN_LOG (no baseline = no claim)
try:
    import torch as _torch

    _torch.save(cal_items, OUT / "cal_items.pt")
    _torch.save(test_items, OUT / "test_items.pt")
    print("saved cal_items.pt + test_items.pt", flush=True)
except Exception as e:
    print(f"save items skip: {e}", flush=True)
run_meta = {
    "model": MODEL_ID, "seed": SEED, "max_seq_len": MAX_SEQ_LEN, "epochs": EPOCHS,
    "mixture": MIXTURE, "n_rows": N_ROWS if MIXTURE else None,
    "augment": AUGMENT, "aug_mode": AUG_MODE, "aug_note": _AUG_NOTE,
    "holdout_taxonomy": HOLDOUT_TAXONOMY, "max_decisions": MAX_DECISIONS,
    "lewidi_flip": LEWIDI_FLIP, "mix_weights": MIX_WEIGHTS if MIXTURE else None,
    "n_rows_built": len(rows) if MIXTURE else None,
    "decisions_total": report.get("total"), "skipped": report.get("skipped"),
    "skip_reason": report.get("reason"), "n_train": len(train_items),
    "n_cal": len(cal_items), "n_test": len(test_items), "resume": resume,
    "calibrate": metrics, "internal_test": test_eval, "external_test": ext_eval,
}
try:
    run_meta["git_hash"] = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    run_meta["git_hash"] = None
(OUT / "run_meta.json").write_text(json.dumps(run_meta, default=str, indent=2))
print("saved run_meta.json — tempel ke docs/RUN_LOG.md", flush=True)

# 6. simpan: adapters+head, merged 16-bit, push HF
model.save_pretrained(str(OUT / "final"))
model.save_pretrained_merged(str(OUT / "merged"))
print(f"saved {OUT / 'final'} + {OUT / 'merged'}", flush=True)
if os.environ.get("HF_TOKEN"):
    try:
        model.push_to_hub(HF_REPO, token=os.environ["HF_TOKEN"])
        print(f"pushed {HF_REPO}", flush=True)
    except Exception as e:
        print(f"push gagal: {e}", flush=True)

# 7. backup dataset Kaggle (ckpt Trainer terakhir + final + metrik)
tok = os.environ.get("KAGGLE_API_TOKEN")
try:
    if tok:
        from kaggle_secrets import UserSecretsClient  # noqa

        tok = tok or UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
except Exception:
    pass
if tok:
    import json as _json

    kdir = Path.home() / ".kaggle"
    kdir.mkdir(exist_ok=True)
    if tok.strip().startswith("{"):
        (kdir / "kaggle.json").write_text(tok)
        os.chmod(kdir / "kaggle.json", 0o600)
        stage = WORKING / "u-ckpt-ds"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        for cp in sorted(OUT.glob("checkpoint-*"))[-1:]:
            shutil.copytree(cp, stage / cp.name)
        for f in ("final", "merged"):
            if (OUT / f).exists():
                shutil.copytree(OUT / f, stage / f)
        for f in ("run_meta.json", "cal_items.pt", "test_items.pt"):
            if (OUT / f).exists():
                shutil.copy2(OUT / f, stage / f)
        (stage / "dataset-metadata.json").write_text(
            _json.dumps({"title": "lfm-2-5-350m-decs1-ckpt", "id": CKPT_DS,
                         "licenses": [{"name": "CC0-1.0"}]}))
        try:
            subprocess.check_call(["kaggle", "datasets", "status", CKPT_DS])
            sh(["kaggle", "datasets", "version", "-p", str(stage), "-m", "unsloth run", "-r", "skip"])
        except subprocess.CalledProcessError:
            sh(["kaggle", "datasets", "create", "-p", str(stage), "-r", "skip"])
    else:
        print("KAGGLE_API_TOKEN bukan kaggle.json; pakai UI: Create Dataset from Output", flush=True)
else:
    print("tanpa KAGGLE_API_TOKEN: backup via UI atau HF push di atas", flush=True)

print("done", flush=True)
