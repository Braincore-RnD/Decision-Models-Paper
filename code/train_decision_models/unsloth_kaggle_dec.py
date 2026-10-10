"""lfm-2.5-350m-decs1 v6 (living) — 1 script Kaggle, format tutorial Unsloth + ckpt/resume.
Arsip: archive/train_v1_typed-only-2ep.py, archive/train_v2_mixture-3ep.py,
archive/train_v3_typed2x-2ep.py, archive/train_v4_30k-noaug-3ep.py,
archive/train_v5_rebalance-3ep.py.
v6 = resep v5 + AUGMENT=1 (uji hipotesis: augmentasi driver transfer, bukan porsi).

Mengikuti https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth
(from_pretrained -> get_peft_model -> build_dataset -> split_holdout ->
DecisionTrainer -> calibrate -> save), ditambah:
  - MODE otomatis: fresh bila tak ada ckpt, resume bila ada (atau FORCE_NEW=1).
  - Checkpoint Trainer tiap epoch (save_total_limit=1) + resume_from_checkpoint.
  - Simpan adapters+head, merged 16-bit, push HF, backup dataset Kaggle.

Jalankan sebagai Kaggle kernel script (GPU T4 x2 disarankan; T4 x1 bisa).
Env:
  MODEL_ID   default LiquidAI/LFM2.5-350M (alternatif: unsloth/LFM2-350M = LFM2 lama)
  DATA       default LocalLLaMA/typed-decisions | SUBSET default all
  EPOCHS     default 2 | MAX_STEPS default 0 (0=pakai epoch; 60=smoke ala Unsloth)
  FORCE_NEW  default 0 ("1" = abaikan ckpt, mulai fresh)
  HF_REPO    default itsdevice/lfm-2.5-350m-decs1 (push bila HF_TOKEN ada)
  CKPT_DS    default amalanfadilgaib/lfm-2-5-350m-decs1-ckpt (backup via KAGGLE_API_TOKEN)

Titik gagal paling mungkin (jujur): FastDecisionModel x arsitektur hybrid LFM.
Sesi smoke (MAX_STEPS=60) dirancang menemukan ini dalam hitungan menit.
"""
from __future__ import annotations

import os
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
EPOCHS = float(os.environ.get("EPOCHS", "3"))  # v4: 3ep (data 2,5x v2, risiko overfit lbh kcl)
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
MIXTURE = os.environ.get("MIXTURE", "1") == "1"  # "0" = typed-decisions saja (v1)
N_ROWS = int(os.environ.get("N_ROWS", "20400"))  # v5: standar (intent w3 di dlm)
AUGMENT = os.environ.get("AUGMENT", "1") == "1"  # v6: 1 = augmentasi ON (uji driver transfer)
MIX_SOURCES = ["ag_news", "arc", "banking77", "boolq", "clinc150", "commonsense_qa",
               "mmlu", "mnli", "prompt_injections", "snli", "sst5", "wanli"]
# v5: banking/clinc w3 (pulihkan porsi intent >=30%); typed normal.
MIX_WEIGHTS = {n: 1.0 for n in MIX_SOURCES} | {"typed_decisions": 1.0,
                                               "banking77": 3.0, "clinc150": 3.0}
# v5 customs: (subset, n). mailroom = triase email, sepupu intent (73k train).
OJ_SUBSETS = [("release-v2-redistributable", 3600, 3407),
              ("mailroom-control-v1", 2400, 3411)]
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
from transformers import TrainingArguments  # noqa: E402

from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported  # noqa: E402

print(f"model={MODEL_ID} data={DATA}:{SUBSET} epochs={EPOCHS} max_steps={MAX_STEPS}", flush=True)

# 1. model + LoRA (format tutorial)
model, tokenizer = FastDecisionModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastDecisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 2. data (format tutorial; v4 = mixture standar + customs lunak, dekontaminasi)
if MIXTURE:
    import random as _random

    from unsloth.models.decision_datasets import (
        Decontaminator,
        build_decision_mixture,
        load_source,
    )

    print(f"mixture {MIX_SOURCES} n={N_ROWS} augment={AUGMENT}", flush=True)
    eval_texts = []
    for _name, _lim in (("typed_decisions", None), ("banking77", 500), ("clinc150", 500)):
        eval_texts += [r["state"] for r in load_source(_name, "test", limit=_lim, seed=3407)]
    rows = build_decision_mixture(
        MIX_WEIGHTS, N_ROWS, seed=3407, split="train", augment=AUGMENT,
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
    # LeWiDi: soft_label = share anotator urut label. conv_abuse deduksi [non_abusive,
    # abusive] (row unanimous benign -> indeks 0); csc = level sarkasme 1-6 ordinal.
    for _cfg, _n, _qt, _keys, _instr in (
        ("conv_abuse", N_LEWIDI, "noul", ["non_abusive", "abusive"],
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
            _txt = r.get("user") or r.get("text") or r.get("response") or ""
            _ctx = r.get("context") or r.get("prev_user") or ""
            _pool.append({"source": f"lewidi-{_cfg}",
                          "state": {"context": _ctx, "text": _txt} if _ctx else {"text": _txt},
                          "questions": {"q": {"type": _qt, "instructions": _instr,
                                              "criteria": (_keys if _qt == "score"
                                                           else {k: k.replace("_", " ") for k in _keys})}},
                          "gold": {"q": _soft_gold(_keys, dist, _qt)}})
        customs += _take(_pool, _n, 3409)
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
    customs += _take(_pool, N_GOEMO, 3410)
    cleaner = Decontaminator(eval_texts)
    customs = [r for r in customs if not cleaner.contaminated(r)]
    _rng = _random.Random(3407)
    _rng.shuffle(customs)
    rows = rows + customs
    _rng.shuffle(rows)
    print(f"customs {len(customs)} (+openjev/lewidi/goemotions)", flush=True)
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
else:
    dataset = load_dataset(DATA, SUBSET, split="train")
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions", flush=True)
if report.get("reason"):
    print(f"skip reason: {report['reason']}", flush=True)
train_items, eval_items = FastDecisionModel.split_holdout(items, seed=3407)
print(f"train {len(train_items)} holdout {len(eval_items)}", flush=True)

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
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    weight_decay=0.01,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=10,
    output_dir=str(OUT),
    report_to="none",
    seed=3407,
)
if MAX_STEPS > 0:
    args["max_steps"] = MAX_STEPS

trainer = DecisionTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_items,
    eval_dataset=eval_items,
    args=TrainingArguments(**args),
)
trainer.train(resume_from_checkpoint=resume)

# 4. calibrate di holdout (format tutorial)
metrics = FastDecisionModel.calibrate(model, tokenizer, eval_items)
print(metrics, flush=True)  # held-out accuracy, calibration error (ece) and loss

# 5. vonis di test terpisah (format tutorial)
try:
    test = load_dataset(DATA, SUBSET, split="test")
    test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model)
    print(FastDecisionModel.evaluate(model, tokenizer, test_items), flush=True)
except Exception as e:
    print(f"test split tak ada, skip evaluate: {e}", flush=True)

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
