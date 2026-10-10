"""ARSIP v1 (rekonstruksi setia run train v1: typed-decisions saja, 2 epoch).
JANGAN DIJALANKAN LANGSUNG — pakai file living terbaru.

lfm-2.5-350m-decs1 via Unsloth — 1 script Kaggle, format tutorial Unsloth + ckpt/resume.

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
EPOCHS = float(os.environ.get("EPOCHS", "2"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
MIXTURE = os.environ.get("MIXTURE", "0") == "1"  # v1: "0" = typed-decisions saja
N_ROWS = int(os.environ.get("N_ROWS", "12000"))
MIX_SOURCES = ["ag_news", "arc", "banking77", "boolq", "clinc150", "commonsense_qa",
               "mmlu", "mnli", "prompt_injections", "snli", "sst5", "wanli",
               "typed_decisions"]
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

# 2. data (format tutorial; v2 = campuran 12 sumber + typed-decisions, dekontaminasi)
if MIXTURE:
    from unsloth.models.decision_datasets import build_decision_mixture, load_source

    print(f"mixture {MIX_SOURCES} n={N_ROWS}", flush=True)
    eval_texts = []
    for _name, _lim in (("typed_decisions", None), ("banking77", 500), ("clinc150", 500)):
        eval_texts += [r["state"] for r in load_source(_name, "test", limit=_lim, seed=3407)]
    rows = build_decision_mixture(
        {n: 1.0 for n in MIX_SOURCES}, N_ROWS, seed=3407, split="train",
        decontaminate_against=eval_texts)
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
else:
    dataset = load_dataset(DATA, SUBSET, split="train")
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions", flush=True)
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
