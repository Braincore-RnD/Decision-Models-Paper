"""#6 A-LOTO 2x2 via Modal (4 train paralel, T4).

Pakai:  modal run code/train_decision_models/modal_train_loto.py            # full 4x
        modal run code/train_decision_models/modal_train_loto.py --smoke   # 60-step validasi pipa
Butuh sekali: modal secret create hf-token HF_TOKEN=hf_...
Estimasi: 4 x ~2.5 jam T4 @ $0.59/jam = ~$6 (Starter $30 gratis), wall ~3 jam.
"""
from __future__ import annotations

import json
import subprocess
import sys

import modal

app = modal.App("jev-loto6")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "unsloth", "datasets", "transformers", "huggingface_hub", "trl", "peft"
).add_local_file(
    "code/train_decision_models/unsloth_kaggle_dec.py",
    remote_path="/root/train/unsloth_kaggle_dec.py",
)
vol = modal.Volume.from_name("jev-loto6", create_if_missing=True)

BASE_ENV = {
    "SEED": "3407", "MAX_SEQ_LEN": "4096", "EPOCHS": "3", "MAX_STEPS": "0",
    "BATCH": "4", "GRAD_ACCUM": "8", "EARLY_STOP": "1", "MIXTURE": "1",
    "N_ROWS": "20400", "FORCE_NEW": "1",
}
# ponytail: 4 sel #6 (AUG x LOTO); kontrol A/E tak-train-ulang sudah ada di RUN_LOG
MATRIX = [
    {"AUG_MODE": "none", "HOLDOUT_TAXONOMY": "banking77",
     "HF_REPO": "itsdevice/lfm-decs1-A-loto-bank"},
    {"AUG_MODE": "none", "HOLDOUT_TAXONOMY": "clinc150",
     "HF_REPO": "itsdevice/lfm-decs1-A-loto-clinc"},
    {"AUG_MODE": "full", "HOLDOUT_TAXONOMY": "banking77",
     "HF_REPO": "itsdevice/lfm-decs1-E-loto-bank"},
    {"AUG_MODE": "full", "HOLDOUT_TAXONOMY": "clinc150",
     "HF_REPO": "itsdevice/lfm-decs1-E-loto-clinc"},
]


@app.function(gpu="T4", timeout=18_000, image=image, volumes={"/mnt/vol": vol},
              secrets=[modal.Secret.from_name("hf-token")])
def train_cell(cfg: dict) -> dict:
    import os
    from pathlib import Path
    job = cfg["HF_REPO"].split("/")[-1]
    work = Path("/mnt/vol") / job
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **BASE_ENV, **cfg)
    r = subprocess.run([sys.executable, "/root/train/unsloth_kaggle_dec.py"],
                       cwd=str(work), env=env, capture_output=True, text=True)
    (work / "train.log").write_text(r.stdout[-50_000:] + "\nSTDERR:\n" + r.stderr[-10_000:])
    vol.commit()
    meta = work / "out-u" / "run_meta.json"
    out = {"job": job, "rc": r.returncode,
           "tail": (r.stdout[-1500:] if r.returncode else "")}
    if meta.exists():
        out["meta"] = json.loads(meta.read_text())
    return out


@app.local_entrypoint()
def main(smoke: bool = False):
    if smoke:
        cfg = dict(MATRIX[0], MAX_STEPS="60", EARLY_STOP="0", EPOCHS="1")
        print(json.dumps(train_cell.remote(cfg), indent=1)[:2000])
        return
    for out in train_cell.map(MATRIX):
        m = out.get("meta", {})
        print(f"===== {out['job']} rc={out['rc']} "
              f"cal={m.get('calibrate')} intest={m.get('internal_test')} "
              f"extest={m.get('external_test')}")
        if out["rc"]:
            print(out["tail"])
