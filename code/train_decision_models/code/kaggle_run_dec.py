"""Kaggle entry lfm-2.5-350m-decs1. kernels push hanya kirim file ini.

Kernel metadata terpisah (code_file=kaggle_run_dec.py), dataset_sources:
  - <user>/dec-src   (code/*.py)
  - opsional <user>/dec-data (train.pt/val.pt jadi; kalau tak ada -> PREP_HF/PREP_SMOKE)
Backbone LiquidAI/LFM2.5-350M selalu dari Hub (tak perlu dataset backbone).
Resume: ckpt_dec.pt (dataset) dulu, lalu HF. Backup: ckpt tiap eval (train_dec.py
--hf_repo) + dataset dec-ckpt di akhir run. Token via env secret, jangan hardcode.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).resolve().parent
ROOT = WORKING
os.chdir(ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONPATH"] = str(ROOT)

IN = Path("/kaggle/input")
OUT = WORKING / "out-dec"
OUT.mkdir(parents=True, exist_ok=True)
DATA = WORKING / "dec-data"
DATA.mkdir(parents=True, exist_ok=True)
CKPT_DS = "amalanfadilgaib/lfm-2-5-350m-decs1-ckpt"
HF_REPO = os.environ.get("HF_REPO_DEC", "itsdevice/lfm-2.5-350m-decs1")
PREP_HF = os.environ.get("PREP_HF", "LocalLLaMA/typed-decisions:all")
PREP_SMOKE = os.environ.get("PREP_SMOKE", "")  # "1" = data sintetik (tiny-overfit DoD)


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


HF_TOKEN = os.environ.get("HF_TOKEN", "")


def hf_token() -> str:
    os.environ["HF_TOKEN"] = HF_TOKEN
    return HF_TOKEN


def kaggle_token() -> str | None:
    tok = os.environ.get("KAGGLE_API_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
    except Exception:
        return None


sh([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.2.0", "accelerate",
    "safetensors", "huggingface_hub", "kaggle", "pandas", "pyarrow"])
try:
    # kernel conv LFM; gagal = fallback reference PyTorch (benar tapi lambat)
    sh([sys.executable, "-m", "pip", "install", "-q", "causal-conv1d"])
except Exception as e:
    print(f"causal-conv1d skip: {e}", flush=True)

# 1. kode dari src dataset (code/*.py, flat-copy)
src = next(IN.glob("**/train_dec.py"), None) if IN.exists() else None
if src is None:
    raise FileNotFoundError("src dataset missing code/train_dec.py under /kaggle/input")
dec = src.parent
for f in ("train_dec.py", "head.py", "data_dec.py"):
    p = dec / f
    if not p.exists():
        raise FileNotFoundError(p)
    shutil.copy(p, ROOT / f)
    print(f"code: {p} -> {ROOT / f}", flush=True)

# 2. data: dataset jadi > PREP_SMOKE > PREP_HF (default typed-decisions)
have = list(IN.glob("**/decision_train.pt")) + list(DATA.glob("train.pt")) if IN.exists() else []
data_dir = str(DATA)
if have and not PREP_SMOKE and not os.environ.get("PREP_HF"):
    data_dir = str(have[0].parent)
    print(f"data: {data_dir}", flush=True)
else:
    prep = [sys.executable, str(ROOT / "data_dec.py"), f"--out_dir={DATA}", "--max_len=1024"]
    if PREP_SMOKE:
        prep += ["--smoke=True"]
    else:
        prep += [f"--from_hf={PREP_HF}"]
        print(f"prep HF {PREP_HF}", flush=True)
    sh(prep)

# 3. resume dec ckpt (dataset dulu, lalu HF). Backbone selalu fresh dari Hub.
hf_token()
init = "fresh"
dec_ckpts = list(IN.glob("**/ckpt_dec.pt")) if IN.exists() else []
if dec_ckpts or (OUT / "ckpt.pt").exists():
    if dec_ckpts:
        shutil.copy(dec_ckpts[0], OUT / "ckpt.pt")
    init = "resume"
    print("resume dec ckpt", flush=True)
elif os.environ.get("HF_TOKEN"):
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(HF_REPO, token=os.environ["HF_TOKEN"],
                                 allow_patterns=["ckpt.pt", "metrics.csv"])
        if (Path(snap) / "ckpt.pt").exists():
            shutil.copy(Path(snap) / "ckpt.pt", OUT / "ckpt.pt")
            init = "resume"
            print(f"resume HF {HF_REPO}", flush=True)
    except Exception as e:
        print(f"HF resume miss: {e}", flush=True)

import torch

n = torch.cuda.device_count()
print(f"cuda devices: {n}", flush=True)
dtype = "float16" if n else "float32"
args = [f"--out_dir={OUT}", f"--data_dir={data_dir}", f"--init_from={init}",
        "--compile=False", f"--dtype={dtype}", f"--hf_repo={HF_REPO}"]
args += ["--eval_interval=0", "--decide_every=0", "--num_train_epochs=2"]
if PREP_SMOKE and not PREP_HF:
    args += ["--max_steps=60"]  # smoke ala Unsloth: 60 step cukup sinyal belajar
train = str(ROOT / "train_dec.py")
if n >= 2:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", train, *args]
else:
    cmd = [sys.executable, train, *args]
sh(cmd)

# 4. backup dataset
ktok = kaggle_token()
if ktok and (OUT / "ckpt.pt").exists():
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(exist_ok=True)
    cred = Path(ktok)
    if cred.exists():
        shutil.copy(cred, kdir / "kaggle.json")
    elif ktok.strip().startswith("{"):
        (kdir / "kaggle.json").write_text(ktok)
    else:
        print("KAGGLE_API_TOKEN bukan kaggle.json; skip dataset version. Pakai UI: Create Dataset from Output", flush=True)
        ktok = None
    if ktok and (kdir / "kaggle.json").exists():
        os.chmod(kdir / "kaggle.json", 0o600)
        stage = WORKING / "dec-ckpt-ds"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        for f in ("ckpt.pt", "metrics.csv", "temps.json"):
            if (OUT / f).exists():
                shutil.copy(OUT / f, stage / ("ckpt_dec.pt" if f == "ckpt.pt" else f))
        if (OUT / "head").is_dir():
            for f in (OUT / "head").iterdir():
                shutil.copy(f, stage / f.name)
        (stage / "dataset-metadata.json").write_text(
            json.dumps({"title": "lfm-2-5-350m-decs1-ckpt", "id": CKPT_DS, "licenses": [{"name": "CC0-1.0"}]}))
        try:
            subprocess.check_call(["kaggle", "datasets", "status", CKPT_DS])
            sh(["kaggle", "datasets", "version", "-p", str(stage), "-m", f"dec ckpt {init}", "-r", "skip"])
        except subprocess.CalledProcessError:
            sh(["kaggle", "datasets", "create", "-p", str(stage), "-r", "skip"])
else:
    print("dataset backup: Create Dataset from Output -> lfm-2-5-350m-decs1-ckpt, or set secret KAGGLE_API_TOKEN=kaggle.json", flush=True)

print("done", OUT / "ckpt.pt", flush=True)
