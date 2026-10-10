"""Decision-head trainer gaya nanoGPT (DDP, FP16, CSV, satu ckpt.pt).
Klon train.py: DDP, FP16, CSV, satu ckpt.pt.

FULL KAGGLE, tanpa training lokal. Dijalankan lewat code/kaggle_run_dec.py:
  PREP_SMOKE=1  -> data sintetik + --max_steps=60 (tiny-overfit DoD)
  PREP_HF=<ds:subset> -> typed-decisions dari HuggingFace
Resume otomatis dari dataset dec-ckpt / HF.
Default freeze_backbone=True: hanya head yang dilatih (paling murah).
Di akhir: calibrate() di val -> temps.json + masuk ckpt.pt.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from ast import literal_eval
from contextlib import nullcontext
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # head.py
from head import DecisionModel, JointHead, calibrate, decision_loss, metrics, systemone_answer
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- defaults ---
out_dir = "out-dec"
data_dir = "code/data"
smoke_data = False
eval_interval = 0  # 0 = auto (10% iters)
eval_batches = 20
log_interval = 1
eval_only = False
always_save_checkpoint = True
init_from = "fresh"  # 'fresh' (head baru di atas backbone pretrained) | 'resume'
backbone_id = "LiquidAI/LFM2.5-350M"
tokenizer_id = "LiquidAI/LFM2.5-350M"
freeze_backbone = True
batch_size = 4
gradient_accumulation_steps = 8
max_len = 1024
head_width = 256
head_routing = 2
head_layers = 4
head_heads = 4
head_ff = 512
learning_rate = 2e-4  # backbone bila unfreeze (Unsloth: 2e-4)
head_lr = 1e-4  # head selalu (Unsloth default head_learning_rate)
label_smoothing = 0.0
brier_weight = 0.0
ordinal_weight = 0.0
# Durasi ala Unsloth DecisionTrainer: epoch-based, bukan iter mentah.
#   num_train_epochs=2 (data kecil -> 3-4); max_steps>0 = smoke ala Unsloth (60).
#   max_iters=0 / lr_decay_iters=0 / warmup_iters=0 = auto dari ukuran data.
num_train_epochs = 2.0
max_steps = 0
max_iters = 0
weight_decay = 1e-2  # Unsloth: 0.01 (train.py pretrain pakai 0.1)
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 0  # 0 = auto: min(10, 10% iters) — Unsloth pakai warmup_steps=10
lr_decay_iters = 0  # 0 = auto = max_iters
min_lr = 2e-5
decide_every = 0  # 0 = auto = eval_interval (samakan contoh jawaban tiap eval)
backend = "nccl"
device = "cuda"
dtype = "float16"
compile = False
seed = 42
hf_repo = ""

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
for arg in sys.argv[1:]:
    if not arg.startswith("--"):
        continue
    key, val = arg[2:].split("=", 1)
    if key not in globals():
        raise ValueError(f"unknown config {key}")
    try:
        attempt = literal_eval(val)
    except (SyntaxError, ValueError):
        attempt = val
    globals()[key] = attempt
    print(f"override {key}={attempt}")
config = {k: globals()[k] for k in config_keys}

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

dec_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size
if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"decisions per iteration: {dec_per_iter}")

if device == "cpu" or not torch.cuda.is_available():
    raise SystemExit("train_dec hanya jalan di GPU (full Kaggle, tanpa training lokal).")
torch.manual_seed(seed + seed_offset)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def load_list(name):
    return torch.load(os.path.join(data_dir, f"{name}.pt"), weights_only=False)


if smoke_data:
    import subprocess

    subprocess.check_call([sys.executable, os.path.join(_HERE, "data_dec.py"),
                           "--smoke=True", f"--out_dir={data_dir}", f"--max_len={max_len}"])
train_items, val_items = load_list("train"), load_list("val")
print(f"train {len(train_items)} val {len(val_items)}")

# Auto-scale durasi dari ukuran data (tak perlu hitung manual).
# Sampling acak-with-replacement, jadi epoch = ekspektasi: iters = epoch * N/dec_per_iter.
import math as _math

_iters_per_epoch = max(1, _math.ceil(len(train_items) / dec_per_iter))
if max_steps > 0:
    max_iters = max_steps  # smoke ala Unsloth (--max_steps=60)
elif max_iters <= 0:
    max_iters = max(1, int(round(num_train_epochs * _iters_per_epoch)))
if warmup_iters <= 0:
    warmup_iters = min(10, max(1, max_iters // 10))
if lr_decay_iters <= 0:
    lr_decay_iters = max_iters
if eval_interval <= 0:
    eval_interval = max(1, max_iters // 10)
if decide_every == 0:
    decide_every = eval_interval
print(f"iters/epoch ~{_iters_per_epoch} -> max_iters={max_iters} warmup={warmup_iters} "
      f"eval_every={eval_interval} (epochs={num_train_epochs})")


_tok = AutoTokenizer.from_pretrained(tokenizer_id)
if _tok.pad_token_id is None:
    _tok.pad_token = _tok.eos_token
PAD_ID = _tok.pad_token_id


def get_batch(split):
    pool = train_items if split == "train" else val_items
    ix = torch.randint(len(pool), (batch_size,)).tolist()
    rows = [pool[i] for i in ix]
    L = max(len(r["input_ids"]) for r in rows)
    nq = max(len(r["rec"]) for r in rows)
    no = max(r["target"].shape[1] for r in rows)
    ids = torch.full((len(rows), L), PAD_ID, dtype=torch.long)
    mask = torch.zeros(len(rows), L, dtype=torch.long)
    target = torch.zeros(len(rows), nq, no)
    valid = torch.zeros(len(rows), nq, no, dtype=torch.bool)
    qtypes = torch.zeros(len(rows), nq, dtype=torch.long)
    ordinal = torch.zeros(len(rows), nq, dtype=torch.bool)
    for b, r in enumerate(rows):
        t = len(r["input_ids"])
        ids[b, :t] = torch.tensor(r["input_ids"])
        mask[b, :t] = 1
        q, o = r["target"].shape
        target[b, :q, :o] = r["target"]
        valid[b, :q, :o] = True
        qtypes[b, :q] = torch.tensor(r["qtypes"])
        ordinal[b, :q] = torch.tensor(r["ordinal"])
    if device_type == "cuda":
        ids, mask = ids.pin_memory().to(device, non_blocking=True), mask.pin_memory().to(device, non_blocking=True)
        target, qtypes = target.pin_memory().to(device, non_blocking=True), qtypes.pin_memory().to(device, non_blocking=True)
    else:
        ids, mask, target, qtypes = ids.to(device), mask.to(device), target.to(device), qtypes.to(device)
    return ids, mask, [{"questions": r["rec"]} for r in rows], target, valid.to(device), qtypes, ordinal.to(device)


iter_num = 0
decisions_seen = 0
best_val_acc = 0.0
temps = {"head": 1.0, "rel": [1.0, 1.0, 1.0]}
head_args = dict(width=head_width, routing=head_routing,
                 layers=head_layers, heads=head_heads, ff=head_ff)

_bb_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]


def load_backbone():
    try:
        bb = AutoModelForCausalLM.from_pretrained(backbone_id, dtype=_bb_dtype)
    except Exception as e:
        print(f"native load gagal ({e}); coba trust_remote_code=True")
        bb = AutoModelForCausalLM.from_pretrained(backbone_id, dtype=_bb_dtype, trust_remote_code=True)
    bb.config.use_cache = False
    return bb


checkpoint = None
if init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    print(f"resume {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if checkpoint.get("backbone_id", backbone_id) != backbone_id:
        raise ValueError(f"ckpt backbone {checkpoint.get('backbone_id')} != {backbone_id}")
    head_args.update(checkpoint.get("head_args", {}))
    temps = checkpoint.get("temps", temps)
elif init_from != "fresh":
    raise ValueError(init_from)

print(f"load backbone {backbone_id}")
backbone = load_backbone()
hidden = backbone.config.hidden_size
if head_args.get("n_embd", hidden) != hidden:
    raise ValueError(f"head n_embd {head_args.get('n_embd')} != backbone hidden {hidden}")
head_args["n_embd"] = hidden
print(f"backbone hidden={hidden} params={sum(p.numel() for p in backbone.parameters()) / 1e6:.0f}M")

head = JointHead(**head_args)
model = DecisionModel(backbone, head)
if checkpoint is not None:
    model.head.load_state_dict(checkpoint["head"])
    iter_num = checkpoint["iter_num"]
    decisions_seen = checkpoint.get("decisions_seen", 0)
    best_val_acc = checkpoint.get("best_val_acc", 0.0)

if freeze_backbone:
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

model.to(device)
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16" and device_type == "cuda"))


def configure_optimizers():
    groups = []
    hd, hn = [], []
    for p in model.head.parameters():
        (hd if p.dim() >= 2 else hn).append(p)
    groups += [{"params": hd, "weight_decay": weight_decay, "lr": head_lr},
               {"params": hn, "weight_decay": 0.0, "lr": head_lr}]
    if not freeze_backbone:
        bd, bn = [], []
        for n, p in model.backbone.named_parameters():
            (bd if p.dim() >= 2 else bn).append(p)
        groups += [{"params": bd, "weight_decay": weight_decay, "lr": learning_rate},
                   {"params": bn, "weight_decay": 0.0, "lr": learning_rate}]
    fused = device_type == "cuda"
    kw = {"fused": True} if fused else {}
    try:
        return torch.optim.AdamW(groups, betas=(beta1, beta2), **kw)
    except TypeError:
        return torch.optim.AdamW(groups, betas=(beta1, beta2))


optimizer = configure_optimizers()
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

if compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=freeze_backbone)
raw_model = model.module if ddp else model


def _push_hf(folder: str) -> None:
    if not hf_repo or not master_process:
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF skip: no HF_TOKEN")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(hf_repo, exist_ok=True, private=True, repo_type="model")
        api.upload_folder(folder_path=folder, repo_id=hf_repo, repo_type="model")
        print(f"HF pushed {folder} -> {hf_repo}")
    except Exception as e:
        print(f"HF push failed (dataset still local): {e}")


def forward_loss(ids, mask, recs, target, valid, qtypes, ordinal):
    logits, v = raw_model(ids, mask, recs)
    v = v & valid
    loss = decision_loss(logits, target, v, qtypes, ordinal, label_smoothing, brier_weight, ordinal_weight)
    return logits, loss


@torch.no_grad()
def estimate():
    out = {}
    was_train = raw_model.training
    raw_model.eval()
    for split in ["train", "val"]:
        losses, accs = [], []
        for _ in range(eval_batches):
            ids, mask, recs, target, valid, qtypes, ordinal = get_batch(split)
            with ctx:
                logits, loss = forward_loss(ids, mask, recs, target, valid, qtypes, ordinal)
            losses.append(loss.item())
            pred = logits.masked_fill(~valid, -1e4).argmax(-1)
            gold = target.argmax(-1)
            m = valid.any(-1)
            accs.append(((pred == gold)[m]).float().mean().item())
        out[split] = (sum(losses) / len(losses), sum(accs) / len(accs))
    if was_train:
        raw_model.train()
        if freeze_backbone:
            raw_model.backbone.eval()
    return out


def get_lr(it, base):
    if it < warmup_iters:
        return base * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (base - min_lr)


@torch.no_grad()
def decide_sample():
    r = val_items[0]
    ids = torch.tensor([r["input_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    raw_model.eval()
    with ctx:
        logits, _ = raw_model(ids, mask, [{"questions": r["rec"]}])
    T, rel = temps["head"], temps["rel"]
    print("decide sample:\n---")
    for j, q in enumerate(r["rec"]):
        row = logits[0, j, : len(q["oids"])] / (T * rel[q["type"]])
        probs = dict(zip(q["oids"], row.softmax(-1).tolist()))
        print(q["id"], systemone_answer(r["asked"][q["id"]], probs))
    print("---")
    if freeze_backbone:
        raw_model.backbone.eval()


def save_ckpt():
    # ckpt.pt = head + optimizer saja (kecil). Backbone selalu di-load ulang dari Hub.
    ckpt = {"head": raw_model.head.state_dict(), "optimizer": optimizer.state_dict(),
            "head_args": head_args, "backbone_id": backbone_id, "tokenizer_id": tokenizer_id,
            "iter_num": iter_num, "decisions_seen": decisions_seen, "best_val_acc": best_val_acc,
            "temps": temps, "config": config}
    tmp = os.path.join(out_dir, "ckpt.pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, os.path.join(out_dir, "ckpt.pt"))
    print(f"saved {out_dir}/ckpt.pt")
    _push_hf(out_dir)


def save_deploy():
    # out-dec/head/ = artefak serve mandiri: bobot head + config + temps + rujukan backbone.
    import json as _json

    d = os.path.join(out_dir, "head")
    os.makedirs(d, exist_ok=True)
    sd = {k: v.float().cpu() for k, v in raw_model.head.state_dict().items()}
    try:
        from safetensors.torch import save_file

        save_file(sd, os.path.join(d, "head.safetensors"))
    except Exception:
        torch.save(sd, os.path.join(d, "head.pt"))
    with open(os.path.join(d, "head_config.json"), "w") as f:
        _json.dump({"head_args": head_args, "backbone_id": backbone_id,
                    "tokenizer_id": tokenizer_id, "temps": temps}, f, indent=2)
    print(f"saved {d}/")


csv_path = os.path.join(out_dir, "metrics.csv")
if master_process and not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["time", "iter", "decisions_seen", "loss", "train_loss", "train_acc",
                                "val_loss", "val_acc", "lr", "grad_norm", "dec_s", "vram_gb"])

t0 = time.time()
base_lrs = [g["lr"] for g in optimizer.param_groups]
while True:
    for g, base in zip(optimizer.param_groups, base_lrs):
        g["lr"] = get_lr(iter_num, base) if decay_lr else base

    if iter_num % eval_interval == 0 and master_process:
        est = estimate()
        print(f"step {iter_num}: train loss {est['train'][0]:.4f} acc {est['train'][1]:.3f} | "
              f"val loss {est['val'][0]:.4f} acc {est['val'][1]:.3f}")
        if est["val"][1] > best_val_acc or always_save_checkpoint:
            best_val_acc = max(best_val_acc, est["val"][1])
            if iter_num > 0:
                save_ckpt()
    if master_process and decide_every > 0 and (iter_num == 0 or iter_num % decide_every == 0):
        decide_sample()
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        ids, mask, recs, target, valid, qtypes, ordinal = get_batch("train")
        with ctx:
            _, loss = forward_loss(ids, mask, recs, target, valid, qtypes, ordinal)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        gn = torch.tensor(0.0)
    if not torch.isfinite(gn):
        optimizer.zero_grad(set_to_none=True)
        scaler.update()
        print(f"iter {iter_num}: non-finite grad, skipped", flush=True)
        iter_num += 1
        decisions_seen += dec_per_iter
        continue
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    lossf = loss.item() * gradient_accumulation_steps
    if not math.isfinite(lossf):
        raise RuntimeError(f"NaN/Inf loss={lossf} at iter {iter_num}")
    decisions_seen += dec_per_iter
    dec_s = dec_per_iter / dt if dt > 0 else 0.0
    vram = torch.cuda.max_memory_allocated() / 1e9 if device_type == "cuda" else 0.0
    if iter_num % log_interval == 0 and master_process:
        print(f"iter {iter_num}: loss {lossf:.4f} {dt * 1e3:.0f}ms {dec_s:.0f} dec/s vram {vram:.2f}G")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), iter_num, decisions_seen,
                                    f"{lossf:.6f}", "", "", "", "", f"{head_lr:.8f}",
                                    f"{float(gn):.4f}", f"{dec_s:.1f}", f"{vram:.3f}"])
    iter_num += 1
    if iter_num > max_iters:
        break

if master_process:
    # kalibrasi di val (cap 400) -> temps.json + ckpt final
    raw_model.eval()
    rows, gold, qtypes = [], [], []
    for r in val_items[:400]:
        ids = torch.tensor([r["input_ids"]], dtype=torch.long, device=device)
        with ctx:
            logits, _ = raw_model(ids, torch.ones_like(ids), [{"questions": r["rec"]}])
        for j, q in enumerate(r["rec"]):
            rows.append(logits[0, j, : len(q["oids"])].detach().float().cpu())
            gold.append(r["gold"][j])
            qtypes.append(q["type"])
    temps = calibrate(rows, gold, qtypes)
    m = metrics(rows, gold, qtypes, temps)
    print(f"calibrated temps={temps} acc={m['acc']:.3f} ece={m['ece']:.4f} loss={m['loss']:.4f}")
    import json as _json

    with open(os.path.join(out_dir, "temps.json"), "w") as f:
        _json.dump({"temps": temps, "metrics": m}, f, indent=2)
    save_deploy()
    save_ckpt()

if ddp:
    destroy_process_group()
