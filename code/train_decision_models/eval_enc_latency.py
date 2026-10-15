"""Latency encoder #18/#21 (Colab T4): bobot HF langsung, tanpa training.

Ukur per-row latency + peak mem vs jumlah opsi (prefill cost), throughput
batch-16, dan sanity acc — untuk ModernBERT-baseline #5a.

Jalan (~20 mnt, download JSON langsung setelah selesai):
  OUT=/content/enc_lat.json nohup python3 eval_enc_latency.py > enc_lat.log 2>&1 &
Env: HF_REPO default itsdevice/modernbert-systemone, SEED 3407.
"""
from __future__ import annotations

import json
import os
import random
import time

HF_REPO = os.environ.get("HF_REPO", "itsdevice/modernbert-systemone")
SEED = int(os.environ.get("SEED", "3407"))
OUT = os.environ.get("OUT", "enc_lat.json")
JOBS = [j.strip() for j in os.environ.get("JOBS", "all").split(",")]
BS = int(os.environ.get("BS", "16"))  # ponytail: BS=32/64 untuk throughput batch #21
print("eval_enc_latency SCRIPT_V=4 keepall-validonly", flush=True)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


class Head(nn.Module):
    def __init__(self, d, proj=256):
        super().__init__()
        self.state_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, proj), nn.GELU(), nn.Linear(proj, proj))
        self.opt_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, proj), nn.GELU(), nn.Linear(proj, proj))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.log_temp = nn.Parameter(torch.zeros(()))


def fmt_row(tok, state, question, criteria, gold=None, max_len=4096):
    mt = tok.mask_token
    lines = [f"State: {state.strip()}", f"Question: {question.strip()}", "Options:"]
    keys = list(criteria.keys())
    for k, v in criteria.items():
        lines.append(f"{mt} {k}: {str(v).strip() if v else k}")
    enc = tok("\n".join(lines), max_length=max_len, truncation=True, return_tensors="pt")
    ids = enc["input_ids"][0]
    mi = (ids == tok.mask_token_id).nonzero(as_tuple=True)[0].tolist()
    if len(mi) != len(keys):
        return None
    return {"input_ids": ids, "mask": enc["attention_mask"][0], "pos": mi,
            "options": keys, "label": keys.index(gold) if gold in keys else -1,
            "ntok": len(ids)}


def collate(batch, pad):
    ms = max(len(b["input_ids"]) for b in batch)
    mo = max(len(b["pos"]) for b in batch)
    B = len(batch)
    ids = torch.full((B, ms), pad, dtype=torch.long)
    am = torch.zeros((B, ms), dtype=torch.long)
    pos = torch.zeros((B, mo), dtype=torch.long)
    val = torch.zeros((B, mo), dtype=torch.bool)
    lab = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = b["input_ids"]
        am[i, :L] = b["mask"]
        n = len(b["pos"])
        pos[i, :n] = torch.tensor(b["pos"])
        val[i, :n] = True
        lab[i] = b["label"]
    return ids, am, pos, val, lab


@torch.no_grad()
def fwd(model, ids, am, pos, val):
    H = model.backbone(ids, attention_mask=am).last_hidden_state
    B, mo, D = pos.shape[0], pos.shape[1], H.size(-1)
    sv = F.normalize(model.head.state_proj(H[:, 0, :]), dim=-1)
    ov = F.normalize(model.head.opt_proj(torch.gather(H, 1, pos.unsqueeze(-1).expand(B, mo, D))), dim=-1)
    return torch.bmm(ov, sv.unsqueeze(-1)).squeeze(-1) * model.head.scale / model.head.log_temp.exp()


class M(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head


def main():
    from datasets import load_dataset

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(HF_REPO)
    bb = AutoModel.from_pretrained(HF_REPO).to(dev).eval()
    head = Head(bb.config.hidden_size)
    from huggingface_hub import hf_hub_download

    hp = hf_hub_download(HF_REPO, "decision_head.pt")
    meta = json.load(open(hf_hub_download(HF_REPO, "decision_meta.json")))
    hsd = torch.load(hp, map_location="cpu")
    with torch.no_grad():
        head.state_proj.load_state_dict(hsd["state_proj"])
        head.opt_proj.load_state_dict(hsd["opt_proj"])
        head.scale.copy_(torch.tensor(float(hsd["scale"])))
        head.log_temp.copy_(torch.tensor(float(__import__("math").log(meta["temperature"]))))
    model = M(bb, head).to(dev).eval()
    rng = random.Random(SEED)
    out = {}

    def bench(name, rows):
        torch.cuda.reset_peak_memory_stats()
        ds = [{"input_ids": r["input_ids"], "mask": r["mask"], "pos": r["pos"],
               "options": r["options"], "label": r["label"], "ntok": r["ntok"]} for r in rows]
        t0 = time.perf_counter()
        ok, n, n_timed = 0, 0, 0
        for i in range(0, len(ds), BS):
            ch = ds[i:i + BS]
            pad = tok.pad_token_id or 0
            ms = max(len(b["input_ids"]) for b in ch)
            B = len(ch)
            ids = torch.full((B, ms), pad, dtype=torch.long)
            am = torch.zeros((B, ms), dtype=torch.long)
            for j, b in enumerate(ch):
                L = len(b["input_ids"])
                ids[j, :L] = b["input_ids"]
                am[j, :L] = b["mask"]
            mo = max(len(b["pos"]) for b in ch)
            pos = torch.zeros((B, mo), dtype=torch.long)
            val = torch.zeros((B, mo), dtype=torch.bool)
            lab = torch.zeros(B, dtype=torch.long)
            for j, b in enumerate(ch):
                nopt = len(b["pos"])
                pos[j, :nopt] = torch.tensor(b["pos"])
                val[j, :nopt] = True
                lab[j] = b["label"]
            lg = fwd(model, ids.to(dev), am.to(dev), pos.to(dev), val.to(dev))
            pr = lg.argmax(-1).cpu()
            good = lab >= 0
            ok += (pr[good] == lab[good]).sum().item()
            n += int(good.sum())
            n_timed += B
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return {"acc": ok / max(n, 1), "n": n, "ms_per_row": 1000 * dt / max(n_timed, 1),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                "mean_ntok": sum(b["ntok"] for b in ds) / max(len(ds), 1)}

    def load(name, limit, k=None):
        rows = []
        if name == "banking77":
            ds = load_dataset("legacy-datasets/banking77", split="test")
            names = ds.features["label"].names
            idx = rng.sample(range(len(ds)), min(limit, len(ds)))
            for i in idx:
                r = ds[i]
                crit = {x: x.replace("_", " ") for x in names}
                if k:
                    crit = dict(list(crit.items())[:k])
                s = fmt_row(tok, r["text"], "Which banking intent?", crit, names[r["label"]])
                if s and (s["label"] != -1 or k):
                    rows.append(s)
        else:
            ds = load_dataset("clinc/clinc_oos", "plus", split="test")
            names = ds.features["intent"].names
            idx = rng.sample(range(len(ds)), min(limit, len(ds)))
            for i in idx:
                r = ds[i]
                crit = {x: x.replace("_", " ") for x in names}
                if k:
                    crit = dict(list(crit.items())[:k])
                s = fmt_row(tok, r["text"], "What is the intent?", crit, names[r["intent"]])
                if s and (s["label"] != -1 or k):
                    rows.append(s)
        return rows

    if JOBS == ["all"] or "bank" in JOBS:
        out["bank_full"] = bench("b", load("banking77", 3080))
    if JOBS == ["all"] or "clinc" in JOBS:
        out["clinc_full"] = bench("c", load("clinc150", 5500))
    if JOBS == ["all"] or "topk" in JOBS:
        out["bank_top8"] = bench("b8", load("banking77", 500, 8))
        out["bank_top24"] = bench("b24", load("banking77", 500, 24))
    for k in sorted(out):
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in out[k].items()
                   if kk != "mean_ntok"}, "ntok", round(out[k]["mean_ntok"]), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"saved {OUT}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
