"""Baseline encoder #5a: ModernBERT-base + OptionMarker head, protokol identik.

Reuse arsitektur dari train_modernbert_dec.py; skrip ini mengisi yang kurang:
mixture resmi (typed-choice + banking77 + clinc150 train), LR diferensiasi,
best-val selection, kalibrasi suhu, eval FULL + subset-500 identik, dump
per-item format items_pred.json (skor via score_preds.py), OOS slice,
run_meta.json buat RUN_LOG.

Aturan main (#5 + AGENTS.md):
- Train split resmi vs test resmi = disjoint by construction (tanpa shingle).
- max_len 4096 + skip overflow tercatat (invarian yang sama dengan paper).
- Subset-500: random.Random(3407), sampler sama dengan embed_baseline.py.
  Full test = eksak, tanpa ambiguitas sampler.

Jalan (Colab T4, ~1-2 jam):
  python train_encoder_baseline.py --out out-modernbert
  python train_encoder_baseline.py --eval-only --final out-modernbert/final
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_modernbert_dec import (  # noqa: E402
    InMemoryDecisionDataset,
    OptionMarkerDecisionModel,
    calibrate_model_temperature,
    collate_decision_batch,
    format_sample,
)

SEED = 3407
MAX_LEN = 4096


def load_mixture(tokenizer):
    """(state, question, criteria, gold) dari 3 train split resmi."""
    from datasets import load_dataset

    rng = random.Random(SEED)
    rows, skipped = [], {"over_ctx": 0}

    def add(state, question, criteria, gold):
        if not criteria or gold not in criteria:
            return
        s = format_sample(tokenizer, state, question, criteria, gold, max_len=MAX_LEN)
        if s is None or s["label"] == -1:
            skipped["over_ctx"] += 1
            return
        rows.append(s)

    td = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    for r in td:
        qs = r["questions"]
        gs = r["gold"]
        if isinstance(qs, str):
            qs = json.loads(qs)
        if isinstance(gs, str):
            gs = json.loads(gs)
        for qid, q in qs.items():
            if not isinstance(q, dict) or q.get("type") != "choice":
                continue
            g = gs.get(qid)
            g = (g.get("label") or g.get("choice") or g.get("answer")) if isinstance(g, dict) else g
            if g and str(g) in q["criteria"]:
                add(r["state"], q.get("instructions", "Select best option:"),
                    {k: str(v) for k, v in q["criteria"].items()}, str(g))

    b77 = load_dataset("legacy-datasets/banking77", split="train")
    names = b77.features["label"].names
    for i in rng.sample(range(len(b77)), len(b77)):
        r = b77[i]
        gold = names[r["label"]]
        add(r["text"], "Which banking intent does this message express?",
            {k: k.replace("_", " ") for k in names}, gold)

    cl = load_dataset("clinc/clinc_oos", "plus", split="train")
    inames = cl.features["intent"].names
    for i in rng.sample(range(len(cl)), len(cl)):
        r = cl[i]
        gold = inames[r["intent"]]
        add(r["text"], "What is the intent of this customer inquiry?",
            {k: k.replace("_", " ") for k in inames}, gold)

    rng.shuffle(rows)
    n_hold = min(400, len(rows) // 10)
    hold, train = rows[:n_hold], rows[n_hold:]
    n_cal = len(hold) // 2
    return train, hold[:n_cal], hold[n_cal:], dict(skipped, n_rows=len(rows))


def make_loaders(tokenizer, train, cal, test, bs):
    pad = tokenizer.pad_token_id or 0
    kw = dict(batch_size=bs, collate_fn=lambda b: collate_decision_batch(b, pad))
    return (DataLoader(InMemoryDecisionDataset(train), shuffle=True, **kw),
            DataLoader(InMemoryDecisionDataset(cal), shuffle=False, **kw),
            DataLoader(InMemoryDecisionDataset(test), shuffle=False, **kw))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ok, n, confs, oks, probs_all, golds = 0, 0, [], [], [], []
    t0 = time.perf_counter()
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["mask_positions"].to(device), batch["mask_valid"].to(device))
        p = out["probabilities"].cpu()
        pred = p.argmax(-1)
        ok += (pred == batch["labels"]).sum().item()
        n += len(pred)
        confs += p.max(-1).values.tolist()
        oks += (pred == batch["labels"]).tolist()
        probs_all += p.tolist()
        golds += batch["labels"].tolist()
    dt = time.perf_counter() - t0
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return {"acc": ok / max(n, 1), "n": n, "ms_per_item": 1000 * dt / max(n, 1),
            "confs": confs, "oks": oks, "probs": probs_all, "golds": golds}


def build_test(tokenizer, bench, limit):
    """Baris test resmi -> samples + gold label string per baris."""
    from datasets import load_dataset

    rng = random.Random(SEED)
    out = []
    if bench == "banking77":
        ds = load_dataset("legacy-datasets/banking77", split="test")
        names = ds.features["label"].names
        idx = rng.sample(range(len(ds)), min(limit or len(ds), len(ds)))
        for i in idx:
            r = ds[i]
            out.append(("banking77", r["text"], "Which banking intent does this message express?",
                        {k: k.replace("_", " ") for k in names}, names[r["label"]]))
    elif bench == "clinc150":
        ds = load_dataset("clinc/clinc_oos", "plus", split="test")
        names = ds.features["intent"].names
        idx = rng.sample(range(len(ds)), min(limit or len(ds), len(ds)))
        for i in idx:
            r = ds[i]
            out.append(("clinc150", r["text"], "What is the intent of this customer inquiry?",
                        {k: k.replace("_", " ") for k in names}, names[r["intent"]]))
    else:
        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
        idx = rng.sample(range(len(ds)), min(limit or len(ds), len(ds)))
        for i in idx:
            r = ds[i]
            qs, gs = r["questions"], r["gold"]
            if isinstance(qs, str):
                qs = json.loads(qs)
            if isinstance(gs, str):
                gs = json.loads(gs)
            for qid, q in qs.items():
                if not isinstance(q, dict) or q.get("type") != "choice":
                    continue
                g = gs.get(qid)
                g = (g.get("label") or g.get("choice") or g.get("answer")) if isinstance(g, dict) else g
                if g and str(g) in q["criteria"]:
                    out.append((f"typed:{qid}", r["state"], q.get("instructions", "Select best option:"),
                                {k: str(v) for k, v in q["criteria"].items()}, str(g)))
    samples = []
    for bench_tag, state, q, crit, gold in out:
        s = format_sample(tokenizer, state, q, crit, gold, max_len=MAX_LEN)
        if s is not None and s["label"] != -1:
            s["_bench"], s["_gold"] = bench, gold
            s["_qid"] = bench_tag.split(":", 1)[1] if ":" in bench_tag else "intent"
            samples.append(s)
    return samples


def dump_items(samples, eval_res, bench):
    """Format items_pred.json (kompatibel score_preds.collect)."""
    rows, k = [], 0
    prob_rows = eval_res["probs"]
    for s in samples:
        p = {str(o): float(v) for o, v in zip(s["options"], prob_rows[k])}
        pred = max(p, key=p.__getitem__)
        qid = s.get("_qid", "intent")
        rows.append({"bench": bench, "gold": {qid: s["_gold"]},
                     "answers": {qid: {"type": "choice", "choice": pred, "confidence": p[pred],
                                       "probabilities": p, "answer": pred}}})
        k += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="out-modernbert")
    ap.add_argument("--eval-only", default="")
    ap.add_argument("--hf-repo", default="")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)

    if args.eval_only:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(args.eval_only)
        backbone = AutoModel.from_pretrained(args.eval_only)
        model = OptionMarkerDecisionModel(backbone, backbone.config.hidden_size)
        model.load_state_dict(torch.load(os.path.join(args.eval_only, "head.pt"), map_location=device),
                              strict=False)
        meta = json.load(open(os.path.join(args.eval_only, "run_meta.json")))
        model.log_temp.copy_(torch.tensor(math.log(meta["temperature"]), device=device))
    else:
        backbone = AutoModel.from_pretrained(args.model_id)
        model = OptionMarkerDecisionModel(backbone, backbone.config.hidden_size).to(device)
        train, cal, test = load_mixture(tok)
        tr_loader, cal_loader, te_loader = make_loaders(tok, train, cal, test, args.bs)
        opt = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": list(model.head.parameters()) + [model.log_temp], "lr": args.lr_head},
        ], weight_decay=0.01)
        best, best_state, patience = -1.0, None, 0
        for ep in range(1, args.epochs + 1):
            model.train()
            tot, n = 0.0, 0
            for b in tr_loader:
                opt.zero_grad()
                loss = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                             b["mask_positions"].to(device), b["mask_valid"].to(device),
                             labels=b["labels"].to(device))["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
                n += 1
            v = evaluate(model, te_loader, device)
            print(f"ep{ep}: loss {tot / max(n,1):.4f} | internal-test {v['acc']:.4f}", flush=True)
            if v["acc"] > best:
                best, patience = v["acc"], 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            else:
                patience += 1
                if patience >= 2:
                    print("early stop", flush=True)
                    break
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        T = calibrate_model_temperature(model, cal_loader, device)
        fin = os.path.join(args.out, "final")
        os.makedirs(fin, exist_ok=True)
        backbone.save_pretrained(fin)
        tok.save_pretrained(fin)
        torch.save(model.head.state_dict(), os.path.join(fin, "head.pt"))
        try:
            gh = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            gh = None
        json.dump({"model": args.model_id, "seed": args.seed, "epochs": args.epochs,
                   "temperature": T, "best_internal": best, "git_hash": gh,
                   "weight_decay": 0.01, "label_smoothing": 0.05},
                  open(os.path.join(fin, "run_meta.json"), "w"), indent=1)
        if args.hf_repo:
            from huggingface_hub import HfApi

            HfApi().upload_folder(folder_path=fin, repo_id=args.hf_repo, create_pr=False)
            print("pushed", args.hf_repo, flush=True)

    model.eval()
    model.to(device)
    dump, summ = [], {}
    for bench, lim in (("banking77", None), ("clinc150", None), ("typed_decisions", None),
                       ("banking77", 500), ("clinc150", 500), ("typed_decisions", 400)):
        tag = f"{bench}-{'full' if lim is None else lim}"
        samples = build_test(tok, bench, lim)
        pad = tok.pad_token_id or 0
        loader = DataLoader(InMemoryDecisionDataset(samples), batch_size=args.bs, shuffle=False,
                            collate_fn=lambda b: collate_decision_batch(b, pad))
        t0 = time.perf_counter()
        r = evaluate(model, loader, device)
        ms = 1000 * (time.perf_counter() - t0) / max(r["n"], 1)
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        oos = None
        if bench == "clinc150":
            oo = [s for s in samples if s["_gold"] == "oos"]
            if oo:
                lo = DataLoader(InMemoryDecisionDataset(oo), batch_size=args.bs, shuffle=False,
                                collate_fn=lambda b: collate_decision_batch(b, pad))
                oos = evaluate(model, lo, device)["acc"]
        summ[tag] = {"acc": r["acc"], "n": r["n"], "ms_per_item": ms, "peak_mem_gb": mem, "oos_recall": oos}
        dump += dump_items(samples, r, bench)
        print(tag, round(r["acc"], 4), r["n"], f"{ms:.0f}ms", flush=True)
    json.dump(dump, open(os.path.join(args.out, "items_pred.json"), "w"), default=str)
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("saved", args.out, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
