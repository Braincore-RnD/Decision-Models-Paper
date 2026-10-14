"""Baseline generatif #5b (Colab T4): backbone LFM2.5-350M yang SAMA, dipakai
generatif (prompt -> JSON) vs JointSchemaHead. Mengisolasi kontribusi head:
akurasi, ECE (confidence verbal), latensi O(T), violation rate.

Bobot base dari HF langsung (tanpa training). Subset-500 SEED=3407
(sampler sama dengan embed_baseline; full test ~5-12 jam, tak feasible
di VM rapuh — disclosed). Tahan-mati: RESUME=1 + save per 100 baris.

Jalan (per bench, ~30-45 mnt):
  BENCH=banking77 OUT=gen_banking.json python eval_generative.py
Env: MODEL default LiquidAI/LFM2.5-350M, SEED 3407, N 500 (typed 400),
  MAX_NEW 256, FEWSHOT 0/1 (default 0: zero-shot eksplisit), RESUME 0/1.
"""
from __future__ import annotations

import json
import os
import random
import re
import time

MODEL = os.environ.get("MODEL", "LiquidAI/LFM2.5-350M")
SEED = int(os.environ.get("SEED", "3407"))
N = int(os.environ.get("N", "500"))
OUT = os.environ.get("OUT", "gen_items.json")
BENCH = os.environ.get("BENCH", "banking77")
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
FEWSHOT = os.environ.get("FEWSHOT", "0") == "1"
RESUME = os.environ.get("RESUME", "0") == "1"

from unsloth import FastLanguageModel  # noqa: E402

model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=4096,
                                               load_in_4bit=True)

FEW = ('Message: "I forgot my PIN, how do I reset it?"\nOptions: [change_pin, card_arrival]\n'
       'Answer: {"choice": "change_pin", "confidence": 0.9}\n\n')


def prompt_for(state, question, options):
    opts = "\n".join(f"- {k}: {v}" for k, v in options.items())
    ex = FEW if FEWSHOT else ""
    return (f"{ex}Message: {state}\n{question}\nOptions:\n{opts}\n"
            f'Answer with JSON only: {{"choice": "<option name>", "confidence": 0.0-1.0}}')


def parse(text, options):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None, "no_json"
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None, None, "bad_json"
    ch = o.get("choice")
    if ch not in options:
        return None, None, "bad_key"
    try:
        cf = float(o.get("confidence"))
        cf = min(max(cf, 0.0), 1.0)
    except Exception:
        return ch, None, "no_conf"
    return ch, cf, "ok"


def load_rows(bench, n):
    from datasets import load_dataset

    rng = random.Random(SEED)
    rows = []
    if bench == "banking77":
        ds = load_dataset("legacy-datasets/banking77", split="test")
        names = ds.features["label"].names
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            rows.append((r["text"], "Which banking intent does this message express?",
                         {k: k.replace("_", " ") for k in names}, names[r["label"]]))
    elif bench == "clinc150":
        ds = load_dataset("clinc/clinc_oos", "plus", split="test")
        names = ds.features["intent"].names
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            r = ds[i]
            rows.append((r["text"], "What is the intent of this customer inquiry?",
                         {k: k.replace("_", " ") for k in names}, names[r["intent"]]))
    else:
        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
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
                    rows.append((r["state"], q.get("instructions", "Select best option:"),
                                 {k: str(v) for k, v in q["criteria"].items()}, str(g)))
                    break
    return rows


def main():
    dump = []
    if RESUME and os.path.exists(OUT):
        dump = json.load(open(OUT))
        print(f"resume: {len(dump)} rows", flush=True)
    rows = load_rows(BENCH, N if BENCH != "typed_decisions" else 400)[len(dump):]
    n_ok, n_viol, confs, oks, dt = 0, 0, [], [], 0.0
    for j, (state, q, opts, gold) in enumerate(rows):
        pr = prompt_for(state, q, opts)
        ids = tok(pr, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        gen = model.generate(**ids, max_new_tokens=MAX_NEW, temperature=0.0,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        dt += time.perf_counter() - t0
        text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        ch, cf, status = parse(text, opts)
        if ch is None:
            n_viol += 1
        ok = (ch == gold)
        n_ok += ok
        if cf is not None:
            confs.append(cf)
            oks.append(ok)
        K = len(opts)
        probs = ({o: (cf if o == ch else (1 - cf) / max(K - 1, 1)) for o in opts}
                 if ch and cf is not None else {})
        dump.append({"bench": BENCH, "gold": {"q": gold},
                     "answers": {"q": {"type": "choice", "choice": ch, "confidence": cf,
                                       "probabilities": probs, "answer": ch}},
                     "gen_status": status, "gen_text": text[:300]})
        if (len(dump)) % 100 == 0:
            json.dump(dump, open(OUT, "w"), default=str)
            print(f"{len(dump)} rows, acc {(n_ok/max(len(dump),1)):.3f}", flush=True)
    n = len(dump)
    ece, nb = 0.0, 10
    for b in range(nb):
        pts = [(c, o) for c, o in zip(confs, oks) if min(int(c * nb), nb - 1) == b]
        if pts:
            ece += len(pts) / max(len(confs), 1) * abs(sum(c for c, _ in pts) / len(pts)
                                                       - sum(o for _, o in pts) / len(pts))
    summ = {"bench": BENCH, "n": n, "acc": n_ok / max(n, 1),
            "violation_rate": n_viol / max(n, 1), "ece10": ece,
            "n_conf": len(confs), "ms_per_row": 1000 * dt / max(n, 1)}
    json.dump(dump, open(OUT, "w"), default=str)
    json.dump(summ, open(OUT.replace(".json", "_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
