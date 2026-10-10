"""Siapkan data decision: JSONL -> train.pt / val.pt berisi record ter-render.

Baris JSONL: {"state": str|dict|JSON-str, "questions": {...}, "gold": {qid: label}}
  label: choice -> key opsi | noul -> true/false | score -> int level.
typed-decisions (parquet) punya state/questions/gold sebagai JSON-string;
--from_hf mengunduhnya (butuh internet + pandas) lalu jalur sama.

Smoke tanpa data/network:  --smoke  (tulis 64 baris sintetik).

Dipakai train_dec.py --data_dir=<dir ini>.
Konvensi CLI = train.py:  --key=value.
"""
from __future__ import annotations

import json
import os
import sys
from ast import literal_eval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from head import QTYPES, encode_record, question_options

src_jsonl = ""
from_hf = ""
tokenizer_id = "LiquidAI/LFM2.5-350M"
out_dir = "code/data"
seed = 42
val_fraction = 0.1
max_val = 400
max_len = 1024
max_rows = 0  # 0 = semua
smoke = False

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

g = torch.Generator().manual_seed(seed)


class HFEnc:
    """Adapter: interface encode_ordinary ala tiktoken di atas tokenizer HF."""

    def __init__(self, tok):
        self.t = tok

    def encode_ordinary(self, s):
        return self.t.encode(s, add_special_tokens=False)


try:
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if _tok.pad_token_id is None:
        _tok.pad_token = _tok.eos_token
    enc = HFEnc(_tok)
    print(f"tokenizer {tokenizer_id} vocab={len(_tok)} pad={_tok.pad_token_id}")
except Exception:
    print("butuh transformers: pip install 'transformers>=5.2.0'")
    raise SystemExit(1)


def as_obj(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def label_to_index(q, label):
    opts = question_options(q)
    if q["type"] == "choice":
        return [o for o, _ in opts].index(str(label))
    if q["type"] == "noul":
        return 0 if (label is True or str(label).lower() == "true") else 1
    return int(label)


def build_rows(jsonl_path):
    items, skipped = [], 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                state, questions = as_obj(r["state"]), as_obj(r["questions"])
                gold = as_obj(r.get("gold", {}))
                rec = encode_record(enc, state, questions, max_len=max_len)
                no = max(len(q["oids"]) for q in rec["questions"])
                nq = len(rec["questions"])
                target = torch.zeros(nq, no)
                qtypes, ordinal, gidx = [], [], []
                for j, q in enumerate(rec["questions"]):
                    asked = questions[q["id"]]
                    gi = label_to_index(asked, gold[q["id"]] if isinstance(gold, dict) else gold)
                    target[j, gi] = 1.0
                    qtypes.append(q["type"])
                    ordinal.append(asked["type"] == "score")
                    gidx.append(gi)
                items.append({"input_ids": rec["input_ids"], "rec": rec["questions"],
                              "target": target, "qtypes": qtypes, "ordinal": ordinal, "gold": gidx,
                              "asked": {q["id"]: questions[q["id"]] for q in rec["questions"]}})
            except Exception as e:
                skipped += 1
                print(f"skip: {e}")
            if max_rows and len(items) >= max_rows:
                break
    print(f"kept {len(items)}, skipped {skipped}")
    return items


def write_splits(items):
    os.makedirs(out_dir, exist_ok=True)
    perm = torch.randperm(len(items), generator=g).tolist()
    nv = min(max_val, int(len(items) * val_fraction))
    val_idx = set(perm[:nv])
    train = [items[i] for i in range(len(items)) if i not in val_idx]
    val = [items[i] for i in perm[:nv]]
    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val, os.path.join(out_dir, "val.pt"))
    print(f"train {len(train)} -> {out_dir}/train.pt, val {len(val)} -> {out_dir}/val.pt")


def main():
    global src_jsonl
    if smoke:
        os.makedirs(out_dir, exist_ok=True)
        demo_q = {"team": {"type": "choice", "instructions": "tim mana?",
                           "criteria": {"a": "tagihan", "b": "teknis"}},
                  "ok": {"type": "noul", "instructions": "minta refund?"}}
        with open(os.path.join(out_dir, "smoke.jsonl"), "w") as f:
            for i in range(64):
                f.write(json.dumps({"state": f"kasus {i}: tagihan ganda" if i % 2 == 0 else f"kasus {i}: error server",
                                    "questions": demo_q,
                                    "gold": {"team": "a" if i % 2 == 0 else "b", "ok": i % 2 == 0}}) + "\n")
        src_jsonl = os.path.join(out_dir, "smoke.jsonl")

    if from_hf:
        # typed-decisions: <subset>/train|test parquet. Butuh: pip install huggingface_hub pandas pyarrow
        from huggingface_hub import hf_hub_download

        ds, subset = (from_hf.split(":", 1) + ["all"])[:2]
        os.makedirs(out_dir, exist_ok=True)
        import pandas as pd

        frames = []
        for split in ("train", "test"):
            pq = hf_hub_download(ds, f"{subset}/{split}.parquet", repo_type="dataset")
            frames.append(pd.read_parquet(pq))
        df = pd.concat(frames, ignore_index=True)
        path = os.path.join(out_dir, "hf.jsonl")
        with open(path, "w") as f:
            for _, r in df.iterrows():
                gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]

                def hard(v):
                    if isinstance(v, dict):
                        if "label" in v:
                            return v["label"]
                        p = v.get("probabilities") or {}
                        return max(p, key=p.__getitem__) if p else v
                    return v

                f.write(json.dumps({"state": r["state"], "questions": r["questions"],
                                    "gold": {k: hard(v) for k, v in gold.items()}}) + "\n")
        src_jsonl = path

    if not src_jsonl:
        print("isi --src_jsonl=<file> | --from_hf=<ds[:subset]> | --smoke")
        raise SystemExit(1)
    write_splits(build_rows(src_jsonl))


if __name__ == "__main__":
    main()
