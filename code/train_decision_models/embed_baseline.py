"""Baseline zero-shot jujur untuk LOTO (#4): cosine similarity embedding.

Utterance test vs nama label taksonomi yang di-hold-out (tanpa training,
tanpa deskripsi label — nama saja, underscore jadi spasi). Klaim transfer
dilaporkan RELATIF ke baseline ini, bukan ke uniform chance.

Model: BAAI/bge-small-en-v1.5 (CPU). Data: HF banking77/clinc150 test رسمی.
SEED sampling = 3407 (subset 500 sama dengan paper) + full.

Jalan: uv run --with sentence-transformers code/train_decision_models/embed_baseline.py
Keluar: docs/eval/embed_baseline.json
"""
import json
import os
import random

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "docs/eval/embed_baseline.json")
SEED = 3407

from datasets import load_dataset  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def norm(name):
    return name.replace("_", " ")


def run(ds_name, cfg, split, text_col, label_col, limit):
    ds = load_dataset(ds_name, cfg, split=split)
    feat = ds.features[label_col]
    names = feat.names if hasattr(feat, "names") else None
    raw_labels = sorted(set(ds[label_col]))
    disp = [str(names[l] if names and isinstance(l, int) else l) for l in raw_labels]
    lab_emb = model.encode([norm(l) for l in disp], normalize_embeddings=True)
    rng = random.Random(SEED)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:limit] if limit else idx
    ok, n = 0, 0
    for i in idx:
        q = model.encode(ds[i][text_col], normalize_embeddings=True)
        pred = raw_labels[int((lab_emb @ q).argmax())]
        ok += pred == ds[i][label_col]
        n += 1
    return {"acc": ok / n, "n": n, "n_labels": len(raw_labels)}


out = {
    "banking77_sub500": run("legacy-datasets/banking77", None, "test", "text", "label", 500),
    "banking77_full": run("legacy-datasets/banking77", None, "test", "text", "label", None),
    "clinc150_sub500": run("clinc/clinc_oos", "plus", "test", "text", "intent", 500),
    "clinc150_full": run("clinc/clinc_oos", "plus", "test", "text", "intent", None),
}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
print(f"saved {OUT}")
