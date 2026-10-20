"""Redo dekontaminasi #10 (CPU-only): audit 13-gram + near-dup TF-IDF.
Replikasi setia logika lib Unsloth (regex \\w+, lowercase, n=13 + fallback whole-string).
Sumber: train/test split kanonis HF yg dipakai mixture (lihat SOURCES di decision_datasets.py).
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

OUT = Path("docs/eval/dekon")
OUT.mkdir(parents=True, exist_ok=True)

def words(s):
    return re.findall(r"\w+", s.lower())

def shingles(text, n=13):
    w = words(text)
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i+n]) for i in range(len(w)-n+1)}

from datasets import load_dataset
bank_tr = load_dataset("legacy-datasets/banking77", split="train")
bank_te = load_dataset("legacy-datasets/banking77", split="test")
clinc_tr = load_dataset("clinc/clinc_oos", "plus", split="train")
clinc_te = load_dataset("clinc/clinc_oos", "plus", split="test")
typed_tr = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
typed_te = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")

bt_tr = [r["text"] for r in bank_tr]; bt_te = [r["text"] for r in bank_te]
ct_tr = [r["text"] for r in clinc_tr]; ct_te = [r["text"] for r in clinc_te]

res = {}
def lenaudit(name, texts):
    n = len(texts); lens = [len(words(t)) for t in texts]
    import statistics
    under = sum(1 for L in lens if L < 13)
    res[name] = {"n": n, "mean_words": round(statistics.mean(lens), 2),
                 "median_words": statistics.median(lens),
                 "max_words": max(lens), "n_lt13": under,
                 "pct_lt13": round(100*under/n, 2)}

lenaudit("bank_test", bt_te); lenaudit("bank_train", bt_tr)
lenaudit("clinc_test", ct_te); lenaudit("clinc_train", ct_tr)

# (a2) lubang cakupan: dekon train hanya lawan 500/500 subsample seed 3407
rng_sub = lambda texts: [texts[i] for i in sorted(__import__("random").Random(3407).sample(range(len(texts)), 500))]
from random import Random
def seedsub(texts, k=500):
    return [texts[i] for i in sorted(Random(3407).sample(range(len(texts)), k))]
# catatan: load_source shuffle via datasets seed — aproksimasi dgn random.Random di sini;
# yg penting: hanya 500 dari 3080/5500 yg masuk eval_texts.
res["coverage_hole"] = {"bank_test_total": len(bt_te), "bank_dekon_only": 500,
                        "clinc_test_total": len(ct_te), "clinc_dekon_only": 500,
                        "note": "train script eval_texts = typed full + banking77/clinc150 limit 500 seed 3407"}

# (b1) exact-match ternormalisasi train-pool vs test (full pool: mixture sampling dari train)
def norm(t): return " ".join(words(t))
for tag, tr, te in [("bank", bt_tr, bt_te), ("clinc", ct_tr, ct_te)]:
    trset = set(norm(t) for t in tr)
    hits = sum(1 for t in te if norm(t) in trset)
    res[f"exact_{tag}_train_vs_test"] = {"hits": hits, "of": len(te),
                                        "pct": round(100*hits/len(te), 2)}
# cross-dataset
bset = set(norm(t) for t in bt_tr); cset = set(norm(t) for t in ct_tr)
res["exact_cross"] = {
    "bank_train_vs_clinc_test": sum(1 for t in ct_te if norm(t) in bset),
    "clinc_train_vs_bank_test": sum(1 for t in bt_te if norm(t) in cset),
    "bank_train_vs_bank_test_dup": res["exact_bank_train_vs_test"]["hits"],
}
# typed: state kompleks (dict) — cek duplikasi id/state serial
import json as J
def tkey(r):
    s = r["state"]
    try: s = J.loads(s) if isinstance(s, str) and s.strip()[:1] in "{[" else s
    except Exception: pass
    return J.dumps(s, sort_keys=True, ensure_ascii=False)
trkeys = set(tkey(r) for r in typed_tr)
thits = sum(1 for r in typed_te if tkey(r) in trkeys)
res["exact_typed_train_vs_test"] = {"hits": thits, "of": len(typed_te)}

# (b2) 8-gram overlap: fraksi test yg berbagi >=1 8-gram dgn train pool
def eight_idx(texts):
    idx = defaultdict(set)
    for j, t in enumerate(texts):
        w = words(t)
        for i in range(len(w)-7):
            idx[" ".join(w[i:i+8])].add(j)
    return idx
for tag, tr, te in [("bank", bt_tr, bt_te), ("clinc", ct_tr, ct_te)]:
    idx = eight_idx(tr)
    hits = sum(1 for t in te if any(" ".join(words(t)[i:i+8]) in idx
                                    for i in range(len(words(t))-7)))
    res[f"eightgram_{tag}_train_vs_test"] = {"hits": hits, "of": len(te),
                                             "pct": round(100*hits/len(te), 2)}

# (b3) TF-IDF cosine > 0.9 train-pool vs test
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
def neardup(tr, te, tag):
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    A = vec.fit_transform(tr); B = vec.transform(te)
    sims = (B @ A.T).tocsr()
    mx = np.asarray(sims.max(axis=1).todense()).ravel()
    for th in (0.9, 0.95, 1.0):
        n = int((mx >= th-1e-9).sum())
        res.setdefault(f"tfidf_{tag}", {})[f"ge_{th}"] = {"n": n, "pct": round(100*n/len(te), 2)}
    res[f"tfidf_{tag}"]["max_mean"] = round(float(mx.mean()), 4)
neardup(bt_tr, bt_te, "bank")
neardup(ct_tr, ct_te, "clinc")
# cross: bank_train vs clinc_test (sinyal mirror antar-taksonomi)
vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
A = vec.fit_transform(bt_tr); B = vec.transform(ct_te)
mx = np.asarray(((B @ A.T).tocsr()).max(axis=1).todense()).ravel()
res["tfidf_banktrain_vs_clinctest"] = {"ge_0.9": int((mx >= 0.9-1e-9).sum()), "max_mean": round(float(mx.mean()), 4)}
A2 = vec.fit_transform(ct_tr); B2 = vec.transform(bt_te)
mx2 = np.asarray(((B2 @ A2.T).tocsr()).max(axis=1).todense()).ravel()
res["tfidf_clinctrain_vs_banktest"] = {"ge_0.9": int((mx2 >= 0.9-1e-9).sum()), "max_mean": round(float(mx2.mean()), 4)}

(OUT/"dekon_redo.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
