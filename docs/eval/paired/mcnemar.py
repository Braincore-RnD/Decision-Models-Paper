"""McNemar exact paired tests for the paper's head-to-head accuracy claims.

Stdlib only. Compares per-item correctness of two dumps sharing the same
test items in the same order (gold sequences verified identical, except
ours-vs-encoder BANKING77 where 80 rows differ only in label-string
normalisation -- pairing is still by row index, correctness judged against
each file's own gold labels).

Run:  python3 docs/eval/paired/mcnemar.py
Out:  docs/eval/paired/results.json
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def correct(item):
    """Prediction choice == gold value, whatever the field names are."""
    gold = list(item["gold"].values())[0]
    ans = item["answers"]
    v = ans[list(ans.keys())[0]]
    pred = v.get("choice") if isinstance(v, dict) else v
    return pred == gold


def subset(items, bench):
    return [x for x in items if x.get("bench") == bench]


def binom_pmf(n, k):
    # log-domain: comb(n,k) overflows float for n~1000
    logp = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    logp -= n * math.log(2)
    return math.exp(logp)


def exact_two_sided(b, c):
    """Two-sided exact McNemar p via Bin(b+c, 0.5); b,c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    p = 2.0 * sum(binom_pmf(n, k) for k in range(m + 1))
    return min(p, 1.0)


def mcnemar(a_corr, b_corr, label_a="A", label_b="B"):
    assert len(a_corr) == len(b_corr)
    n = len(a_corr)
    b = sum(1 for a, e in zip(a_corr, b_corr) if a and not e)  # A right, B wrong
    c = sum(1 for a, e in zip(a_corr, b_corr) if not a and e)  # A wrong, B right
    acc_a = sum(a_corr) / n
    acc_b = sum(b_corr) / n
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    p_asymp = math.erfc(math.sqrt(chi2 / 2)) if (b + c) > 0 else 1.0
    return {
        "n": n,
        label_a + "_acc": round(acc_a, 4),
        label_b + "_acc": round(acc_b, 4),
        "b_" + label_a + "_only": b,
        "c_" + label_b + "_only": c,
        "chi2_cc": round(chi2, 3),
        "p_asymp": round(p_asymp, 4) if p_asymp >= 0.0001 else "<0.0001",
        "p_exact": round(exact_two_sided(b, c), 4)
        if exact_two_sided(b, c) >= 0.0001
        else "<0.0001",
    }


results = {}

# 1. A-vs-E LOTO BANKING77 (paper's +8.6pt transfer-credit claim).
a_bank = subset(load("docs/eval/kaggle/A-loto/items_A_loto_bank.json"), "banking77")
e_bank = subset(load("docs/eval/colab/loto-banking/items_pred.json"), "banking77")
ga = [list(x["gold"].values())[0] for x in a_bank]
ge = [list(x["gold"].values())[0] for x in e_bank]
assert ga == ge, "A/E BANKING77 gold sequences differ -- not pairable"
results["loto_bank_A_vs_E"] = mcnemar(
    [correct(x) for x in a_bank], [correct(x) for x in e_bank],
    label_a="A_canon", label_b="E_aug",
)

# 2. A-vs-E LOTO CLINC150: E per-item dump not archived (only aggregate
# eval_results.json), so no paired test is possible. Recorded as n/a.
results["loto_clinc_A_vs_E"] = {"n/a": "E LOTO-clinc per-item dump not archived"}

# 3. Ours (E) vs fine-tuned ModernBERT on official full tests.
ours = load("docs/eval/kaggle/E-full/full13.json")
enc = load("docs/eval/encoder-modernbert/items_pred.json")
for bench in ("banking77", "clinc150"):
    o = subset(ours, bench)
    m = subset(enc, bench)
    assert len(o) == len(m), (bench, len(o), len(m))
    go = [list(x["gold"].values())[0] for x in o]
    gm = [list(x["gold"].values())[0] for x in m]
    note = "gold order identical" if go == gm else "80 rows differ in label-string normalisation only; paired by row index"
    r = mcnemar([correct(x) for x in o], [correct(x) for x in m],
                label_a="ours_E", label_b="modernbert")
    r["note"] = note
    results["fulltest_" + bench[:4] + "_ours_vs_enc"] = r

# 4. Hybrid (E in-dist 500-subset) vs dense SmolLM2-360M same-recipe subset.
abl = subset(load("docs/eval/colab/ablasi-full/items_pred.json"), "banking77"), \
    subset(load("docs/eval/colab/ablasi-full/items_pred.json"), "clinc150")
smo = load("docs/eval/kaggle/smollm2/items_smollm2.json")
for bench, ab in (("banking77", abl[0]), ("clinc150", abl[1])):
    sm = subset(smo, bench)
    ga = [list(x["gold"].values())[0] for x in ab]
    gs = [list(x["gold"].values())[0] for x in sm]
    assert ga == gs, bench + " hybrid/dense gold sequences differ"
    results["subset_" + bench[:4] + "_hybrid_vs_dense"] = mcnemar(
        [correct(x) for x in ab], [correct(x) for x in sm],
        label_a="hybrid_LFM", label_b="dense_SmolLM2",
    )

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=1)

for k, v in results.items():
    print(k, "->", json.dumps(v))
