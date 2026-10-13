"""Figure RQ3 dari items_pred.json lokal: reliability + risk-coverage.
E-banking (in-dist) vs LOTO-banking (shift), model Decision masing-masing.

Jalan (matplotlib ephemera, tanpa install): uv run --with matplotlib code/train_decision_models/fig_rq3.py
Keluar: paper/figures/fig_reliability.pdf + fig_riskcover.pdf
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_preds import collect  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.join(BASE, "docs/eval/colab/ablasi-full/items_pred.json")
LOTO = os.path.join(BASE, "docs/eval/colab/loto-banking/items_pred.json")
FIG = os.path.join(BASE, "paper/figures")
os.makedirs(FIG, exist_ok=True)


def series(path, bench):
    by, _ = collect(json.load(open(path)))
    return [(q["conf"], q["ok"]) for q in by[bench]]


def reliability(s, nb=10):
    xs, ys = [], []
    for b in range(nb):
        pts = [ok for c, ok in s if min(int(c * nb), nb - 1) == b]
        if pts:
            xs.append((b + 0.5) / nb)
            ys.append(sum(pts) / len(pts))
    return xs, ys


def risk_cover(s):
    s = sorted(s, key=lambda t: -t[0])
    err, xs, ys = 0, [0.0], [0.0]
    for i, (_, ok) in enumerate(s, 1):
        err += not ok
        xs.append(i / len(s))
        ys.append(err / i)
    return xs, ys


e_id = series(E, "banking77")
l_sh = series(LOTO, "banking77")

plt.figure(figsize=(3.4, 2.6))
plt.plot([0, 1], [0, 1], "k--", lw=1)
for s, lab, mk in ((e_id, "E in-dist", "o"), (l_sh, "LOTO shift", "s")):
    x, y = reliability(s)
    plt.plot(x, y, marker=mk, ms=4, lw=1.2, label=lab)
plt.xlabel("confidence")
plt.ylabel("accuracy")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_reliability.pdf"))
print("saved fig_reliability.pdf")

plt.figure(figsize=(3.4, 2.6))
for s, lab, mk in ((e_id, "E in-dist", "o"), (l_sh, "LOTO shift", "s")):
    x, y = risk_cover(s)
    plt.plot(x, y, lw=1.2, label=lab)
plt.xlabel("coverage")
plt.ylabel("selective risk")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_riskcover.pdf"))
print("saved fig_riskcover.pdf")
