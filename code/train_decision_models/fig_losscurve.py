"""Figure kurva eval-loss dari docs/eval/kaggle/*/loss_series.json (traceable, tanpa GPU).

Jalan: uv run --with matplotlib code/train_decision_models/fig_losscurve.py
Keluar: paper/figures/fig_losscurve.pdf
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(BASE, "paper/figures")
os.makedirs(FIG, exist_ok=True)

plt.figure(figsize=(3.4, 2.6))
for f in sorted(glob.glob(os.path.join(BASE, "docs/eval/kaggle/*/loss_series.json"))):
    name = os.path.basename(os.path.dirname(f))
    ev = json.load(open(f))["eval"]
    if ev:
        x, y = zip(*sorted(ev))
        plt.plot(x, y, marker="o", ms=4, lw=1.2, label=name)
plt.xlabel("epoch")
plt.ylabel("eval loss")
plt.xticks([1, 2, 3])
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_losscurve.pdf"))
print("saved fig_losscurve.pdf")
