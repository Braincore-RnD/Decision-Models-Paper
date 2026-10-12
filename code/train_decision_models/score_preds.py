"""Skor metrik RQ3 dari items_pred.json -> metrics.json. Murni lokal (tanpa GPU).

Metrik per question (choice/score/noul diperlakukan sama: distribusi atas opsi):
  acc       = fraksi argmax == gold
  brier     = mean Σ_k (p_k - y_k)^2  (strictly proper; rendah bagus)
  nll       = mean -log p_gold (rendah bagus)
  ece       = 10-bin |conf - acc| berbobot (rendah bagus)
  auroc     = P(maxprob benar > maxprob salah): daya bedakan benar-vs-salah
              (skor confidence, bukan akurasi; lih. System Switch)
  cov@risk5 = coverage (% di atas threshold) terbesar dgn selective risk ≤5%

Jalan: python3 score_preds.py items_pred.json [metrics.json]
"""
from __future__ import annotations

import json
import math
import sys


def _gold_label(g):
    if isinstance(g, dict):
        return g.get("label", g.get("answer"))
    return g


def _probs(ans):
    if isinstance(ans, dict):
        p = ans.get("probabilities", ans.get("probs", {}))
        if isinstance(p, dict):
            return {str(k): float(v) for k, v in p.items()}
        if isinstance(p, (list, tuple)):
            return {str(i): float(v) for i, v in enumerate(p)}
    return {}


def _auroc(scores):
    # scores: [(s, benar?)]; Mann-Whitney (tanpa numpy). 0.5 = acak, 1.0 = sempurna.
    pos = sorted(s for s, y in scores if y)
    neg = sorted(s for s, y in scores if not y)
    if not pos or not neg:
        return float("nan")
    wins = sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main(inp, outp=None):
    rows = json.load(open(inp))
    per_q, errs = [], 0
    for r in rows:
        if "error" in r:
            errs += 1
            continue
        gold, answers = r.get("gold") or {}, r.get("answers") or {}
        for qid, g in gold.items():
            a = answers.get(qid)
            if a is None:
                errs += 1
                continue
            gl, probs = _gold_label(g), _probs(a)
            if not probs or gl is None:
                errs += 1
                continue
            gl = str(gl)
            pred = max(probs, key=probs.__getitem__)
            conf = probs[pred]
            ok = pred == gl
            brier = sum((p - (1.0 if k == gl else 0.0)) ** 2 for k, p in probs.items())
            nll = -math.log(max(probs.get(gl, 0.0), 1e-12))
            per_q.append({"bench": r.get("bench"), "ok": ok, "conf": conf,
                          "brier": brier, "nll": nll})
    n = len(per_q)
    acc = sum(q["ok"] for q in per_q) / n
    brier = sum(q["brier"] for q in per_q) / n
    nll = sum(q["nll"] for q in per_q) / n
    bins, ece = [[] for _ in range(10)], 0.0  # ECE 10-bin atas maxprob
    for q in per_q:
        bins[min(int(q["conf"] * 10), 9)].append(q)
    for b in bins:
        if b:
            ece += len(b) / n * abs(sum(q["conf"] for q in b) / len(b) - sum(q["ok"] for q in b) / len(b))
    auroc = _auroc([(q["conf"], q["ok"]) for q in per_q])
    by_conf = sorted(per_q, key=lambda q: -q["conf"])  # coverage@risk≤5%: sapu prefix teratas
    errs_sel, cov_best = 0, 0.0
    for i, q in enumerate(by_conf, 1):
        errs_sel += not q["ok"]
        if errs_sel / i <= 0.05:
            cov_best = i / n
    metrics = {"n": n, "errors_skipped": errs, "acc": acc, "brier": brier, "nll": nll,
               "ece10": ece, "auroc_error": auroc, "coverage_at_risk5": cov_best}
    print(json.dumps(metrics, indent=1))
    if outp:
        json.dump(metrics, open(outp, "w"), indent=1)
        print(f"saved {outp}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
