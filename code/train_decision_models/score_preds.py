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


def collect(rows):
    """Kelompokkan per bench: {bench: [q]}; q = ok/conf/brier/nll/probs/gold/qtype."""
    by, errs = {}, 0
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
            gl, probs = _gold_label(g), _probs(a if isinstance(a, dict) else {})
            if not probs or gl is None:
                errs += 1
                continue
            gl = str(gl)
            pred = max(probs, key=probs.__getitem__)
            by.setdefault(r.get("bench"), []).append({
                "ok": pred == gl, "conf": probs[pred], "gold": gl,
                "probs": probs, "qtype": (a.get("type") if isinstance(a, dict) else None),
                "brier": sum((p - (1.0 if k == gl else 0.0)) ** 2 for k, p in probs.items()),
                "nll": -math.log(max(probs.get(gl, 0.0), 1e-12))})
    return by, errs


def metrics_of(qs):
    n = len(qs)
    bins, ece = [[] for _ in range(10)], 0.0
    for q in qs:
        bins[min(int(q["conf"] * 10), 9)].append(q)
    for b in bins:
        if b:
            ece += len(b) / n * abs(sum(q["conf"] for q in b) / len(b) - sum(q["ok"] for q in b) / len(b))
    by_conf = sorted(qs, key=lambda q: -q["conf"])
    errs_sel, cov_best = 0, 0.0
    for i, q in enumerate(by_conf, 1):
        errs_sel += not q["ok"]
        if errs_sel / i <= 0.05:
            cov_best = i / n
    return {"n": n, "acc": sum(q["ok"] for q in qs) / n,
            "brier": sum(q["brier"] for q in qs) / n,
            "nll": sum(q["nll"] for q in qs) / n, "ece10": ece,
            "auroc_error": _auroc([(q["conf"], q["ok"]) for q in qs]),
            "coverage_at_risk5": cov_best}


def boot_ci(qs, n_boot=2000, seed=0):
    """CI persentil acc + ece10. O(n) per resample (tanpa AUROC: O(n^2))."""
    import random as _r
    rng, n = _r.Random(seed), len(qs)
    accs, eces = [], []
    for _ in range(n_boot):
        s = [qs[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(q["ok"] for q in s) / n)
        eces.append(metrics_of(s)["ece10"])
    accs.sort(), eces.sort()
    lo, hi = accs[int(0.025 * n_boot)], accs[min(int(0.975 * n_boot), n_boot - 1)]
    el, eh = eces[int(0.025 * n_boot)], eces[min(int(0.975 * n_boot), n_boot - 1)]
    return {"acc": [lo, hi], "ece10": [el, eh]}


def _scaled(probs, T):
    z = {k: math.log(max(p, 1e-12)) for k, p in probs.items()}  # shift-invariant: tanpa C
    m = max(v / T for v in z.values())
    e = {k: math.exp(v / T - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def fit_T(qs, iters=60):
    """Golden-section atas mean NLL di logT in [-4, 4]."""
    def nll(logT):
        T = math.exp(logT)
        return sum(-math.log(max(_scaled(q["probs"], T).get(q["gold"], 0.0), 1e-12)) for q in qs) / len(qs)
    a, b, gr = -4.0, 4.0, (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return math.exp((a + b) / 2)


def apply_T(qs, T_by_type):
    out = []
    for q in qs:
        T = T_by_type.get(q["qtype"], T_by_type.get(None, 1.0))
        p = _scaled(q["probs"], T)
        pred = max(p, key=p.__getitem__)
        out.append({"ok": pred == q["gold"], "conf": p[pred], "gold": q["gold"],
                    "probs": p, "qtype": q["qtype"],
                    "brier": sum((v - (1.0 if k == q["gold"] else 0.0)) ** 2 for k, v in p.items()),
                    "nll": -math.log(max(p.get(q["gold"], 0.0), 1e-12))})
    return out


def fit_by_type(qs):
    by, T = {}, {}
    for q in qs:
        by.setdefault(q["qtype"], []).append(q)
    for t, sub in by.items():
        T[t] = fit_T(sub) if len(sub) >= 10 else 1.0
    return T


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


def ece_equal_mass(qs, nb=10):
    s = sorted(qs, key=lambda q: q["conf"])
    ece, n = 0.0, len(s)
    for i in range(nb):
        b = s[i * n // nb:(i + 1) * n // nb]
        if b:
            ece += len(b) / n * abs(sum(q["conf"] for q in b) / len(b) - sum(q["ok"] for q in b) / len(b))
    return ece


def from_cal_logits(path):
    """cal_logits.json (logits+label per question) -> q dicts ala collect()."""
    qs = []
    for x in json.load(open(path)):
        p = _scaled({str(i): v for i, v in enumerate(x["logits"])}, 1.0)
        gl = str(x["label"])
        pred = max(p, key=p.__getitem__)
        qs.append({"ok": pred == gl, "conf": p[pred], "gold": gl, "probs": p,
                   "qtype": x["qtype"],
                   "brier": sum((v - (1.0 if k == gl else 0.0)) ** 2 for k, v in p.items()),
                   "nll": -math.log(max(p.get(gl, 0.0), 1e-12))})
    return qs


if __name__ == "__main__":
    toks, boot, seed, ts_from = sys.argv[1:], 0, 0, None
    args, cal = [], False
    i = 0
    while i < len(toks):
        a = toks[i]
        if a == "--boot":
            boot = int(toks[i + 1]); i += 2
        elif a == "--seed":
            seed = int(toks[i + 1]); i += 2
        elif a == "--ts-from":
            ts_from = toks[i + 1]; i += 2
        elif a == "--cal":
            cal = True; i += 1
        else:
            args.append(a); i += 1
    if not boot and ts_from is None and not cal:
        main(args[0], args[1] if len(args) > 1 else None)
    elif cal:
        qs = from_cal_logits(args[0])
        m = metrics_of(qs)
        m["ece_equal_mass"] = ece_equal_mass(qs)
        m["ci95"] = boot_ci(qs, boot or 2000, seed)
        T = fit_by_type(qs)
        m["ts_fit_on_cal"] = {"T": T, "ece_ts": metrics_of(apply_T(qs, T))["ece10"]}
        print(json.dumps(m, indent=1))
    else:
        by, errs = collect(json.load(open(args[0])))
        fit_qs = by.get(ts_from, []) if ts_from else []
        T_id = fit_by_type(fit_qs) if fit_qs else {}
        out = {"errors_skipped": errs}
        for bench, qs in sorted(by.items()):
            m = metrics_of(qs)
            m["errors_skipped"] = 0
            if boot:
                m["ci95"] = boot_ci(qs, boot, seed)
            if ts_from:
                T_or = fit_by_type(qs)
                m["ts"] = {"fit_from": ts_from, "T": T_id,
                           "ece_ts_indist": metrics_of(apply_T(qs, T_id))["ece10"],
                           "ece_ts_oracle": metrics_of(apply_T(qs, T_or))["ece10"],
                           "T_oracle": T_or}
            out[bench] = m
        print(json.dumps(out, indent=1))
        if len(args) > 1:
            json.dump(out, open(args[1], "w"), indent=1)
            print(f"saved {args[1]}")
