"""Clef-style joint decision head untuk backbone decoder-only. Tanpa dependensi Unsloth/HF.

Matematika = port text-only dari Cloudflare joint_schema_model.py:
render (state + skema) -> 1 prefill -> head skor tiap opsi tiap pertanyaan
-> softmax per pertanyaan. Tokenizer: tiktoken GPT-2 (marker teks biasa,
karena vocab GPT-2 tak punya token chat khusus).

Jalankan self-check:  python decision/head.py
"""
from __future__ import annotations

import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

QTYPES = {"noul": 0, "choice": 1, "score": 2}


def render(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(q):
    t = q["type"]
    if t == "noul":
        crit = {"true": "yes", "false": "no"}
        crit.update(q.get("criteria") or {})
        return [("true", crit["true"]), ("false", crit["false"])]
    if t == "choice":
        return sorted((str(k), v) for k, v in q["criteria"].items())
    return [(str(i), v) for i, v in enumerate(q["criteria"])]


def encode_record(enc, state, questions, max_len=1024):
    """Render satu record -> {input_ids, questions:[{id,type,span,ospans,oids}]}.

    Tiap potongan di-tokenize terpisah (span stabil). State dipotong dari
    ekor bila kepanjangan; skema tak pernah dipotong (error bila muat tidak).
    """
    tok = lambda s: enc.encode_ordinary(s)
    pre = tok("STATE:\n")
    suf = tok("\nDECIDE:\n")
    schema, qmeta = [], []
    for i, (qid, q) in enumerate(questions.items()):
        schema += tok(f"\nQ{i + 1} ID:{qid} TYPE:{q['type']} INST:")
        qs, schema = len(schema), schema + tok(render(q.get("instructions") or qid))
        qe = len(schema)
        schema += tok("\nOPTS:\n")
        ospans, oids = [], []
        for oid, desc in question_options(q):
            schema += tok(f"- {oid}: ")
            os_, schema = len(schema), schema + tok(render(desc))
            schema += tok("\n")
            ospans.append((os_, len(schema) - 1))
            oids.append(oid)
        schema += tok("ENDQ\n")
        qmeta.append({"id": str(qid), "type": QTYPES[q["type"]], "span": (qs, qe), "ospans": ospans, "oids": oids})
    fixed = len(pre) + len(schema) + len(suf)
    if fixed > max_len:
        raise ValueError(f"schema {fixed} tok > max_len {max_len}")
    sids = tok(render(state))[: max_len - fixed]
    off = len(pre) + len(sids)
    for m in qmeta:
        a, b = m["span"]
        m["span"] = (a + off, b + off)
        m["ospans"] = [(a + off, b + off) for a, b in m["ospans"]]
    return {"input_ids": pre + sids + schema + suf, "questions": qmeta}


class EvLayer(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.qn, self.mn = nn.LayerNorm(w), nn.LayerNorm(w)
        self.attn = nn.MultiheadAttention(w, heads, batch_first=True)
        self.fn, self.ff = nn.LayerNorm(w), nn.Sequential(nn.Linear(w, w * 2), nn.GELU(), nn.Linear(w * 2, w))

    def forward(self, q, mem):
        q = q + self.attn(self.qn(q), self.mn(mem), self.mn(mem), need_weights=False)[0]
        return q + self.ff(self.fn(q))


class JointHead(nn.Module):
    """Skor tiap opsi: prior leksikal + joint reasoning. ~6M param di width=256."""

    def __init__(self, n_embd, width=256, routing=2, layers=4, heads=4, ff=512):
        super().__init__()
        self.hn = nn.LayerNorm(n_embd)
        self.mem = nn.Linear(n_embd, width, bias=False)
        self.qp = nn.Linear(n_embd, width, bias=False)
        self.cp = nn.Linear(n_embd, width, bias=False)
        self.lp = nn.Linear(n_embd, width, bias=False)
        self.gp = nn.Linear(n_embd, width, bias=False)
        self.temb = nn.Embedding(3, width)
        self.ev = nn.ModuleList([EvLayer(width, heads) for _ in range(routing)])
        self.sn = nn.LayerNorm(width)
        dec = nn.TransformerDecoderLayer(width, heads, ff, 0.0, "gelu", batch_first=True, norm_first=True)
        self.dec = nn.ModuleList([copy.deepcopy(dec) for _ in range(layers)])
        self.fn, self.on = nn.LayerNorm(width), nn.LayerNorm(width)
        self.res = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.prior_s = nn.Parameter(torch.zeros(()))
        self.joint_s = nn.Parameter(torch.zeros(()))
        self.gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean(h, s, e):
        return h[s:e].mean(dim=0)

    def forward(self, hidden, ids, mask, records, emb):
        """-> logits (B,Qmax,Omax), valid (B,Qmax,Omax bool). records: list meta encode_record."""
        H = self.hn(hidden)
        B = hidden.size(0)
        nq = max(len(r["questions"]) for r in records)
        no = max(len(q["oids"]) for r in records for q in r["questions"])
        logits = hidden.new_full((B, nq, no), -1e4)
        valid = torch.zeros(B, nq, no, dtype=torch.bool, device=hidden.device)
        for b, rec in enumerate(records):
            L = int(mask[b].sum().item())
            h, mem = H[b, :L], self.mem(H[b, :L]).unsqueeze(0)
            glo = h[-1]
            qs = rec["questions"]
            qv = torch.stack([self._mean(h, *q["span"]) for q in qs])
            ctxs, lexs = [], []
            for q in qs:
                ctxs.append(torch.stack([self._mean(h, *s) for s in q["ospans"]]))
                lex = torch.stack([emb[ids[b, s:e]].mean(dim=0) for s, e in q["ospans"]])
                lexs.append(lex)
            qry = torch.cat(
                [self.cp(c) + self.lp(l) + self.qp(v).unsqueeze(0) for c, l, v in zip(ctxs, lexs, qv)]
            ).unsqueeze(0)
            for layer in self.ev:
                qry = layer(qry, mem)
            qry = qry[0].split([len(q["oids"]) for q in qs])
            sums = []
            for f, o in zip(self.qp(qv), qry):
                w = torch.softmax(o @ f / (o.shape[-1] ** 0.5), dim=0)
                sums.append((w.unsqueeze(-1) * o).sum(dim=0))
            tids = torch.tensor([q["type"] for q in qs], device=hidden.device)
            fld = (self.qp(qv) + self.sn(torch.stack(sums)) + self.gp(glo).unsqueeze(0) + self.temb(tids)).unsqueeze(0)
            for layer in self.dec:
                fld = layer(fld, mem)
            fld = self.fn(fld[0])
            ps = self.prior_s.clamp(max=math_log100()).exp()
            js = self.joint_s.clamp(max=math_log100()).exp()
            g = torch.sigmoid(self.gate)
            for j, (f, q, lex, ro) in enumerate(zip(fld, qs, lexs, qry)):
                prior = ps * (F.normalize(lex, dim=-1) @ F.normalize(qv[j] + glo, dim=-1))
                o = self.on(ro)
                cos = F.cosine_similarity(f.unsqueeze(0).expand_as(o), o, dim=-1)
                feat = torch.cat([f.unsqueeze(0).expand_as(o), o, f.unsqueeze(0).expand_as(o) * o,
                                  (f.unsqueeze(0).expand_as(o) - o).abs()], dim=-1)
                row = prior + g * (js * cos + self.res(feat).squeeze(-1))
                n = len(q["oids"])
                logits[b, j, :n] = row
                valid[b, j, :n] = True
        return logits, valid


def math_log100():
    import math
    return math.log(100.0)


def decision_loss(logits, target, valid, qtypes, ordinal, ls=0.0, bw=0.0, ow=0.0):
    """logits/target/valid (B,Q,O); qtypes (B,Q) int; ordinal (B,Q) bool khusus score."""
    logp = torch.log_softmax(logits.masked_fill(~valid, -1e4).float(), -1)
    tgt = target
    if ls:
        uni = valid.float() / valid.sum(-1, keepdim=True).clamp(min=1)
        tgt = (1 - ls) * target + ls * uni
    loss = -(tgt * logp).sum(-1)[valid.any(-1)].mean()
    if bw:
        loss = loss + bw * (((logp.exp() - target) ** 2 * valid).sum(-1)[valid.any(-1)]).mean()
    if ow and ordinal.any():
        p = logp.exp()
        O = p.shape[-1]
        lv = torch.arange(O, device=p.device, dtype=p.dtype)
        dist = (lv[:, None] - lv[None, :]).abs()
        span = (valid.sum(-1) - 1).clamp(min=1).to(p.dtype)
        expd = torch.einsum("bqi,ij,bqj->bq", p, dist, target) / span
        m = ordinal & valid.any(-1)
        loss = loss + ow * (expd[m]).mean() if m.any() else loss
    return loss


def _fit_temp(rows, gold):
    """Satu temperatur via LBFGS pada NLL hard label. rows: list tensor (O,) beda panjang ok."""
    logt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=50)
    Z = [z.detach() for z in rows]
    G = torch.tensor(gold)

    def closure():
        opt.zero_grad()
        l = sum(F.cross_entropy((z / logt.exp()).unsqueeze(0), g.unsqueeze(0)) for z, g in zip(Z, G)) / len(Z)
        l.backward()
        return l

    opt.step(closure)
    return float(logt.exp().clamp(0.05, 20.0).detach().item())


def calibrate(rows, gold, qtypes):
    """rows: list (O,) logit mentah; gold: list int; qtypes: list int. -> temps + metrik."""
    T = _fit_temp(rows, gold)
    scaled = [z / T for z in rows]
    rel = [1.0, 1.0, 1.0]
    for t in range(3):
        idx = [i for i in range(len(rows)) if qtypes[i] == t]
        if len(idx) >= 10:
            rel[t] = _fit_temp([rows[i] for i in idx], [gold[i] for i in idx])
    return {"head": T, "rel": rel}


def metrics(rows, gold, qtypes, temps=None):
    """-> {acc, acc_per_type, ece, loss}. rows logit mentah; temps hasil calibrate (opsional)."""
    rows = [z.detach() for z in rows]
    T, rel = (temps["head"], temps["rel"]) if temps else (1.0, [1.0, 1.0, 1.0])
    probs = [(z / (T * rel[t])).softmax(-1) for z, t in zip(rows, qtypes)]
    pred = [int(p.argmax()) for p in probs]
    ok = [p == g for p, g in zip(pred, gold)]
    acc = sum(ok) / len(ok)
    by = {}
    for t, n in QTYPES.items():
        ii = [i for i in range(len(gold)) if qtypes[i] == n]
        by[t] = sum(ok[i] for i in ii) / len(ii) if ii else 0.0
    conf = [float(p.max()) for p in probs]
    ece, M = 0.0, 10
    for m in range(M):
        ii = [i for i in range(len(gold)) if conf[i] > m / M and conf[i] <= (m + 1) / M]
        if ii:
            ece += len(ii) / len(gold) * abs(sum(ok[i] for i in ii) / len(ii) - sum(conf[i] for i in ii) / len(ii))
    loss = sum(float(F.cross_entropy((r / (T * rel[t])), torch.tensor(g))) for r, g, t in zip(rows, gold, qtypes)) / len(gold)
    return {"acc": acc, "acc_per_type": by, "ece": ece, "loss": loss}


class DecisionModel(nn.Module):
    """Backbone HF (output_hidden_states) + JointHead. Import-aman (tanpa efek samping)."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, ids, mask, records):
        bb = self.backbone
        kw = dict(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        if next(bb.parameters()).requires_grad:
            h = bb(**kw).hidden_states[-1]
        else:
            with torch.no_grad():
                h = bb(**kw).hidden_states[-1]
        return self.head(h, ids, mask, records, bb.get_input_embeddings().weight)


def systemone_answer(q, probs):
    if q["type"] == "noul":
        return {"type": "noul", "noul": round(probs["true"], 4)}
    if q["type"] == "choice":
        opts = [str(k) for k in q["criteria"]]
        c = max(opts, key=probs.__getitem__)
        return {"type": "choice", "choice": c, "confidence": round(probs[c], 4),
                "probabilities": {o: round(probs[o], 4) for o in opts}}
    lv = [str(i) for i in range(len(q["criteria"]))]
    return {"type": "score", "score": round(sum(i * probs[l] for i, l in enumerate(lv)), 4),
            "confidence": round(max(probs[l] for l in lv), 4),
            "legend": dict(zip(lv, q["criteria"])),
            "probabilities": {l: round(probs[l], 4) for l in lv}}


if __name__ == "__main__":
    # ponytail: self-check CPU — encode, forward, overfit head, calibrate.
    class FakeEnc:
        def encode_ordinary(self, s):
            return [ord(c) % 128 + 1 for c in s]

    enc = FakeEnc()
    Q = {"team": {"type": "choice", "instructions": "pilih tim",
                  "criteria": {"a": "satu", "b": "dua", "c": "tiga"}},
         "ok": {"type": "noul", "instructions": "benar?"},
         "lv": {"type": "score", "instructions": "level", "criteria": ["lo", "mid", "hi"]}}
    rec = encode_record(enc, {"msg": "tagihan ganda, minta refund"}, Q, max_len=512)
    assert len(rec["questions"]) == 3 and rec["questions"][0]["oids"] == ["a", "b", "c"]
    torch.manual_seed(0)
    head = JointHead(n_embd=64, width=32, routing=1, layers=1, heads=2, ff=64)
    C, V = 64, 128
    emb = nn.Embedding(V, C)
    base = torch.tensor(rec["input_ids"]) % V
    T, B = len(base), 4
    ids = base.unsqueeze(0).repeat(B, 1)
    H = torch.randn(B, T, C)
    mask = torch.ones(B, T, dtype=torch.long)
    recs = [{**rec, "questions": rec["questions"]}] * B
    logits, valid = head(H, ids, mask, recs, emb.weight)
    assert logits.shape == (4, 3, 3) and valid.sum() == 4 * (3 + 2 + 3)
    prow = logits.masked_fill(~valid, -1e4).softmax(-1).sum(-1)
    assert torch.allclose(prow[valid.any(-1)], torch.ones(12), atol=1e-4)
    # overfit: hidden & target tetap, loss harus ambruk
    tgt = torch.zeros(4, 3, 3)
    gold = [0, 1, 2]
    for b in range(4):
        for j, g in enumerate(gold):
            tgt[b, j, g % (3 if j == 0 else 2 if j == 1 else 3)] = 1.0
    qtypes = torch.zeros(4, 3, dtype=torch.long)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    l0 = float(decision_loss(logits, tgt, valid, qtypes, torch.zeros(4, 3, dtype=torch.bool)).detach())
    for _ in range(60):
        logits, _ = head(H, ids, mask, recs, emb.weight)
        l = decision_loss(logits, tgt, valid, qtypes, torch.zeros(4, 3, dtype=torch.bool))
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
    l1 = float(decision_loss(head(H, ids, mask, recs, emb.weight)[0], tgt, valid, qtypes,
                             torch.zeros(4, 3, dtype=torch.bool)).detach())
    assert l1 < l0 / 2, (l0, l1)
    rows = [logits[b, j, : len(rec["questions"][j]["oids"])] for b in range(1) for j in range(3)]
    temps = calibrate(rows, gold, [0, 0, 0])
    m = metrics(rows, gold, [0, 0, 0], temps)
    assert 0.0 <= m["ece"] <= 1.0 and temps["head"] > 0
    print(f"ok loss {l0:.3f}->{l1:.3f} temps={temps} acc={m['acc']:.2f}")
