from unsloth.models.decision_datasets import canonical_row, load_source
import itertools

rows = load_source("clinc150", "test", limit=2000, seed=3407)
print("n:", len(rows), flush=True)
print("row keys:", list(rows[0].keys()), flush=True)
c = canonical_row(rows[0])
print("canon keys:", list(c.keys()), flush=True)
print("gold:", str(c.get("gold"))[:200], flush=True)
qs = c.get("questions", {})
q0 = next(iter(qs.values()))
print("q keys:", list(q0.keys()) if isinstance(q0, dict) else type(q0), flush=True)
print("instr:", str(q0.get("instructions"))[:120], flush=True)
cr = q0.get("criteria")
print("ncrit:", len(cr) if isinstance(cr, dict) else type(cr).__name__, flush=True)
print("crit sample:", list(itertools.islice(cr.items(), 2)), flush=True)
golds = [str(canonical_row(r).get("gold"))[:60] for r in rows[:20]]
print("golds sample:", golds[:6], flush=True)
print("INSPECT DONE", flush=True)
