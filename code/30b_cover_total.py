"""Gross proceeds by consensus of three readings of the cover of the final prospectus. Offer price x securities sold
(the reading used so far) fails when the cover states the number of securities in an unusual way (units, ADSs, share
counts split across classes or sellers), so each IPO gets three estimates in $ million:
  A  offer price x securities sold, as read by 12_universe_prospectus.py and 30a_cover_fill.py;
  B  the total on the price line of the cover table: '(initial) public offering price  $ P  $ TOTAL';
  C  price x (total underwriting discount / discount per security), from the discount line of the same table.
For B and C, in the first 12,000 characters, the figures within 220 characters after the anchor phrase are read: the
first with a decimal point below 1,000 is the per-security figure, the first of at least 10,000 is the total.
proceeds = the median of the estimates that agree with at least one other within 2%; missing when no two agree
(when only one estimate exists it is kept and flagged single=1).
Output: data/universe/cover_total.csv"""
import gzip, os, pathlib, re, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
TXT = pathlib.Path(os.environ.get("TEXT_DIR", ROOT / "data" / "text"))
pr = pd.read_csv(ROOT / "data" / "universe" / "prospectus.csv").set_index("key")
cf = pd.read_csv(ROOT / "data" / "universe" / "cover_fill.csv").set_index("key")
offer = pr.offer_424.fillna(cf.offer_fill.reindex(pr.index))
shares = pr.shares_offered.where(pr.shares_offered > 1000).fillna(cf.shares_fill.reindex(pr.index))
D = pd.read_csv(TXT / "docs.csv"); D = D[D.role.eq("final_424")]
NUM = re.compile(r"\$ ?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
def line(head, pat):
    for m in re.finditer(pat, head):
        vals = [v.replace(",", "") for v in NUM.findall(head[m.end():m.end() + 220])]
        ps = next((float(v) for v in vals if "." in v and float(v) < 1000), None)
        tot = next((float(v) for v in vals if float(v) >= 1e4), None)
        if ps and tot: return ps, tot
    return None, None
rows = []
for x in D.itertuples():
    f = TXT / "docs" / f"{x.cik}_{x.acc}.txt.gz"
    if not f.exists(): continue
    head = " ".join(gzip.open(f, "rt", encoding="utf-8").read()[:20000].replace("\xa0", " ").split()).lower()[:12000]
    px, tot = line(head, r"public offering price|price to (?:the )?public")
    dps, dtot = line(head, r"underwriting discounts?(?: and commissions?)?|underwriting commissions?")
    p0 = offer.get(x.key, np.nan)
    A = p0 * shares.get(x.key, np.nan) / 1e6 if p0 == p0 else np.nan
    B = tot / 1e6 if tot else np.nan
    pxx = px if px else (p0 if p0 == p0 else None)
    C = pxx * (dtot / dps) / 1e6 if (pxx and dps and dtot and dps < pxx) else np.nan
    est = [v for v in (A, B, C) if v == v and v > 0]
    ok = [v for v in est if any(abs(v - w) / w < 0.02 for w in est if w is not v)]
    proc = float(np.median(ok)) if ok else (est[0] if len(est) == 1 else np.nan)
    rows.append({"key": x.key, "proc_A": A, "proc_B": B, "proc_C": C, "proceeds_m": proc, "single": int(len(est) == 1), "n_est": len(est),
                 "changed": int(proc == proc and A == A and abs(proc - A) / proc > 0.02)})
r = pd.DataFrame(rows); r.to_csv(ROOT / "data" / "universe" / "cover_total.csv", index=False)
print("prospectuses", len(r), "| proceeds set", int(r.proceeds_m.notna().sum()), "| single estimate", int(r.single.sum()),
      "| no agreement", int((r.proceeds_m.isna() & (r.n_est >= 2)).sum()), "| differs from A by >2%", int(r.changed.sum()), "| A missing but set", int((r.proc_A.isna() & r.proceeds_m.notna()).sum()))
