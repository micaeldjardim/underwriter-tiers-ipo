"""Fill the offer price and the number of securities sold for IPOs where 12_universe_prospectus.py left them blank,
mostly partnerships selling common units and deals selling ordinary shares or ADSs. Reads the first 4,000 characters
of the final prospectus (the cover): securities = first number followed by 'shares', 'units', 'common units',
'ordinary shares', 'american depositary shares' or 'adss'; price = first '$X per share/unit/ADS', or the
'initial public offering price' figure. Output: data/universe/cover_fill.csv"""
import gzip, os, pathlib, re, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
TXT = pathlib.Path(os.environ.get("TEXT_DIR", ROOT / "data" / "text"))
pr = pd.read_csv(ROOT / "data" / "universe" / "prospectus.csv")
need = pr[~(pr.offer_424.notna() & (pr.shares_offered > 1000))].key
D = pd.read_csv(TXT / "docs.csv"); D = D[D.role.eq("final_424") & D.key.isin(need)]
SEC = re.compile(r"(\d{1,3}(?:,\d{3}){1,3})\s+(?:common\s+units|units|shares|ordinary\s+shares|class\s+a\s+(?:common\s+)?shares|american\s+depositary\s+shares|adss)")
PX = [re.compile(r"\$\s?(\d{1,4}(?:\.\d{1,2})?)\s+per\s+(?:share|common\s+unit|unit|ads|american\s+depositary|ordinary\s+share)"),
      re.compile(r"(?:initial\s+)?public\s+offering\s+price[^$]{0,60}\$\s?(\d{1,4}(?:\.\d{1,2})?)")]
rows = []
for x in D.itertuples():
    f = TXT / "docs" / f"{x.cik}_{x.acc}.txt.gz"
    if not f.exists(): continue
    head = " ".join(gzip.open(f, "rt", encoding="utf-8").read()[:6000].replace("\xa0", " ").split()).lower()[:4000]
    m = SEC.search(head); n = float(m.group(1).replace(",", "")) if m else np.nan
    px = np.nan
    for P in PX:
        q = P.search(head)
        if q and 0.5 <= float(q.group(1)) <= 500: px = float(q.group(1)); break
    rows.append({"key": x.key, "offer_fill": px, "shares_fill": n if n == n and n > 1000 else np.nan})
r = pd.DataFrame(rows); r.to_csv(ROOT / "data" / "universe" / "cover_fill.csv", index=False)
print("to fill", len(need), "| read", len(r), "| price found", int(r.offer_fill.notna().sum()), "| securities found", int(r.shares_fill.notna().sum()))
