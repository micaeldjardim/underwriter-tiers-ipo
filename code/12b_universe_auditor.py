"""Auditor of the IPO financial statements, from the 'Experts' section of the pricing prospectus (424B):
'... have been audited by <firm>, independent registered public accounting firm'. Big4 = EY, Deloitte, KPMG, PwC.
Also the lock-up length stated in the prospectus (days). Output: data/universe/auditor.csv"""
import re, json, pathlib, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "universe"
src = (ROOT / "code" / "12_universe_prospectus.py").read_text(encoding="utf-8").split("def find_range")[0]
g = {"__file__": str(ROOT / "code" / "12_universe_prospectus.py")}; exec(src, g)    # text_of() and the timeline only
text_of, t = g["text_of"], g["t"]
CACHE = OUT / "auditor_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

AUD = re.compile(r"audited\s+by\s+(.{3,120}?),?\s+(?:an\s+)?independent\s+(?:registered\s+)?(?:public\s+)?(?:accounting|auditors?|certified)", re.I)
FIRM = re.compile(r"([A-Z][A-Za-z&.,'\- ]{2,70}?(?:LLP|L\.L\.P\.|LLC|PLLC|P\.C\.|P\.A\.|CPAs?|Ltd\.?|Limited|S\.A\.|AG|GmbH|Inc\.))\s*,?\s*(?:an?\s+)?independent", re.S)
BIG4 = {"ey": r"ernst\s*&\s*young|\bey\b", "deloitte": r"deloitte", "kpmg": r"kpmg", "pwc": r"pricewaterhouse"}
LOCK = re.compile(r"(?:for\s+a\s+period\s+of|during\s+the\s+period\s+ending|for)\s+(\d{2,3})\s+days\s+after\s+the\s+date\s+of\s+this\s+prospectus", re.I)

rows = []
for i, r in enumerate(t.itertuples()):
    if r.key in cache: rows.append(cache[r.key]); continue
    p = text_of(r.cik, r.acc_424, r.doc_424, 3_000_000)
    # the Experts section sits near the end, before the F-pages; its wording varies ('audited by X, independent ...' or
    # 'X, independent registered public accounting firm, has audited ...'), so look for a firm name around the last mentions
    ks = [m.start() for m in re.finditer(r"\bexperts\b", p, re.I)][-3:]
    firm = None
    for k in reversed(ks):
        win = p[max(0, k - 1500):k + 1500]
        hit = AUD.search(win) or FIRM.search(win)
        if hit: firm = hit.group(1).strip(); break
    if not firm:
        hit = AUD.search(p); firm = hit.group(1).strip() if hit else None
    big = next((b for b, rx in BIG4.items() if firm and re.search(rx, firm, re.I)), None)
    locks = [int(x) for x in LOCK.findall(p) if 60 <= int(x) <= 720]
    out = {"key": r.key, "auditor": firm, "big4": int(big is not None) if firm else None, "big4_firm": big,
           "lockup_days": max(set(locks), key=locks.count) if locks else None}
    rows.append(out); cache[r.key] = out
    if i % 50 == 0: CACHE.write_text(json.dumps(cache)); print(i, out, flush=True)
CACHE.write_text(json.dumps(cache))
d = pd.DataFrame(rows); d.to_csv(OUT / "auditor.csv", index=False)
print("auditor found:", d.auditor.notna().mean().round(3), "big4:", d.big4.mean().round(3)); print(d.lockup_days.value_counts().head())
