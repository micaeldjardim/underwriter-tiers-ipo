"""Universe of US IPOs 2012-2022 from Ritter's public founding-dates file (IPO-age.xlsx, Field-Ritter, N=13,945
1975-2022; already excludes units, closed-end funds, SPACs, ADRs, partnerships and penny stocks per Ritter).
For each IPO: find its CIK (our sample's CIK, SEC ticker map, EDGAR name search) and rebuild the filing timeline
exactly as 02b does, anchored on Ritter's offer date. Also keeps SIC code, state/country of incorporation.
Output: data/universe/timeline.csv"""
import re, json, time, difflib, pathlib, requests, urllib.parse, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external"  # see README: IPO-age.xlsx (Ritter, public); IPO Bloomberg.xlsx (licensed, not included)
SUB = ROOT / "data" / "edgar" / "submissions"
OUT = ROOT / "data" / "universe"; OUT.mkdir(exist_ok=True)
UA = {"User-Agent": __import__("os").environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")}  # SEC asks for a contact

r = pd.read_excel(DATA / "IPO-age.xlsx").iloc[1:, :11]
r.columns = ["offer_date", "name", "tickersymbol", "cusip", "rollup", "vc_r", "dual", "post_shares", "internet", "permno", "founding"]
r["t_offer"] = pd.to_datetime(r.offer_date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
u = r[(r.t_offer >= "2012-01-01") & (r.t_offer <= "2022-12-31")].reset_index(drop=True)
f_old = ROOT / "data" / "edgar_timeline.csv"  # CIKs of an earlier hand-checked sample; optional
old = pd.read_csv(f_old) if f_old.exists() else pd.DataFrame(columns=["status", "tickersymbol", "cik"])
old = old[old.status == "ok"].drop_duplicates("tickersymbol").set_index("tickersymbol").cik
tick = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA).json()
tmap = {v["ticker"].upper(): v["cik_str"] for v in tick.values()}
STOP = r"\b(inc|corp|corporation|ltd|limited|plc|co|company|holdings?|group|the|nv|n v|bv|b v|sa|ag|llc|lp|parent|international|technologies|technology)\b"
norm = lambda x: re.sub(r"\s+", " ", re.sub(STOP, " ", re.sub(r"[^a-z ]", " ", str(x).lower()))).strip()

def get(url):
    for k in range(4):
        try:
            x = requests.get(url, headers=UA, timeout=30); time.sleep(0.12)
            if x.status_code == 200: return x.json()
            if x.status_code == 404: return None
        except requests.RequestException: pass
        time.sleep(2 ** k)
    return None

def sub(cik):
    f = SUB / f"CIK{int(cik):010d}.json"
    if not f.exists():
        j = get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
        if j is None: return None
        for extra in j.get("filings", {}).get("files", []):
            x = get("https://data.sec.gov/submissions/" + extra["name"])
            if x: j.setdefault("_extra", []).append(x)
        f.write_text(json.dumps(j))
    return json.loads(f.read_text())

def filings(j):
    cols = ["form", "filingDate", "accessionNumber", "primaryDocument"]
    parts = [pd.DataFrame({c: j["filings"]["recent"][c] for c in cols})] + [pd.DataFrame({c: x[c] for c in cols}) for x in j.get("_extra", [])]
    f = pd.concat(parts, ignore_index=True); f["date"] = pd.to_datetime(f.filingDate)
    return f.sort_values("date").reset_index(drop=True)

def names(j): return [j.get("name", "")] + [n["name"] for n in j.get("formerNames", [])]
def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b: return 0.0
    return max(difflib.SequenceMatcher(None, a, b).ratio(), 1.0 if (a in b or b in a) else 0.0)

def search(name):
    j = get(f"https://efts.sec.gov/LATEST/search-index?keysTyped={urllib.parse.quote(norm(name) or str(name))}")
    return [int(h["_id"]) for h in j.get("hits", {}).get("hits", [])][:5] if j else []

def search_424(name, t):
    """Full-text search: pricing prospectuses filed in [-5, +15] days of the offer that contain the issuer's name.
    Finds issuers that changed name or CIK after the IPO (e.g. Cempra -> Melinta)."""
    q = urllib.parse.quote(f'"{norm(name)}"')
    j = get(f"https://efts.sec.gov/LATEST/search-index?q={q}&forms=424B4,424B1,424B3&startdt={(t - pd.Timedelta(days=5)).date()}&enddt={(t + pd.Timedelta(days=15)).date()}")
    hits = j.get("hits", {}).get("hits", []) if j else []
    return list(dict.fromkeys(int(c) for h in hits for c in h["_source"].get("ciks", [])))[:5]

PRICING = {"424B4", "424B1", "424B3", "424B2", "424B5"}
PUB = {"S-1", "F-1", "S-11"}
REG = PUB | {"S-1/A", "F-1/A", "S-11/A"}

def fit(cik, t):
    j = sub(cik)
    if j is None: return None
    f = filings(j)
    p = f[f.form.isin(PRICING) & (f.date >= t - pd.Timedelta(days=5)) & (f.date <= t + pd.Timedelta(days=15))]
    if p.empty: return None
    if f[f.form.isin(PUB | {"DRS"}) & (f.date < t) & (f.date >= t - pd.Timedelta(days=4 * 365))].empty: return None
    return j, f, p.iloc[0]

CK = OUT / "timeline_cache.json"
cache = json.loads(CK.read_text()) if CK.exists() else {}
rows = []
for i, x in enumerate(u.itertuples()):
    key = f"{x.tickersymbol}|{x.offer_date}"
    if key in cache: rows.append(cache[key]); continue
    out = {"key": key, "tickersymbol": x.tickersymbol, "t_issue": str(x.t_offer.date()), "ritter_name": x.name}
    cands = [c for c in [old.get(x.tickersymbol), tmap.get(str(x.tickersymbol).upper())] if pd.notna(c)]
    cands = list(dict.fromkeys(int(c) for c in cands))
    hits = [(c, h) for c in cands for h in [fit(c, x.t_offer)] if h]
    src = "candidate"
    if not hits:
        hits = [(c, h) for c in search(x.name) if c not in cands for h in [fit(c, x.t_offer)] if h]; src = "name_search"
    if not hits:
        hits = [(c, h) for c in search_424(x.name, x.t_offer) if c not in cands for h in [fit(c, x.t_offer)] if h]; src = "fulltext_424"
    if not hits:
        out["status"] = "no_match"
    else:
        c, (j, f, p) = max(hits, key=lambda ch: max(sim(x.name, n) for n in names(ch[1][0])))
        before = f[(f.date < x.t_offer) & (f.date >= x.t_offer - pd.Timedelta(days=4 * 365))]
        rw = before[before.form == "RW"]
        if len(rw): before = before[before.date > rw.date.max()]
        chain = before[before.form.isin(REG | {"DRS", "DRS/A"})]
        gaps = chain.date.diff().dt.days
        # a silence > 270 days restarts the attempt only if what follows is a NEW registration (S-1/F-1/DRS without an RW,
        # e.g. CZR); an amendment after the silence revives the same registration (e.g. SUPN) and is flagged as dormant
        restart = (gaps > 270) & chain.form.isin(PUB | {"DRS"})
        if restart.any(): before = before[before.date >= chain.date[restart].max()]
        chain = before[before.form.isin(REG | {"DRS", "DRS/A"})]
        dormant = int((chain.date.diff().dt.days > 270).any())
        pub, drs = before[before.form.isin(PUB)], before[before.form == "DRS"]
        t_pub = pub.date.min() if len(pub) else pd.NaT
        t_drs = drs.date.min() if len(drs) else pd.NaT
        if pd.notna(t_drs) and pd.notna(t_pub) and t_drs > t_pub: t_drs = pd.NaT
        out |= {"status": "ok" if pd.notna(t_pub) else "no_public_filing", "cik": c, "cik_source": src, "edgar_name": j.get("name"),
                "name_score": round(max(sim(x.name, n) for n in names(j)), 2), "sic": j.get("sic"), "sic_desc": j.get("sicDescription"),
                "state_inc": j.get("stateOfIncorporation"), "biz_state": (j.get("addresses", {}).get("business") or {}).get("stateOrCountry"),
                "prior_rw": int(len(rw) > 0), "dormant": dormant, "t_drs": str(t_drs.date()) if pd.notna(t_drs) else None,
                "t_pub": str(t_pub.date()) if pd.notna(t_pub) else None, "t_424": str(p.date.date()), "form_424": p.form,
                "acc_424": p.accessionNumber, "doc_424": p.primaryDocument, "form_reg": pub.form.iloc[0] if len(pub) else None,
                "acc_reg": ";".join(before[before.form.isin(REG)].accessionNumber), "doc_reg": ";".join(before[before.form.isin(REG)].primaryDocument),
                "form_seq": ";".join(before[before.form.isin(REG)].form), "date_seq": ";".join(before[before.form.isin(REG)].filingDate),
                "n_drs_all": int(before.form.isin(["DRS", "DRS/A"]).sum()),
                "n_pub_amend": int(before[before.date >= (t_pub if pd.notna(t_pub) else x.t_offer)].form.isin(REG - PUB).sum())}
    rows.append(out); cache[key] = out
    if i % 50 == 0:
        CK.write_text(json.dumps(cache)); print(i, key, out.get("status"), out.get("edgar_name"), flush=True)
CK.write_text(json.dumps(cache))
t = pd.DataFrame(rows).merge(u.assign(key=u.tickersymbol.astype(str) + "|" + u.offer_date.astype(str)).drop(columns=["tickersymbol", "name"]), on="key", how="left")
for k in ["t_issue", "t_drs", "t_pub", "t_424"]: t[k] = pd.to_datetime(t[k])
t["pub_window"] = (t.t_issue - t.t_pub).dt.days
t["conf_period"] = (t.t_pub - t.t_drs).dt.days
t["total_dur"] = (t.t_issue - t[["t_drs", "t_pub"]].min(axis=1)).dt.days
t["confidential"] = t.t_drs.notna().astype(int)
t.to_csv(OUT / "timeline.csv", index=False)
print(t.status.value_counts()); print(t.cik_source.value_counts()); print(t[["pub_window", "conf_period", "total_dur", "confidential"]].describe().round(1))
