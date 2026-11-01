"""SEC review workload when each IPO's registration becomes public, from the EDGAR quarterly master indexes (free).
Workload = registration statements and amendments (S-1, F-1, S-4, S-11, Form 10, and their amendments) filed in the 30
days before the IPO's public flip: the queue the reviewing staff is working through (cf. Kuechling, Schmidtke and Posch 2021).
Two versions:
  wl_all    every such filing on EDGAR;
  wl_office S-1/F-1 filings and amendments by registrants assigned to the same Division of Corporation Finance office as
            the issuer (submissions JSON field 'ownerOrg', e.g. '03 Life Sciences'; the current assignment, applied to
            the whole period). Restricted to S-1/F-1 filers, the IPO-type registrations, to keep the lookup to ~10,000 firms.
Output: data/universe/workload.csv (key, wl_all, wl_office, office); cache data/edgar/index/, data/edgar/owner_org.json"""
import gzip, io, json, time, pathlib, requests, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
IDX = ROOT / "data" / "edgar" / "index"; IDX.mkdir(parents=True, exist_ok=True)
SUB = ROOT / "data" / "edgar" / "submissions"
ORG = ROOT / "data" / "edgar" / "owner_org.json"
UA = {"User-Agent": __import__("os").environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")}  # SEC asks for a contact
FORMS = {"S-1", "S-1/A", "F-1", "F-1/A", "S-4", "S-4/A", "S-11", "S-11/A", "10-12B", "10-12B/A", "10-12G", "10-12G/A"}

def get(url, binary=False):
    for k in range(5):
        try:
            x = requests.get(url, headers=UA, timeout=60); time.sleep(0.12)
            if x.status_code == 200: return x.content if binary else x.json()
            if x.status_code == 404: return None
        except requests.RequestException: pass
        time.sleep(2 ** k)
    return None

rows = []
for y in range(2011, 2024):
    for q in range(1, 5):
        f = IDX / f"{y}Q{q}.csv"
        if not f.exists():
            raw = get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/master.gz", binary=True)
            if raw is None: continue
            lines = gzip.decompress(raw).decode("latin-1").splitlines()
            start = next(i for i, l in enumerate(lines) if l.startswith("-----")) + 1
            d = pd.DataFrame([l.split("|")[:4] for l in lines[start:]], columns=["cik", "name", "form", "date"])
            d[d.form.isin(FORMS)].to_csv(f, index=False); print(y, q, flush=True)
        rows.append(pd.read_csv(f))
ix = pd.concat(rows, ignore_index=True); ix["date"] = pd.to_datetime(ix.date)
print("registration filings 2011-2023:", len(ix), "registrants:", ix.cik.nunique())

# office of each registrant (only the ownerOrg field is kept; full submissions of our IPOs are already cached by 11)
org = json.loads(ORG.read_text()) if ORG.exists() else {}
IPO_FORMS = {"S-1", "S-1/A", "F-1", "F-1/A"}
from concurrent.futures import ThreadPoolExecutor
def office(c):
    f = SUB / f"CIK{int(c):010d}.json"
    j = json.loads(f.read_text()) if f.exists() else get(f"https://data.sec.gov/submissions/CIK{int(c):010d}.json")
    return str(int(c)), (j or {}).get("ownerOrg") or None
todo = [c for c in ix[ix.form.isin(IPO_FORMS)].cik.unique() if str(int(c)) not in org]
with ThreadPoolExecutor(6) as ex:                    # 6 workers x ~1 request per 0.6 s stays under the SEC's 10/s
    for i, (k, o) in enumerate(ex.map(office, todo)):
        org[k] = o
        if i % 300 == 0: ORG.write_text(json.dumps(org)); print("owner org", i, "of", len(todo), flush=True)
ORG.write_text(json.dumps(org))
ix["office"] = ix.cik.astype(int).astype(str).map(org)

a = pd.read_csv(ROOT / "data" / "universe" / "analysis.csv", parse_dates=["t_pub"])
a["office"] = a.cik.astype(int).astype(str).map(org)
day = ix.groupby("date").size()
dayo = ix[ix.form.isin(IPO_FORMS)].dropna(subset=["office"]).groupby(["office", "date"]).size()
def window(series, d):
    return int(series[(series.index > d - pd.Timedelta(days=30)) & (series.index <= d)].sum())
a["wl_all"] = [window(day, d) for d in a.t_pub]
a["wl_office"] = [window(dayo.loc[o], d) if isinstance(o, str) and o in dayo.index.get_level_values(0) else None for o, d in zip(a.office, a.t_pub)]
a[["key", "office", "wl_all", "wl_office"]].to_csv(ROOT / "data" / "universe" / "workload.csv", index=False)
print(a[["wl_all", "wl_office"]].describe().round(1).to_string()); print(a.office.value_counts().to_string())
