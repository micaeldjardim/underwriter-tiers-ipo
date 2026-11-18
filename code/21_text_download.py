"""Prospectus text and SEC comment letters for the universe (data/universe/timeline.csv), EDGAR only, free.
- docs: first public registration statement, last pre-pricing amendment, pricing prospectus (424B)
- letters: UPLOAD (SEC staff) and CORRESP (issuer replies) filed from 60 days before the first submission
  to two years after pricing (staff letters are released at least 20 business days after effectiveness)
Text is kept with line breaks at block tags, gzipped (TEXT_DIR, default data/text).
Resumable: files already on disk are skipped. Output manifests: TEXT_DIR/docs.csv, TEXT_DIR/letters.csv"""
import os, re, sys, gzip, html, time, json, pathlib, requests, pandas as pd
import fitz  # PyMuPDF, for the PDF letters
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUB = ROOT / "data" / "edgar" / "submissions"
TXT = pathlib.Path(os.environ.get("TEXT_DIR", ROOT / "data" / "text"))
(TXT / "docs").mkdir(parents=True, exist_ok=True); (TXT / "letters").mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")}  # SEC asks for a contact

def get(url):
    for k in range(5):
        try:
            r = requests.get(url, headers=UA, timeout=90); time.sleep(0.12)
            if r.status_code == 200: return r.content
            if r.status_code == 404: return None
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    return None

def to_text(raw, name):
    if name.lower().endswith(".pdf") or raw[:5] == b"%PDF-":
        try:
            with fitz.open(stream=raw, filetype="pdf") as d: return "\n".join(p.get_text() for p in d)
        except Exception:
            return ""
    raw = re.sub(rb"data:[\w/+.-]+;base64,[A-Za-z0-9+/=\s]+", b"", raw)  # inline images bloat some filings past memory
    s = raw[:40_000_000].decode("utf-8", "ignore")
    if not re.search(r"(?i)<(html|body|p|div|table)\b", s): return s  # plain-text filing
    s = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", s)
    s = re.sub(r"(?i)<(br|/p|/div|/tr|/h\d|/li|/table)\b[^>]*>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()

def save(path, cik, acc, doc):
    if path.exists(): return path.stat().st_size, "cached"          # gz bytes; not reread, to spare memory
    raw = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}")
    if raw is None: return 0, "missing"
    try:
        s = to_text(raw, doc)
    except MemoryError:
        return len(raw), "too_big"
    with gzip.open(path, "wt", encoding="utf-8") as f: f.write(s)
    return len(s), "ok"

def filings(cik):
    p = SUB / f"CIK{int(cik):010d}.json"
    if not p.exists(): return pd.DataFrame()
    d = json.loads(p.read_text())
    parts = [d["filings"]["recent"]] + [e for e in d.get("_extra", []) if isinstance(e, dict)]
    f = pd.concat([pd.DataFrame({k: v for k, v in x.items() if isinstance(v, list)}) for x in parts], ignore_index=True)
    return f.drop_duplicates("accessionNumber")

t = pd.read_csv(ROOT / "data" / "universe" / "timeline.csv")
t = t[t.status == "ok"].reset_index(drop=True).head(int(os.environ.get("LIMIT", 10**6)))
docs, lets = [], []
for i, r in t.iterrows():
    accs, dnames, forms = str(r.acc_reg).split(";"), str(r.doc_reg).split(";"), str(r.form_seq).split(";")
    jobs = [("first", accs[0], dnames[0])]
    if len(accs) > 1: jobs.append(("last_pre", accs[-1], dnames[-1]))
    if isinstance(r.acc_424, str): jobs.append(("final_424", r.acc_424, r.doc_424))
    for role, acc, doc in jobs:
        n, st = save(TXT / "docs" / f"{int(r.cik)}_{acc}.txt.gz", r.cik, acc, doc)
        docs.append(dict(key=r.key, role=role, cik=int(r.cik), acc=acc, doc=doc, n_chars=n, status=st))
    f = filings(r.cik)
    if len(f):
        start = pd.to_datetime(r.t_drs if isinstance(r.t_drs, str) else r.t_pub) - pd.Timedelta(days=60)
        end = pd.to_datetime(r.t_424) + pd.Timedelta(days=730)
        dt = pd.to_datetime(f.filingDate)
        f = f[f.form.isin(["UPLOAD", "CORRESP"]) & (dt >= start) & (dt <= end)]
        for _, x in f.iterrows():
            n, st = save(TXT / "letters" / f"{int(r.cik)}_{x.accessionNumber}.txt.gz", r.cik, x.accessionNumber, x.primaryDocument)
            lets.append(dict(key=r.key, cik=int(r.cik), acc=x.accessionNumber, form=x.form, date=x.filingDate,
                             doc=x.primaryDocument, n_chars=n, status=st))
    if i % 25 == 0:
        pd.DataFrame(docs).to_csv(TXT / "docs.csv", index=False); pd.DataFrame(lets).to_csv(TXT / "letters.csv", index=False)
        print(f"{i + 1}/{len(t)} IPOs | docs {len(docs)} | letters {len(lets)}", flush=True)
pd.DataFrame(docs).to_csv(TXT / "docs.csv", index=False); pd.DataFrame(lets).to_csv(TXT / "letters.csv", index=False)
print("done", len(t), "IPOs |", len(docs), "docs |", len(lets), "letters")
