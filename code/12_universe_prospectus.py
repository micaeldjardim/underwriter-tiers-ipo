"""Cover-page variables for the universe (data/universe/timeline.csv), from EDGAR text only:
- pricing prospectus (424B): offer price, shares offered, over-allotment shares, secondary shares (selling holders),
  EGC status, listing exchange, and the bookrunners named on the cover (matched to Ritter's public underwriter ranks)
- registration filings: first stated price range and last range before pricing (same rules as 03)
Underwriter rank: Ritter's Underwriter-Rank.xls (public), period column matched to the IPO year.
Output: data/universe/prospectus.csv"""
import re, sys, html, time, json, pathlib, requests, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "universe"
UA = {"User-Agent": __import__("os").environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")}  # SEC asks for a contact
CACHE = OUT / "prospectus_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
t = pd.read_csv(OUT / "timeline.csv")
t = t[t.status == "ok"]

def text_of(cik, acc, doc, limit=600_000):
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
    for k in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=40, stream=True)
            if r.status_code == 200:
                raw = b""
                for chunk in r.iter_content(65536):
                    raw += chunk
                    if len(raw) >= limit: break
                r.close(); time.sleep(0.12)
                s = raw.decode("utf-8", "ignore")
                s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
                s = re.sub(r"<[^>]+>", " ", s)
                return re.sub(r"\s+", " ", html.unescape(s).replace("\xa0", " "))
            if r.status_code == 404: return ""
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    return ""

NUM = r"(?:US)?\$\s?(\d{1,3}(?:\.\d{1,2})?)"
RANGE = re.compile(r"between\s+" + NUM + r"\s+and\s+" + NUM, re.I)
RANGE2 = re.compile(NUM + r"\s+(?:to|and)\s+" + NUM + r"\s+per\s+(?:share|ADS|American Depositary Share)", re.I)
EGC = re.compile(r"we\s+are\s+an?\s+[\"“']?\s*emerging\s+growth\s+compan", re.I)
OFFER = re.compile(r"(?:initial\s+)?public\s+offering\s+price\s+(?:of|is|per\s+share\s+is)?\s*" + NUM, re.I)
SHARES = re.compile(r"([\d,]{5,})\s+(?:shares|American Depositary Shares|ADSs)\b", re.I)
GREEN = re.compile(r"(?:option|right)\s+to\s+purchase\s+(?:up\s+to\s+)?(?:an\s+)?(?:additional\s+)?([\d,]{4,})\s+(?:additional\s+)?(?:shares|ADSs)", re.I)
SELL = re.compile(r"selling\s+(?:stock|share)holders?\s+(?:named\s+in\s+this\s+prospectus\s+)?(?:are|is)\s+offering\s+([\d,]{4,})", re.I)
EXCH = re.compile(r"(Nasdaq|New York Stock Exchange|NYSE American|NYSE MKT|NYSE)", re.I)

def find_range(s):
    for rx in (RANGE, RANGE2):
        for m in rx.finditer(s):
            lo, hi = float(m.group(1)), float(m.group(2))
            if 1 <= lo <= hi <= 200 and hi - lo <= 0.5 * hi: return lo, hi
    return None

# Ritter underwriter ranks -> searchable aliases
rk = pd.read_excel(ROOT / "data" / "external" / "Underwriter-Rank.xls", sheet_name="Sheet1").set_index("Underwriter Name")
PERIODS = [("Rank1011", 2010, 2011), ("Rank1217", 2012, 2017), ("Rank1820", 2018, 2020), ("Rank2122", 2021, 2022)]
# how each bank is printed on the cover (lowercase, punctuation stripped) -> Ritter row(s); ranks = max over the rows listed
BANKS = {
    r"goldman sachs": ["Goldman Sachs & Co"], r"morgan stanley": ["Morgan Stanley & Co"], r"j ?p ?morgan": ["JP Morgan (JPM)"],
    r"bofa|bank of america|baml": ["Bank of America-Merrill Lynch (BOA-Merrill)"], r"merrill lynch": ["Merrill Lynch & Co Inc", "Bank of America-Merrill Lynch (BOA-Merrill)"],
    r"citigroup|citi\b": ["Citigroup", "Citigroup Global Markets Inc"], r"credit suisse": ["Credit Suisse"], r"deutsche bank": ["Deutsche Bank Securities Corp"],
    r"barclays": ["Barclays Capital"], r"ubs": ["UBS Investment Bank"], r"wells fargo": ["Wells Fargo"], r"rbc": ["RBC Capital Markets"],
    r"jefferies": ["Jefferies & Co Inc"], r"evercore": ["Evercore"], r"cowen": ["Cowen", "TD Cowen"], r"piper sandler": ["Piper-Sandler"],
    r"piper jaffray": ["Piper Jaffray Inc", "Piper Jaffray Cos"], r"svb leerink|leerink": ["SVB-Leerink", "Leerink Swann & Co.", "Leerink-SVB"],
    r"stifel": ["Stifel", "Stifel Nicolaus & Co Inc"], r"baird": ["Robert W Baird & Co Inc"], r"william blair": ["William Blair & Co"],
    r"raymond james": ["Raymond James & Associates Inc"], r"keybanc": ["KeyBanc Capital Markets "], r"suntrust": ["SunTrust Robinson Humphrey"],
    r"truist": ["Truist Securities"], r"oppenheimer": ["Oppenheimer & Co Inc"], r"canaccord": ["Canaccord Genuity"], r"needham": ["Needham & Co Inc"],
    r"btig": ["BTIG"], r"guggenheim": ["Guggenheim Securities"], r"nomura": ["Nomura Securities"], r"hsbc": ["HSBC"], r"mizuho": ["Mizuho Securities"],
    r"macquarie": ["Macquarie Bank"], r"bmo": ["BMO Capital Markets"], r"jmp": ["JMP-Sec"], r"wedbush": ["Wedbush Morgan Securities"],
    r"roth": ["Roth Capital Partners Inc"], r"craig hallum": ["Craig-Hallum, Inc."], r"b riley|b ?riley": ["B-Riley"], r"fbr": ["Friedman Billings Ramsey Group (FBR Group)"],
    r"ladenburg": ["Ladenburg Thalmann & Co"], r"maxim": ["Maxim Group LLC"], r"aegis": ["Aegis Capital"], r"thinkequity|think equity": ["Thinkequity Partners LLC"],
    r"ef hutton": ["EF Hutton & Co Inc (EFH)"], r"boustead": ["Boustead Securities"], r"network 1": ["Network 1 Financial Securities"],
    r"joseph gunnar": ["Joseph Gunnar & Co., LLC"], r"alexander capital": ["Alexander Capital, LP"], r"univest": ["Univest Securities"],
    r"benchmark": ["The Benchmark Company"], r"lake street": ["Lake Street Capital Markets"], r"chardan": ["Chardan Capital Markets, LLC"],
    r"h ?c wainwright": ["HC Wainwright & Co Inc"], r"cantor": ["Cantor, Fitzgerald & Co., Inc."], r"keefe|kbw": ["Keefe Bruyette & Woods Inc"],
    r"sandler o ?neill": ["Sandler O'Neill Partners"], r"janney": ["Janney Montgomery Scott Inc"], r"northland": ["Northland Capital Markets"],
    r"imperial capital": ["Imperial-Cap"], r"allen": ["Allen & Co Inc"], r"lazard": ["Lazard"], r"moelis": ["Moelis"], r"pjt": ["PJT Partners"],
    r"loop capital": ["Loop Capital"], r"drexel hamilton": ["Drexel Hamilton"], r"stephens": ["Stephens Inc"], r"davidson": ["DA Davidson & Co Inc"],
    r"tudor pickering": ["Tudor-Pickering"], r"simmons": ["Simmons & Co"], r"societe generale|soci t g n rale": ["Societe Generale"],
    r"bnp": ["BNP Paribas SA (BNP-CA)"], r"credit agricole": ["Credit Agricole Securities (CNCANY)"], r"santander": ["Santander Investment Bank"],
    r"itau": ["Itau BBA"], r"bradesco": ["Banco Bradesco"], r"btg pactual": ["BTG Pactual"], r"scotia": ["Scotia Capital Inc"], r"td securities": ["TD Securities Inc"],
    r"cibc": ["CIBC Oppenheimer"], r"smbc|nikko": ["SMBC Nikko"], r"kkr": ["KKR Capital"], r"tpg": ["TPG Capital"], r"china international capital|cicc": ["China International Capital Corp (CICC)"],
    r"china renaissance": ["China Renaissance (CRS)"], r"cmb international": ["CMB International Capital Ltd"], r"haitong": ["Haitong International"],
    r"huatai": ["Huatai Securities"], r"futu": ["Futu Securities International (HK)"], r"tiger brokers": ["Tiger Brokers (NZ)"], r"us tiger": ["US Tiger"],
    r"amtd": ["AMTD Global Markets"], r"clsa": ["CLSA"], r"berenberg": ["Berenberg"], r"academy securities": [], r"siebert": [], r"ramirez": [],
    r"national securities": ["National Securities Corp"], r"dawson james": ["Dawson\xa0James\xa0Securities,\xa0Inc."], r"kingswood": ["Kingswood"],
    r"spartan": ["Spartan Group (Spartan Capital)"], r"brookline": ["Brookline"], r"jones ?trading": ["Jones Trading"], r"lifesci": ["LifeSci Capital"],
    r"earlybird": ["EarlyBirdCapital Inc"], r"i ?bankers": ["I-Bankers Securities Inc"], r"paulson": ["Paulson Investment Co"], r"feltl": ["Feltl & Co"],
    r"a ?g ?p|alliance global": ["A.G.P/Alliance Global Partners"], r"wallachbeth": ["WallachBeth Capital"], r"laidlaw": ["Laidlaw & Co (UK)", "Laidlaw Global Securities"],
    r"newbridge": ["Newbridge Securities"], r"r ?f lafferty": ["R. F. Lafferty"], r"axiom": ["Axiom-Cap"], r"mdb capital": ["MDB-Capital"], r"valuable capital": ["Valuable Capital"],
    r"viewtrade": ["VIEWT-SEC"], r"prime number": ["Prime Number Capital (PNCPS)"], r"revere": ["Revere Securities"], r"pacific century": ["Pacific Century"],
    r"craft capital": ["Craft Capital"], r"joseph stone": ["Joseph Stone Capital"], r"west park": ["West Park Capital"], r"odeon": ["Odeon Capital Group"],
    r"seaport global": ["Seaport Global Securities"], r"oak ridge": ["Oak Ridge Financial"], r"hovde": ["Hovde Group"], r"fig partners": ["FIG Partners"],
    r"performance trust": ["Performance Trust"], r"pacific crest": ["Pacific Crest Securities Inc"], r"zelman": ["Zelman"], r"summer street": ["Summer Street"],
    r"burnham": ["Burnham Securities Inc"], r"sterne agee": ["Sterne Agee & Leach Inc"], r"wunderlich": ["Wunderlich Securities, Inc."], r"rbs": ["RBS Securities (Royal Bk Scotland)"],
    r"liontree": ["Liontree"], r"code advisors": ["Code Advisors"], r"sofi": ["SoFi Securities"], r"rabo": ["Rabo Securities"], r"ameriprise": ["Ameriprise-Fin"],
    r"dougherty": [], r"capitol securities": [], r"mitsubishi|mufg": ["Mitsubishi"], r"ing\b": [], r"natixis": [], r"ubs securities": ["UBS Investment Bank"],
}
def rank_rows(rows, col):
    v = pd.to_numeric(rk.loc[[x for x in rows if x in rk.index], col], errors="coerce") if rows else pd.Series(dtype=float)
    v = v[v >= 0]
    return round(v.max()) if len(v) else None     # Ritter adds .001/.501 as tie-breakers; keep the integer scale
ranks = {k: {p: rank_rows(v, p) for p, *_ in PERIODS} for k, v in BANKS.items()}
ARX = re.compile(r"\b(" + "|".join(sorted(BANKS, key=len, reverse=True)) + r")\b")
MONTH = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
DELIV = re.compile(r"\bon (?:or about )?" + MONTH + r" \d{1,2},? \d{4}[^.]{0,200}\.")
END = re.compile(r"the date of this prospectus|prospectus dated|" + MONTH + r" \d{1,2},? \d{4}|table of contents")
TABLE = re.compile(r"underwriters?\s+number\s+of\s+(?:shares|adss|american depositary)")

def match_banks(block):
    block = re.sub(r"\s+", " ", re.sub(r"[^a-z& ]", " ", block.replace(".", "")))
    found = []
    for x in ARX.finditer(block):
        k = next(p for p in sorted(BANKS, key=len, reverse=True) if re.fullmatch(p, x.group(1)))
        if k not in found: found.append(k)
    return found

def banks_on_cover(s, full=""):
    """Banks on the cover: right after the delivery sentence ('on or about <date>.'), up to the prospectus dateline,
    a bare date or the table of contents. Covers that show the banks only as logos fall back to the first
    'Underwriter / Number of shares' table in the underwriting section (lead-left bank listed first)."""
    low = s.lower()
    for d in DELIV.finditer(low):
        seg = low[d.end():d.end() + 700]
        e = END.search(seg)
        found = match_banks(seg[:e.start() if e else 400])
        if found: return found, "cover"
    lowf = full.lower()
    for m in TABLE.finditer(lowf):
        found = match_banks(lowf[m.end():m.end() + 1200].split("total")[0])
        if found: return found, "table"
    return [], None

def num(x): return float(x.replace(",", "")) if x else None

rows = []
for i, r in enumerate(t.itertuples()):
    key = r.key
    if key in cache: rows.append(cache[key]); continue
    out = {"key": key}
    seq = list(zip(r.acc_reg.split(";"), r.doc_reg.split(";"), r.form_seq.split(";"), r.date_seq.split(";"))) if isinstance(r.acc_reg, str) else []
    ranges, first_i = [], None
    for k, (acc, doc, fm, dt) in enumerate(seq):
        rg = find_range(text_of(r.cik, acc, doc, 250_000))
        if rg: ranges.append((dt, fm) + rg); first_i = k; break
    if ranges:
        for acc, doc, fm, dt in reversed(seq[first_i + 1:]):
            rg = find_range(text_of(r.cik, acc, doc, 250_000))
            if rg: ranges.append((dt, fm) + rg); break
        out |= {"range_lo0": ranges[0][2], "range_hi0": ranges[0][3], "range_date0": ranges[0][0],
                "range_lo1": ranges[-1][2], "range_hi1": ranges[-1][3], "n_ranges": len(ranges)}
    p = text_of(r.cik, r.acc_424, r.doc_424, 600_000)
    if p:
        cover = p[:40_000]
        out["egc"] = int(bool(EGC.search(p)))
        m = OFFER.search(p); out["offer_424"] = float(m.group(1)) if m else None
        m = SHARES.search(cover); out["shares_offered"] = num(m.group(1)) if m else None
        m = GREEN.search(p[:200_000]); out["greenshoe"] = num(m.group(1)) if m else None
        m = SELL.search(cover); out["secondary_shares"] = num(m.group(1)) if m else 0.0
        m = EXCH.search(cover); out["exch_424"] = m.group(1).upper().replace("NEW YORK STOCK EXCHANGE", "NYSE") if m else None
        b, how = banks_on_cover(cover, p)
        out["banks"] = "|".join(b); out["n_bookrunners"] = len(b) if how == "cover" else None; out["banks_src"] = how
    rows.append(out); cache[key] = out
    if i % 25 == 0:
        CACHE.write_text(json.dumps(cache)); print(i, key, out, flush=True)
CACHE.write_text(json.dumps(cache))
d = pd.DataFrame(rows).merge(t[["key", "t_issue"]], on="key")
yr = pd.to_datetime(d.t_issue).dt.year
col = yr.map(lambda y: next(p for p, a, b in PERIODS if a <= y <= b) if 2010 <= y <= 2022 else None)
def rank_of(banks, c, how):
    v = [ranks[a][c] for a in str(banks).split("|") if a in ranks and ranks[a][c] is not None]
    return (v[0] if how == "lead" else max(v)) if v else None
d["uw_lead"] = [rank_of(b, c, "lead") for b, c in zip(d.banks, col)]
d["uw_max"] = [rank_of(b, c, "max") for b, c in zip(d.banks, col)]
d.drop(columns="t_issue").to_csv(OUT / "prospectus.csv", index=False)
print(d.describe().round(2).T.to_string()); print("bank found:", d.banks.fillna("").ne("").mean().round(3))
