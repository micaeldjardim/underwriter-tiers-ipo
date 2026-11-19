"""Issuer-side variables read from the final prospectus (424B) of each IPO in the universe. No downloads: uses the
texts cached by 21_text_download.py (data/text, or TEXT_DIR).
- spread: underwriting discount per share / offer price (%), from the cover table (first figure after the words
  'underwriting discount(s)/commission(s)' in the opening pages that lies between 1% and 15% of the offer price).
- uw_warrants: the underwriter or its representative receives warrants as compensation (placement-agent warrants are
  not counted: in these prospectuses they mostly belong to private rounds before the IPO).
- best_efforts: the opening pages say the offering is on a best-efforts basis.
- own_pre, own_post: beneficial ownership (%) of all directors and executive officers as a group, before and after
  the offering, from the principal-stockholders table; group_n = number of persons in that group.
- founder_ceo: the Chief Executive Officer is described as founder or co-founder in the same clause.
- going_concern: the prospectus mentions substantial doubt about the ability to continue as a going concern.
Every rule finds an anchor phrase with str.find and reads a short window around it (plain regex over the whole
document is slow when the phrase is absent). Output: data/universe/issuer_prospectus.csv"""
import gzip, os, pathlib, re, sys, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
TXT = pathlib.Path(os.environ.get("TEXT_DIR", ROOT / "data" / "text"))
# every matched IPO (1,999), not only the 1,039 of the analysis file: paper B uses a wider sample
pr = pd.read_csv(ROOT / "data" / "universe" / "prospectus.csv")
D = pd.read_csv(TXT / "docs.csv"); D = D[D.role.eq("final_424") & D.key.isin(pr.key)]
offer = pr.set_index("key")["offer_424"]
cf = ROOT / "data" / "universe" / "cover_fill.csv"          # from 30a, when it has been run
if cf.exists(): offer = offer.fillna(pd.read_csv(cf).set_index("key")["offer_fill"])
offer = offer.to_dict()
NUM = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)")
PCT = re.compile(r"(\d{1,3}(?:\.\d+)?) ?%")
GRPN = re.compile(r"^ ?\( ?(\d+|[a-z]+) (?:persons?|individuals?|people)")
WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}

def anchors(low, phrase, limit=None):
    i = low.find(phrase); n = 0
    while i != -1 and (limit is None or n < limit):
        yield i; n += 1
        i = low.find(phrase, i + 1)

def spread(low, p):
    if not p or np.isnan(p): return np.nan
    head = low[:70000]
    for ph in ("underwriting discount", "underwriting commission", "underwriters' discount", "underwriter discount", "placement agent fee"):
        for i in anchors(head, ph):
            for n in NUM.finditer(head[i + len(ph):i + len(ph) + 260]):
                if "." not in n.group(1): continue
                v = float(n.group(1).replace(",", ""))
                if 0.01 * p <= v <= 0.15 * p: return v / p * 100
    return np.nan

def owners(low):
    out = (np.nan, np.nan, np.nan)
    for i in anchors(low, "as a group"):
        seg = low[i + 10:i + 430]
        m = GRPN.match(seg)
        if not m: continue
        g = m.group(1); n = int(g) if g.isdigit() else WORDS.get(g, np.nan)
        cut = [seg.find(s) for s in (" 5% ", "other selling", "selling stockholders", "selling shareholders", "less than", "represents", " * represents")]
        cut = [c for c in cut if c > 60]
        if cut: seg = seg[:min(cut)]
        pct = [float(x) for x in PCT.findall(seg)]; pct = [x for x in pct if x <= 100]
        if pct: return n, pct[0], (pct[-1] if len(pct) > 1 else np.nan)
        out = (n, np.nan, np.nan)
    return out

def founder_ceo(low):
    for i in anchors(low, "chief executive officer"):
        w0 = max(0, i - 130); left = low[w0:i]; right = low[i + 23:i + 153]
        for stop in ".;":                       # stay inside the clause
            if stop in left: left = left[left.rfind(stop) + 1:]
            if stop in right: right = right[:right.find(stop)]
        if "founder" in left or "founded" in left or "founder" in right or "founded" in right:
            return 1, (left + "chief executive officer" + right).strip()[:220]
    return 0, ""

def warrants(low):
    for ph in ("underwriters' warrant", "underwriter's warrant", "underwriter warrant", "representative's warrant", "representatives' warrant",
               "representative warrant", "underwriters warrant"):
        if ph in low: return 1
    for i in anchors(low, "warrant"):
        seg = low[i:i + 170]
        if re.search(r"warrants? (?:to purchase [^.]{0,110} )?(?:issued |issuable |to be issued )?to the (?:underwriter|representative)", seg): return 1
    return 0

def going_concern(low):
    for i in anchors(low, "substantial doubt"):
        if "going concern" in low[max(0, i - 220):i + 240]: return 1
    return 0

rows = []; T0 = time.time()
for k, x in enumerate(D.itertuples()):
    if k % 200 == 0: print(k, round(time.time() - T0), flush=True)
    p = TXT / "docs" / f"{x.cik}_{x.acc}.txt.gz"
    if not p.exists(): continue
    tx = gzip.open(p, "rt", encoding="utf-8").read()
    low = " ".join(tx.replace("\u2019", "'").replace("\ufffd", "'").replace("\xa0", " ").split()).lower()
    n, pre, post = owners(low); fc, fct = founder_ceo(low)
    rows.append({"key": x.key, "spread": spread(low, offer.get(x.key)), "uw_warrants": warrants(low),
                 "best_efforts": int("best efforts" in low[:30000] or "best-efforts" in low[:30000]),
                 "group_n": n, "own_pre": pre, "own_post": post, "founder_ceo": fc, "founder_ceo_text": fct,
                 "founder_any": int("founder" in low), "going_concern": going_concern(low), "n_chars": len(tx)})
r = pd.DataFrame(rows)
r.to_csv(ROOT / "data" / "universe" / "issuer_prospectus.csv", index=False)
print("N", len(r)); print(r.drop(columns=["key", "founder_ceo_text"]).describe().T[["count", "mean", "50%", "min", "max"]].round(2).to_string())
