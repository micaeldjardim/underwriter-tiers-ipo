"""Terms of the underwriter warrants and of the non-accountable expense allowance, for the IPOs flagged by
27_issuer_prospectus.py. Read from the final prospectus, in a window after each mention of the underwriter's or
representative's warrants:
- w_cover: warrants as % of the shares sold in the offering, stated as a percentage or as a number of shares divided by
  the shares offered on the cover (accepted between 1% and 15%);
- w_strike: exercise price as % of the offer price (accepted between 100% and 200%);
- w_years: term in years, when stated as 'N years' / 'N-year' in the same window (accepted 3 to 10);
- nonacc: non-accountable expense allowance, % of gross proceeds (accepted 0.25% to 3%), for every IPO.
- w_value: Black-Scholes value of the warrants as % of gross proceeds = w_cover x value of one warrant per dollar of
  offer price, with the stated strike (125% if missing), five years (or the stated term), volatility 60%, r = 2%,
  no dividends. An illustration under stated assumptions, not a market price.
Output: data/universe/warrant_terms.csv"""
import gzip, math, os, pathlib, re, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
TXT = pathlib.Path(os.environ.get("TEXT_DIR", ROOT / "data" / "text"))
p = pd.read_csv(ROOT / "data" / "universe" / "issuer_prospectus.csv")
sh = pd.read_csv(ROOT / "data" / "universe" / "prospectus.csv").set_index("key")["shares_offered"]
cf = ROOT / "data" / "universe" / "cover_fill.csv"
if cf.exists(): sh = sh.where(sh > 1000).fillna(pd.read_csv(cf).set_index("key")["shares_fill"])
ct = ROOT / "data" / "universe" / "cover_total.csv"         # 30b: securities sold = confirmed proceeds / offer price
if ct.exists():
    c = pd.read_csv(ct).set_index("key"); pr0 = pd.read_csv(ROOT / "data" / "universe" / "prospectus.csv").set_index("key")["offer_424"]
    sh = (c.proceeds_m * 1e6 / pr0.reindex(c.index)).round().fillna(sh.reindex(c.index)).combine_first(sh)
sh = sh.to_dict()
ABS = re.compile(r"warrants? (?:to purchase|exercisable for) (?:up to |an aggregate of |a total of )?(\d{1,3}(?:,\d{3})+|\d{4,})(?: shares| ordinary shares| units| common shares| class a| american depositary)")
D = pd.read_csv(TXT / "docs.csv"); D = D[D.role.eq("final_424")].merge(p[["key", "uw_warrants"]], on="key")
PH = ("underwriters' warrant", "underwriter's warrant", "underwriter warrant", "representative's warrant", "representatives' warrant",
      "representative warrant", "underwriters warrant", "compensation warrant",
      "warrants to the underwriter", "warrants to the representative", "to the underwriters warrants", "to the representative warrants")
COV = re.compile(r"(\d{1,2}(?:\.\d+)?) ?%\)? of the (?:aggregate |total )?(?:number of )?(?:shares|units|securities|ordinary shares|adss|american depositary shares|common stock|common shares|class a)")
STR = re.compile(r"(\d{3}(?:\.\d+)?) ?% of the (?:initial )?(?:public )?(?:offering|offer) price")
YRS = re.compile(r"(\d{1,2}|three|four|five|six|seven|ten)[- ]years?")
NW = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10}
NA = re.compile(r"(\d(?:\.\d+)?) ?%")
def N(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def bs(k, t, sig=0.6, r=0.02):                      # call value per $1 of offer price, strike k x offer price
    d1 = (math.log(1 / k) + (r + sig * sig / 2) * t) / (sig * math.sqrt(t)); d2 = d1 - sig * math.sqrt(t)
    return N(d1) - k * math.exp(-r * t) * N(d2)
def mode(v):
    return pd.Series(v).round(2).mode().iloc[0] if v else np.nan
rows = []
for x in D.itertuples():
    f = TXT / "docs" / f"{x.cik}_{x.acc}.txt.gz"
    if not f.exists(): continue
    low = " ".join(gzip.open(f, "rt", encoding="utf-8").read().replace("\u2019", "'").replace("\ufffd", "'").replace("\xa0", " ").split()).lower()
    cov, stk, yrs, na = [], [], [], []
    if x.uw_warrants == 1:
        for ph in PH:
            i = low.find(ph)
            while i != -1:
                seg = low[i:i + 700]
                cov += [float(v) for v in COV.findall(seg) if 1 <= float(v) <= 15]
                n_sh = sh.get(x.key)
                if n_sh and n_sh == n_sh and n_sh > 0:
                    cov += [100 * float(v.replace(",", "")) / n_sh for v in ABS.findall(seg) if 1 <= 100 * float(v.replace(",", "")) / n_sh <= 15]
                stk += [float(v) for v in STR.findall(seg) if 100 <= float(v) <= 200]
                yrs += [int(v) if v.isdigit() else NW[v] for v in YRS.findall(seg) if 3 <= (int(v) if v.isdigit() else NW[v]) <= 10]
                i = low.find(ph, i + 1)
    i = low.find("non-accountable")
    while i != -1:
        na += [float(v) for v in NA.findall(low[max(0, i - 160):i + 200]) if 0.25 <= float(v) <= 3]
        i = low.find("non-accountable", i + 1)
    c, k, t = mode(cov), mode(stk), mode(yrs)
    val = c * bs((k if k == k else 125) / 100, t if t == t else 5) if c == c else np.nan
    rows.append({"key": x.key, "w_cover": c, "w_strike": k, "w_years": t, "nonacc": mode(na), "w_value": val})
r = pd.DataFrame(rows); r.to_csv(ROOT / "data" / "universe" / "warrant_terms.csv", index=False)
m = r.merge(p[["key", "uw_warrants", "spread"]], on="key")
print("with warrants:", int(m.uw_warrants.sum()), "| coverage read:", int(m.w_cover.notna().sum()), "| strike read:", int(m.w_strike.notna().sum()), "| term read:", int(m.w_years.notna().sum()))
print(m[m.uw_warrants == 1][["w_cover", "w_strike", "w_years", "w_value", "nonacc", "spread"]].describe().T.round(2).to_string())
print("coverage values:", m.w_cover.value_counts().head(8).to_dict()); print("strike values:", m.w_strike.value_counts().head(6).to_dict())
print("non-accountable allowance: share of IPOs", round(m.nonacc.notna().mean(), 3), "values", m.nonacc.value_counts().head(5).to_dict())
print("BS value of one warrant per $ of offer (strike 125%, 5y): vol 40%", round(bs(1.25, 5, .4), 3), "60%", round(bs(1.25, 5, .6), 3), "80%", round(bs(1.25, 5, .8), 3))
