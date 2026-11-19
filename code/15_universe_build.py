"""Analysis file for the universe: timeline (11) + cover page (12) + auditor (12b) + XBRL financials (13) + prices (14)
+ Ritter founding-date variables. Sample filters follow the JOBS Act literature (Chaplinsky-Hanley-Moon 2017, Dambra-Field-
Gustafson 2015, Even-Tov et al.): drop financials (SIC 6xxx), offer price < $5, foreign issuers (F-1 or business address
outside the US), registrations started before the JOBS Act (5 Apr 2012) and IPOs without a first-day return.
Every drop is logged in output/universe_sample_log.csv. Output: data/universe/analysis.csv"""
import pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
U = ROOT / "data" / "universe"
t = pd.read_csv(U / "timeline.csv", parse_dates=["t_issue", "t_pub", "t_drs"])
t = t[t.status == "ok"]
for f in ["prospectus", "auditor", "financials", "prices"]:
    t = t.merge(pd.read_csv(U / f"{f}.csv"), on="key", how="left")
mk = pd.read_csv(ROOT / "data" / "market_index.csv", index_col=0, parse_dates=True)
# third source for the first-day return: the Eikon-based return of the original 712-IPO file (same first trading day)
f_ek = ROOT / "data" / "analysis.csv"  # licensed (Refinitiv Eikon), not included
ek = (pd.read_csv(f_ek, parse_dates=["t_issue"]) if f_ek.exists() else pd.DataFrame(columns=["tickersymbol", "t_issue", "ir"]))[["tickersymbol", "t_issue", "ir"]].rename(columns={"ir": "ir_ek"})
t = t.merge(ek.dropna().drop_duplicates(["tickersymbol", "t_issue"]), on=["tickersymbol", "t_issue"], how="left")
t["ir_src"] = t.ir_src.where(t.ir.notna(), np.where(t.ir_ek.notna(), "eikon", None))
t["ir"] = t.ir.fillna(t.ir_ek)

US = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR".split())
t["sic"] = pd.to_numeric(t.sic, errors="coerce")
t["foreign"] = ((t.form_reg == "F-1") | ~t.biz_state.isin(US)).astype(int)
t["offer"] = t.offer_424
t["start"] = t[["t_drs", "t_pub"]].min(axis=1)
log = [("matched to EDGAR (Ritter universe 2012-2022)", len(t))]
def keep(mask, why):
    global t
    t = t[mask]; log.append((why, len(t)))
keep(~t.sic.between(6000, 6799), "drop financials (SIC 6000-6799)")
keep(t.offer.ge(5), "drop offer price < $5 or missing")
keep(t.foreign.eq(0), "drop foreign issuers (F-1 / non-US address)")
keep(t.start.ge("2012-04-05"), "drop registrations started before the JOBS Act")
keep(t.ir.notna(), "drop IPOs without first-day return")
pd.DataFrame(log, columns=["step", "N"]).to_csv(ROOT / "output" / "universe_sample_log.csv", index=False)
print(pd.DataFrame(log, columns=["step", "N"]).to_string(index=False))

# clocks and review
t["pub"] = t.pub_window
t["lnpub"] = np.log(t.pub); t["lntotal"] = np.log(t.total_dur)
t["lnconf"] = np.log(t.conf_period.where(t.conf_period > 0))
t["pub_share"] = 100 * t.pub / t.total_dur
t["lnamend"] = np.log1p(t.n_pub_amend); t["lndrs"] = np.log1p(t.n_drs_all)
# underwriter: Ritter rank of the first bookrunner on the cover (lead-left); top tier = rank 9
t["uw"] = t.uw_lead; t["top"] = (t.uw >= 9).astype(int)
t["lnbanks"] = np.log(t.n_bookrunners)
# pricing
t["mid0"] = (t.range_lo0 + t.range_hi0) / 2
t["rev"] = 100 * (t.offer / t.mid0 - 1)
t.loc[~t.rev.between(-80, 150), "rev"] = np.nan
t["rev_pos"] = t.rev.clip(lower=0); t["rev_neg"] = t.rev.clip(upper=0)
t["width"] = 100 * (t.range_hi0 - t.range_lo0) / t.mid0
# firm
yr = t.t_issue.dt.year
t["founding"] = pd.to_numeric(t.founding, errors="coerce")
t["lnage"] = np.log1p((yr - t.founding).clip(lower=0))
vc = pd.to_numeric(t.vc_r, errors="coerce")       # Ritter: 1 = VC-backed, 2 = the growth-capital subset of VC
t["vc"] = (vc >= 1).astype(float).where(vc.notna()); t["growth_cap"] = (vc == 2).astype(int); t["dual"] = pd.to_numeric(t.dual, errors="coerce")
t["internet"] = pd.to_numeric(t.internet, errors="coerce")
usd = t.assets_unit.eq("USD")
t["lnassets"] = np.log(t.assets.where(usd & (t.assets > 0)))
t["loss"] = (t.net_income < 0).astype(float).where(t.net_income.notna())
t["rnd_int"] = (t.rnd.fillna(0) / t.assets).where(usd & (t.assets > 0)).clip(upper=5)
t["rnd_pos"] = (t.rnd.fillna(0) > 0).astype(int)
t["lev"] = (t.liabilities / t.assets).where(usd & (t.assets > 0)).clip(upper=5)
t["shares"] = t.shares_offered.where(t.shares_offered > 1000)
t["lnproc"] = np.log(t.offer * t.shares / 1e6)
# Correction of 19 Nov 2026: offer price x shares fails when the cover states the securities in an unusual way (units,
# ADSs, several classes). Where 30b_cover_total.py confirms gross proceeds by two readings of the cover and the value
# differs, the confirmed value replaces the product. IPOs whose proceeds were missing stay missing, so the sample of
# IPOs with every control is unchanged.
_ct = U / "cover_total.csv"
if _ct.exists():
    _c = pd.read_csv(_ct); _c = _c[_c.proceeds_m.notna() & (_c.single == 0)].set_index("key")["proceeds_m"]
    _new = np.log(t.key.map(_c))
    _fix = t.lnproc.notna() & _new.notna() & ((np.exp(_new) - np.exp(t.lnproc)).abs() / np.exp(_new) > 0.02)
    print("proceeds corrected from the cover:", int(_fix.sum()))
    t.loc[_fix, "lnproc"] = _new[_fix]
t["secondary"] = (t.secondary_shares > 0).astype(int)
t["nasdaq"] = t.exch_424.fillna("").str.contains("NASDAQ").astype(int)
t["sic2"] = (t.sic // 100).astype(int)
BIO = {2834, 2835, 2836, 8731}
TECH = {3571, 3572, 3575, 3577, 3578, 3661, 3663, 3669, 3812, 3823, 3825, 3826, 3827, 3829, 3841, 3845, 4812, 4813, 4899} | set(range(3671, 3680)) | set(range(7370, 7380))
t["biotech"] = t.sic.isin(BIO).astype(int)
t["tech"] = (t.sic.isin(TECH) | t.internet.eq(1)).astype(int)
# industry for fixed effects: 2-digit SIC with biotech as its own group (Dambra et al. convention)
t["ind"] = np.where(t.biotech == 1, "bio", t.sic2.astype(str))
# market conditions
def ret(col, d, days):
    a = mk[col].asof(d - pd.Timedelta(days=days)); b = mk[col].asof(d - pd.Timedelta(days=1))
    return 100 * (b / a - 1)
t["nasdaq90"] = [ret("^IXIC", d, 90) for d in t.t_issue]
t = t.sort_values("t_issue").reset_index(drop=True)
t["n_ipo90"] = [((t.t_issue < d) & (t.t_issue >= d - pd.Timedelta(days=90))).sum() for d in t.t_issue]
t["lnipo90"] = np.log1p(t.n_ipo90)
t["ir90"] = [t.ir[(t.t_issue < d) & (t.t_issue >= d - pd.Timedelta(days=90))].mean() for d in t.t_issue]
# JOBS Act status and the 2017 extension
t["post17"] = (t.start >= "2017-07-10").astype(int)
t["nonegc"] = (t.egc == 0).astype(int)
t["yq"] = t.t_issue.dt.to_period("Q").astype(str)
t["ipo_year"] = t.t_issue.dt.year
# aftermarket (survivors only, Yahoo): first-day close -> ~90 days, minus IWM
t["r90"] = 100 * (t.yf_close90 / t.yf_close1 - 1)
t["m90"] = [100 * (mk.IWM.asof(b) / mk.IWM.asof(a) - 1) if pd.notna(b) else np.nan for a, b in zip(t.t_issue, pd.to_datetime(t.yf_day90))]
t["ar90"] = t.r90 - t.m90
t.to_csv(U / "analysis.csv", index=False)
cols = ["pub", "total_dur", "conf_period", "uw", "ir", "rev", "lnage", "vc", "lnassets", "loss", "rnd_int", "lnproc", "big4", "nasdaq", "biotech", "tech", "egc", "confidential", "lockup_days", "ar90"]
print(t[cols].describe().T.round(2).to_string())
print("missing share:\n", t[cols].isna().mean().round(3).to_string())
