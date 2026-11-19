"""Paper B sample. Paper A's filters (offer price >= $5, U.S. issuers, registrations after the JOBS Act, a first-day
return) serve its study of SEC review; they remove the smallest offerings, which are the subject of paper B. Here the
sample is every IPO in the Field-Ritter list for 2012-2022 that is matched to EDGAR and has a readable final
prospectus, minus financial firms (SIC 6000-6799), with the offer price and shares sold on the cover.
Variables are built as in 15_universe_build.py; compensation comes from 27 and 27b. Assets are used only when XBRL
reports them in U.S. dollars. Lead bank = first bookrunner on the cover (lead-left), as in 12.
Output: data/universe/paper_b.csv and output/paper_b_sample_log.csv"""
import pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
U = ROOT / "data" / "universe"
t = pd.read_csv(U / "timeline.csv", parse_dates=["t_issue", "t_pub", "t_drs"])
t = t[t.status == "ok"]
for f in ["prospectus", "financials"]:
    t = t.merge(pd.read_csv(U / f"{f}.csv"), on="key", how="left")
ip = pd.read_csv(U / "issuer_prospectus.csv").drop(columns=["founder_ceo_text"])
wt = pd.read_csv(U / "warrant_terms.csv")
t = t.merge(ip, on="key", how="left").merge(wt, on="key", how="left")
cf = pd.read_csv(U / "cover_fill.csv")              # 30a: price and securities read again from the cover where 12 left them blank
t = t.merge(cf, on="key", how="left")
t["offer_424"] = t.offer_424.fillna(t.offer_fill)
t["shares_offered"] = t.shares_offered.where(t.shares_offered > 1000).fillna(t.shares_fill)
log = [("matched to EDGAR (Ritter universe 2012-2022)", len(t))]
def keep(mask, why):
    global t
    t = t[mask]; log.append((why, len(t)))
t["sic"] = pd.to_numeric(t.sic, errors="coerce")
keep(~t.sic.between(6000, 6799), "drop financials (SIC 6000-6799)")
keep(t.n_chars.notna() & (t.n_chars > 20000), "final prospectus readable")
ct = pd.read_csv(U / "cover_total.csv")[["key", "proceeds_m", "single"]]      # 30b: proceeds by consensus of three cover readings
t = t.merge(ct, on="key", how="left")
keep(t.offer_424.notna() & t.proceeds_m.notna() & (t.single == 0), "offer price read and gross proceeds confirmed by two cover readings")
keep(t.spread.notna(), "underwriting discount read on the cover")
US = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR".split())
t["foreign"] = ((t.form_reg == "F-1") | ~t.biz_state.isin(US)).astype(int)
t["offer"] = t.offer_424; t["proc_m"] = t.proceeds_m; t["lnproc"] = np.log(t.proc_m)
t["penny"] = (t.offer < 5).astype(int)
t["uw"] = t.uw_lead
t["lead_bank"] = t.banks.fillna("").str.split("|").str[0].replace("", np.nan)
yr = t.t_issue.dt.year; t["ipo_year"] = yr; t["yq"] = t.t_issue.dt.to_period("Q").astype(str)
t["founding"] = pd.to_numeric(t.founding, errors="coerce")
t["lnage"] = np.log1p((yr - t.founding).clip(lower=0))
vc = pd.to_numeric(t.vc_r, errors="coerce")
t["vc"] = (vc >= 1).astype(float).where(vc.notna()); t["growth_cap"] = (vc == 2).astype(float).where(vc.notna())
t["dual"] = pd.to_numeric(t.dual, errors="coerce")
usd = t.assets_unit.eq("USD")
t["assets_m"] = (t.assets / 1e6).where(usd & (t.assets > 0)); t["lnassets"] = np.log(t.assets_m * 1e6)
t["loss"] = (t.net_income < 0).astype(float).where(t.net_income.notna())
t["rnd_int"] = (t.rnd.fillna(0) / t.assets).where(usd & (t.assets > 0)).clip(upper=5)
t["nasdaq"] = t.exch_424.fillna("").str.contains("NASDAQ").astype(int)
t["sic2"] = (t.sic // 100).astype("Int64")
BIO = {2834, 2835, 2836, 8731}
TECH = {3571, 3572, 3575, 3577, 3578, 3661, 3663, 3669, 3812, 3823, 3825, 3826, 3827, 3829, 3841, 3845, 4812, 4813, 4899} | set(range(3671, 3680)) | set(range(7370, 7380))
t["biotech"] = t.sic.isin(BIO).astype(int)
t["tech"] = (t.sic.isin(TECH) | pd.to_numeric(t.internet, errors="coerce").eq(1)).astype(int)
t["ind"] = np.where(t.biotech == 1, "bio", t.sic2.astype(str))
# compensation
t["above7"] = (t.spread > 7.001).astype(int); t["at7"] = t.spread.between(6.999, 7.001).astype(int)
med = t.loc[t.uw_warrants == 1, "w_value"].median()
t["wv"] = np.where(t.uw_warrants == 1, t.w_value.fillna(med), 0.0)
t["na"] = t.nonacc.fillna(0.0)
t["direct"] = t.spread + t.wv + t.na
t["direct_usd_m"] = t.direct / 100 * t.proc_m
t["small_contract"] = ((t.uw_warrants == 1) | (t.above7 == 1)).astype(int)
t["band"] = pd.cut(t.proc_m, [0, 10, 25, 50, 100, 250, 1e9], labels=["<10", "10-25", "25-50", "50-100", "100-250", ">250"])
t["second"] = (t.uw < 5).astype(float).where(t.uw.notna()); t["elite"] = (t.uw >= 9).astype(float).where(t.uw.notna())
t["tier"] = np.select([t.uw.isna(), t.uw < 5, t.uw >= 9], ["unranked", "bottom", "elite"], "middle")
pd.DataFrame(log, columns=["step", "N"]).to_csv(ROOT / "output" / "paper_b_sample_log.csv", index=False)
print(pd.DataFrame(log, columns=["step", "N"]).to_string(index=False))
print(f"median warrant value used where terms unread: {med:.2f}")
t.to_csv(U / "paper_b.csv", index=False)
print(t[["proc_m", "assets_m", "vc", "foreign", "penny", "uw", "spread", "uw_warrants", "w_cover", "direct", "small_contract"]].describe().T.round(2).to_string())
print("tier:", t.tier.value_counts().to_dict()); print("band:", t.band.value_counts().sort_index().to_dict())
