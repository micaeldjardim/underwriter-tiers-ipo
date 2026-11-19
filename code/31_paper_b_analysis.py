"""Paper B analysis on the wider sample of 30_paper_b_data.py. Writes output/paper_b_results.txt
1 contract by offering size; 2 where the step is (grid of cutoffs); 3 dollar floor (mechanism); 4 bank or size
(lead-bank fixed effects, within-bank comparisons, variance shares); 5 venture backing, access and cost by thirds of
assets, with a decomposition and matching; 6 robustness (paper A's filters, without 2021-22, U.S. only).
OLS; standard errors clustered two ways by industry and issue quarter, and by IPO year."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pathlib, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
s = pd.read_csv(ROOT / "data" / "universe" / "paper_b.csv")
s["band"] = pd.Categorical(s.band, ["<10", "10-25", "25-50", "50-100", "100-250", ">250"], ordered=True)
lines = []
# warrant bank: the lead bank took warrants in more than half of its OTHER IPOs in the sample (leave-one-out, so an
# IPO's own contract does not define its bank); missing when the bank leads no other IPO
g = s.groupby("lead_bank").uw_warrants
s["n_other"] = g.transform("size") - 1
s["w_other"] = ((g.transform("sum") - s.uw_warrants) / s.n_other).where(s.n_other > 0)
s["wbank"] = (s.w_other > 0.5).astype(float).where(s.w_other.notna())
def say(*a):
    t = " ".join(str(v) for v in a); print(t); lines.append(t)
def fit(f, d, cl="2way"):
    m = smf.ols(f, d); rl = m.data.row_labels
    gr = d.loc[rl, ["ind", "yq"]].apply(lambda c: pd.factorize(c)[0]).values if cl == "2way" else pd.factorize(d.loc[rl, cl])[0]
    return m.fit(cov_type="cluster", cov_kwds={"groups": gr})
def both(f, d, keys, lab="", show=True):
    a, b = fit(f, d), fit(f, d, "ipo_year")
    if show:
        for k in keys:
            say(f"  {lab:34s} {k:22s} {a.params[k]:8.3f}  t2way {a.tvalues[k]:6.2f}  tyear {b.tvalues[k]:6.2f}  N={int(a.nobs)}  R2={a.rsquared:.3f}")
    return a
say(f"N={len(s)}; foreign {int(s.foreign.sum())}; offer < $5 {int(s.penny.sum())}; ranked lead bank {int(s.uw.notna().sum())}; assets in USD {int(s.assets_m.notna().sum())}; VC known {int(s.vc.notna().sum())}")
say("warrants:", int(s.uw_warrants.sum()), "| terms read:", int((s.uw_warrants.eq(1) & s.w_cover.notna()).sum()),
    "| median cover/strike/years among read:", s.loc[s.uw_warrants == 1, ["w_cover", "w_strike", "w_years", "w_value"]].median().round(2).to_dict())

say("\n== 1 contract by size of the offering")
say(s.groupby("band", observed=True).agg(N=("key", "size"), proceeds_med=("proc_m", "median"), spread=("spread", "mean"), at7=("at7", "mean"), above7=("above7", "mean"),
    warrants=("uw_warrants", "mean"), allowance=("na", lambda x: (x > 0).mean()), direct=("direct", "mean"), direct_med=("direct", "median"),
    usd_m=("direct_usd_m", "median"), foreign=("foreign", "mean"), penny=("penny", "mean"), vc=("vc", "mean"), bottom=("second", "mean")).round(3).to_string())
say("by tier:")
say(s.groupby("tier").agg(N=("key", "size"), proceeds=("proc_m", "median"), warrants=("uw_warrants", "mean"), above7=("above7", "mean"), small=("small_contract", "mean"), direct=("direct", "mean")).round(3).to_string())
say("by year: share of IPOs below $25m and share on the small contract")
say(s.groupby("ipo_year").agg(N=("key", "size"), below25=("proc_m", lambda x: (x < 25).mean()), small=("small_contract", "mean")).round(3).T.to_string())

say("\n== 2 where is the step? LPM of warrants (and small contract) on 1[proceeds < c] + year FE, cutoffs $10-100m")
for y in ["uw_warrants", "small_contract"]:
    res = []
    for c in [10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40, 45, 50, 60, 75, 100]:
        s["_b"] = (s.proc_m < c).astype(int)
        m = smf.ols(f"{y} ~ _b + C(ipo_year)", s).fit(); res.append((c, m.rsquared, m.params["_b"]))
    best = max(res, key=lambda r: r[1])
    say(f"  {y}: " + "  ".join(f"{c:g}:{r2:.3f}" for c, r2, _ in res) + f"  -> best cutoff ${best[0]:g}m (R2 {best[1]:.3f}, jump {best[2]:.2f})")
    # bootstrap the location of the best cutoff
    rng = np.random.default_rng(7); locs = []
    for b in range(300):
        d = s.sample(len(s), replace=True, random_state=int(rng.integers(1e9)))
        r = []
        for c in [10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40, 45, 50, 60]:
            d["_b"] = (d.proc_m < c).astype(int); r.append((c, smf.ols(f"{y} ~ _b", d).fit().rsquared))
        locs.append(max(r, key=lambda z: z[1])[0])
    say(f"    bootstrap (300) location of the best cutoff: median {np.median(locs):g}, 90% range {np.percentile(locs, 5):g}-{np.percentile(locs, 95):g}")
say("  linear-in-log-proceeds vs step: R2 of warrants on lnproc + year FE",
    round(smf.ols("uw_warrants ~ lnproc + C(ipo_year)", s).fit().rsquared, 3), "| on bands + year FE", round(smf.ols("uw_warrants ~ C(band) + C(ipo_year)", s).fit().rsquared, 3))

say("\n== 3 mechanism: is there a floor in dollars?")
s["cash_usd_m"] = s.spread / 100 * s.proc_m + s.na / 100 * s.proc_m
say(s.groupby("band", observed=True).agg(N=("key", "size"), proceeds=("proc_m", "median"), cash_usd=("cash_usd_m", "median"), warrant_usd=("wv", lambda x: np.nan),
    total_usd=("direct_usd_m", "median"), total_p10=("direct_usd_m", lambda x: x.quantile(.1)), total_p90=("direct_usd_m", lambda x: x.quantile(.9)), direct_pct=("direct", "median")).round(2).to_string())
s["lnusd"] = np.log(s.direct_usd_m); s["lncash"] = np.log(s.cash_usd_m)
for lab, d in [("proceeds < $25m", s[s.proc_m < 25]), ("$25-50m", s[(s.proc_m >= 25) & (s.proc_m < 50)]), ("$50-250m", s[(s.proc_m >= 50) & (s.proc_m < 250)]), ("> $250m", s[s.proc_m >= 250])]:
    both("lnusd ~ lnproc + C(ipo_year)", d, ["lnproc"], f"elasticity of total $ | {lab}")
    both("lncash ~ lnproc + C(ipo_year)", d, ["lnproc"], f"elasticity of cash $ | {lab}")
say("  among warrant IPOs: does the warrant share of compensation rise as proceeds fall?")
w = s[(s.uw_warrants == 1) & s.w_cover.notna()].copy()
w["wshare"] = w.wv / w.direct
both("wshare ~ lnproc + C(ipo_year)", w, ["lnproc"], "warrant share of comp")
both("w_cover ~ lnproc + C(ipo_year)", w, ["lnproc"], "coverage % of shares")
both("spread ~ lnproc + C(ipo_year)", w, ["lnproc"], "spread among warrant IPOs")

say("\n== 4 bank or size?")
b = s.dropna(subset=["lead_bank"]).copy()
nb = b.groupby("lead_bank").key.transform("size"); b = b[nb >= 3]
say(f"  IPOs with a lead bank that leads >= 3 IPOs: {len(b)}, banks {b.lead_bank.nunique()}")
r_band = smf.ols("uw_warrants ~ C(band) + C(ipo_year)", b).fit().rsquared
r_bank = smf.ols("uw_warrants ~ C(lead_bank) + C(ipo_year)", b).fit().rsquared
r_both = smf.ols("uw_warrants ~ C(band) + C(lead_bank) + C(ipo_year)", b).fit().rsquared
r_year = smf.ols("uw_warrants ~ C(ipo_year)", b).fit().rsquared
say(f"  R2 warrants: year {r_year:.3f} | + size bands {r_band:.3f} | + lead bank FE {r_bank:.3f} | both {r_both:.3f}")
say(f"  increment of bank FE over bands {r_both - r_band:.3f}; increment of bands over bank FE {r_both - r_bank:.3f}")
wb = b.groupby("lead_bank").agg(N=("key", "size"), warr=("uw_warrants", "mean"), small=("proc_m", lambda x: (x < 25).mean()), med=("proc_m", "median"), rank=("uw", "median"))
users = wb[wb.warr > 0]
say(f"  banks that ever take warrants: {len(users)}; their IPOs {int(users.N.sum())}")
say(users.sort_values("N", ascending=False).head(20).round(2).to_string())
u = b[b.lead_bank.isin(users.index)]
say("  within the banks that ever take warrants: share with warrants by size band")
say(u.groupby("band", observed=True).agg(N=("key", "size"), warrants=("uw_warrants", "mean"), banks=("lead_bank", "nunique")).round(3).to_string())
both("uw_warrants ~ C(band, Treatment('25-50')) + C(lead_bank) + C(ipo_year)", u, ["C(band, Treatment('25-50'))[T.10-25]", "C(band, Treatment('25-50'))[T.50-100]"], "within warrant banks, bank FE")
both("uw_warrants ~ lnproc + C(lead_bank) + C(ipo_year)", u, ["lnproc"], "within warrant banks, bank FE")
mix = u.groupby("lead_bank").filter(lambda g: (g.proc_m < 25).any() and (g.proc_m >= 25).any())
say(f"  banks with deals on both sides of $25m: {mix.lead_bank.nunique()}, IPOs {len(mix)}")
say(mix.assign(side=np.where(mix.proc_m < 25, "below 25", "25+")).groupby("side").agg(N=("key", "size"), warrants=("uw_warrants", "mean"), direct=("direct", "mean")).round(3).to_string())
both("uw_warrants ~ I(proc_m < 25) + C(lead_bank) + C(ipo_year)", mix, ["I(proc_m < 25)[T.True]"], "same bank, below vs above $25m")
say("  tier given size, all IPOs with a ranked lead bank (middle omitted):")
r = s.dropna(subset=["uw"])
for y in ["uw_warrants", "small_contract", "direct"]:
    both(f"{y} ~ second + elite + C(band) + C(ipo_year)", r, ["second", "elite"], f"{y} | bands")
    both(f"{y} ~ second + elite + C(band) + foreign + penny + C(ipo_year)", r, ["second", "elite"], f"{y} | bands + foreign + penny")
say("  unranked lead banks (not in Ritter's ranking):", s.tier.eq("unranked").sum(), "IPOs; warrants", round(s[s.tier == "unranked"].uw_warrants.mean(), 3), "median proceeds", round(s[s.tier == "unranked"].proc_m.median(), 1))

say("  leave-one-out warrant banks: IPOs led by one", int(s.wbank.sum()), "of", int(s.wbank.notna().sum()), "with the measure; banks", s.loc[s.wbank == 1, "lead_bank"].nunique())
say(s.dropna(subset=["wbank"]).groupby(["band", "wbank"], observed=True).agg(N=("key", "size"), warrants=("uw_warrants", "mean"), direct=("direct", "mean")).round(3).to_string())
for y in ["uw_warrants", "direct"]:
    both(f"{y} ~ wbank + C(band) + C(ipo_year)", s.dropna(subset=["wbank"]), ["wbank"], f"{y}: warrant bank given bands")
    both(f"{y} ~ wbank + C(band) + C(ipo_year)", s.dropna(subset=["wbank"])[lambda x: x.proc_m < 50], ["wbank"], f"{y}: warrant bank, deals < $50m")

say("\n== 5 venture backing (U.S. dollar assets and VC known)")
v = s.dropna(subset=["vc", "lnassets", "lnage", "loss"]).copy()
v["sz"] = pd.qcut(v.lnassets, 3, labels=["small", "mid", "large"])
say(f"  N={len(v)}; tercile cutoffs ($m): {np.exp(v.lnassets.quantile([1/3, 2/3])).div(1e6).round(1).tolist()}")
say(v.groupby(["sz", "vc"], observed=True).agg(N=("key", "size"), assets=("assets_m", "median"), proceeds=("proc_m", "median"), bottom=("second", "mean"), elite=("elite", "mean"),
    warrants=("uw_warrants", "mean"), small=("small_contract", "mean"), direct=("direct", "mean"), foreign=("foreign", "mean")).round(3).to_string())
X0 = "lnassets + lnage + loss + rnd_int + biotech + tech + foreign + C(ipo_year)"
for z in ["small", "mid", "large"]:
    d = v[v.sz == z]
    for y in ["lnproc", "second", "elite", "uw_warrants", "small_contract", "direct"]:
        dd = d.dropna(subset=["uw"]) if y in ("second", "elite") else d
        both(f"{y} ~ vc + {X0}", dd, ["vc"], f"{z}: {y}")
say("  decomposition in the small tercile:")
d = v[v.sz == "small"].copy()
for y in ["uw_warrants", "small_contract", "direct"]:
    a0 = both(f"{y} ~ vc + {X0}", d, ["vc"], f"{y} (1) total")
    a1 = both(f"{y} ~ vc + C(band) + {X0}", d, ["vc"], f"{y} (2) + size bands")
    a2 = both(f"{y} ~ vc + lnproc + C(band) + {X0}", d, ["vc"], f"{y} (3) + log proceeds")
    say(f"    share of (1) gone with size bands {1 - a1.params['vc'] / a0.params['vc']:.2f}; with bands and log proceeds {1 - a2.params['vc'] / a0.params['vc']:.2f}")
say("  the bank channel in the small tercile (leave-one-out warrant bank):")
dw = d.dropna(subset=["wbank"])
both(f"wbank ~ vc + {X0}", dw, ["vc"], "warrant bank on VC")
both(f"wbank ~ vc + C(band) + {X0}", dw, ["vc"], "warrant bank on VC, given bands")
for y in ["uw_warrants", "direct"]:
    a0 = both(f"{y} ~ vc + {X0}", dw, ["vc"], f"{y} (1) total, same N")
    a1 = both(f"{y} ~ vc + wbank + {X0}", dw, ["vc", "wbank"], f"{y} (2) + warrant bank")
    a2 = both(f"{y} ~ vc + wbank + C(band) + {X0}", dw, ["vc", "wbank"], f"{y} (3) + warrant bank + bands")
    say(f"    share of (1) gone with the warrant bank {1 - a1.params['vc'] / a0.params['vc']:.2f}; with bank and bands {1 - a2.params['vc'] / a0.params['vc']:.2f}")
say("  matching in the small tercile: each VC firm to the nearest non-VC firm on standardized lnassets, lnage, loss, rnd_int, biotech, tech, foreign and IPO year (with replacement)")
Z = ["lnassets", "lnage", "loss", "rnd_int", "biotech", "tech", "foreign", "ipo_year"]
dz = d.dropna(subset=Z).reset_index(drop=True); M = (dz[Z] - dz[Z].mean()) / dz[Z].std()
TR, CO = dz[dz.vc == 1], dz[dz.vc == 0]
dist = ((M.loc[TR.index].values[:, None, :] - M.loc[CO.index].values[None, :, :]) ** 2).sum(-1)
nn = CO.index.values[dist.argmin(1)]
for y in ["lnproc", "uw_warrants", "small_contract", "direct"]:
    diff = dz.loc[TR.index, y].values - dz.loc[nn, y].values
    say(f"    ATT {y:15s} {np.nanmean(diff):7.3f}  (se {np.nanstd(diff) / np.sqrt(np.isfinite(diff).sum()):.3f}; treated {len(TR)}, distinct controls {len(set(nn))})")
say("    balance (treated mean, matched control mean):", {z: (round(dz.loc[TR.index, z].mean(), 2), round(dz.loc[nn, z].mean(), 2)) for z in Z[:4]})
say("  sensitivity (Oster 2019, delta for beta=0, Rmax = 1.3 R2) for VC -> small contract in the small tercile:")
m0 = smf.ols("small_contract ~ vc", d).fit(); m1 = smf.ols(f"small_contract ~ vc + {X0}", d).fit()
b0, r0, b1, r1 = m0.params["vc"], m0.rsquared, m1.params["vc"], m1.rsquared; rmax = min(1, 1.3 * r1)
delta = (b1 * (r1 - r0)) / ((b0 - b1) * (rmax - r1)) if (b0 - b1) * (rmax - r1) != 0 else np.nan
say(f"    beta uncontrolled {b0:.3f} (R2 {r0:.3f}), controlled {b1:.3f} (R2 {r1:.3f}), delta {delta:.2f}")

say("\n== 6 robustness of the step (warrants and direct compensation below $25m vs $50-250m)")
s["lt25"] = (s.proc_m < 25).astype(int); s["mid"] = ((s.proc_m >= 25) & (s.proc_m < 50)).astype(int); s["gt250"] = (s.proc_m >= 250).astype(int)
for lab, d in [("all", s), ("U.S. issuers", s[s.foreign == 0]), ("offer >= $5", s[s.penny == 0]), ("U.S., >= $5", s[(s.foreign == 0) & (s.penny == 0)]),
               ("to 2020", s[s.ipo_year <= 2020]), ("not biotech", s[s.biotech == 0]), ("ranked lead bank, with tier", s.dropna(subset=["uw"]))]:
    extra = " + second + elite" if lab.startswith("ranked") else ""
    for y in ["uw_warrants", "direct"]:
        both(f"{y} ~ lt25 + mid + gt250{extra} + C(ipo_year)", d, ["lt25"], f"{lab}: {y}")
say("  with lead-bank fixed effects (banks leading >= 3 IPOs): the size step within banks")
bb = s[s.groupby("lead_bank").key.transform("size") >= 3]
for y in ["uw_warrants", "direct"]:
    both(f"{y} ~ lt25 + mid + gt250 + C(lead_bank) + C(ipo_year)", bb, ["lt25", "mid"], f"bank FE: {y}")
say("\n== 7 robustness of the venture results (bottom third by assets, thirds recomputed in each sample)")
XR = "lnassets + lnage + loss + rnd_int + biotech + tech + C(ipo_year)"
for lab, d0 in [("all", s), ("U.S. issuers", s[s.foreign == 0]), ("offer >= $5", s[s.penny == 0]), ("U.S., >= $5", s[(s.foreign == 0) & (s.penny == 0)]),
                ("to 2020", s[s.ipo_year <= 2020]), ("without 2022", s[s.ipo_year <= 2021]), ("not biotech", s[s.biotech == 0])]:
    vv = d0.dropna(subset=["vc", "lnassets", "lnage", "loss"]).copy()
    vv = vv[vv.lnassets <= vv.lnassets.quantile(1 / 3)]
    xr = XR + (" + foreign" if vv.foreign.nunique() > 1 else "")
    xr = xr.replace(" + biotech", "") if vv.biotech.nunique() < 2 else xr
    dw = vv.dropna(subset=["wbank"])
    say(f" {lab}: N={len(vv)} (bank classified {len(dw)}); no-VC {int((vv.vc == 0).sum())}, VC {int((vv.vc == 1).sum())}; warrants no-VC {vv[vv.vc == 0].uw_warrants.mean():.2f} vs VC {vv[vv.vc == 1].uw_warrants.mean():.2f}")
    both(f"wbank ~ vc + {xr}", dw, ["vc"], f"{lab}: warrant bank")
    a0 = both(f"uw_warrants ~ vc + {xr}", dw, ["vc"], f"{lab}: warrants")
    a1 = both(f"uw_warrants ~ vc + wbank + {xr}", dw, ["vc"], f"{lab}: warrants | bank")
    c0 = both(f"direct ~ vc + {xr}", dw, ["vc"], f"{lab}: direct")
    c1 = both(f"direct ~ vc + wbank + {xr}", dw, ["vc"], f"{lab}: direct | bank")
    say(f"    share through the bank: warrants {1 - a1.params['vc'] / a0.params['vc']:.2f}, direct {1 - c1.params['vc'] / c0.params['vc']:.2f}")
say("\n== 8 classification check: hand codes of 100 prospectuses (33_validation_sample.py)")
hc = ROOT / "data" / "universe" / "validation_handcodes.csv"
if hc.exists():
    h = pd.read_csv(hc); say(pd.crosstab(h.uw_warrants, h.hand, margins=True).to_string()); say("  agreement:", (h.uw_warrants == h.hand).mean())
(ROOT / "output" / "paper_b_results.txt").write_text("\n".join(lines), encoding="utf-8")
