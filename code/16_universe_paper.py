"""Paper tables on the universe (data/universe/analysis.csv), with the controls used across the JOBS Act / IPO literature
and two-way clustered errors (industry x issue quarter). Adds what the 712-IPO version lacked: entropy balancing of
top-tier vs other banks, an event study of the July 2017 extension, a placebo date, and alternative rank measures.
Writes output/universe_results.txt and output/universe_estimates.csv"""
import pathlib, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
x = pd.read_csv(ROOT / "data" / "universe" / "analysis.csv", parse_dates=["t_issue", "t_pub", "t_drs", "start"])
if "ir" not in x:  # the published file omits Bloomberg/Eikon returns (see README); Yahoo covers part of the sample
    x["ir"] = x.ir_yf
lines, rows = [], []
def say(*a):
    s = " ".join(str(v) for v in a); print(s); lines.append(s)
w = lambda v: v.clip(*v.quantile([.01, .99]))

x["big4_na"] = x.big4.isna().astype(int); x["big4"] = x.big4.fillna(0)
x["rnd_int"] = x.rnd_int.fillna(0)
X = ["lnage", "vc", "lnassets", "loss", "rnd_int", "lnproc", "big4", "big4_na", "nasdaq", "tech", "dual", "secondary", "nasdaq90", "lnipo90", "ir90"]
s = x.dropna(subset=["uw", "lnpub"] + X).copy()
s["irw"] = w(s.ir); s["ar90w"] = w(s.ar90)
CTL = " + ".join(X) + " + C(ind) + C(ipo_year)"
say(f"Sample N={len(s)} of {len(x)}  {s.t_issue.min().date()} - {s.t_issue.max().date()}  confidential={s.confidential.mean():.2f}  EGC={s.egc.mean():.2f}  top={s.top.mean():.2f}")
say("by year:", s.ipo_year.value_counts().sort_index().to_dict())

def fit(f, d=None, cl="2way"):
    d = s if d is None else d
    m = smf.ols(f, d); rl = m.data.row_labels
    g = d.loc[rl, ["ind", "yq"]].apply(lambda c: pd.factorize(c)[0]).values if cl == "2way" else pd.factorize(d.loc[rl, cl])[0]
    return m.fit(cov_type="cluster", cov_kwds={"groups": g})
def r(m, k): return f"{m.params[k]:8.3f} (t={m.tvalues[k]:5.2f})"
def keep(tab, name, m, k):
    rows.append({"table": tab, "model": name, "var": k, "coef": m.params[k], "se": m.bse[k], "t": m.tvalues[k], "p": m.pvalues[k], "N": int(m.nobs), "R2": m.rsquared})
def oster(y, d, xv="uw"):
    d = d.dropna(subset=[y])
    ms = smf.ols(f"{y} ~ {xv} + C(ipo_year)", d).fit(); mf = smf.ols(f"{y} ~ {xv} + {CTL}", d).fit()
    bs, bf, rs, rf = ms.params[xv], mf.params[xv], ms.rsquared, mf.rsquared
    return (bf * (rf - rs)) / ((bs - bf) * (min(1, 1.3 * rf) - rf)) if bs != bf else np.inf

say("\n== T1  medians by tier (top = Ritter rank 9)")
med = s.groupby("top")[["total_dur", "conf_period", "pub", "pub_share", "n_drs_all", "n_pub_amend", "rev", "ir"]].median()
say(med.round(1).to_string()); med.to_csv(OUT / "universe_medians_by_tier.csv")

say("\n== T2  prestige and the length of each phase")
c = s[s.confidential == 1]
for y, d, lab in [("lntotal", s, "ln total"), ("lnconf", c, "ln confidential (DRS filers)"), ("lnpub", s, "ln public window"), ("pub_share", s, "public share of total, %")]:
    for k in ["uw", "top"]:
        m = fit(f"{y} ~ {k} + {CTL}", d); keep("T2", lab, m, k)
        say(f"{lab:32s} {k:4s} {r(m, k)}  N={int(m.nobs)}" + (f"  Oster delta={oster(y, d):.2f}" if k == "uw" else ""))

say("\n== T3  review rounds")
for y, d, lab in [("lndrs", c, "ln(1+confidential drafts)"), ("lnamend", s, "ln(1+public amendments)")]:
    m = fit(f"{y} ~ uw + {CTL}", d); keep("T3", lab, m, "uw"); say(f"{lab:32s} uw {r(m, 'uw')}  N={int(m.nobs)}")
m0 = fit(f"lnpub ~ uw + {CTL}"); m = fit(f"lnpub ~ uw + lnamend + {CTL}")
keep("T3", "lnpub | amendments", m, "uw"); keep("T3", "lnpub | amendments", m, "lnamend")
say(f"lnpub with amendments: uw {r(m, 'uw')}  amendments {r(m, 'lnamend')}  -> uw falls by {100 * (1 - m.params['uw'] / m0.params['uw']):.0f}%")

say("\n== T4  July 2017: confidential filing extended to non-EGCs (triple difference, by registration start)")
m = fit(f"lnpub ~ uw * nonegc * post17 + {CTL}")
for k in ["uw:nonegc:post17", "uw:nonegc", "uw:post17", "uw"]: keep("T4", "lnpub triple diff", m, k); say(f"  {k:20s} {r(m, k)}")
b, V = m.params, m.cov_params()
for lab, ks in [("EGC before", ["uw"]), ("EGC after", ["uw", "uw:post17"]), ("non-EGC before", ["uw", "uw:nonegc"]), ("non-EGC after", ["uw", "uw:nonegc", "uw:post17", "uw:nonegc:post17"])]:
    e = sum(b[k] for k in ks); se = np.sqrt(sum(V.loc[i, j] for i in ks for j in ks)); say(f"  uw effect, {lab:15s} {e:7.3f} (t={e / se:5.2f})")
say("  N non-EGC before/after:", int(((s.nonegc == 1) & (s.post17 == 0)).sum()), int(((s.nonegc == 1) & (s.post17 == 1)).sum()),
    " non-EGC confidential share before/after:", s[s.nonegc == 1].groupby("post17").confidential.mean().round(2).to_dict())
# event study: uw x non-EGC by registration-start year, 2016 = reference
s["sy"] = s.start.dt.year
es = fit(f"lnpub ~ uw * nonegc + uw:nonegc:C(sy, Treatment(2016)) + {CTL}")
say("  event study uw x nonEGC x start-year (ref 2016):")
for k in [k for k in es.params.index if k.startswith("uw:nonegc:C(sy")]:
    keep("T4es", "event study", es, k); say(f"    {k[-5:-1]} {r(es, k)}")
# placebo: pretend the extension happened in July 2015
s["post15"] = ((s.start >= "2015-07-10") & (s.start < "2017-07-10")).astype(int)
p = fit(f"lnpub ~ uw * nonegc * post15 + {CTL}", s[s.start < "2017-07-10"])
keep("T4", "placebo July 2015", p, "uw:nonegc:post15"); say(f"  placebo July 2015 (pre-2017 sample) {r(p, 'uw:nonegc:post15')}")

say("\n== T5  consequences of a short public window")
for y, rhs, lab in [("rev", "lnpub + uw", "price revision"), ("irw", "lnpub + uw", "first-day return"), ("irw", "lnpub + uw + rev_pos + rev_neg", "first-day | revision"),
                    ("ar90w", "lnpub + uw + irw", "90-day abnormal (survivors)")]:
    m = fit(f"{y} ~ {rhs} + {CTL}")
    for k in ["lnpub", "uw"]: keep("T5", lab, m, k)
    say(f"{lab:30s} lnpub {r(m, 'lnpub')}  uw {r(m, 'uw')}  N={int(m.nobs)}")

say("\n== T6  entropy balancing: top-tier vs other banks (Hainmueller 2012; first moments of all controls)")
def ebal(d, cols, treat):
    Xc = d.loc[d[treat] == 0, cols].values; mt = d.loc[d[treat] == 1, cols].values.mean(0)
    Z = (Xc - mt) / Xc.std(0).clip(1e-9)
    f = lambda l: np.log(np.exp(Z @ l).sum()); gr = lambda l: (np.exp(Z @ l) / np.exp(Z @ l).sum()) @ Z
    l = minimize(f, np.zeros(Z.shape[1]), jac=gr, method="BFGS").x
    wc = np.exp(Z @ l); wc = wc / wc.sum() * len(wc)
    out = pd.Series(1.0, index=d.index); out[d[treat] == 0] = wc
    return out
EB = [k for k in X if s[k].std() > 0]
s["wt"] = ebal(s, EB, "top")
bal = pd.DataFrame({"top": s[s.top == 1][EB].mean(), "other raw": s[s.top == 0][EB].mean(),
                    "other weighted": s[s.top == 0][EB].multiply(s.wt[s.top == 0], axis=0).sum() / s.wt[s.top == 0].sum()})
say(bal.round(3).to_string()); bal.to_csv(OUT / "universe_balance.csv")
for y, d, lab in [("lnpub", s, "ln public window"), ("lnconf", s[s.confidential == 1], "ln confidential"), ("lnamend", s, "ln(1+amendments)"), ("rev", s, "price revision"), ("irw", s, "first-day return")]:
    d = d.dropna(subset=[y]); mm = smf.wls(f"{y} ~ top + C(ind) + C(ipo_year)", d, weights=d.wt)
    g = d.loc[mm.data.row_labels, ["ind", "yq"]].apply(lambda c: pd.factorize(c)[0]).values
    m = mm.fit(cov_type="cluster", cov_kwds={"groups": g}); keep("T6", lab, m, "top"); say(f"  {lab:20s} top {r(m, 'top')}  N={int(m.nobs)}")

say("\n== R   robustness of uw -> ln public window")
for lab, f, d, cl in [("baseline", f"lnpub ~ uw + {CTL}", s, "2way"), ("cluster by year only", f"lnpub ~ uw + {CTL}", s, "ipo_year"),
                      ("max rank of bookrunners", f"lnpub ~ uw_max + {CTL}", s, "2way"), ("+ ln bookrunners, lockup", f"lnpub ~ uw + lnbanks + lockup_days + {CTL}", s, "2way"),
                      ("drop biotech", f"lnpub ~ uw + {CTL}", s[s.biotech == 0], "2way"), ("drop 2020-21", f"lnpub ~ uw + {CTL}", s[~s.ipo_year.isin([2020, 2021])], "2way"),
                      ("total <= 550 days", f"lnpub ~ uw + {CTL}", s[s.total_dur <= 550], "2way"), ("no prior RW, not dormant", f"lnpub ~ uw + {CTL}", s[(s.prior_rw == 0) & (s.dormant == 0)], "2way"),
                      ("EGC only", f"lnpub ~ uw + {CTL}", s[s.egc == 1], "2way"), ("confidential filers only", f"lnpub ~ uw + {CTL}", s[s.confidential == 1], "2way"),
                      ("year-quarter FE", f"lnpub ~ uw + {CTL.replace('C(ipo_year)', 'C(yq)')}", s, "2way")]:
    k = "uw_max" if "uw_max" in f else "uw"
    m = fit(f, d, cl); keep("R", lab, m, k); say(f"  {lab:28s} {r(m, k)}  N={int(m.nobs)}")

say("\n== S   chain uw -> lnpub -> revision -> first-day return (product of coefficients, bootstrap B=500)")
def chain(d):
    a = smf.ols(f"lnpub ~ uw + {CTL}", d).fit().params["uw"]
    b = smf.ols(f"rev ~ lnpub + uw + {CTL}", d).fit().params["lnpub"]
    cc = smf.ols(f"irw ~ rev + lnpub + uw + {CTL}", d).fit().params["rev"]
    return a * b * cc
sc = s.dropna(subset=["rev"]); est = chain(sc); rng = np.random.default_rng(1)
bs = [chain(sc.iloc[rng.integers(0, len(sc), len(sc))]) for _ in range(500)]
say(f"  indirect effect {est:.3f}  95% CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]  N={len(sc)}")
rows.append({"table": "S", "model": "serial chain", "var": "a*b*c", "coef": est, "se": np.std(bs), "t": None, "p": None, "N": len(sc), "R2": None})

pd.DataFrame(rows).to_csv(OUT / "universe_estimates.csv", index=False)
(OUT / "universe_results.txt").write_text("\n".join(lines), encoding="utf-8")
