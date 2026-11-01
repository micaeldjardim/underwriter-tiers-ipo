"""Prestige as divisions, not a scale. Same sample, controls, fixed effects and two-way clustering as 16.
D1  threshold regression (Hansen 2000): estimate where the line between divisions falls, with a bootstrap test that a
    threshold exists and a likelihood-ratio confidence set for it, instead of choosing the cut by eye.
D2  division effects on every outcome: second division (below the estimated cut) and elite (rank 9) vs the middle.
D3  bank or deal size? second vs first division among deals of similar size, proceeds bins, entropy balancing.
D4  robustness of the second-division effect.
D5  survival after the IPO from EDGAR exits (18): any exit, merger, failure, deficiency notice within 3 years.
Writes output/divisions_results.txt and output/divisions_estimates.csv"""
import pathlib, numpy as np, pandas as pd, statsmodels.formula.api as smf, patsy
ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / "code" / "16_universe_paper.py").read_text(encoding="utf-8").split('say("\\n== T1')[0]
g = {"__file__": str(ROOT / "code" / "16_universe_paper.py")}; exec(src, g)          # sample s, X, CTL, fit(), r()
s, X, CTL, fit, r, OUT = g["s"], g["X"], g["CTL"], g["fit"], g["r"], g["OUT"]
lines, rows = [], []
def say(*a):
    t = " ".join(str(v) for v in a); print(t); lines.append(t)
def keep(tab, name, m, k):
    rows.append({"table": tab, "model": name, "var": k, "coef": m.params[k], "se": m.bse[k], "t": m.tvalues[k], "p": m.pvalues[k], "N": int(m.nobs)})
w = lambda v: v.clip(*v.quantile([.01, .99]))
ex = pd.read_csv(ROOT / "data" / "universe" / "exits.csv")
s = s.merge(ex[["key", "exit3", "exit_merger3", "exit_fail3", "def301_3"]], on="key", how="left")
s["proc"] = np.exp(s.lnproc)
fit = lambda f, d=None, cl="2way": g["fit"](f, s if d is None else d, cl)   # 16's fit defaults to its own s
s["irw"] = w(s.ir)
say(f"N={len(s)}  ranks:", s.uw.round().value_counts().sort_index().to_dict())

say("\n== D1  threshold regression: y = a*1[uw < c] + controls + FE; c chosen by least squares (Hansen 2000)")
CUTS = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
def threshold(y, B=500, seed=7, ctl=None):
    d = s.dropna(subset=[y]); Y, Z = patsy.dmatrices(f"{y} ~ {ctl or CTL}", d, return_type="dataframe")
    u = d.loc[Y.index, "uw"].values; Y, Z = Y.values.ravel(), Z.values; n = len(Y)
    def ssr(yv, A): return float(((yv - A @ np.linalg.lstsq(A, yv, rcond=None)[0]) ** 2).sum())
    D = {c: np.column_stack([Z, (u < c).astype(float)]) for c in CUTS}
    s0 = ssr(Y, Z); sc = {c: ssr(Y, D[c]) for c in CUTS}; c_hat = min(sc, key=sc.get); s1 = sc[c_hat]
    F = n * (s0 - s1) / s1
    # fixed-regressor bootstrap under the null of no threshold (Hansen 1996, 2000)
    e0 = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]; rng = np.random.default_rng(seed); Fb = []
    for _ in range(B):
        yb = e0 * rng.standard_normal(n); sb0 = ssr(yb, Z); sb1 = min(ssr(yb, D[c]) for c in CUTS); Fb.append(n * (sb0 - sb1) / sb1)
    p = float(np.mean(np.array(Fb) >= F))
    LR = {c: n * (sc[c] - s1) / s1 for c in CUTS}
    cs = [c for c in CUTS if LR[c] <= 7.35]                 # Hansen (2000) 95% asymptotic critical value
    return c_hat, F, p, LR, cs, n
for y in ["lnpub", "lnamend", "rev", "irw"]:
    # pricing outcomes without proceeds (offer price x shares moves with the revision itself)
    c_hat, F, p, LR, cs, n = threshold(y, ctl=CTL.replace("lnproc + ", "") if y in ("rev", "irw") else None)
    say(f"  {y:8s} cut at uw < {c_hat}  sup-F={F:6.1f}  bootstrap p={p:.3f}  95% set for the cut: {cs}  N={n}")
    say("           LR by cut: " + "  ".join(f"{c}:{v:.1f}" for c, v in LR.items()))
    rows.append({"table": "D1", "model": y, "var": "cut", "coef": c_hat, "se": None, "t": F, "p": p, "N": n})

say("\n== D2  divisions: second = rank < 5, elite = rank 9, reference = ranks 5-8")
s["second"] = (s.uw < 5).astype(int); s["elite"] = (s.uw >= 9).astype(int)
say("  N second / middle / elite:", int(s.second.sum()), int(((s.second == 0) & (s.elite == 0)).sum()), int(s.elite.sum()),
    "  median proceeds $M:", s.groupby(np.select([s.second == 1, s.elite == 1], ["second", "elite"], "middle")).proc.median().round(1).to_dict())
s["ar90w"] = w(s.ar90); s["irw"] = w(s.ir)
c = s[s.confidential == 1]
OUTC = [("lntotal", s, "ln total registration"), ("lnconf", c, "ln confidential phase"), ("lnpub", s, "ln public window"),
        ("lndrs", c, "ln(1+confidential drafts)"), ("lnamend", s, "ln(1+public amendments)"), ("rev", s, "price revision, %"),
        ("irw", s, "first-day return, %"), ("ar90w", s, "90-day abnormal, survivors"), ("exit3", s, "exit within 3y"),
        ("exit_merger3", s, "acquired within 3y"), ("exit_fail3", s, "failed/delisted within 3y"), ("def301_3", s, "deficiency notice within 3y")]
for y, d, lab in OUTC:
    m = fit(f"{y} ~ second + elite + {CTL}", d); keep("D2", lab, m, "second"); keep("D2", lab, m, "elite")
    say(f"  {lab:30s} second {r(m, 'second')}  elite {r(m, 'elite')}  mean={d[y].mean():6.2f}  N={int(m.nobs)}")
# proceeds = offer price x shares, so it moves with the revision: re-estimate pricing outcomes with pre-IPO size only
CTL_NP = CTL.replace("lnproc + ", "")
for y, lab in [("rev", "price revision, % (no proceeds control)"), ("irw", "first-day return, % (no proceeds control)")]:
    m = fit(f"{y} ~ second + elite + {CTL_NP}"); keep("D2", lab, m, "second"); keep("D2", lab, m, "elite")
    say(f"  {lab:30s} second {r(m, 'second')}  elite {r(m, 'elite')}  N={int(m.nobs)}")
for y, lab in [("exit_fail3", "failed within 3y"), ("def301_3", "deficiency notice 3y"), ("exit_merger3", "acquired within 3y")]:
    m = fit(f"{y} ~ second + elite + C(ind) + C(ipo_year)"); say(f"  {lab:30s} FE only: second {r(m, 'second')}  elite {r(m, 'elite')}")
m = fit(f"lnpub ~ second + elite + uw + {CTL}"); say(f"  lnpub with the linear rank too: second {r(m, 'second')}  elite {r(m, 'elite')}  uw {r(m, 'uw')}")
m = fit(f"lnpub ~ uw + {CTL}", s[s.second == 0])
say(f"  linear rank within ranks 5-9 only: uw {r(m, 'uw')}  N={int(m.nobs)}")

say("\n== D3  the bank or the deal size?")
for cap in [20, 40, 60]:
    d = s[s.proc <= cap]
    m = fit(f"lnpub ~ second + {CTL}", d); keep("D3", f"proceeds <= {cap}M", m, "second")
    ma = fit(f"lnamend ~ second + {CTL}", d)
    say(f"  proceeds <= ${cap}M: N second/first {int(d.second.sum())}/{int((1 - d.second).sum())}  lnpub second {r(m, 'second')}  lnamend second {r(ma, 'second')}")
s["procbin"] = pd.qcut(s.proc, 10, labels=False, duplicates="drop")
m = fit(f"lnpub ~ second + elite + {CTL} + C(procbin)"); keep("D3", "proceeds-decile FE", m, "second")
say(f"  proceeds-decile fixed effects (non-linear size): second {r(m, 'second')}  elite {r(m, 'elite')}")
m = fit(f"lnpub ~ second + elite + {CTL} + C(procbin) + np.log(post_shares) + width", s.assign(post_shares=pd.to_numeric(s.post_shares, errors="coerce")))
say(f"  + ln post-IPO shares, range width: second {r(m, 'second')}  N={int(m.nobs)}")
# entropy balancing of second vs first division among small deals (proceeds <= $40M)
from scipy.optimize import minimize
def eb(d, cols, treat):
    Xc = d.loc[d[treat] == 0, cols].values; mt = d.loc[d[treat] == 1, cols].values.mean(0)
    Zc = (Xc - mt) / Xc.std(0).clip(1e-9)
    f = lambda l: np.log(np.exp(Zc @ l).sum()); gr = lambda l: (np.exp(Zc @ l) / np.exp(Zc @ l).sum()) @ Zc
    l = minimize(f, np.zeros(Zc.shape[1]), jac=gr, method="BFGS").x
    wc = np.exp(Zc @ l); wc = wc / wc.sum() * len(wc); out = pd.Series(1.0, index=d.index); out[d[treat] == 0] = wc
    return out
sm = s[s.proc <= 40].copy(); EBC = [k for k in X if sm[k].std() > 0] + ["biotech"]
sm["wt"] = eb(sm, EBC, "second")
bal = pd.DataFrame({"second": sm[sm.second == 1][EBC].mean(), "first raw": sm[sm.second == 0][EBC].mean(),
                    "first weighted": sm[sm.second == 0][EBC].multiply(sm.wt[sm.second == 0], axis=0).sum() / sm.wt[sm.second == 0].sum()})
say("  entropy balancing, proceeds <= $40M (max weight share of one control IPO: "
    f"{(sm.wt[sm.second == 0] / sm.wt[sm.second == 0].sum()).max():.2f}):"); say(bal.round(3).to_string())
bal.to_csv(OUT / "divisions_balance_small.csv")
for y, lab in [("lnpub", "ln public window"), ("lnamend", "ln(1+amendments)"), ("rev", "price revision"), ("irw", "first-day return"), ("exit_fail3", "failed within 3y")]:
    d = sm.dropna(subset=[y]); mm = smf.wls(f"{y} ~ second + C(ipo_year)", d, weights=d.wt)
    m = mm.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.loc[mm.data.row_labels, "yq"])[0]})
    keep("D3eb", lab, m, "second"); say(f"    {lab:20s} second {r(m, 'second')}  N={int(m.nobs)}")

say("\n== D4  robustness of the second-division effect on ln public window")
for lab, f, d, cl in [("baseline", f"lnpub ~ second + elite + {CTL}", s, "2way"), ("cluster by year only", f"lnpub ~ second + elite + {CTL}", s, "ipo_year"),
                      ("cut at rank < 6", f"lnpub ~ low6 + elite + {CTL}", s.assign(low6=(s.uw < 6).astype(int)), "2way"),
                      ("max rank of bookrunners", f"lnpub ~ second_m + elite + {CTL}", s.assign(second_m=(s.uw_max < 5).astype(int)), "2way"),
                      ("drop biotech", f"lnpub ~ second + elite + {CTL}", s[s.biotech == 0], "2way"),
                      ("drop 2020-21", f"lnpub ~ second + elite + {CTL}", s[~s.ipo_year.isin([2020, 2021])], "2way"),
                      ("EGC only", f"lnpub ~ second + elite + {CTL}", s[s.egc == 1], "2way"),
                      ("confidential filers only", f"lnpub ~ second + elite + {CTL}", s[s.confidential == 1], "2way"),
                      ("public filers only", f"lnpub ~ second + elite + {CTL}", s[s.confidential == 0], "2way"),
                      ("no prior RW, not dormant", f"lnpub ~ second + elite + {CTL}", s[(s.prior_rw == 0) & (s.dormant == 0)], "2way"),
                      ("year-quarter FE", f"lnpub ~ second + elite + {CTL.replace('C(ipo_year)', 'C(yq)')}", s, "2way")]:
    k = [v for v in ["low6", "second_m", "second"] if f"~ {v} " in f][0]
    try:
        m = fit(f, d, cl); keep("D4", lab, m, k); say(f"  {lab:28s} {r(m, k)}  N={int(m.nobs)}  n_second={int(d[k].sum()) if k in d else ''}")
    except Exception as e: say(f"  {lab:28s} failed: {e}")
say("\n  medians by division:")
med = s.groupby(np.select([s.second == 1, s.elite == 1], ["second", "elite"], "middle"))[["pub", "conf_period", "total_dur", "n_pub_amend", "rev", "ir", "proc", "exit3", "exit_fail3", "def301_3"]].agg(["median"]).round(2)
med2 = s.groupby(np.select([s.second == 1, s.elite == 1], ["second", "elite"], "middle"))[["exit3", "exit_merger3", "exit_fail3", "def301_3", "confidential", "egc", "biotech"]].mean().round(3)
say(med.to_string()); say("  shares:"); say(med2.to_string())
pd.DataFrame(rows).to_csv(OUT / "divisions_estimates.csv", index=False)
(OUT / "divisions_results.txt").write_text("\n".join(lines), encoding="utf-8")
