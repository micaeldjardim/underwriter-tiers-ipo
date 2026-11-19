"""Descriptive statistics and inference checks for the paper. Same sample, controls and fixed effects as 16.
S1  summary statistics of every variable in the models (the 828 IPOs with every control), and of the comment-letter
    and prospectus measures from 22 and 24
S2  the tier effects under four error structures: two-way (industry x issue quarter, the baseline), issue quarter,
    IPO year, and a wild cluster bootstrap by IPO year (Rademacher weights, null imposed); R2 and cluster counts
Writes output/descriptives_paper.csv, output/inference_checks.txt"""
import pathlib, runpy, numpy as np, pandas as pd, statsmodels.formula.api as smf
ROOT = pathlib.Path(__file__).resolve().parents[1]
g = runpy.run_path(str(ROOT / "code" / "22_sec_letters.py"))            # s with the letter counts, CTL, fit(), r()
s, CTL = g["s"], g["CTL"]
fit = g["g"]["fit"]
lines = []
def say(*a):
    x = " ".join(str(v) for v in a); print(x); lines.append(x)
tone = pd.read_csv(ROOT / "data" / "universe" / "prospectus_tone.csv")
s = s.merge(tone, on="key", how="left")
s["proc"] = np.exp(s.lnproc); s["age"] = np.exp(s.lnage) - 1; s["assets_m"] = np.exp(s.lnassets) / 1e6 if s.lnassets.mean() > 12 else np.exp(s.lnassets)
s["words_k"] = s.words_first / 1000; s["n_ipo90v"] = np.exp(s.lnipo90) - 1 if "n_ipo90" not in s else s.n_ipo90

V = [("pub", "Public window (days)"), ("conf_period", "Confidential phase (days)"), ("total_dur", "Total registration (days)"),
     ("n_pub_amend", "Public amendments"), ("n_drs_all", "Confidential drafts"), ("rev", "Price revision (%)"), ("irw", "First-day return (%)"),
     ("uw", "Lead underwriter rank"), ("second", "Second division (rank 2-4)"), ("elite", "Elite (rank 9)"),
     ("proc", "Proceeds ($ million)"), ("assets_m", "Total assets ($ million)"), ("age", "Firm age (years)"), ("vc", "VC-backed"),
     ("loss", "Loss"), ("rnd_int", "R&D / assets"), ("big4", "Big Four auditor"), ("dual", "Dual-class shares"),
     ("nasdaq", "Nasdaq listing"), ("secondary", "Secondary shares"), ("tech", "Technology firm"), ("biotech", "Biotech"),
     ("confidential", "Confidential filer"), ("egc", "Emerging growth company"), ("nasdaq90", "Nasdaq return, prior 90 days (%)"),
     ("n_ipo90v", "IPOs in the prior 90 days"), ("ir90", "Mean first-day return, prior 90 days (%)"),
     ("comments_all", "SEC comments, all"), ("comments_conf", "SEC comments, confidential phase"), ("comments_pub", "SEC comments, public phase"),
     ("rounds_all", "SEC letter rounds, all"), ("rounds_pub", "SEC letter rounds, public phase"), ("first_letter", "Comments in the first letter"),
     ("words_k", "Prospectus words (thousands)"), ("negative_first", "Negative words (%)"), ("uncertainty_first", "Uncertainty words (%)"),
     ("litigious_first", "Litigious words (%)"), ("positive_first", "Positive words (%)")]
rows = []
for v, lab in V:
    if v not in s: say("missing", v); continue
    x = pd.to_numeric(s[v], errors="coerce").dropna()
    rows.append({"var": v, "label": lab, "N": len(x), "mean": x.mean(), "sd": x.std(), "p25": x.quantile(.25), "median": x.median(), "p75": x.quantile(.75)})
D = pd.DataFrame(rows); D.to_csv(ROOT / "output" / "descriptives_paper.csv", index=False)
say("== S1  summary statistics"); say(D.round(3).to_string(index=False))

say("\n== S2  clusters in the estimation sample:", "industries", s.ind.nunique(), " issue quarters", s.yq.nunique(), " IPO years", s.ipo_year.nunique())
def wild(f, d, k, B=999, seed=11):
    """Wild cluster bootstrap-t by IPO year with the null imposed on k (Cameron, Gelbach and Miller 2008)."""
    d = d.dropna(subset=[f.split("~")[0].strip()]).reset_index(drop=True)
    full = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ipo_year)[0]})
    t0 = full.tvalues[k]
    fr = f.replace(f"{k} + ", "", 1)
    res = smf.ols(fr, d).fit(); yhat, u = res.fittedvalues.values, res.resid.values
    Xm = full.model.exog; names = full.model.exog_names; j = names.index(k)
    gid = pd.factorize(d.ipo_year, sort=True)[0]; G = gid.max() + 1   # sorted, so the draws do not depend on row order
    XtXi = np.linalg.pinv(Xm.T @ Xm); rng = np.random.default_rng(seed); ts = []
    n, p = Xm.shape; adj = (G / (G - 1)) * ((n - 1) / (n - p))
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], G)[gid]; yb = yhat + u * w
        b = XtXi @ (Xm.T @ yb); e = yb - Xm @ b
        meat = np.zeros((p, p))
        for c in range(G):
            sc = Xm[gid == c].T @ e[gid == c]; meat += np.outer(sc, sc)
        se = np.sqrt(adj * (XtXi @ meat @ XtXi)[j, j]); ts.append(b[j] / se)
    return t0, float((np.abs(ts) >= abs(t0)).mean())

s["second"] = (s.uw < 5).astype(int); s["elite"] = (s.uw >= 9).astype(int)
c = s[s.confidential == 1]; NP = CTL.replace("lnproc + ", ""); t = s[s.has_letters == 1]
SPEC = [("ln public window", "lnpub", s, CTL, "second"), ("ln(1+public amendments)", "lnamend", s, CTL, "second"),
        ("ln confidential phase", "lnconf", c, CTL, "second"), ("ln(1+confidential drafts)", "lndrs", c, CTL, "second"),
        ("ln total registration", "lntotal", s, CTL, "second"), ("price revision, %", "rev", s, NP, "elite"),
        ("first-day return, %", "irw", s, NP, "elite"),
        ("ln(1+comments), all", "ln_comments_all", t, CTL, "second"), ("ln(1+comments), public", "ln_comments_pub", t, CTL, "second"),
        ("ln(1+rounds), public", "ln_rounds_pub", t, CTL, "second"), ("ln first letter", "ln_first_letter", t, CTL, "second")]
say(f"{'outcome':28s} {'var':6s} {'coef':>7s} {'t 2way':>7s} {'t qtr':>7s} {'t year':>7s} {'wild p':>7s} {'R2':>6s} {'N':>5s}")
for lab, y, d, ctl, k in SPEC:
    f = f"{y} ~ {k} + {'elite' if k == 'second' else 'second'} + {ctl}"
    d = d.dropna(subset=[y])
    m2, mq, my = fit(f, d, "2way"), fit(f, d, "yq"), fit(f, d, "ipo_year")
    t0, pw = wild(f, d, k)
    say(f"{lab:28s} {k:6s} {m2.params[k]:7.3f} {m2.tvalues[k]:7.2f} {mq.tvalues[k]:7.2f} {my.tvalues[k]:7.2f} {pw:7.3f} {m2.rsquared:6.3f} {int(m2.nobs):5d}")
(ROOT / "output" / "inference_checks.txt").write_text("\n".join(lines), encoding="utf-8")

say("\n== S3  Table 'public comment load': second-division t under IPO-year clustering")
for y in ["lnpub", "lnamend"]:
    for lab, extra in [("base", ""), ("+ public load", "ln_comments_pub + ln_rounds_pub + "), ("+ first letter", "ln_first_letter + ")]:
        f = f"{y} ~ second + elite + {extra}{CTL}"; d = t.dropna(subset=["ln_first_letter"]) if "first" in lab else t
        m2, my = fit(f, d, "2way"), fit(f, d, "ipo_year")
        say(f"  {y:8s} {lab:15s} second {m2.params['second']:6.3f}  t 2way {m2.tvalues['second']:6.2f}  t year {my.tvalues['second']:5.2f}  R2 {m2.rsquared:.3f}  N {int(m2.nobs)}")
(ROOT / "output" / "inference_checks.txt").write_text("\n".join(lines), encoding="utf-8")
