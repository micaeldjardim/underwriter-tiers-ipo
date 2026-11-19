"""Estimates quoted in the paper that the other scripts do not save: Panel C of the comment-letter table (confidential
filers only), the association between the public window and the offer-price revision, the size-fixed-effects row of
the deal-size table under the two-way and year clusterings, and the statistics of the 2019 listing-rule paragraph.
Same sample, controls, fixed effects and clustering as 22. Writes output/extra_estimates.txt"""
import pathlib, runpy, numpy as np, pandas as pd, statsmodels.formula.api as smf
ROOT = pathlib.Path(__file__).resolve().parents[1]
g = runpy.run_path(str(ROOT / "code" / "22_sec_letters.py"))
s, CTL = g["s"].copy(), g["CTL"]
lines = []
def say(*a):
    x = " ".join(str(v) for v in a); print(x); lines.append(x)
def fit(f, d, cl="2way"):
    m = smf.ols(f, d); rl = m.data.row_labels
    gr = d.loc[rl, ["ind", "yq"]].apply(lambda c: pd.factorize(c)[0]).values if cl == "2way" else pd.factorize(d.loc[rl, cl])[0]
    return m.fit(cov_type="cluster", cov_kwds={"groups": gr})
say("columns with comment counts:", [c for c in s.columns if "comment" in c or "round" in c or "letter" in c])
c = s[s.confidential == 1].copy()
if "share_pub" not in c:
    tot = c.comments_conf.fillna(0) + c.comments_pub.fillna(0)
    c["share_pub"] = (c.comments_pub / tot).where(tot > 0)
for v in ["rounds_conf", "comments_conf", "comments_pub", "rounds_pub"]:
    if "ln_" + v not in c and v in c: c["ln_" + v] = np.log1p(c[v])
if "ln_comments_all" not in c: c["ln_comments_all"] = np.log1p(c.comments_conf.fillna(0) + c.comments_pub.fillna(0))
say("\n== Panel C: confidential filers only (IPOs with letters)")
c = c[c.has_letters == 1] if "has_letters" in c else c
say("N confidential filers with letters:", len(c), "| second division:", int(c.second.sum()), "| raw share of comments in the public phase by tier:",
    c.groupby(np.select([c.second == 1, c.elite == 1], ["second", "elite"], "middle")).share_pub.mean().round(3).to_dict())
for y in ["ln_comments_all", "ln_comments_conf", "ln_comments_pub", "ln_rounds_conf", "ln_rounds_pub", "share_pub"]:
    m = fit(f"{y} ~ second + elite + {CTL}", c)
    say(f"  {y:18s} second {m.params['second']:7.3f} (t={m.tvalues['second']:5.2f})  elite {m.params['elite']:7.3f} (t={m.tvalues['elite']:5.2f})  N={int(m.nobs)}")
say("\n== public window and price revision, all IPOs")
for lab, f in [("rev ~ lnpub + tiers + controls", f"rev ~ lnpub + second + elite + {CTL}"), ("rev ~ lnpub + controls", f"rev ~ lnpub + {CTL}"),
               ("irw ~ lnpub + tiers + controls", f"irw ~ lnpub + second + elite + {CTL}")]:
    m = fit(f, s); say(f"  {lab:34s} lnpub {m.params['lnpub']:7.3f} (t={m.tvalues['lnpub']:5.2f})  N={int(m.nobs)}")
say("\n== proceeds-decile fixed effects")
s["pdec"] = pd.qcut(s.lnproc, 10, labels=False)
f = f"lnpub ~ second + elite + C(pdec) + {CTL}"      # as in 17: decile indicators in addition to log proceeds
for cl in ["2way", "yq", "ipo_year"]:
    try:
        m = fit(f, s, cl); say(f"  cluster {cl:9s} second {m.params['second']:6.3f} (t={m.tvalues['second']:5.2f})  elite {m.params['elite']:6.3f} (t={m.tvalues['elite']:5.2f})")
    except Exception as e: say("  failed", cl, e)
say("\n== second division by size and year")
s["proc"] = np.exp(s.lnproc)
say("  median proceeds by tier:", s.groupby(np.select([s.second == 1, s.elite == 1], ["second", "elite"], "middle")).proc.median().round(1).to_dict())
say("  second-division IPOs priced in 2021-2022:", int(((s.second == 1) & s.ipo_year.isin([2021, 2022])).sum()), "of", int(s.second.sum()))
for cap in [20, 40, 60]:
    d = s[s.proc <= cap]; say(f"  proceeds <= ${cap}M: second {int(d.second.sum())}, ranked 5 or above {int((d.second == 0).sum())}")
(ROOT / "output" / "extra_estimates.txt").write_text("\n".join(lines), encoding="utf-8")
