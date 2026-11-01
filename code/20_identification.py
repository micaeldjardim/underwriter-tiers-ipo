"""Two tests of where the second-division delay comes from. Same sample, controls, FE and clustering as 16/17.
I1  SEC workload (19): if the delay is a review bottleneck for weak filings, the second-division gap should widen when the
    reviewing staff is busy. lnpub on division dummies x standardized log workload (EDGAR-wide and same-office).
I2  Nasdaq's liquidity rule (SR-NASDAQ-2019-009, approved 5 Jul 2019, effective 5 Aug 2019): restricted shares no longer
    count toward float, and half the round-lot holders must hold unrestricted shares worth $2,500 or more. It raised the
    bar for micro-cap listings without changing their banks. Second division x post-rule, with year FE.
Writes output/identification_results.txt and output/identification_estimates.csv"""
import pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / "code" / "16_universe_paper.py").read_text(encoding="utf-8").split('say("\\n== T1')[0]
g = {"__file__": str(ROOT / "code" / "16_universe_paper.py")}; exec(src, g)
s, CTL, r = g["s"], g["CTL"], g["r"]
fit = lambda f, d=None, cl="2way": g["fit"](f, s if d is None else d, cl)
lines, rows = [], []
def say(*a):
    t = " ".join(str(v) for v in a); print(t); lines.append(t)
def keep(tab, name, m, k):
    rows.append({"table": tab, "model": name, "var": k, "coef": m.params[k], "se": m.bse[k], "t": m.tvalues[k], "p": m.pvalues[k], "N": int(m.nobs)})

s = s.merge(pd.read_csv(ROOT / "data" / "universe" / "workload.csv"), on="key", how="left")
s["second"] = (s.uw < 5).astype(int); s["elite"] = (s.uw >= 9).astype(int)
z = lambda v: (v - v.mean()) / v.std()
s["wl"] = z(np.log(s.wl_all)); s["wlo"] = z(np.log1p(s.wl_office))

say("== I1  SEC workload in the 30 days before the public flip (standardized log count of registration filings)")
say("  workload by division (median filings, all / same office):", s.groupby("second")[["wl_all", "wl_office"]].median().to_dict())
for lab, f, d in [("EDGAR-wide", f"lnpub ~ second * wl + elite * wl + {CTL}", s),
                  ("same office", f"lnpub ~ second * wlo + elite * wlo + {CTL} + C(office)", s.dropna(subset=["wlo", "office"])),
                  ("EDGAR-wide, amendments", f"lnamend ~ second * wl + elite * wl + {CTL}", s)]:
    m = fit(f, d); w = "wlo" if "wlo" in f else "wl"
    for k in ["second", w, f"second:{w}", f"elite:{w}"]: keep("I1", lab, m, k)
    say(f"  {lab:24s} second {r(m, 'second')}  workload {r(m, w)}  second x workload {r(m, f'second:{w}')}  elite x workload {r(m, f'elite:{w}')}  N={int(m.nobs)}")

say("\n== I2  Nasdaq liquidity rule, effective 5 Aug 2019 (IPOs priced on or after)")
s["post19"] = (s.t_issue >= "2019-08-05").astype(int)
say("  second-division IPOs before/after:", int(((s.second == 1) & (s.post19 == 0)).sum()), int(((s.second == 1) & (s.post19 == 1)).sum()),
    "  share of all IPOs before/after:", s.groupby("post19").second.mean().round(3).to_dict())
say("  median proceeds $M of second division before/after:", s[s.second == 1].groupby("post19").lnproc.apply(lambda v: np.exp(v).median()).round(1).to_dict())
for y in ["lnpub", "lnamend", "lnconf"]:
    d = s if y != "lnconf" else s[s.confidential == 1]
    m = fit(f"{y} ~ second * post19 + elite + {CTL}", d); keep("I2", y, m, "second:post19"); keep("I2", y, m, "second")
    say(f"  {y:8s} second {r(m, 'second')}  second x post-rule {r(m, 'second:post19')}  N={int(m.nobs)}")
w2 = s[(s.t_issue >= "2017-08-05") & (s.t_issue < "2021-08-05")]
m = fit(f"lnpub ~ second * post19 + elite + {CTL}", w2); keep("I2", "lnpub, 2 years each side", m, "second:post19")
say(f"  lnpub, two years each side: second x post-rule {r(m, 'second:post19')}  N={int(m.nobs)}  n_second={int(w2.second.sum())}")
pd.DataFrame(rows).to_csv(ROOT / "output" / "identification_estimates.csv", index=False)
(ROOT / "output" / "identification_results.txt").write_text("\n".join(lines), encoding="utf-8")
