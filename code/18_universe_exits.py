"""Exits from the exchange after the IPO, from the issuer's EDGAR filing history (no CRSP). Free substitute for CRSP
delisting codes: who left the exchange, when, and why, within 1, 2 and 3 years of the IPO.
- Exit date (see the class filter in the loop): the first of
    Form 25-NSE (the exchange removes the security; counts even if the firm keeps filing, e.g. moved to OTC),
    Form 15-12B / 15-12G / 15-15D (deregistration),
    Form 25 filed by the issuer, only if it is a real exit: a Form 15 follows within a year or periodic reports
    (10-K/10-Q) stop within a year. Otherwise it is a transfer between exchanges (e.g. NYSE -> Nasdaq), not an exit.
- Reason: merger = 8-K item 2.01 (completion of acquisition) or 5.01 (change in control) in [-180, +30] days around
  the exit, or a merger proxy / tender-offer filing (DEFM14A, PREM14A, SC 14D9, SC TO-T, SC 13E3) in the prior year;
  bankruptcy = 8-K item 1.03 before the exit; other = everything else (exchange delisting for non-compliance,
  going dark). 'Failure' exits = bankruptcy + other.
- Distress short of exit: 8-K item 3.01 (notice of delisting or failure to satisfy a listing rule) within 3 years.
Uses the submissions JSON cached by 11 (data/edgar/submissions, pulled 30 Oct 2026, so every 3-year window of IPOs
up to Dec 2022 is fully observed). Output: data/universe/exits.csv"""
import json, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUB = ROOT / "data" / "edgar" / "submissions"
a = pd.read_csv(ROOT / "data" / "universe" / "analysis.csv", parse_dates=["t_issue"])

def hist(cik):
    f = SUB / f"CIK{int(cik):010d}.json"
    if not f.exists(): return None
    j = json.loads(f.read_text())
    cols = ["form", "filingDate", "items"]
    parts = [pd.DataFrame({c: j["filings"]["recent"][c] for c in cols})] + [pd.DataFrame({c: x[c] for c in cols}) for x in j.get("_extra", [])]
    h = pd.concat(parts, ignore_index=True); h["date"] = pd.to_datetime(h.filingDate); h["items"] = h["items"].fillna("")
    return h.sort_values("date").reset_index(drop=True), j.get("tickers", [])

PERIODIC = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"}
DEREG = {"15-12B", "15-12G", "15-15D"}
MERGER_DOCS = {"DEFM14A", "PREM14A", "SC 14D9", "SC TO-T", "SC 13E3", "DEFM14C", "PREM14C"}
rows = []
for r in a.itertuples():
    out = {"key": r.key, "exit_date": None, "exit_form": None, "exit_reason": None, "listed_now": None, "def301_date": None}
    x = hist(r.cik) if pd.notna(r.cik) else None
    if x is None: rows.append(out); continue
    h, tick = x; out["listed_now"] = int(len(tick) > 0)
    h = h[h.date > r.t_issue]
    per = h[h.form.isin(PERIODIC)].date
    # Forms 25/15 are filed per security class, so a delisted warrant, preferred, note or unit (AMC, ONSIU) or an
    # exchange transfer also produces one. A filing counts as the exit of the common stock only if the firm stops
    # filing periodic reports within a year after it, or (25-NSE) the exchange had sent a 3.01 deficiency notice in
    # the prior two years, i.e. a removal for non-compliance even if the firm keeps reporting from the OTC.
    d301 = h[h.form.str.startswith("8-K") & h["items"].str.contains(r"\b3\.01\b")]
    cands = []
    for e in h[h.form.isin({"25", "25-NSE"} | DEREG)].itertuples():
        stops = not (per > e.date + pd.Timedelta(days=365)).any()
        noncomp = e.form == "25-NSE" and d301.date.between(e.date - pd.Timedelta(days=730), e.date).any()
        if stops or noncomp: cands.append((e.date, e.form))
    if len(d301): out["def301_date"] = d301.date.iloc[0]
    if cands:
        d, frm = min(cands)
        eightk = h[h.form.str.startswith("8-K")]
        near = eightk[eightk.date.between(d - pd.Timedelta(days=180), d + pd.Timedelta(days=30))]
        prior = h[h.date.between(d - pd.Timedelta(days=365), d + pd.Timedelta(days=30))]
        if near["items"].str.contains(r"\b(?:2\.01|5\.01)\b").any() or prior.form.isin(MERGER_DOCS).any(): why = "merger"
        elif eightk[eightk.date <= d + pd.Timedelta(days=30)]["items"].str.contains(r"\b1\.03\b").any(): why = "bankruptcy"
        else: why = "other"
        out.update(exit_date=d, exit_form=frm, exit_reason=why)
    rows.append(out)
e = pd.DataFrame(rows)
e = e.merge(a[["key", "t_issue"]], on="key")
days = (pd.to_datetime(e.exit_date) - e.t_issue).dt.days
for y in [1, 2, 3]:
    w = days.le(365 * y)
    e[f"exit{y}"] = w.astype(int)
    e[f"exit_merger{y}"] = (w & e.exit_reason.eq("merger")).astype(int)
    e[f"exit_fail{y}"] = (w & e.exit_reason.isin(["bankruptcy", "other"])).astype(int)
e["def301_3"] = ((pd.to_datetime(e.def301_date) - e.t_issue).dt.days.le(365 * 3)).astype(int)
e["days_listed"] = days
# consistency: a firm with no exit filing should still have a ticker today, unless it exited by other routes
e["no_exit_but_unlisted"] = (e.exit_date.isna() & e.listed_now.eq(0)).astype(int)
e.drop(columns="t_issue").to_csv(ROOT / "data" / "universe" / "exits.csv", index=False)
print("N", len(e), "| exit within 1/2/3y:", [round(e[f'exit{y}'].mean(), 3) for y in (1, 2, 3)])
print("reasons (any time):", e.exit_reason.value_counts().to_dict(), "| forms:", e.exit_form.value_counts().to_dict())
print("3y: merger", e.exit_merger3.mean().round(3), "fail", e.exit_fail3.mean().round(3), "| 3.01 notice within 3y", e.def301_3.mean().round(3))
print("no exit filing but no ticker today:", e.no_exit_but_unlisted.sum(), "| exit filing but still has ticker:", (e.exit_date.notna() & e.listed_now.eq(1)).sum())
