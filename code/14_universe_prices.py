"""First-day return for the universe. Sources, in order:
(1) Bloomberg 'Offer To 1st Close' (IPO Bloomberg.xlsx, % return), matched by ticker with pricing date within 10 days of
    the first trade (validated: corr 0.996 with our Eikon-based return on 548 overlapping IPOs);
(2) Yahoo Finance first-day close (split-unadjusted) over the 424B offer price, for tickers still listed and whose Yahoo
    history starts within [-1, +5] days of the first trade. Also keeps Yahoo's close ~90 calendar days later (survivors only;
    long-run returns wait for CRSP). Run with yfinance on PYTHONPATH. Output: data/universe/prices.csv"""
import pathlib, warnings, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external"  # see README: IPO-age.xlsx (Ritter, public); IPO Bloomberg.xlsx (licensed, not included)
OUT = ROOT / "data" / "universe"
t = pd.read_csv(OUT / "timeline.csv", parse_dates=["t_issue"])
t = t[t.status == "ok"].merge(pd.read_csv(OUT / "prospectus.csv")[["key", "offer_424"]], on="key", how="left")

f_bb = DATA / "IPO Bloomberg.xlsx"  # licensed; without it only the Yahoo return is built
b = pd.read_excel(f_bb, sheet_name="IPO Bloomberg - Ticker") if f_bb.exists() else pd.DataFrame(columns=["Issuer Ticker", "Pricing Date", "Offer To 1st Close", "Offer Size (M)"])
b["tk"] = b["Issuer Ticker"].astype(str).str.split().str[0].str.upper()
b["t_bb"] = pd.to_datetime(b["Pricing Date"], errors="coerce")
b = b[b["Offer To 1st Close"].notna()][["tk", "t_bb", "Offer To 1st Close", "Offer Size (M)"]]
m = t[["key", "tickersymbol", "t_issue"]].assign(tk=t.tickersymbol.astype(str).str.upper()).merge(b, on="tk")
m = m[(m.t_bb - m.t_issue).dt.days.abs() <= 10].sort_values("key").drop_duplicates("key")
t = t.merge(m[["key", "Offer To 1st Close", "Offer Size (M)"]].rename(columns={"Offer To 1st Close": "ir_bb", "Offer Size (M)": "size_bb"}), on="key", how="left")

rows = []
for r in t.itertuples():
    out = {"key": r.key}
    try:
        tk = yf.Ticker(str(r.tickersymbol))
        h = tk.history(start=str((r.t_issue - pd.Timedelta(days=3)).date()), end=str((r.t_issue + pd.Timedelta(days=130)).date()), auto_adjust=False)
        if len(h):
            h.index = h.index.tz_localize(None).normalize()
            d0 = h.index[0]
            sp = tk.splits
            fac = 1.0
            if len(sp):
                sp.index = sp.index.tz_localize(None)
                fac = float(sp[sp.index > d0].prod()) if (sp.index > d0).any() else 1.0
            h90 = h[h.index <= d0 + pd.Timedelta(days=90)]
            out |= {"yf_first_day": d0, "yf_close1": float(h.Close.iloc[0]) * fac, "yf_close90": float(h90.Close.iloc[-1]) * fac,
                    "yf_day90": h90.index[-1], "days_from_issue": (d0 - r.t_issue).days}
    except Exception as e:
        out["err"] = str(e)[:80]
    rows.append(out)
y = pd.DataFrame(rows)
t = t.merge(y, on="key", how="left")
ok = t.days_from_issue.between(-1, 5) & t.offer_424.gt(0)
t["ir_yf"] = (t.yf_close1 / t.offer_424 - 1).where(ok) * 100
t["ir"] = t.ir_bb.fillna(t.ir_yf)
t["ir_src"] = t.ir_bb.notna().map({True: "bloomberg", False: None}).fillna(t.ir_yf.notna().map({True: "yahoo", False: None}))
t[["key", "ir", "ir_src", "ir_bb", "ir_yf", "size_bb", "yf_close1", "yf_close90", "yf_day90", "days_from_issue"]].to_csv(OUT / "prices.csv", index=False)
both = t[t.ir_bb.notna() & t.ir_yf.notna()]
print("ir found:", t.ir.notna().mean().round(3), t.ir_src.value_counts().to_dict())
print("bloomberg vs yahoo on overlap:", len(both), both[["ir_bb", "ir_yf"]].corr().iloc[0, 1].round(3), ((both.ir_bb - both.ir_yf).abs() < 2).mean().round(3))
