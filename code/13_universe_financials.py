"""Pre-IPO accounting controls from EDGAR XBRL company facts (free API).
Registration statements were not XBRL-tagged for most of the period, but the first 10-Q/10-K after the IPO reports the
pre-IPO fiscal year as comparatives. So for each IPO we take, per concept, the latest value whose period ENDS on or before
the first trading day: balance-sheet items at the last pre-IPO instant, flow items over the last pre-IPO fiscal year
(duration 330-400 days). US-GAAP first, IFRS for foreign private issuers.
Output: data/universe/financials.csv"""
import json, time, pathlib, requests, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "universe"
FACTS = ROOT / "data" / "edgar" / "companyfacts"; FACTS.mkdir(exist_ok=True)
UA = {"User-Agent": __import__("os").environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")}  # SEC asks for a contact
t = pd.read_csv(OUT / "timeline.csv", parse_dates=["t_issue"])
t = t[t.status == "ok"]

STOCK = {"assets": ["Assets"], "liabilities": ["Liabilities"], "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalents"],
         "equity": ["StockholdersEquity", "Equity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
         "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt", "NoncurrentBorrowings"]}
FLOW = {"revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenue", "RevenueFromContractsWithCustomers"],
        "net_income": ["NetIncomeLoss", "ProfitLoss", "ProfitLossAttributableToOwnersOfParent"],
        "op_income": ["OperatingIncomeLoss", "ProfitLossFromOperatingActivities"],
        "rnd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost", "ResearchAndDevelopmentExpenseRecognisedAsExpense"],
        "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss"]}

def facts(cik):
    f = FACTS / f"CIK{int(cik):010d}.json"
    if not f.exists():
        for k in range(4):
            try:
                r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json", headers=UA, timeout=60); time.sleep(0.12)
                if r.status_code == 200: f.write_text(r.text); break
                if r.status_code == 404: f.write_text("{}"); break
            except requests.RequestException: pass
            time.sleep(2 ** k)
    return json.loads(f.read_text()) if f.exists() else {}

def pick(j, names, t_issue, flow):
    for tax in ("us-gaap", "ifrs-full"):
        for n in names:
            node = j.get("facts", {}).get(tax, {}).get(n)
            if not node: continue
            vals = [v | {"unit": u} for u, lst in node["units"].items() for v in lst if u in ("USD",) or tax == "ifrs-full"]
            d = pd.DataFrame(vals)
            if d.empty: continue
            d["end"] = pd.to_datetime(d.end)
            d = d[(d.end <= t_issue) & (d.end >= t_issue - pd.Timedelta(days=550))]
            if flow:
                if "start" not in d: continue
                d = d[(d.end - pd.to_datetime(d.start)).dt.days.between(330, 400)]
            if d.empty: continue
            d = d.sort_values(["end", "filed"])
            last = d[d.end == d.end.max()].iloc[0]          # earliest filing reporting that period
            return last.val, str(last.end.date()), last.unit, tax
    return None, None, None, None

rows = []
for i, r in enumerate(t.itertuples()):
    j = facts(r.cik)
    out = {"key": r.key}
    for k, names in (STOCK | FLOW).items():
        v, end, unit, tax = pick(j, names, r.t_issue, k in FLOW)
        out[k] = v
        if k in ("assets", "revenue"): out[f"{k}_end"] = end; out[f"{k}_unit"] = unit
    rows.append(out)
    if i % 100 == 0: print(i, r.key, out, flush=True)
d = pd.DataFrame(rows)
d.to_csv(OUT / "financials.csv", index=False)
print(d.notna().mean().round(3).to_string())
