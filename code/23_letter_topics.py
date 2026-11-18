"""What the public-phase comments are about. Letters and sample from 22; each numbered comment is split out and tagged
by keyword rules (a comment can carry several topics). Compares topic counts per IPO by division, public and
confidential phase, with the controls, fixed effects and clustering of 16.
Writes output/letter_topics.txt"""
import re, gzip, runpy, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
g = runpy.run_path(str(ROOT / "code" / "22_sec_letters.py"))
L, t, fit, r, CTL, TXT = g["L"], g["t"], g["fit"], g["r"], g["CTL"], g["TXT"]
lines = []
def say(*a):
    x = " ".join(str(v) for v in a); print(x); lines.append(x)

TOPICS = {
    "price_range":   r"price range|bona fide|rule 430a|blanks?\b|omitted information|fill in",
    "financials":    r"financial statements|audit(ed|or)|rule 3-12|staleness|updated? (your )?financial|interim period",
    "revenue":       r"revenue recognition|asc 606|asc 605|deferred revenue|gross versus net|principal versus agent",
    "stock_comp":    r"stock-based compensation|share-based compensation|cheap stock|fair value of (your )?common stock|409a|grant date",
    "mdna":          r"management.s discussion|liquidity|results of operations|key (performance )?metrics|non-gaap",
    "risk":          r"risk factor",
    "business":      r"\bbusiness\b section|competitive|market (size|opportunity)|industry data|third.party (data|sources)",
    "governance":    r"controlled company|dual.class|board of directors|related.party|executive compensation|lock-up",
    "underwriting":  r"underwrit|directed share|selling (stock|share)holder",
    "legal_exhibit": r"exhibit|legal opinion|material (agreement|contract)|confidential treatment",
    "egc_testing":   r"testing.the.waters|emerging growth|written communications|section 5\(d\)",
}
NUMLINE = re.compile(r"(?m)^\s*(\d{1,3})\.\s+")
rows = []
for x in L[L.review].itertuples():
    tx = gzip.open(TXT / "letters" / f"{x.cik}_{x.acc}.txt.gz", "rt", encoding="utf-8").read()
    marks, k = [], 0
    for m in NUMLINE.finditer(tx):
        if int(m.group(1)) == k + 1: k += 1; marks.append(m.start())
    for i, a in enumerate(marks):
        c = tx[a:marks[i + 1] if i + 1 < len(marks) else len(tx)][:4000].lower()
        rows.append({"key": x.key, "phase": x.phase, **{k2: int(bool(re.search(v, c))) for k2, v in TOPICS.items()}})
C = pd.DataFrame(rows)
say(f"comments split: {len(C)} from {C.key.nunique()} IPOs; share tagged with any topic {C[list(TOPICS)].max(axis=1).mean():.2f}")
say("share of comments per topic, all phases:", C[list(TOPICS)].mean().round(3).to_dict())

P = C.groupby(["key", "phase"])[list(TOPICS)].sum().unstack("phase", fill_value=0)
P.columns = [f"{a}_{b}" for a, b in P.columns]
d = t.merge(P.reset_index(), on="key", how="left").fillna({c: 0 for c in P.columns})
grp = np.select([d.second == 1, d.elite == 1], ["second", "elite"], "middle")
say("\n== mean comments per IPO by topic, PUBLIC phase")
say(d.groupby(grp)[[f"{k}_pub" for k in TOPICS if f"{k}_pub" in d]].mean().round(2).T.to_string())
say("\n== mean comments per IPO by topic, CONFIDENTIAL phase")
say(d.groupby(grp)[[f"{k}_conf" for k in TOPICS if f"{k}_conf" in d]].mean().round(2).T.to_string())
say("\n== second-division effect on ln(1 + comments on topic), public phase; controls + FE, two-way clusters")
for k in TOPICS:
    for ph in ["pub", "conf"]:
        c = f"{k}_{ph}"
        if c not in d: continue
        d["y"] = np.log1p(d[c]); m = fit(f"y ~ second + elite + {CTL}", d)
        say(f"  {c:20s} second {r(m, 'second')}  elite {r(m, 'elite')}  mean={d[c].mean():.2f}")
(ROOT / "output" / "letter_topics.txt").write_text("\n".join(lines), encoding="utf-8")
