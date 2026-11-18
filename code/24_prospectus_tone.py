"""Prospectus text by division: length and Loughran-McDonald tone (share of words, not counts) in the first public
registration statement and the pricing prospectus, the risk-factor section's length, and the change between the two.
Text from 21 (TEXT_DIR), dictionary TEXT_DIR/lm/LM_MasterDictionary_1993-2025.csv. Same sample and model as 16.
Writes data/universe/prospectus_tone.csv, output/prospectus_tone.txt"""
import os, re, gzip, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
TXT = pathlib.Path(os.environ.get("TEXT_DIR", ROOT / "data" / "text"))
src = (ROOT / "code" / "16_universe_paper.py").read_text(encoding="utf-8").split('say("\\n== T1')[0]
g = {"__file__": str(ROOT / "code" / "16_universe_paper.py")}; exec(src, g)
s, CTL, r = g["s"], g["CTL"], g["r"]
fit = lambda f, d: g["fit"](f, d, "2way")
lines = []
def say(*a):
    x = " ".join(str(v) for v in a); print(x); lines.append(x)

lm = pd.read_csv(TXT / "lm" / "LM_MasterDictionary_1993-2025.csv", keep_default_na=False)
CAT = {c: set(lm.loc[lm[c] > 0, "Word"].str.lower()) for c in ["Negative", "Positive", "Uncertainty", "Litigious", "Weak_Modal"]}
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
def risk_len(tx):
    # the risk-factor section: from the last "RISK FACTORS" heading line before "You should carefully consider"
    # to the next major heading; fall back to 0 when not found
    m = re.search(r"(?is)\n\s*risk factors\s*\n(.{0,400}?)(investing in our|an investment in|you should carefully consider)", tx)
    if not m: return np.nan
    rest = tx[m.start():]
    e = re.search(r"\n\s*(special note regarding forward|cautionary note regarding forward|forward-looking statements|use of proceeds)\s*\n", rest, re.I)
    return len(WORD.findall(rest[:e.start()].lower())) if e else np.nan
def measures(path):
    tx = gzip.open(path, "rt", encoding="utf-8").read()
    w = WORD.findall(tx.lower()); n = len(w) or 1
    out = {"words": len(w), "risk_words": risk_len(tx)}
    for c, S in CAT.items(): out[c.lower()] = sum(x in S for x in w) / n * 100
    return out

D = pd.read_csv(TXT / "docs.csv")
D = D[D.key.isin(s.key) & D.role.isin(["first", "final_424"])]
R = []
for x in D.itertuples():
    p = TXT / "docs" / f"{x.cik}_{x.acc}.txt.gz"
    if p.exists(): R.append({"key": x.key, "role": x.role, **measures(p)})
T = pd.DataFrame(R).pivot_table(index="key", columns="role").sort_index(axis=1)
T.columns = [f"{a}_{b}" for a, b in T.columns]; T = T.reset_index()
T.to_csv(ROOT / "data" / "universe" / "prospectus_tone.csv", index=False)
d = s.merge(T, on="key", how="inner")
d["second"] = (d.uw < 5).astype(int); d["elite"] = (d.uw >= 9).astype(int)
for v in ["words", "risk_words"]:
    for ro in ["first", "final_424"]: d[f"ln_{v}_{ro}"] = np.log(d[f"{v}_{ro}"])
    d[f"dln_{v}"] = d[f"ln_{v}_final_424"] - d[f"ln_{v}_first"]
for c in [k.lower() for k in CAT]: d[f"d_{c}"] = d[f"{c}_final_424"] - d[f"{c}_first"]
say(f"IPOs with both documents: {len(d)}  (second {int(d.second.sum())}); risk section found first/final:",
    int(d.risk_words_first.notna().sum()), int(d.risk_words_final_424.notna().sum()))
grp = np.select([d.second == 1, d.elite == 1], ["second", "elite"], "middle")
cols = ["words_first", "words_final_424", "risk_words_first", "risk_words_final_424", "negative_first", "negative_final_424",
        "uncertainty_first", "uncertainty_final_424", "litigious_first", "positive_first"]
say(d.groupby(grp)[cols].median().round(2).T.to_string())
say("\n== division effects; controls + FE, two-way clusters")
ys = ["ln_words_first", "ln_risk_words_first", "negative_first", "uncertainty_first", "litigious_first", "positive_first",
      "dln_words", "dln_risk_words", "d_negative", "d_uncertainty", "d_litigious", "d_positive"]
for y in ys:
    e = d.dropna(subset=[y]); m = fit(f"{y} ~ second + elite + {CTL}", e)
    say(f"  {y:22s} second {r(m, 'second')}  elite {r(m, 'elite')}  mean={e[y].mean():7.3f}  N={int(m.nobs)}")
(ROOT / "output" / "prospectus_tone.txt").write_text("\n".join(lines), encoding="utf-8")
