"""What the SEC asks the second division. Staff comment letters (UPLOAD) from 21, same sample, controls, fixed effects
and two-way clustering as 16.
L1  letters on the IPO registration (draft or public), split by the phase they review; comments per letter = the
    highest numbered comment in it (staff number comments 1..k within each letter)
L2  division effects on rounds and comments, confidential and public phase
L3  does the public-phase comment load carry the second-division gap in the public window and in public amendments?
Reads TEXT_DIR (default data/text). Writes data/universe/sec_letters.csv, output/letters_results.txt"""
import os, re, gzip, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
TXT = pathlib.Path(os.environ.get("TEXT_DIR", ROOT / "data" / "text"))
src = (ROOT / "code" / "16_universe_paper.py").read_text(encoding="utf-8").split('say("\\n== T1')[0]
g = {"__file__": str(ROOT / "code" / "16_universe_paper.py")}; exec(src, g)          # sample s, CTL, fit(), r()
s, CTL, r = g["s"], g["CTL"], g["r"]
fit = lambda f, d=None, cl="2way": g["fit"](f, s if d is None else d, cl)
lines = []
def say(*a):
    t = " ".join(str(v) for v in a); print(t); lines.append(t)

MON = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
DATE = re.compile(MON + r"\s+(\d{1,2}),\s*(\d{4})")
NUM = re.compile(r"(?m)^\s*(\d{1,3})\.\s+\S")
def parse(path):
    t = gzip.open(path, "rt", encoding="utf-8").read()
    head = t[:2500]
    m = DATE.search(head)
    d = pd.to_datetime(" ".join(m.groups()), errors="coerce") if m else pd.NaT
    re_blk = head[head.find("Re:"):][:900] if "Re:" in head else head[:1200]
    reg = bool(re.search(r"(?i)registration\s+statement|form\s+[sf]-1|offering\s+statement", re_blk))
    draft = bool(re.search(r"(?i)draft\s+registration|confidential(ly)?\s+submi", re_blk))
    nums = [int(x) for x in NUM.findall(t) if int(x) < 300]
    # comments run 1..k: take the longest run that starts at 1
    k = 0
    for x in nums:
        if x == k + 1: k = x
    return d, reg, draft, k, len(t)

L = pd.read_csv(TXT / "letters.csv")
L = L[(L.form == "UPLOAD") & (L.status != "missing") & L.key.isin(s.key)].drop_duplicates("acc")
P = [parse(TXT / "letters" / f"{c}_{a}.txt.gz") for c, a in zip(L.cik, L.acc)]
L[["ldate", "reg", "draft", "ncom", "nchar"]] = pd.DataFrame(P, index=L.index)
L = L.merge(s[["key", "start", "t_pub", "t_offer"]].rename(columns={"t_pub": "pubd", "t_offer": "offd"}), on="key", how="left")
for c in ["start", "pubd", "offd"]: L[c] = pd.to_datetime(L[c])
L["ldate"] = L.ldate.fillna(pd.to_datetime(L.date))
L = L[L.reg & (L.ldate >= L.start - pd.Timedelta(days=30)) & (L.ldate <= L.offd + pd.Timedelta(days=5))]
L["phase"] = np.where(L.draft | (L.ldate < L.pubd), "conf", "pub")
L["review"] = L.ncom > 0                                  # letters with comments, not acceleration or "no review" notes
say(f"UPLOAD letters on the IPO registration: {len(L)} for {L.key.nunique()} of {len(s)} IPOs; with comments {int(L.review.sum())}")
A = L[L.review].groupby(["key", "phase"]).agg(rounds=("acc", "size"), comments=("ncom", "sum"), first=("ncom", "first")).unstack("phase")
A.columns = [f"{a}_{b}" for a, b in A.columns]
A = A.reindex(s.key).fillna({c: 0 for c in A.columns}).reset_index().rename(columns={"index": "key"})
F = L[L.review].sort_values("ldate").groupby("key").ncom.first().rename("first_letter")
A = A.merge(F, on="key", how="left")
A["has_letters"] = A.key.isin(L.key).astype(int)
A.to_csv(ROOT / "data" / "universe" / "sec_letters.csv", index=False)
s = s.merge(A, on="key", how="left")
for c in ["comments_conf", "comments_pub", "rounds_conf", "rounds_pub"]:
    if c not in s: s[c] = 0
s["comments_all"] = s.comments_conf + s.comments_pub; s["rounds_all"] = s.rounds_conf + s.rounds_pub
for c in ["comments_all", "comments_conf", "comments_pub", "rounds_all", "rounds_pub", "first_letter"]: s["ln_" + c] = np.log1p(s[c])
s["second"] = (s.uw < 5).astype(int); s["elite"] = (s.uw >= 9).astype(int)
t = s[s.has_letters == 1].copy()
say(f"IPOs with letters found: {len(t)}  (second {int(t.second.sum())}, elite {int(t.elite.sum())})")

say("\n== L1  means by division (IPOs with letters)")
grp = np.select([t.second == 1, t.elite == 1], ["second (2-4)", "elite (9)"], "middle (5-8)")
say(t.groupby(grp)[["first_letter", "comments_all", "comments_conf", "comments_pub", "rounds_all", "rounds_pub", "pub"]].mean().round(1).to_string())

say("\n== L2  division effects (second, elite vs middle), controls + FE, two-way clusters")
for y in ["ln_first_letter", "ln_comments_all", "ln_comments_conf", "ln_comments_pub", "ln_rounds_all", "ln_rounds_pub"]:
    d = t.dropna(subset=[y]); m = fit(f"{y} ~ second + elite + {CTL}", d)
    say(f"  {y:18s} second {r(m, 'second')}  elite {r(m, 'elite')}  mean={d[y].mean():5.2f}  N={int(m.nobs)}")

say("\n== L3  public window and public amendments, with and without the public-phase comment load")
for y in ["lnpub", "lnamend"]:
    m0 = fit(f"{y} ~ second + elite + {CTL}", t)
    m1 = fit(f"{y} ~ second + elite + ln_comments_pub + ln_rounds_pub + {CTL}", t)
    m2 = fit(f"{y} ~ second + elite + ln_first_letter + {CTL}", t)
    say(f"  {y:8s} base       second {r(m0, 'second')}  N={int(m0.nobs)}")
    say(f"  {y:8s} + pub load second {r(m1, 'second')}  comments {r(m1, 'ln_comments_pub')}  rounds {r(m1, 'ln_rounds_pub')}")
    say(f"  {y:8s} + 1st ltr  second {r(m2, 'second')}  first letter {r(m2, 'ln_first_letter')}")
(ROOT / "output" / "letters_results.txt").write_text("\n".join(lines), encoding="utf-8")
