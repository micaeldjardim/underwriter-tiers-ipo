"""Validation sample for the underwriter-warrant indicator of paper B. Draws 100 IPOs from the paper B sample with a
fixed seed: 50 flagged and 50 not flagged, the latter over-sampled among offerings below $50 million (30 of 50), where
a missed warrant would matter. For each it prints the passages a reader needs to code the IPO by hand: every place in
the final prospectus where 'warrant' occurs within 250 characters of 'underwriter', 'representative' or 'placement
agent' (up to two passages; 'warranty' and 'warranties' are skipped). The hand codes are stored in data/universe/validation_handcodes.csv.
Usage: python 33_validation_sample.py [start] [end]"""
import gzip, os, pathlib, re, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
TXT = pathlib.Path(os.environ.get("TEXT_DIR", ROOT / "data" / "text"))
s = pd.read_csv(ROOT / "data" / "universe" / "paper_b.csv")
D = pd.read_csv(TXT / "docs.csv"); D = D[D.role.eq("final_424")][["key", "cik", "acc"]].rename(columns={"cik": "dcik", "acc": "dacc"})
s = s.merge(D, on="key")
a = s[s.uw_warrants == 1].sample(50, random_state=11)
b1 = s[(s.uw_warrants == 0) & (s.proc_m < 50)].sample(30, random_state=11)
b2 = s[(s.uw_warrants == 0) & (s.proc_m >= 50)].sample(20, random_state=11)
v = pd.concat([a, b1, b2]).sample(frac=1, random_state=3).reset_index(drop=True)      # shuffled, so the flag is not visible by position
v[["key", "ritter_name", "uw_warrants", "proc_m"]].to_csv(ROOT / "data" / "universe" / "validation_sample.csv", index=False)
lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, len(v))
W = re.compile(r"warrants?(?![a-z])")
for i, x in v.iloc[lo:hi].iterrows():
    low = " ".join(gzip.open(TXT / "docs" / f"{x.dcik}_{x.dacc}.txt.gz", "rt", encoding="utf-8").read().replace("\u2019", "'").replace("\ufffd", "'").replace("\xa0", " ").split()).lower()
    out, last = [], -10**9
    for m in W.finditer(low):
        if m.start() - last < 600: continue
        seg = low[max(0, m.start() - 250):m.start() + 250]
        if re.search(r"underwriter|representative|placement agent", seg):
            out.append(low[max(0, m.start() - 170):m.start() + 230]); last = m.start()
        if len(out) == 2: break
    print(f"#{i} {x.ritter_name[:28]} | ${x.proc_m:.0f}m")
    for o in out: print("   >", o)
    if not out: print("   > NONE (no 'warrant' near underwriter/representative/placement agent;", len(W.findall(low)), "mentions of 'warrant(s)' in all)")
