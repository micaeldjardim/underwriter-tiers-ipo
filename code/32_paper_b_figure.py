"""Paper B, Figure 1: share of IPOs that grant warrants to the underwriter and mean direct compensation, by size of the
offering, for IPOs led by a warrant bank (leave-one-out, as in 31) and by other banks. Writes output/paper_b_fig1_contracts.pdf"""
import pathlib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
s = pd.read_csv(ROOT / "data" / "universe" / "paper_b.csv")
g = s.groupby("lead_bank").uw_warrants; n = g.transform("size") - 1
s["wbank"] = (((g.transform("sum") - s.uw_warrants) / n).where(n > 0) > 0.5).astype(float).where(n > 0)
edges = [0, 5, 10, 15, 25, 35, 50, 75, 100, 250, 1e9]; labs = ["<5", "5-10", "10-15", "15-25", "25-35", "35-50", "50-75", "75-100", "100-250", ">250"]
s["bin"] = pd.cut(s.proc_m, edges, labels=labs)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k, (lab, d, st) in enumerate([("All IPOs", s, "-"), ("Led by a warrant bank", s[s.wbank == 1], "--"), ("Led by another bank", s[s.wbank == 0], ":")]):
    t = d.groupby("bin", observed=False).agg(N=("key", "size"), w=("uw_warrants", "mean"), c=("direct", "mean"))
    t = t[t.N >= 5]
    x = [labs.index(i) for i in t.index]
    ax[0].plot(x, t.w, st, marker="o", color="0.15" if k == 0 else ("0.35" if k == 1 else "0.55"), label=lab, ms=4)
    ax[1].plot(x, t.c, st, marker="o", color="0.15" if k == 0 else ("0.35" if k == 1 else "0.55"), label=lab, ms=4)
for a, yl in zip(ax, ["Share granting underwriter warrants", "Direct compensation (% of proceeds)"]):
    a.set_xticks(range(len(labs))); a.set_xticklabels(labs, rotation=45, fontsize=8); a.set_xlabel("Proceeds ($ million)"); a.set_ylabel(yl)
    a.spines[["top", "right"]].set_visible(False)
ax[1].axhline(7, color="0.7", lw=0.8); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); out = ROOT / "output" / "paper_b_fig1_contracts.pdf"; fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=110)
print("saved", out)
