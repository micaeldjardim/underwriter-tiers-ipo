# Prestige comes in tiers: replication files

Code and derived data for:

> Jardim, M., Caldieraro, F., Parente, R. *Prestige comes in tiers: Underwriter rank, SEC review time and IPO pricing.* Working paper, 2026.

The paper rebuilds the SEC registration history of 1,039 U.S. IPOs priced from 2012 to 2022. It uses a threshold regression (Hansen, 2000) to locate where the 1–9 underwriter rank actually breaks.

## What is here

| Folder | Contents |
|---|---|
| `code/` | Python scripts `11`–`20`, run in numeric order (see below) |
| `data/universe/` | Derived data, one row per IPO, keyed by `key` |
| `data/market_index.csv` | Daily S&P 500, Russell 2000 (IWM) and Nasdaq Composite closes from Yahoo Finance |
| `output/` | The estimates reported in the paper |

| File | Source | Built by |
|---|---|---|
| `timeline.csv` | EDGAR filing histories (DRS, S-1/A, 424B) matched to the Field–Ritter IPO list | `11` |
| `prospectus.csv` | Prospectus cover: price range, offer price, shares, bookrunners, Ritter rank of the lead | `12` |
| `auditor.csv` | Auditor and lock-up from the prospectus | `12b` |
| `financials.csv` | XBRL company facts (assets, revenue, net income, R&D) | `13` |
| `prices.csv` | Yahoo Finance prices; Bloomberg columns removed (see below) | `14` |
| `analysis.csv` | Merged file after the sample filters, with every variable used in the paper | `15` |
| `exits.csv` | Exchange exits within one, two and three years, from EDGAR Forms 25, 25-NSE and 15 and 8-K items 1.03, 2.01, 3.01 and 5.01 | `18` |
| `workload.csv` | SEC registration workload in the 30 days before each public filing, EDGAR-wide and by Corporation Finance office | `19` |

## What is not here, and why

- **First-day returns from Bloomberg (1,008 IPOs) and Refinitiv Eikon (11 IPOs)** are licensed and cannot be redistributed. The columns `ir`, `ir_bb`, `size_bb` and `ir_ek` were removed. `ir_src` still records which source each IPO used. `ir_yf`, the Yahoo Finance return, is kept but covers only 572 of the 1,039 IPOs.
- **Jay Ritter's files.** Download `IPO-age.xlsx` (founding dates) and `Underwriter-Rank.xls` from <https://site.warrington.ufl.edu/ritter/ipo-data/> into `data/external/`. You only need them to rebuild from scratch.
- **Raw EDGAR caches** (about 2.7 GB). The scripts download what they need from EDGAR.

## Reproducing the paper

```
pip install -r requirements.txt
python code/17_universe_divisions.py   # Tables 2-5 and Fig. 1 (runs 16 first)
python code/20_identification.py       # Table 6
```

On the published files, every estimate for review time, amendments, price revisions, exits, SEC workload and the Nasdaq 2019 rule reproduces the paper exactly. That covers 46 of the 52 rows in `divisions_estimates.csv` and all 19 rows in `identification_estimates.csv`. The six first-day-return rows need the licensed returns. If `analysis.csv` has no `ir` column, `16` falls back to the Yahoo return. To match the paper, add your own `ir` (offer price to first close, in percent) from Bloomberg or CRSP.

To rebuild everything from EDGAR, set a contact string first, as the SEC requires:

```
export SEC_USER_AGENT="Your Name your.email@example.com"
python code/11_universe_timeline.py
python code/12_universe_prospectus.py
python code/12b_universe_auditor.py
python code/13_universe_financials.py
python code/14_universe_prices.py
python code/15_universe_build.py
python code/18_universe_exits.py
python code/19_sec_workload.py
```

Scripts `11`, `12`, `13`, `18` and `19` download several thousand EDGAR documents within the SEC's rate limit, so expect them to run for hours.

## Licence

Code: MIT. Derived data: CC BY 4.0. EDGAR data are public. Ritter's files belong to Jay Ritter, and the variables taken from them should be credited to him.
