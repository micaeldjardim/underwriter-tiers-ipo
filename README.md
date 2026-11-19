# Prestige comes in tiers: replication files

Code and derived data for:

> Jardim, M. *Prestige comes in tiers: Underwriter rank, SEC review and IPO pricing.* Working paper, 2026.

The paper rebuilds the SEC registration history of 1,039 U.S. IPOs priced from 2012 to 2022. It uses a threshold regression (Hansen, 2000) to locate where the 1–9 underwriter rank actually breaks. It then reads the SEC comment letters and the prospectuses of these IPOs to see where in the review the tiers differ.

## What is here

| Folder | Contents |
|---|---|
| `code/` | Python scripts `11`–`25b` for the first paper and `27`–`33` for the second (see below) |
| `data/universe/` | Derived data, one row per IPO, keyed by `key` |
| `data/text/` | Lists of every EDGAR document and SEC letter used, with accession numbers (`docs.csv`, `letters.csv`) |
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
| `sec_letters.csv` | SEC staff comment letters on each IPO registration: rounds and numbered comments, confidential and public phase, comments in the first letter | `22` |
| `prospectus_tone.csv` | Length, risk-factor length and Loughran–McDonald tone (shares of words) of the first public registration statement and the pricing prospectus | `24` |

In `data/text/`, `docs.csv` lists the first public registration statement, the last pre-pricing amendment and the pricing prospectus of each IPO; `letters.csv` lists every SEC staff letter (UPLOAD) and issuer response (CORRESP) in the window that `21` searches. Both are produced by `21`; `n_chars` is the size of the stored text file.

## What is not here, and why

- **First-day returns from Bloomberg (1,008 IPOs) and Refinitiv Eikon (11 IPOs)** are licensed and cannot be redistributed. The columns `ir`, `ir_bb`, `size_bb` and `ir_ek` were removed. `ir_src` still records which source each IPO used. `ir_yf`, the Yahoo Finance return, is kept but covers only 572 of the 1,039 IPOs.
- **Jay Ritter's files.** Download `IPO-age.xlsx` (founding dates) and `Underwriter-Rank.xls` from <https://site.warrington.ufl.edu/ritter/ipo-data/> into `data/external/`. You only need them to rebuild from scratch.
- **Raw EDGAR caches** (about 2.7 GB) and **the text of the prospectuses and SEC letters** (about 1 GB). They are public on EDGAR; `21` downloads them again from the accession numbers.
- **The Loughran–McDonald master dictionary.** Download `Loughran-McDonald_MasterDictionary_1993-2025.csv` from <https://sraf.nd.edu/loughranmcdonald-master-dictionary/> into `data/text/lm/` and save it as `LM_MasterDictionary_1993-2025.csv`.

## Reproducing the paper

```
pip install -r requirements.txt
python code/17_universe_divisions.py   # Tables 3-5, 10 and 12, Fig. 1 (runs 16 first)
python code/20_identification.py       # Table 11
```

The comment-letter and prospectus tables (6 to 9), the summary statistics (Table 2) and the year-clustered and bootstrap columns need the texts. Download them, then run the analyses:

```
export SEC_USER_AGENT="Your Name your.email@example.com"
python code/21_text_download.py   # about 6,000 documents and 24,000 letters; hours
python code/22_sec_letters.py     # Tables 6 and 8
python code/23_letter_topics.py   # Table 7
python code/24_prospectus_tone.py # Table 9
python code/25_descriptives_inference.py # Table 2; year-clustered t and wild-bootstrap p in Tables 5, 6 and 8
```

`21` reads the EDGAR submission files that `11` saves in `data/edgar/submissions/`, so run `11` first.

`25` needs the licensed first-day returns for its first-day-return rows, like `16`. On the published files its wild-bootstrap p-values can differ from the paper in the third decimal; coefficients and t-statistics match.

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

## Correction of 19 November 2026: gross proceeds

Proceeds were computed as offer price times the number of shares read on the prospectus cover. That fails when the cover states the securities in an unusual way (units, ADSs, several classes of shares). `30b_cover_total.py` now reads gross proceeds three ways from the cover (price times securities, the total on the price line, and price times total discount over discount per security) and keeps the value on which two readings agree within 2%. `15_universe_build.py` replaces the old value where the confirmed one differs: 18 of the 1,039 IPOs in `analysis.csv`, 13 of them among the 828 with every control. Proceeds enter the first paper as a control (`lnproc`). The sample is unchanged and the estimates move slightly; the second-division effect on the public window goes from 0.758 to 0.779. Every file in `output/` was regenerated, and `25b_extra_estimates.py` saves the estimates quoted in the text that the other scripts did not write to disk (`output/extra_estimates.txt`).

## Second paper: underwriter compensation

The repository also holds the files of a second paper that uses the same collection of filings and none of the first paper's outcomes:

> Jardim, M. *The ten percent offering: Endorsement, bank segments, and what small firms pay to go public.* Working paper, 2026.

It reads from each final prospectus what the lead bank was paid (cash spread, underwriter warrants, non-accountable expense allowance) and relates it to the size of the offering, the identity of the lead bank and venture backing, in 1,597 IPOs (foreign issuers and offerings priced below $5 are kept).

| File | Contents | Built by |
|---|---|---|
| `data/universe/issuer_prospectus.csv` | Spread, underwriter-warrant indicator and other fields read from the final prospectus, all 1,999 matched IPOs | `27` |
| `data/universe/warrant_terms.csv` | Warrant coverage, exercise price, term, expense allowance and Black-Scholes value | `27b` |
| `data/universe/cover_fill.csv`, `cover_total.csv` | Offer price, securities sold and gross proceeds read again from the cover | `30a`, `30b` |
| `data/universe/paper_b.csv` | The sample of the second paper with every variable | `30` |
| `data/universe/validation_sample.csv`, `validation_handcodes.csv` | 100 prospectuses drawn at random and coded by hand to check the warrant indicator (97 agree) | `33` |
| `output/paper_b_results.txt`, `paper_b_sample_log.csv`, `paper_b_fig1_contracts.pdf` | Every estimate, the sample steps and Figure 1 | `31`, `32` |

To reproduce the tables from the published files:

```
python code/31_paper_b_analysis.py   # all tables; about 15 minutes (bootstrap of the size cutoff)
python code/32_paper_b_figure.py     # Figure 1
```

`27`, `27b`, `30a`, `30b` and `33` read the text of the prospectuses, which `21` downloads into `data/text/`. Run them in that order, then `30`, before `31`.

## Licence

Code: MIT. Derived data: CC BY 4.0. EDGAR data are public. Ritter's files belong to Jay Ritter, and the variables taken from them should be credited to him.
