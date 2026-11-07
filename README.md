# Google Play review panel

A reproducible research pipeline that maps **listed consumer-internet companies → their
consumer apps → Play developers → parent companies → tickers**, backfills timestamped Google
Play reviews, builds a company-week panel, and backtests review-based signals against
market-adjusted equity returns.

The point is to test the idea properly rather than wait weeks for data to accumulate — but
also to be honest about where the public data runs out.

```text
32 companies · 32 apps · ~2,000 reviews per 4 seconds of crawling
        │
        ├── reviews.sqlite        review-level, deduplicated, resumable
        ├── prices/*.csv          weekly adjusted closes + regional benchmarks
        ├── panel_weekly.parquet  company-week signals + forward returns
        ├── analysis/*.csv        regressions, long-short, event study, power
        └── reports/report.md     rendered findings and figures
```

## The honest headline

The brief assumes **30 companies × 12+ quarters**. That is achievable for part of the panel,
not all of it. Google Play's review endpoint paginates **newest → oldest only**. To reach a
date you must read every review since that date. A mega-app such as YouTube produces ~5,000
reviews per day, so three years of history would require millions of rows. Smaller listed
apps (Carvana, Shopee, Zillow, Alibaba, Coupang) yield 3–15 years at a modest crawl budget.

The pipeline therefore measures coverage instead of assuming it, assigns each company to a
tier, and restricts time-series claims to the deep-history tier. `reports/report.md` reports
the actual N, T, minimum detectable effect, and the share of the original 30×12 design that
survives.

**Snapshot at 2026-11-07:** 1,237,605 reviews across 32 apps; only **10 of 32 companies clear
12+ quarters**. No review signal survives Benjamini–Hochberg at q < 0.10, and the best raw
long-short result (rating momentum, gross Sharpe 0.69) falls to 0.35 when a single extreme
week is removed. The honest conclusion is that public Play reviews, at this panel size, do not
produce a reliable next-week equity signal — and the binding constraint is data depth, not the
statistical machinery.

## Quickstart

Python 3.11–3.13. The repo is tested on 3.13.

```bash
cd play-review-panel
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e .

python -m prpanel ingest --since 2023-11-07 --max-reviews 50000
python -m prpanel enrich
python -m prpanel prices
python -m prpanel build
python -m prpanel analyze --n-perm 300
python -m prpanel report
```

Or the whole chain:

```bash
python -m prpanel all --since 2023-11-07 --max-reviews 50000
```

A small smoke run (two apps, 4k reviews each):

```bash
python -m prpanel ingest --apps carvana,duolingo --max-reviews 4000 --restart
python -m prpanel enrich && python -m prpanel prices && python -m prpanel build
```

## Commands

| Command | What it does |
|---|---|
| `prpanel ingest` | Backfill (or `--mode refresh`) reviews. Resumable via a checkpointed continuation token. |
| `prpanel enrich` | English-only VADER sentiment; idempotent and resumable. |
| `prpanel prices` | Weekly adjusted closes and market-adjusted returns; cached to CSV. |
| `prpanel build` | Company-week panel, trailing signals, forward returns, coverage table. |
| `prpanel analyze` | Two-way FE regressions, long-short backtests, event study, power analysis. |
| `prpanel report` | Render `reports/report.md` and figures. |
| `prpanel status` | Row counts and per-app ingestion state. |

Useful flags: `--apps grab,sea` to restrict ingestion, `--since YYYY-MM-DD`, `--max-reviews`,
`--restart` to ignore a checkpoint, `--n-perm` for randomization draws, `--mc` for a Monte
Carlo power check.

Continuation tokens are short-lived (roughly an hour). If a resumed fetch returns nothing and
you are raising `--max-reviews`, use `--restart`: duplicate review ids are ignored, so the only
cost is re-reading the pages, not duplicate storage.

## Panel construction

`config/panel.csv` is the single source of truth. Every row states the app, package id,
developer shown on Play, parent company, ticker, Yahoo symbol, regional benchmark, and a note
where the ownership chain is not obvious. Examples:

| App | Play developer | Parent | Ticker |
|---|---|---|---|
| Grab | Grab Holdings | Grab Holdings Limited | GRAB |
| Shopee | Shopee | Sea Limited | SE |
| Tinder | Tinder LLC | Match Group, Inc. | MTCH |
| Temu | Temu | PDD Holdings Inc. | PDD |
| WeChat | WeChat | Tencent Holdings Limited | 0700.HK |
| Cash App | Block, Inc. | Block, Inc. | XYZ |

`load_panel()` validates uniqueness, package-id syntax, role values, and benchmark presence
before any scraping begins. The mapping is deliberately *not* inferred from Play metadata at
runtime: Play's developer string is not a legal-entity identifier.

## Data model

`reviews` stores `review_id` (primary key), `package_id`, `company_id`, 1–5 `score`, UTC
`at_utc`, text, thumbs-up count, app version, developer reply, requested locale, and an
ingestion timestamp. **Author names, avatars and other identifiers are not stored.** Re-runs
are idempotent. `ingest_state` checkpoints the continuation token, oldest/newest timestamps
and status after every committed batch.

`panel_weekly.parquet` is one row per company-week with: review count, mean/SD rating, 1/2/5-star
shares, reply rate, English share, VADER sentiment, six trailing signals, forward 1/2/4-week
returns, market-adjusted returns, and a one-week-lagged variant.

## Methodology (short version)

- **Timing.** Reviews are binned into weeks ending Friday 23:59 UTC. Signals use week *t* and
  trailing history only. Forward returns run Friday close *t* → Friday close *t+1*; a lagged
  variant (trade one week later) is the conservative check.
- **Signals.** `rating_z` (level vs own 26-week history), `rating_mom_z` (4-week momentum),
  `one_star_z` (complaint share), `sent_z` (English VADER), `vol_z` (review-volume surprise),
  and a pre-specified `composite` averaging the first four with the complaint sign flipped.
- **Inference.** Two-way (company + week) fixed effects, company-clustered standard errors,
  cross-sectional signal standardisation, 1st/99th percentile outcome winsorisation, and
  Benjamini–Hochberg q-values across every signal/horizon pair.
- **Portfolio.** Weekly quintile long–short, equal weight, 10 bp per side on measured turnover,
  Newey–West t-stat, and a randomization-inference p-value that permutes signals across
  companies within each week.
- **Event study.** Rating shocks at ≤ −1.5 within-company SD; 1/2/4-week abnormal returns
  versus control weeks.
- **Power.** Analytic MDE for an AR(1) two-way FE panel plus a Monte Carlo check using the
  same estimator.

## Outputs

- `reports/report.md` — the rendered study, with coverage, descriptives, regressions,
  long-short, event study, power, limitations.
- `reports/figures/` — coverage, ratings, long-short equity curve, event study, power.
- `data/analysis/` — machine-readable results.
- `data/panel_weekly.csv` — flat CSV of the panel for non-Python inspection.
- `data/reviews.sqlite` — review-level store (not redistributed).
- `docs/DATA_DICTIONARY.md` — column-level definitions.
- `logs/prpanel.log`, `logs/ingest_full.log` — run logs.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite covers panel validation, weekly binning, trailing-only signal construction,
forward-return offsets, FE coefficient recovery on synthetic data, long-short behaviour,
event-study execution, and the power formulas.

## Compliance and limits

- Google Play's terms restrict automated collection. This is an **internal research
  prototype**; do not redistribute review text or author data. For production, license review
  data from a vendor (Sensor Tower, AppFollow, data.ai) or obtain written permission.
- Android only; reviewers are self-selected and often prompted. Ratings are not a census of
  user satisfaction.
- The review timestamp does not reveal edits/deletions; this is measurement error.
- Parent/subsidiary and ticker mappings change. Re-validate `config/panel.csv` periodically.
- Backtest returns are close-to-close, market-adjusted with a regional ETF, and exclude
  borrow costs, slippage beyond the stated cost, and survivorship effects.
- Nothing here is investment advice.

## Layout

```text
play-review-panel/
├── config/panel.csv          # audited app → developer → parent → ticker map
├── docs/DATA_DICTIONARY.md   # column definitions
├── src/prpanel/
│   ├── config.py             # paths + panel validation
│   ├── db.py                 # SQLite schema, idempotent writes, checkpoints
│   ├── ingest.py             # resumable newest→oldest backfill
│   ├── text.py               # English VADER enrichment
│   ├── prices.py             # weekly prices + regional benchmarks
│   ├── features.py           # weekly aggregation, signals, forward returns
│   ├── backtest.py           # FE regressions, long-short, event study
│   ├── power.py              # analytic + Monte Carlo power
│   ├── report.py             # markdown + figures
│   └── cli.py                # command line
├── tests/
├── data/                     # generated, gitignored
├── reports/                  # generated report + figures
├── RUNLOG.md                 # decisions, runs, breakages, fixes
└── pyproject.toml
```
