# RUNLOG

Chronological record of decisions, runs, failures and fixes. Times are Asia/Hong_Kong.

## 2026-11-06 — scope and feasibility

- Read the existing workspace. `feasibility_report.md` and `phase0/` are an unrelated Hong
  Kong store-locator study; left untouched. This project lives in `play-review-panel/`.
- System Python 3.14 has no data stack. Created a project-local Python 3.13 venv with `uv`.
  `uv`'s default cache and Python install directories are under `~/.local`, which the file
  sandbox blocks, so both were redirected into the project (`.uv-cache`, `.uv-python`).
- Installed pandas, numpy, scipy, statsmodels, matplotlib, google-play-scraper, yfinance,
  vaderSentiment, pyarrow, tabulate, pytest.

## 2026-11-06 — data feasibility

- Google Play review API works and is fast: ~2,000 reviews per 3–4 s.
- **Critical finding:** the review endpoint is newest-first with no date filter. Five thousand
  reviews cover only ~4 days for Duolingo, ~7 days for Spotify, ~1 day for YouTube, but
  ~3.4 years for Carvana, ~2.8 years for Alibaba and ~1.5 years for Shopee.
- Country parameters do **not** partition the review pool: Duolingo US/NZ/HK returned
  identical review ids and timestamps. Smaller storefronts are not a workaround.
- Wayback Machine CDX returned HTTP 429 again (same as the earlier unrelated study), so
  historical aggregate-rating snapshots are not available as a fallback.
- Consequence: a 32-company panel is buildable, but 12+ quarters is only available for the
  lower-volume apps. The design now measures coverage in tiers instead of assuming it.

## 2026-11-07 — panel design

- Fixed the panel at **32 companies / 32 apps**: Grab, Sea (Shopee), Uber, DoorDash, Spotify,
  Duolingo, Airbnb, Booking, eBay, Etsy, Netflix, Meta (Instagram), Alphabet (YouTube), Amazon,
  Shopify (Shop), Pinterest, Snap, Reddit, Roblox, Match (Tinder), Bumble, Zillow, Carvana,
  Coinbase, Block (Cash App), Alibaba (Taobao), PDD (Temu), JD, Tencent (WeChat), Coupang,
  MercadoLibre, Wise.
- Mapping decisions that need a note:
  - PDD uses **Temu**, not Pinduoduo: the China package is not in the US storefront and its
    review stream is stale (last usable reviews 2023).
  - Meta uses **Instagram** only, to avoid double-counting the Facebook app.
  - Sea uses the **Shopee SG** package; the review pool is global, so the package choice is
    about app identity, not geography.
  - Shopify uses the consumer **Shop** app, not the merchant admin.
  - Block's ticker changed SQ → XYZ; Yahoo carries the full price history under XYZ.
- Benchmarks are regional to keep currency consistent: SPY for USD names, MCHI for China ADRs,
  2800.HK for Tencent, EWY for Coupang, ^FTSE for Wise, URTH as a fallback.
- Ingestion policy: `--since 2023-11-07` (12 quarters to 2026-11-07), cap 50,000 reviews/app,
  batch 2,000, 0.7–1.2 s jitter between batches. On reaching the date, the app stops early.
- Privacy: author names, avatars and user ids are dropped at the parsing boundary; only
  review-level public fields are stored.
- Sentiment is English-only VADER. Non-English reviews keep NULL sentiment and are excluded
  from sentiment means, not imputed. Star ratings remain the primary, language-independent
  signal.

## 2026-11-07 — implementation

- Built `config.py` (validated CSV mapping), `db.py` (SQLite + resumable state), `ingest.py`
  (checkpointed backfill), `text.py`, `prices.py`, `features.py`, `backtest.py`, `power.py`,
  `report.py`, `cli.py`.
- Tests: 17 passing. Covered panel uniqueness/validation, Friday weekly bins, trailing-only
  signal construction, forward-return offsets, FE coefficient recovery on synthetic data,
  long-short behaviour, event-study execution and the power formulas.
- Bugs found and fixed during testing:
  - `cs_z` used pandas' default sample SD (ddof=1), so cross-sectional z-scores did not have
    unit variance; switched to ddof=0.
  - `fe_regression` selected `company_id` twice when it was also the cluster variable, then
    renamed it away; fixed by copying to `_cluster` instead of renaming.
  - `event_study` called the FE estimator on a constant, which has no within variation; the
    CAR t-stat now comes from a company-clustered mean test.
  - Forward-return lag offset was applied with the wrong sign; lagged returns were looking
    backwards. Replaced with an explicit `offset` argument.
  - Coverage table had a no-op quarter aggregation; replaced with a clean count of quarters
    containing at least five reviews.
- Smoke run (Carvana + Duolingo, 4,000 reviews each) passed ingest → enrich → prices → build.
  Carvana's 4,000 reviews covered 13 quarters; Duolingo's covered one. This previewed the
  coverage problem that the full run later quantifies.

## 2026-11-07 — full backfill

Command:

```bash
python -m prpanel ingest --since 2023-11-07 --max-reviews 50000 --batch 2000 --restart
```

Result: **1,237,604 reviews, 32/32 apps, 0 errors**, 51 minutes wall-clock, 640 MB SQLite.
No rate-limit or parse failures. Per-app coverage after the first pass:

| App | Reviews | First review | Coverage |
|---|---:|---|---|
| Coinbase | 42,089 | 2023-11-07 | full 3 years |
| Airbnb | 41,463 | 2023-11-07 | full 3 years |
| Carvana | 4,233 | 2023-11-07 | full 3 years |
| Tencent / WeChat | 16,508 | 2023-11-07 | full 3 years |
| Sea / Shopee | 11,188 | 2023-11-07 | full 3 years |
| MercadoLibre | 5,610 | 2023-11-07 | full 3 years |
| Zillow | 10,405 | 2023-11-07 | full 3 years |
| JD | 250 | 2023-11-07 | full span, very sparse |
| Alibaba / Taobao | 5,506 | 2023-11-07 | full 3 years |
| Coupang | 352 | 2023-11-08 | full span, very sparse |
| Etsy | 50,000 | 2023-11-25 | full 3 years |
| Bumble | 50,000 | 2023-11-26 | full 3 years |
| Booking | 50,000 | 2024-05-02 | 2.5 years |
| Wise | 50,000 | 2024-09-19 | 2.1 years |
| Grab | 50,000 | 2024-11-19 | 2.0 years |
| Match / Tinder | 50,000 | 2025-02-16 | 1.7 years |
| Shopify | 50,000 | 2025-03-03 | 1.7 years |
| eBay | 50,000 | 2025-03-17 | 1.6 years |
| Amazon | 50,000 | 2025-05-02 | 1.5 years |
| Reddit | 50,000 | 2025-07-02 | 1.3 years |
| Block / Cash App | 50,000 | 2025-08-04 | 1.3 years |
| DoorDash | 50,000 | 2025-08-08 | 1.3 years |
| Pinterest | 50,000 | 2026-02-08 | 0.7 years |
| Netflix | 50,000 | 2026-03-05 | 0.7 years |
| Snap | 50,000 | 2026-05-22 | 0.5 years |
| Uber | 50,000 | 2026-07-14 | 0.3 years |
| PDD / Temu | 50,000 | 2026-08-12 | 0.2 years |
| Spotify | 50,000 | 2026-08-26 | 0.2 years |
| Duolingo | 50,000 | 2026-09-20 | 0.1 years |
| Meta / Instagram | 50,000 | 2026-09-23 | 0.1 years |
| Roblox | 50,000 | 2026-10-05 | 0.1 years |
| Alphabet / YouTube | 50,000 | 2026-10-12 | 0.07 years |

The cap binds for exactly the apps where reviews arrive faster than the crawl: a three-year
history for YouTube would be millions of rows. Six names were resumed from their saved
continuation tokens with a higher cap (Grab/Booking/Wise → 80k, eBay/Match → 90k,
Amazon → 110k) to push them over 12 quarters without re-downloading the first 50k.

## 2026-11-07 — checkpoint expiry during top-up

- Resuming the six capped apps from their saved continuation tokens returned zero items: the
  tokens had expired (they are short-lived, roughly an hour). The first top-up therefore
  wrote nothing and marked the apps `done_end`.
- Fix: `ingest_app` now detects "resumed from a checkpoint but got an empty first page" and
  restarts once from newest, continuing to the higher cap. Duplicate review ids are ignored,
  so this only costs re-reading.
- Because `done_end` is terminal in the scheduler, the second attempt still skipped. A forced
  `--restart` was used for Grab/Booking/Wise (80k) and eBay (90k). This is documented in the
  README: use `--restart` when raising a cap after a failed/expired resume.

## Known issues / maintenance

- The Play endpoint is unofficial. A field or endpoint change breaks ingestion; the parser
  keeps only known keys and the DB is idempotent, so a re-run is safe.
- `JD`, `NetEase`, `Baidu`-type China packages have very few international reviews; their
  history is long but weekly counts can fall below the 5-review signal floor. They contribute
  to coverage, not necessarily to every week's cross-section.
- App updates, prompted-review campaigns and rating-prompt timing create volume shocks that
  are not user-demand shocks. The volume signal is therefore reported but not included in the
  composite.
- Report figures are regenerated by `prpanel report`; `reports/figures/` is gitignored.


## 2026-11-07 — final build, analysis and report

- Final panel: **2,936 company-weeks**, 1,237,605 reviews, 32 companies. Tiers: **10 A**
  (12+ quarters), **13 B** (4–11), **9 C** (<4). The top-up attempt did not materially change
  coverage because Play continuation tokens expire; the 10/32 result is the honest ceiling of
  this crawl budget.
- Sentiment: 1,229,605 reviews scored in 78 s; 86.0% of all reviews carry a VADER score.
- Regressions: 18 signal × horizon specifications; **no coefficient survives BH at q < 0.10**
  (minimum q = 0.72). The strongest raw coefficient is `rating_z` on next-week abnormal return
  (+0.11% per 1 SD, p = 0.22).
- Long-short: composite gross Sharpe 0.03 (p_perm = 0.97); best raw signal is `rating_mom_z`
  at 0.69 gross / 0.30 net (p_perm = 0.33). That result is **halved to 0.35** by removing one
  week (Bumble, 2026-10-09, +134% single-name move); annualised gross falls from 24.5% to
  10.2%. No signal is statistically distinguishable from zero.
- Event study: negative rating shocks (n=199) earn -1.27% abnormal over 4 weeks vs -0.02% for
  controls; t = -0.93, p = 0.35. Not significant.
- Power: realised N=20 companies, median T=87 weeks, residual σ=6.2%/week → MDE ≈21.8%
  annualised per 1 SD, long-short Sharpe MDE ≈2.17. A 3-year version of the panel would still
  have a Sharpe MDE of ~1.62.
- Report: `reports/report.md`; figures in `reports/figures/`.
- Tests: 17 passing after the final code changes.
