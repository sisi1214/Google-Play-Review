# Google Play review panel: app sentiment as a consumer-internet signal

**Prepared:** 2026-11-07 · **Data through:** 2026-11-06 · **Scope:** 32 listed companies, 1,237,605 timestamped reviews

> Research prototype. Public Google Play reviews are collected for aggregate, non-redistributed analysis. Review text is not republished. See the compliance section before using this commercially.

## Executive summary

- The panel maps **32 apps → developers → parents → tickers** and contains **1,237,605 reviews** from 2023-11-10 to 2026-11-06.
- **Coverage is the binding constraint, not statistical power.** 10 companies clear 12+ quarters; 13 have 4–11 quarters; 9 have less than four. Mega-apps (YouTube, Spotify, Duolingo, Uber) generate reviews faster than the public API can be paginated backwards.
- The pre-specified composite signal's long–short portfolio earned **0.8% annualised gross** (Sharpe 0.03, net Sharpe -0.46), with a randomization-inference p-value of 0.970.
- **No signal/horizon regression survives Benjamini–Hochberg at q < 0.10.** The honest reading is that review sentiment does not reliably predict next-week or next-month abnormal returns in this sample.
- With the actual panel (N=20, median T=87 weeks, residual σ=6.2%/week), the MDE is **21.8% annualised per 1 SD** of signal. The long–short Sharpe MDE is about 2.17.

## 1. What was built

A reproducible pipeline with six stages:

| Stage | Command | Output |
|---|---|---|
| Backfill reviews | `python -m prpanel ingest` | SQLite review store, resumable by continuation token |
| Enrich text | `python -m prpanel enrich` | English VADER sentiment per review |
| Prices | `python -m prpanel prices` | Weekly adjusted closes and market-adjusted returns |
| Features | `python -m prpanel build` | Company-week panel with signals and forward returns |
| Analysis | `python -m prpanel analyze` | FE regressions, long-short, event study, power |
| Report | `python -m prpanel report` | This document and figures |

The app → developer → parent → ticker map is `config/panel.csv`; every row is explicit and validated at load time. Author names and avatars are never stored.

## 2. Coverage: the 30×12 premise is only partly achievable

Google Play's review endpoint paginates **newest to oldest only**. Reaching a date requires reading every review since that date. For a mega-app that is millions of reviews, so 12 quarters of history is not obtainable at any reasonable crawl budget. The panel is therefore unbalanced, and the report separates deep-history from short-history names.

![Coverage](figures/coverage.png)

| company                       | ticker   |   n_reviews | first_week   | last_week   |   quarters_ge5 | tier              |
|:------------------------------|:---------|------------:|:-------------|:------------|---------------:|:------------------|
| Etsy, Inc.                    | ETSY     |       50000 | 2023-12-01   | 2026-11-06  |             13 | A (12+ quarters)  |
| Bumble Inc.                   | BMBL     |       50000 | 2023-12-01   | 2026-11-06  |             13 | A (12+ quarters)  |
| Coinbase Global Inc.          | COIN     |       42089 | 2023-11-10   | 2026-11-06  |             13 | A (12+ quarters)  |
| Airbnb Inc.                   | ABNB     |       41463 | 2023-11-10   | 2026-11-06  |             13 | A (12+ quarters)  |
| Tencent Holdings Limited      | 0700.HK  |       16508 | 2023-11-10   | 2026-11-06  |             13 | A (12+ quarters)  |
| Sea Limited                   | SE       |       11188 | 2023-11-10   | 2026-11-06  |             13 | A (12+ quarters)  |
| Zillow Group Inc.             | ZG       |       10405 | 2023-11-10   | 2026-11-06  |             13 | A (12+ quarters)  |
| MercadoLibre, Inc.            | MELI     |        5610 | 2023-11-10   | 2026-11-06  |             13 | A (12+ quarters)  |
| Alibaba Group Holding Limited | BABA     |        5506 | 2023-11-10   | 2026-11-06  |             13 | A (12+ quarters)  |
| Carvana Co.                   | CVNA     |        4233 | 2023-11-10   | 2026-11-06  |             13 | A (12+ quarters)  |
| Booking Holdings Inc.         | BKNG     |       50000 | 2024-05-03   | 2026-11-06  |             11 | B (4-11 quarters) |
| Wise plc                      | WISE.L   |       50000 | 2024-09-20   | 2026-11-06  |             10 | B (4-11 quarters) |
| Grab Holdings Limited         | GRAB     |       50001 | 2024-11-22   | 2026-11-06  |              9 | B (4-11 quarters) |
| Coupang, Inc.                 | CPNG     |         352 | 2023-11-10   | 2026-11-06  |              9 | B (4-11 quarters) |
| eBay Inc.                     | EBAY     |       50000 | 2025-03-21   | 2026-11-06  |              8 | B (4-11 quarters) |
| Shopify Inc.                  | SHOP     |       50000 | 2025-03-07   | 2026-11-06  |              8 | B (4-11 quarters) |
| Match Group, Inc.             | MTCH     |       50000 | 2025-02-21   | 2026-11-06  |              8 | B (4-11 quarters) |
| Amazon.com, Inc.              | AMZN     |       50000 | 2025-05-02   | 2026-11-06  |              7 | B (4-11 quarters) |
| DoorDash Inc.                 | DASH     |       50000 | 2025-08-08   | 2026-11-06  |              6 | B (4-11 quarters) |
| Reddit, Inc.                  | RDDT     |       50000 | 2025-07-04   | 2026-11-06  |              6 | B (4-11 quarters) |
| Block, Inc.                   | XYZ      |       50000 | 2025-08-08   | 2026-11-06  |              6 | B (4-11 quarters) |
| Netflix, Inc.                 | NFLX     |       50000 | 2026-03-06   | 2026-11-06  |              4 | B (4-11 quarters) |
| Pinterest, Inc.               | PINS     |       50000 | 2026-02-13   | 2026-11-06  |              4 | B (4-11 quarters) |
| Snap Inc.                     | SNAP     |       50000 | 2026-05-22   | 2026-11-06  |              3 | C (<4 quarters)   |
| JD.com, Inc.                  | JD       |         250 | 2023-11-10   | 2026-11-06  |              3 | C (<4 quarters)   |
| Uber Technologies Inc.        | UBER     |       50000 | 2026-07-17   | 2026-11-06  |              2 | C (<4 quarters)   |
| Spotify Technology S.A.       | SPOT     |       50000 | 2026-08-28   | 2026-11-06  |              2 | C (<4 quarters)   |
| Duolingo Inc.                 | DUOL     |       50000 | 2026-09-25   | 2026-11-06  |              2 | C (<4 quarters)   |
| Meta Platforms Inc.           | META     |       50000 | 2026-09-25   | 2026-11-06  |              2 | C (<4 quarters)   |
| PDD Holdings Inc.             | PDD      |       50000 | 2026-08-14   | 2026-11-06  |              2 | C (<4 quarters)   |
| Alphabet Inc.                 | GOOGL    |       50000 | 2026-10-16   | 2026-11-06  |              1 | C (<4 quarters)   |
| Roblox Corporation            | RBLX     |       50000 | 2026-10-09   | 2026-11-06  |              1 | C (<4 quarters)   |

**Tier A** (12+ quarters) carries the time-series tests. **Tier B** contributes cross-sectional observations. **Tier C** is displayed for completeness and excluded from any claim about 12-quarter power.

### App → developer → parent → ticker

This is the auditable mapping used throughout. The Play developer string is not a legal-entity identifier, so the parent and ticker columns are curated and validated rather than inferred.

| company                       | ticker   | app_name                       | developer                             | parent                        | package_id                        |
|:------------------------------|:---------|:-------------------------------|:--------------------------------------|:------------------------------|:----------------------------------|
| Grab Holdings Limited         | GRAB     | Grab - Taxi & Food Delivery    | Grab Holdings                         | Grab Holdings Limited         | com.grabtaxi.passenger            |
| Sea Limited                   | SE       | Shopee 11.11 Big Sale          | Shopee                                | Sea Limited                   | com.shopee.sg                     |
| Uber Technologies Inc.        | UBER     | Uber: Rides, eats, hotels      | Uber Technologies, Inc.               | Uber Technologies Inc.        | com.ubercab                       |
| DoorDash Inc.                 | DASH     | DoorDash: Food, Grocery, More  | DoorDash                              | DoorDash Inc.                 | com.dd.doordash                   |
| Spotify Technology S.A.       | SPOT     | Spotify: Music and Podcasts    | Spotify AB                            | Spotify Technology S.A.       | com.spotify.music                 |
| Duolingo Inc.                 | DUOL     | Duolingo: Learn Languages      | Duolingo                              | Duolingo Inc.                 | com.duolingo                      |
| Airbnb Inc.                   | ABNB     | Airbnb                         | Airbnb                                | Airbnb Inc.                   | com.airbnb.android                |
| Booking Holdings Inc.         | BKNG     | Booking.com: Hotels & Travel   | Booking.com Hotels & Vacation Rentals | Booking Holdings Inc.         | com.booking                       |
| eBay Inc.                     | EBAY     | eBay online shopping & selling | eBay Mobile                           | eBay Inc.                     | com.ebay.mobile                   |
| Etsy, Inc.                    | ETSY     | Etsy: Shop from Real People    | Etsy, Inc                             | Etsy Inc.                     | com.etsy.android                  |
| Netflix, Inc.                 | NFLX     | Netflix                        | Netflix, Inc.                         | Netflix, Inc.                 | com.netflix.mediaclient           |
| Meta Platforms Inc.           | META     | Instagram                      | Instagram                             | Meta Platforms Inc.           | com.instagram.android             |
| Alphabet Inc.                 | GOOGL    | YouTube                        | Google LLC                            | Alphabet Inc.                 | com.google.android.youtube        |
| Amazon.com, Inc.              | AMZN     | Amazon Shopping                | Amazon Mobile LLC                     | Amazon.com, Inc.              | com.amazon.mShop.android.shopping |
| Shopify Inc.                  | SHOP     | Shop: Track, Pay, Discover     | Shopify Inc.                          | Shopify Inc.                  | com.shopify.arrive                |
| Pinterest, Inc.               | PINS     | Pinterest                      | Pinterest                             | Pinterest Inc.                | com.pinterest                     |
| Snap Inc.                     | SNAP     | Snapchat                       | Snap Inc                              | Snap Inc.                     | com.snapchat.android              |
| Reddit, Inc.                  | RDDT     | Reddit                         | reddit Inc.                           | Reddit Inc.                   | com.reddit.frontpage              |
| Roblox Corporation            | RBLX     | Roblox                         | Roblox Corporation                    | Roblox Corporation            | com.roblox.client                 |
| Match Group, Inc.             | MTCH     | Tinder: Match & Meet Singles   | Tinder LLC                            | Match Group Inc.              | com.tinder                        |
| Bumble Inc.                   | BMBL     | Bumble Dating App: Meet & Date | Bumble Holding Limited                | Bumble Inc.                   | com.bumble.app                    |
| Zillow Group Inc.             | ZG       | Zillow: Homes for Sale & Rent  | Zillow Group                          | Zillow Group Inc.             | com.zillow.android.zillowmap      |
| Carvana Co.                   | CVNA     | Carvana: Buy/Sell Used Cars    | Carvana                               | Carvana Co.                   | com.carvana.carvana               |
| Coinbase Global Inc.          | COIN     | Coinbase: Buy Crypto & Stocks  | Coinbase Inc                          | Coinbase Global Inc.          | com.coinbase.android              |
| Block, Inc.                   | XYZ      | Cash App                       | Block, Inc.                           | Block Inc.                    | com.squareup.cash                 |
| Alibaba Group Holding Limited | BABA     | Taobao                         | Taobao                                | Alibaba Group Holding Limited | com.taobao.taobao                 |
| PDD Holdings Inc.             | PDD      | Temu: Shop Like a Billionaire  | Temu                                  | PDD Holdings Inc.             | com.einnovation.temu              |
| JD.com, Inc.                  | JD       | JD.com                         | 京东                                    | JD.com, Inc.                  | com.jingdong.app.mall             |
| Tencent Holdings Limited      | 0700.HK  | WeChat                         | WeChat                                | Tencent Holdings Limited      | com.tencent.mm                    |
| Coupang, Inc.                 | CPNG     | Coupang - Mobile Shopping      | Coupang Corp.                         | Coupang Inc.                  | com.coupang.mobile                |
| MercadoLibre, Inc.            | MELI     | Mercado Libre: Compras online  | Mercado Libre                         | MercadoLibre Inc.             | com.mercadolibre                  |
| Wise plc                      | WISE.L   | Wise: International Transfers  | Wise Payments Ltd.                    | Wise plc                      | com.transferwise.android          |

## 3. Signal construction and timing

Reviews are binned into weeks ending Friday 23:59 UTC. Signals for week *t* use only reviews in week *t* and trailing history, so nothing is computed from the future. Forward returns run from the Friday close of week *t* to the Friday close of week *t+1*. A one-week-lagged variant is also tested.

| Signal | Definition | Hypothesis |
|---|---|---|
| `rating_z` | Weekly mean star rating vs own trailing 26-week mean/SD | Better product → positive |
| `rating_mom_z` | 4-week rating momentum, standardised | Improving product → positive |
| `one_star_z` | 1-star share vs own trailing 26-week mean/SD | Complaints → negative |
| `sent_z` | English VADER sentiment vs own trailing 26-week mean/SD | Better text tone → positive |
| `vol_z` | log(review count) vs trailing 8-week mean/SD | Attention/growth, sign ambiguous |
| `composite` | Mean of rating, momentum, −1-star and sentiment z-scores | Pre-specified ensemble |

A signal requires at least 5 reviews in the week; sentiment requires at least 3 English reviews. Weeks with too little data are missing, not zero-filled.

![Ratings](figures/ratings.png)

## 4. Panel descriptives

| metric                            | value   |
|:----------------------------------|:--------|
| company-weeks                     | 2,936   |
| weeks with ≥5 reviews             | 2,640   |
| mean reviews/week (active)        | 429.3   |
| mean rating                       | 3.333   |
| mean 1-star share                 | 0.343   |
| English share of text             | 87.6%   |
| sentiment coverage (scored / all) | 86.0%   |

**Signal correlation (average weekly Spearman):**

| signal       |   rating_z |   rating_mom_z |   one_star_z |   sent_z |   vol_z |   composite |
|:-------------|-----------:|---------------:|-------------:|---------:|--------:|------------:|
| rating_z     |      1.000 |          0.730 |       -0.922 |    0.718 |   0.081 |       0.940 |
| rating_mom_z |      0.730 |          1.000 |       -0.687 |    0.520 |   0.051 |       0.797 |
| one_star_z   |     -0.922 |         -0.687 |        1.000 |   -0.699 |  -0.080 |      -0.917 |
| sent_z       |      0.718 |          0.520 |       -0.699 |    1.000 |   0.059 |       0.811 |
| vol_z        |      0.081 |          0.051 |       -0.080 |    0.059 |   1.000 |       0.077 |
| composite    |      0.940 |          0.797 |       -0.917 |    0.811 |   0.077 |       1.000 |

## 5. Predictive panel regressions

Specification: outcome = signal + company fixed effects + week fixed effects, with company-clustered standard errors. Signals are standardised cross-sectionally within each week; outcomes are clipped at ±50% weekly to limit outlier influence without estimating tails from the full sample. Coefficients are weekly abnormal returns per 1 SD of signal.

| signal       |   coef_% |   se_% |      t |     p |   q_value |    n |   n_entities |   r2_within |
|:-------------|---------:|-------:|-------:|------:|----------:|-----:|-------------:|------------:|
| rating_z     |    0.111 |  0.088 |  1.261 | 0.219 |     0.722 | 2364 |           26 |       0.000 |
| one_star_z   |   -0.094 |  0.080 | -1.179 | 0.249 |     0.722 | 2364 |           26 |       0.000 |
| sent_z       |    0.104 |  0.094 |  1.102 | 0.281 |     0.722 | 2377 |           27 |       0.000 |
| vol_z        |   -0.119 |  0.133 | -0.895 | 0.378 |     0.757 | 2777 |           30 |       0.000 |
| composite    |    0.075 |  0.097 |  0.772 | 0.447 |     0.796 | 2364 |           26 |       0.000 |
| rating_mom_z |   -0.007 |  0.101 | -0.074 | 0.942 |     0.977 | 2287 |           25 |       0.000 |

Full results for all three horizons (fast 1-week, 4-week, and lagged 1-week) are in `data/analysis/regressions.csv`.

![Long-short](figures/long_short.png)

## 6. Long–short backtest

Each week, companies with valid signals are ranked; the top and bottom quintiles are held equal-weighted, long versus short. Returns are market-adjusted. Costs are 10 bp per side applied to measured turnover. The randomization p-value permutes the signal across companies within each week (300 draws).

| signal       |   weeks |   ann_ret_% |   sharpe_gross |   sharpe_net |   t_hac |   p_perm |   max_dd_% |   turnover |   hit_rate |
|:-------------|--------:|------------:|---------------:|-------------:|--------:|---------:|-----------:|-----------:|-----------:|
| rating_z     |     103 |      -0.954 |         -0.041 |       -0.526 |  -0.052 |    0.963 |    -26.871 |      1.098 |      0.466 |
| rating_mom_z |     100 |      24.497 |          0.694 |        0.299 |   1.049 |    0.329 |    -21.248 |      1.341 |      0.540 |
| one_star_z   |     103 |     -12.008 |         -0.436 |       -0.866 |  -0.654 |    0.561 |    -45.239 |      1.139 |      0.476 |
| sent_z       |     103 |      -0.852 |         -0.033 |       -0.522 |  -0.044 |    0.990 |    -25.777 |      1.194 |      0.476 |
| vol_z        |     149 |     -26.453 |         -0.853 |       -1.278 |  -1.380 |    0.183 |    -65.634 |      1.270 |      0.450 |
| composite    |     103 |       0.792 |          0.031 |       -0.457 |   0.041 |    0.970 |    -26.330 |      1.208 |      0.515 |

**Single-event sensitivity.** The best gross Sharpe is not robust to one corporate event: the largest weekly long-short move was 27.7% in the week ending 2026-10-09, when Bumble (a long-leg name) rose 134% in one week. Removing that single week cuts annualised gross return from 24.5% to 10.2% and Sharpe from 0.69 to 0.35. The panel regressions clip weekly outcomes at ±50%, so the event does not create significance there.

**Composite detail:** gross annualised 0.8%, net -11.8%, max drawdown -26.3%, weekly turnover 1.21, 103 weeks. Best gross Sharpe across signals: rating_mom_z (0.69).

**Lagged execution** (signal at week t, trade one week later; the conservative timing check):

| signal    |   n_weeks |   ann_ret_% |   sharpe_gross |   sharpe_net |   t_hac |   p_perm |
|:----------|----------:|------------:|---------------:|-------------:|--------:|---------:|
| composite |       102 |      -6.903 |         -0.301 |       -0.848 |  -0.408 |    0.751 |
| rating_z  |       102 |      -1.546 |         -0.058 |       -0.489 |  -0.090 |    0.924 |

A Sharpe estimated on ~2–3 years of weekly data has a standard error of roughly 0.71. Even a true Sharpe of 0.5 is not distinguishable from zero at this sample length.

## 7. Rating-shock event study

Events are weeks where `rating_z` ≤ -1.5 (a drop of 1.5 within-company SD), after at least 20 weeks of history. Abnormal returns are measured from the following week.

| group          |    n |   CAR_1w |   CAR_2w |   CAR_4w |   t_4w |
|:---------------|-----:|---------:|---------:|---------:|-------:|
| negative shock |  199 |   -0.408 |   -0.722 |   -1.272 | -0.932 |
| positive shock |  124 |    0.651 |    0.800 |    0.442 |  0.419 |
| control        | 1505 |   -0.017 |   -0.008 |   -0.023 | -0.063 |

FE-adjusted difference (negative shock − control, company + week fixed effects), 4-week CAR: -0.31% (t=-0.28, p=0.786, N=1704).

![Event study](figures/event_study.png)

## 8. Statistical power

![Power](figures/power.png)

Using the realised panel (N=20 companies with ≥52 weeks, median T=87 weeks, residual σ=6.2%/week, AR(1) ρ=-0.01), the analytic MDE at 80% power / 5% two-sided is **0.42% per week** (≈21.8% annualised) per 1 SD of signal.

Design MDE for the full 32-name panel at alternative histories (the realised-panel MDE above is the relevant one):

|   years |   n_entities |   n_weeks |   rho |   MDE_%_weekly |   MDE_%_annual |
|--------:|-------------:|----------:|------:|---------------:|---------------:|
|    1.00 |        32.00 |     52.00 |  0.00 |           0.35 |          18.14 |
|    1.00 |        32.00 |     52.00 |  0.20 |           0.43 |          22.22 |
|    1.00 |        32.00 |     52.00 |  0.40 |           0.53 |          27.71 |
|    2.00 |        32.00 |    104.00 |  0.00 |           0.25 |          12.83 |
|    2.00 |        32.00 |    104.00 |  0.20 |           0.30 |          15.71 |
|    2.00 |        32.00 |    104.00 |  0.40 |           0.38 |          19.60 |
|    3.00 |        32.00 |    156.00 |  0.00 |           0.20 |          10.47 |
|    3.00 |        32.00 |    156.00 |  0.20 |           0.25 |          12.83 |
|    3.00 |        32.00 |    156.00 |  0.40 |           0.31 |          16.00 |
|    4.00 |        32.00 |    208.00 |  0.00 |           0.17 |           9.07 |
|    4.00 |        32.00 |    208.00 |  0.20 |           0.21 |          11.11 |
|    4.00 |        32.00 |    208.00 |  0.40 |           0.27 |          13.86 |

**Monte Carlo check** (same estimator, simulated panels):

|   effect_%_weekly |   effect_%_annual |   power |   n_sims |
|------------------:|------------------:|--------:|---------:|
|             0.000 |             0.000 |   0.067 |  150.000 |
|             0.005 |            26.000 |   0.873 |  150.000 |
|             0.010 |            52.000 |   1.000 |  150.000 |
|             0.020 |           104.000 |   1.000 |  150.000 |
|             0.030 |           156.000 |   1.000 |  150.000 |

The Sharpe MDE matters as much as the regression MDE: over the realised ~1.7-year window the Sharpe MDE is about 2.2; even with a full three years it would be about 1.6. A backtest Sharpe of 0.5 looks economically meaningful but is statistically indistinguishable from zero at either length. That is a property of the sample length, not a fixable modelling choice.

## 9. Limitations and compliance

1. **Public-data ceiling.** The Play review endpoint is newest-first and unpaginated by date. Mega-app history is truncated; Tier C companies cannot support 12-quarter claims.
2. **Platform selection.** Android only; app-store reviewers are self-selected and prompted, so levels and changes are not representative of all users.
3. **Review edits and deletions.** The timestamp is the review's recorded time; edits and removals are not separately observable, which adds measurement error.
4. **Mapping risk.** Parent/subsidiary structures and tickers change. The map is curated and validated, but corporate actions (e.g., Block's SQ→XYZ change) require maintenance.
5. **Returns.** Weekly close-to-close, market-adjusted with a regional ETF; no transaction costs beyond the stated 10 bp, no borrow costs, no slippage, no survivorship correction.
6. **Not sector-neutral.** With 32 names, the top and bottom quintiles can concentrate in one sector; part of any long-short result may be a sector exposure rather than a review effect.
7. **Selected panel, selected apps.** Companies were included only if they have a listed parent, a consumer app, and usable Play data. PDD is represented by Temu (its international app), not Pinduoduo; WeChat reviews come from the global/US storefront and are not a census of mainland usage.
8. **Single-name events.** Equal-weight quintile portfolios can be dominated by one corporate action. The best long-short Sharpe here is roughly halved by removing one takeover week (Bumble, October 2026); treat any single-name-driven result as a confound, not a review signal.
9. **Multiple testing.** Six signals × three horizons are reported with BH q-values; the long-short search is also reported in full. A single q<0.10 result is not a tradable discovery.
10. **Terms of service.** Google Play's terms restrict automated collection. This prototype is for internal research; for production, license data from a vendor (Sensor Tower, AppFollow, data.ai) or obtain written permission. Do not redistribute review text or author data.
11. **Not investment advice.** Nothing here is a recommendation to buy or sell a security.

## 10. Reproduction

```bash
cd play-review-panel
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e .
python -m prpanel ingest --since 2023-11-07 --max-reviews 50000
python -m prpanel enrich
python -m prpanel prices
python -m prpanel build
python -m prpanel analyze --n-perm 300
python -m prpanel report
```

Run metadata: `{"python": "3.13.15", "pandas": "3.1.1", "numpy": "2.5.4", "statsmodels": "0.15.0", "companies": 32, "weeks": 157, "reviews": 1237605, "first_week": "2023-11-10", "last_week": "2026-11-06", "signals": ["rating_z", "rating_mom_z", "one_star_z", "sent_z", "vol_z", "composite"], "n_perm": 300}`
