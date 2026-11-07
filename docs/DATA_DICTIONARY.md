# Data dictionary

## `config/panel.csv`

| Column | Meaning |
|---|---|
| `company_id` | Stable slug used as the panel key. |
| `company` | Legal/listed company name. |
| `ticker` | Display ticker, including exchange suffix where relevant (`0700.HK`, `WISE.L`). |
| `yahoo_symbol` | Symbol passed to Yahoo Finance. |
| `exchange` | Listing venue. |
| `country` | Country of listing/headquarters used for grouping, not a review filter. |
| `sector` | Coarse consumer-internet sector. |
| `app_name` | Play listing title at curation time. |
| `package_id` | Android package id; unique and validated. |
| `developer` | Developer string shown on the Play listing. |
| `parent` | Listed parent company. Curated; Play does not provide a legal-entity map. |
| `app_role` | `primary` or `secondary`. |
| `benchmark_symbol` | Currency-consistent market benchmark. |
| `notes` | Why the app/developer/parent mapping is what it is. |

## `reviews` table

| Column | Meaning |
|---|---|
| `review_id` | Play review id (primary key). |
| `package_id`, `company_id` | App and company keys. |
| `score` | 1–5 stars. |
| `at_utc` | Review timestamp, assumed UTC (ISO-8601 with `Z`). |
| `content` | Review text, truncated to 8,000 chars. Author identity is not stored. |
| `thumbs_up` | Helpful votes at collection time. |
| `app_version` | `reviewCreatedVersion` (fallback `appVersion`). |
| `reply_content`, `replied_at_utc` | Developer reply, if any. |
| `lang`, `country` | Requested store locale, not detected language. |
| `ingested_at` | Collection timestamp. |
| `sentiment` | VADER compound in [-1, 1], English reviews only. |
| `is_english` | Heuristic flag used for sentiment eligibility. |

## `data/panel_weekly.parquet` / `.csv`

One row per `(company_id, week_end)`.

| Column | Meaning |
|---|---|
| `week_end` | Friday date of the ISO-style week ending Friday 23:59 UTC. |
| `n_reviews` | Reviews in the week. |
| `mean_score`, `sd_score` | Mean and SD of 1–5 star ratings. |
| `share_1`, `share_2`, `share_neg`, `share_5` | Share of 1-star, 2-star, 1–2 star, 5-star reviews. |
| `mean_thumbs`, `reply_rate` | Mean helpful votes; share with a developer reply. |
| `sent_mean`, `sent_n` | Mean VADER sentiment and number of scored English reviews. |
| `english_share` | Share of reviews flagged English. |
| `rating_obs`, `one_star_obs`, `sent_obs` | Signal-eligible observations (n≥5, sentiment n≥3). |
| `log_n` | `log1p(n_reviews)`. |
| `rating_z` | Weekly mean rating vs own trailing 26-week mean/SD. |
| `rating_mom`, `rating_mom_z` | 4-week rating momentum and its standardised version. |
| `one_star_z` | 1-star share vs own trailing 26-week mean/SD (higher = worse). |
| `sent_z` | Sentiment vs own trailing 26-week mean/SD. |
| `vol_z` | Review-volume surprise vs trailing 8-week mean/SD. |
| `composite_n`, `composite` | Mean of available components (rating, momentum, −1-star, sentiment); requires ≥2. |
| `quarter` | Calendar quarter of `week_end`. |
| `ret`, `mkt_ret` | Weekly close-to-close return and benchmark return. |
| `abn_ret` | `ret - mkt_ret`. |
| `world_abn_ret` | `ret - URTH` return (global robustness). |
| `fwd_ret_{h}w` | Compounded return from close *t* to close *t+h*. |
| `fwd_ret_{h}w_lag` | Compounded return from close *t+1* to close *t+1+h*. |
| `fwd_abn_{h}w`, `fwd_abn_{h}w_lag` | Same, using `abn_ret`. |
