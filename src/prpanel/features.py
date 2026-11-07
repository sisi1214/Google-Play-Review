"""Build the company-week panel used by the backtest.

Timing convention
-----------------
Reviews are binned into weeks ending Friday 23:59 UTC. A signal for week t uses
only reviews timestamped in week t (and trailing history). Forward returns are
measured from the Friday close of week t to the Friday close of week t+1. That
assumes execution at/after the Friday close, which is feasible because the
review cutoff is the same close. A one-week-lagged variant is also produced as
a robustness check and is the conservative specification.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import PANEL_WEEKLY, PanelRow, ensure_dirs
from .prices import build_price_panel

log = logging.getLogger("prpanel.features")

MIN_WEEK_REVIEWS = 5
SIGNAL_COLS = ["rating_z", "rating_mom_z", "one_star_z", "sent_z", "vol_z", "composite"]

REVIEW_QUERY = """
SELECT company_id, package_id, at_utc, score, thumbs_up,
       (reply_content IS NOT NULL) AS replied, sentiment, is_english
FROM reviews
"""


def load_reviews(conn) -> pd.DataFrame:
    df = pd.read_sql_query(REVIEW_QUERY, conn)
    if df.empty:
        return df
    df["at_utc"] = pd.to_datetime(df["at_utc"], utc=True)
    df["replied"] = df["replied"].astype(float)
    return df


def aggregate_weekly(reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        return pd.DataFrame()
    df = reviews.copy()
    ts = df["at_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["week_end"] = ts.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    df["is_1"] = (df["score"] == 1).astype(float)
    df["is_2"] = (df["score"] == 2).astype(float)
    df["is_neg"] = (df["score"] <= 2).astype(float)
    df["is_5"] = (df["score"] == 5).astype(float)
    df["eng"] = df["is_english"].fillna(0).astype(float)

    g = df.groupby(["company_id", "week_end"], sort=True)
    out = g.agg(
        n_reviews=("score", "size"),
        mean_score=("score", "mean"),
        sd_score=("score", "std"),
        share_1=("is_1", "mean"),
        share_2=("is_2", "mean"),
        share_neg=("is_neg", "mean"),
        share_5=("is_5", "mean"),
        mean_thumbs=("thumbs_up", "mean"),
        reply_rate=("replied", "mean"),
        sent_mean=("sentiment", "mean"),
        sent_n=("sentiment", "count"),
        english_share=("eng", "mean"),
    ).reset_index()
    return out


def _week_scaffold(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return weekly
    start = weekly["week_end"].min()
    end = weekly["week_end"].max()
    frames = []
    for cid, g in weekly.groupby("company_id", sort=True):
        idx = pd.date_range(g["week_end"].min(), end, freq="W-FRI")
        s = g.set_index("week_end").reindex(idx).rename_axis("week_end").reset_index()
        s["company_id"] = cid
        for c in ("n_reviews",):
            s[c] = s[c].fillna(0)
        frames.append(s)
    out = pd.concat(frames, ignore_index=True)
    cols = ["company_id", "week_end"] + [c for c in out.columns if c not in {"company_id", "week_end"}]
    out = out[cols].sort_values(["company_id", "week_end"]).reset_index(drop=True)
    out["n_reviews"] = out["n_reviews"].astype(int)
    return out


def _roll(panel: pd.DataFrame, col: str, window: int, min_periods: int, stat: str = "mean") -> pd.Series:
    """Trailing rolling stat computed on data strictly before the current week."""
    def f(s: pd.Series) -> pd.Series:
        r = s.shift(1).rolling(window, min_periods=min_periods)
        return getattr(r, stat)()
    return panel.groupby("company_id")[col].transform(f)


def _z(obs: pd.Series, base: pd.Series, sd: pd.Series) -> pd.Series:
    return (obs - base) / sd.replace(0, np.nan)


def add_signals(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["company_id", "week_end"]).copy()
    p["rating_obs"] = p["mean_score"].where(p["n_reviews"] >= MIN_WEEK_REVIEWS)
    p["one_star_obs"] = p["share_1"].where(p["n_reviews"] >= MIN_WEEK_REVIEWS)
    p["sent_obs"] = p["sent_mean"].where(p["sent_n"] >= 3)
    p["log_n"] = np.log1p(p["n_reviews"])

    # Rating level relative to its own trailing 26-week distribution.
    base = _roll(p, "rating_obs", 26, 8, "mean")
    sd = _roll(p, "rating_obs", 26, 8, "std")
    p["rating_z"] = _z(p["rating_obs"], base, sd)

    # Four-week rating momentum, standardised by its own trailing volatility.
    mom = p["rating_obs"] - _roll(p, "rating_obs", 4, 3, "mean")
    p["rating_mom"] = mom
    mom_sd = _roll(p, "rating_mom", 26, 8, "std")
    p["rating_mom_z"] = mom / mom_sd.replace(0, np.nan)

    # One-star share (positive = deterioration).
    os_base = _roll(p, "one_star_obs", 26, 8, "mean")
    os_sd = _roll(p, "one_star_obs", 26, 8, "std")
    p["one_star_z"] = _z(p["one_star_obs"], os_base, os_sd)

    # Sentiment (positive = better).
    sent_base = _roll(p, "sent_obs", 26, 8, "mean")
    sent_sd = _roll(p, "sent_obs", 26, 8, "std")
    p["sent_z"] = _z(p["sent_obs"], sent_base, sent_sd)

    # Volume surprise: log reviews vs trailing 8-week mean/std.
    vol_base = _roll(p, "log_n", 8, 4, "mean")
    vol_sd = _roll(p, "log_n", 8, 4, "std")
    p["vol_z"] = _z(p["log_n"], vol_base, vol_sd)

    components = pd.concat(
        [p["rating_z"], p["rating_mom_z"], -p["one_star_z"], p["sent_z"]], axis=1
    )
    p["composite_n"] = components.notna().sum(axis=1)
    p["composite"] = components.mean(axis=1, skipna=True).where(p["composite_n"] >= 2)

    p["quarter"] = p["week_end"].dt.to_period("Q")
    return p


def _compound(g: pd.Series, h: int, offset: int = 0) -> pd.Series:
    """Cumulative return over h weeks starting offset+1 weeks ahead."""
    out = pd.Series(1.0, index=g.index)
    for i in range(1, h + 1):
        out = out * (1.0 + g.shift(-(i + offset)))
    return out - 1.0


def add_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["company_id", "week_end"]).copy()
    for base, prefix in (("ret", "fwd_ret"), ("abn_ret", "fwd_abn")):
        g = p.groupby("company_id")[base]
        for h in (1, 2, 4):
            p[f"{prefix}_{h}w"] = g.transform(lambda s, h=h: _compound(s, h, 0))
            p[f"{prefix}_{h}w_lag"] = g.transform(lambda s, h=h: _compound(s, h, 1))
    return p


def coverage_table(panel: pd.DataFrame, rows: list[PanelRow]) -> pd.DataFrame:
    meta = pd.DataFrame([r.as_dict() for r in rows])[
        ["company_id", "company", "ticker", "app_name", "package_id", "sector", "exchange"]
    ]
    if panel.empty:
        out = meta.copy()
        for c in ("n_reviews", "first_week", "last_week", "weeks", "weeks_ge5", "quarters_ge5", "tier"):
            out[c] = np.nan
        return out
    g = panel.groupby("company_id")
    cov = g.agg(
        n_reviews=("n_reviews", "sum"),
        first_week=("week_end", "min"),
        last_week=("week_end", "max"),
        weeks=("week_end", "nunique"),
        weeks_ge5=("n_reviews", lambda s: int((s >= MIN_WEEK_REVIEWS).sum())),
    ).reset_index()
    # quarters with at least 5 reviews, computed cleanly
    q = panel[panel["n_reviews"] >= MIN_WEEK_REVIEWS].groupby("company_id")["quarter"].nunique()
    cov["quarters_ge5"] = cov["company_id"].map(q).fillna(0).astype(int)
    cov = meta.merge(cov, on="company_id", how="left")
    cov["n_reviews"] = cov["n_reviews"].fillna(0).astype(int)
    cov["weeks_ge5"] = cov["weeks_ge5"].fillna(0).astype(int)
    cov["quarters_ge5"] = cov["quarters_ge5"].fillna(0).astype(int)
    cov["tier"] = np.select(
        [cov["quarters_ge5"] >= 12, cov["quarters_ge5"] >= 4],
        ["A (12+ quarters)", "B (4-11 quarters)"],
        default="C (<4 quarters)",
    )
    cov["history_days"] = (pd.to_datetime(cov["last_week"]) - pd.to_datetime(cov["first_week"])).dt.days
    return cov.sort_values(["quarters_ge5", "n_reviews"], ascending=False).reset_index(drop=True)


def build_panel(conn, rows: list[PanelRow], *, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    reviews = load_reviews(conn)
    if reviews.empty:
        log.warning("no reviews in database")
        return pd.DataFrame(), coverage_table(pd.DataFrame(), rows)
    log.info("loaded %d reviews", len(reviews))
    weekly = aggregate_weekly(reviews)
    panel = _week_scaffold(weekly)
    log.info("scaffolded %d company-weeks", len(panel))
    panel = add_signals(panel)

    prices = build_price_panel(rows)
    panel = panel.merge(
        prices[["company_id", "date", "ret", "mkt_ret", "abn_ret", "world_abn_ret"]],
        left_on=["company_id", "week_end"], right_on=["company_id", "date"], how="left",
    ).drop(columns=["date"])
    panel = add_forward_returns(panel)
    panel["n_reviews"] = panel["n_reviews"].astype(int)
    panel = panel.sort_values(["company_id", "week_end"]).reset_index(drop=True)

    cov = coverage_table(panel, rows)
    if save:
        panel.to_parquet(PANEL_WEEKLY, index=False)
        panel.to_csv(PANEL_WEEKLY.with_suffix(".csv"), index=False)
        cov.to_csv(PANEL_WEEKLY.with_name("coverage.csv"), index=False)
        log.info("saved panel: %s rows -> %s", len(panel), PANEL_WEEKLY)
    return panel, cov
