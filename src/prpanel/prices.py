"""Weekly price data and market-adjusted returns.

Prices are downloaded once and cached as CSV so that the analysis is
reproducible without hitting Yahoo again. Adjusted close is used, so splits and
dividends are handled. Each company is benchmarked against a currency-consistent
index: SPY for USD names, MCHI for China ADRs, 2800.HK for the HK listing,
EWY for Coupang, ^FTSE for Wise, and URTH as a global fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import PRICE_DIR, PanelRow, ensure_dirs, symbols

log = logging.getLogger("prpanel.prices")

DAILY_CSV = PRICE_DIR / "daily_close.csv"
WEEKLY_CSV = PRICE_DIR / "weekly_close.csv"
RETURNS_CSV = PRICE_DIR / "weekly_returns.csv"
PANEL_PRICES_CSV = PRICE_DIR / "panel_prices.csv"


def _flatten_close(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance >= 0.2 returns (field, ticker) for multi-ticker downloads.
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"].copy()
        else:
            close = df.xs("Close", axis=1, level=-1).copy()
    else:
        close = df[["Close"]].copy()
        close.columns = ["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def download_prices(start: str = "2019-01-01", end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if DAILY_CSV.exists() and not refresh:
        close = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)
        log.info("loaded cached daily prices: %s", close.shape)
        return close

    import yfinance as yf

    syms = symbols()
    log.info("downloading %d symbols from Yahoo Finance", len(syms))
    raw = yf.download(
        syms, start=start, end=end, interval="1d", auto_adjust=True,
        progress=False, threads=True, group_by="column", timeout=30,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo returned no price data")
    close = _flatten_close(raw)
    close = close.dropna(how="all")
    close.to_csv(DAILY_CSV)
    missing = [s for s in syms if s not in close.columns or close[s].dropna().empty]
    if missing:
        log.warning("no usable price history for: %s", missing)
    log.info("saved daily prices: %s", close.shape)
    return close


def weekly_frames(close: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    if close is None:
        close = download_prices()
    weekly = close.resample("W-FRI").last()
    weekly = weekly[weekly.index <= pd.Timestamp.today().normalize()]
    returns = weekly.pct_change(fill_method=None)
    weekly.to_csv(WEEKLY_CSV)
    returns.to_csv(RETURNS_CSV)
    return weekly, returns


def build_price_panel(rows: list[PanelRow], returns: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    if returns is None:
        if RETURNS_CSV.exists():
            returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)
        else:
            _, returns = weekly_frames()
    if "URTH" not in returns.columns:
        log.warning("URTH unavailable; world return set to NaN")
        returns = returns.assign(URTH=float("nan"))

    frames = []
    for row in rows:
        if row.yahoo_symbol not in returns.columns or row.benchmark_symbol not in returns.columns:
            log.warning("skipping prices for %s (%s / %s)", row.company_id, row.yahoo_symbol, row.benchmark_symbol)
            continue
        ret = returns[row.yahoo_symbol]
        mkt = returns[row.benchmark_symbol]
        world = returns["URTH"]
        f = pd.DataFrame({
            "date": returns.index,
            "company_id": row.company_id,
            "ticker": row.ticker,
            "ret": ret.values,
            "mkt_ret": mkt.values,
            "world_ret": world.values,
        })
        f["abn_ret"] = f["ret"] - f["mkt_ret"]
        f["world_abn_ret"] = f["ret"] - f["world_ret"]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = out.dropna(subset=["ret"]).sort_values(["company_id", "date"]).reset_index(drop=True)
    out.to_csv(PANEL_PRICES_CSV, index=False)
    log.info("price panel: %s rows, %s companies, %s..%s", len(out), out.company_id.nunique(), out.date.min(), out.date.max())
    return out
