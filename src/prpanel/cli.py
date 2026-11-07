"""Command line interface for the Google Play review panel."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import db
from .config import DB_PATH, LOG_DIR, PANEL_WEEKLY, ensure_dirs, load_panel
from .features import build_panel, coverage_table
from .ingest import ingest_all
from .text import enrich_sentiment

log = logging.getLogger("prpanel.cli")

ANALYSIS_DIR = DB_PATH.parent / "analysis"


def setup_logging(verbose: bool = False) -> None:
    ensure_dirs()
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "prpanel.log")],
        force=True,
    )


def _load_panel_or_fail() -> list:
    rows = load_panel()
    log.info("panel: %d companies, %d apps", len(rows), len({r.package_id for r in rows}))
    return rows


def cmd_ingest(args) -> None:
    rows = _load_panel_or_fail()
    only = set(args.apps.split(",")) if args.apps else None
    conn = db.connect(DB_PATH)
    db.init_schema(conn)
    results = ingest_all(
        conn, rows, since=date.fromisoformat(args.since),
        max_reviews=args.max_reviews, mode=args.mode, restart=args.restart,
        only=only, batch=args.batch,
    )
    print("\ncompany    status       added    total  oldest")
    for r in results:
        print(f"{r.company_id:10s} {r.status:12s} {r.added:7d} {r.total:8d}  {str(r.oldest)[:10]}")
    print(f"\nDB: {db.counts(conn)}")
    conn.close()


def cmd_enrich(args) -> None:
    conn = db.connect(DB_PATH)
    db.init_schema(conn)
    res = enrich_sentiment(conn, batch_size=args.batch, limit=args.limit)
    print(res)
    conn.close()


def cmd_prices(args) -> None:
    from .prices import build_price_panel, download_prices, weekly_frames

    rows = _load_panel_or_fail()
    close = download_prices(refresh=args.refresh)
    weekly, returns = weekly_frames(close)
    panel = build_price_panel(rows, returns)
    print(f"weekly prices: {weekly.shape}; price panel: {panel.shape}")


def cmd_build(args) -> None:
    rows = _load_panel_or_fail()
    conn = db.connect(DB_PATH)
    db.init_schema(conn)
    panel, cov = build_panel(conn, rows)
    conn.close()
    if panel.empty:
        print("no reviews; run ingest first")
        return
    print(f"panel: {panel.shape}")
    print(cov[["company", "ticker", "n_reviews", "quarters_ge5", "tier"]].to_string(index=False))
    print(f"tiers: {cov['tier'].value_counts().to_dict()}")


def cmd_analyze(args) -> None:
    from .backtest import SIGNALS, event_study, long_short, regression_table, signal_correlations
    from .power import actual_panel_power, mde_table, monte_carlo_power

    ensure_dirs()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if not PANEL_WEEKLY.exists():
        raise SystemExit("panel not built; run `python -m prpanel build` first")
    panel = pd.read_parquet(PANEL_WEEKLY)
    rows = _load_panel_or_fail()
    cov = coverage_table(panel, rows)
    signals = [s.strip() for s in args.signals.split(",")] if args.signals else SIGNALS

    log.info("running panel regressions")
    reg = regression_table(panel, signals=signals)
    reg.to_csv(ANALYSIS_DIR / "regressions.csv", index=False)

    log.info("running long-short backtests (%d signals, %d permutations)", len(signals), args.n_perm)
    ls_stats = []
    for s in signals:
        r = long_short(panel, s, "fwd_abn_1w", quantiles=args.quantiles,
                       min_names=args.min_names, cost_bps=args.cost_bps, n_perm=args.n_perm)
        r.series.to_csv(ANALYSIS_DIR / f"ls_{s}.csv", index=False)
        st = dict(r.stats)
        st["signal"] = s
        ls_stats.append(st)
        log.info("LS %-12s sharpe=%.2f net=%.2f p_perm=%.3f", s,
                 st.get("sharpe_gross", float("nan")), st.get("sharpe_net", float("nan")),
                 st.get("p_perm", float("nan")))
    pd.DataFrame(ls_stats).to_csv(ANALYSIS_DIR / "ls_stats.csv", index=False)

    log.info("lagged-execution robustness for composite and rating")
    lag_stats = []
    for s in [x for x in ("composite", "rating_z") if x in signals]:
        r = long_short(panel, s, "fwd_abn_1w_lag", quantiles=args.quantiles,
                       min_names=args.min_names, cost_bps=args.cost_bps, n_perm=args.n_perm)
        r.series.to_csv(ANALYSIS_DIR / f"ls_lag_{s}.csv", index=False)
        st = dict(r.stats); st["signal"] = s
        lag_stats.append(st)
    pd.DataFrame(lag_stats).to_csv(ANALYSIS_DIR / "ls_stats_lag.csv", index=False)

    log.info("event study")
    ev = event_study(panel)
    ev_events = ev.pop("events")
    ev_events.to_csv(ANALYSIS_DIR / "event_events.csv", index=False)
    (ANALYSIS_DIR / "event_study.json").write_text(json.dumps(ev, indent=2, default=float))

    corr = signal_correlations(panel, signals=signals)
    corr.to_csv(ANALYSIS_DIR / "signal_correlations.csv")

    power = actual_panel_power(panel)
    (ANALYSIS_DIR / "power.json").write_text(json.dumps(power, indent=2, default=float))
    mde = mde_table(n_entities=max(panel["company_id"].nunique(), 2))
    mde.to_csv(ANALYSIS_DIR / "mde_table.csv", index=False)

    if args.mc:
        log.info("Monte Carlo power (%d sims per effect)", args.mc_sims)
        mc = monte_carlo_power(
            [0.0, 0.005, 0.01, 0.02, 0.03],
            n_sims=args.mc_sims, seed=23,
        )
        mc.to_csv(ANALYSIS_DIR / "mc_power.csv", index=False)

    import platform
    import numpy as _np
    import pandas as _pd
    import statsmodels as _sm
    meta = {
        "python": platform.python_version(),
        "pandas": _pd.__version__,
        "numpy": _np.__version__,
        "statsmodels": _sm.__version__,
        "companies": int(panel["company_id"].nunique()),
        "weeks": int(panel["week_end"].nunique()),
        "reviews": int(panel["n_reviews"].sum()),
        "first_week": str(panel["week_end"].min().date()),
        "last_week": str(panel["week_end"].max().date()),
        "signals": signals,
        "n_perm": args.n_perm,
    }
    (ANALYSIS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"analysis written to {ANALYSIS_DIR}")
    if power:
        print(f"actual power: N={power.get('n_entities')} T={power.get('n_weeks')} "
              f"MDE={power.get('mde_annualized', float('nan'))*100:.1f}%/yr "
              f"Sharpe MDE={power.get('sharpe_mde', float('nan')):.2f}")


def cmd_report(args) -> None:
    from .report import generate_report

    panel = pd.read_parquet(PANEL_WEEKLY)
    rows = _load_panel_or_fail()
    cov = coverage_table(panel, rows)
    reg = pd.read_csv(ANALYSIS_DIR / "regressions.csv")
    stats = pd.read_csv(ANALYSIS_DIR / "ls_stats.csv")
    ls_results = {}
    for _, row in stats.iterrows():
        s = row["signal"]
        series = pd.read_csv(ANALYSIS_DIR / f"ls_{s}.csv", parse_dates=["week_end"])
        ls_results[s] = SimpleNamespace(signal=s, series=series, stats=row.to_dict())
    ev = json.loads((ANALYSIS_DIR / "event_study.json").read_text())
    power = json.loads((ANALYSIS_DIR / "power.json").read_text())
    mde = pd.read_csv(ANALYSIS_DIR / "mde_table.csv")
    corr = pd.read_csv(ANALYSIS_DIR / "signal_correlations.csv", index_col=0)
    meta = json.loads((ANALYSIS_DIR / "meta.json").read_text())
    mc = None
    if (ANALYSIS_DIR / "mc_power.csv").exists():
        mc = pd.read_csv(ANALYSIS_DIR / "mc_power.csv")
    ls_lag = None
    if (ANALYSIS_DIR / "ls_stats_lag.csv").exists():
        ls_lag = pd.read_csv(ANALYSIS_DIR / "ls_stats_lag.csv")
    mapping = pd.DataFrame([r.as_dict() for r in rows])[
        ["company", "ticker", "app_name", "developer", "parent", "package_id", "sector"]
    ]
    out = generate_report(panel, cov, reg, ls_results, ev, power, mde, corr,
                          meta=meta, mc_power=mc, ls_lag=ls_lag, mapping=mapping)
    print(f"report: {out}")


def cmd_status(args) -> None:
    conn = db.connect(DB_PATH)
    db.init_schema(conn)
    print(db.counts(conn))
    print("\nper app:")
    rows = conn.execute("""
        SELECT a.company_id, s.package_id, s.status, s.n_reviews, s.oldest_at, s.newest_at, s.last_error
        FROM ingest_state s JOIN apps a ON a.package_id = s.package_id
        ORDER BY a.company_id
    """).fetchall()
    for r in rows:
        print(dict(r))
    conn.close()


def cmd_all(args) -> None:
    cmd_ingest(args)
    cmd_enrich(args)
    cmd_prices(args)
    cmd_build(args)
    cmd_analyze(args)
    cmd_report(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prpanel", description="Google Play review panel for listed consumer-internet companies")
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="backfill or refresh reviews")
    ing.add_argument("--since", default="2023-11-07", help="oldest review date (YYYY-MM-DD)")
    ing.add_argument("--max-reviews", type=int, default=50_000, help="per-app cap")
    ing.add_argument("--mode", choices=["backfill", "refresh"], default="backfill")
    ing.add_argument("--restart", action="store_true", help="ignore checkpoint and restart from newest")
    ing.add_argument("--apps", default="", help="comma-separated company_ids")
    ing.add_argument("--batch", type=int, default=2000)
    ing.set_defaults(func=cmd_ingest)

    en = sub.add_parser("enrich", help="compute English VADER sentiment")
    en.add_argument("--batch", type=int, default=4000)
    en.add_argument("--limit", type=int, default=None)
    en.set_defaults(func=cmd_enrich)

    pr = sub.add_parser("prices", help="download/cached weekly prices")
    pr.add_argument("--refresh", action="store_true")
    pr.set_defaults(func=cmd_prices)

    bu = sub.add_parser("build", help="build the company-week panel")
    bu.set_defaults(func=cmd_build)

    an = sub.add_parser("analyze", help="regressions, long-short, event study, power")
    an.add_argument("--signals", default="")
    an.add_argument("--n-perm", type=int, default=300)
    an.add_argument("--quantiles", type=int, default=5)
    an.add_argument("--min-names", type=int, default=12)
    an.add_argument("--cost-bps", type=float, default=10.0)
    an.add_argument("--mc", action="store_true")
    an.add_argument("--mc-sims", type=int, default=150)
    an.set_defaults(func=cmd_analyze)

    re = sub.add_parser("report", help="render reports/report.md")
    re.set_defaults(func=cmd_report)

    st = sub.add_parser("status", help="database counts and per-app state")
    st.set_defaults(func=cmd_status)

    al = sub.add_parser("all", help="run the full pipeline")
    al.add_argument("--since", default="2023-11-07")
    al.add_argument("--max-reviews", type=int, default=50_000)
    al.add_argument("--mode", choices=["backfill", "refresh"], default="backfill")
    al.add_argument("--restart", action="store_true")
    al.add_argument("--apps", default="")
    al.add_argument("--batch", type=int, default=2000)
    al.add_argument("--n-perm", type=int, default=300)
    al.add_argument("--quantiles", type=int, default=5)
    al.add_argument("--min-names", type=int, default=12)
    al.add_argument("--cost-bps", type=float, default=10.0)
    al.add_argument("--signals", default="")
    al.add_argument("--mc", action="store_true")
    al.add_argument("--mc-sims", type=int, default=150)
    al.set_defaults(func=cmd_all)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
