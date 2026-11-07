"""Statistical power for the panel design.

The headline question is whether ~32 companies x 12+ quarters can detect a
plausible review-signal effect. Two answers are produced:

1. An analytic minimum detectable effect (MDE) for a two-way fixed-effects
   panel with AR(1) residuals and company-clustered errors.
2. A Monte Carlo rejection rate using the same estimator as the real backtest,
   so the analytic approximation can be checked rather than trusted.

Both are reported in weekly return units per one standard-deviation signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import _two_way_demean, fe_regression

log = logging.getLogger("prpanel.power")


def analytic_mde(
    n_entities: int,
    n_weeks: int,
    residual_sd: float,
    *,
    rho: float = 0.1,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict:
    t_eff = n_weeks * (1.0 - rho) / (1.0 + rho)
    se = residual_sd / np.sqrt(max(n_entities * t_eff, 1.0))
    se *= np.sqrt(n_entities / max(n_entities - 1, 1))  # cluster correction
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    mde = z * se
    return {
        "n_entities": n_entities,
        "n_weeks": n_weeks,
        "residual_sd": residual_sd,
        "rho": rho,
        "effective_weeks": t_eff,
        "se": se,
        "mde_weekly": mde,
        "mde_annualized": mde * 52,
        "mde_pct_weekly": mde * 100,
    }


def sharpe_mde(n_weeks: int, *, alpha: float = 0.05, power: float = 0.8) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z / np.sqrt(max(n_weeks / 52.0, 1e-9)))


def mde_table(n_entities: int = 32) -> pd.DataFrame:
    rows = []
    for weeks in (52, 104, 156, 208):
        for rho in (0.0, 0.2, 0.4):
            r = analytic_mde(n_entities, weeks, 0.05, rho=rho)
            r["years"] = weeks / 52
            rows.append(r)
    return pd.DataFrame(rows)


def simulate_panel(
    n_entities: int,
    n_weeks: int,
    effect: float,
    *,
    residual_sd: float = 0.05,
    rho: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2023-11-03", periods=n_weeks, freq="W-FRI")
    week_fe = rng.normal(0, 0.02, size=n_weeks)
    rows = []
    for i in range(n_entities):
        cid = f"c{i:02d}"
        x = np.zeros(n_weeks)
        e = np.zeros(n_weeks)
        for t in range(1, n_weeks):
            x[t] = rho * x[t - 1] + rng.normal()
            e[t] = rho * e[t - 1] + rng.normal(0, residual_sd)
        y = effect * x + e + week_fe + rng.normal(0, 0.02)
        rows.append(pd.DataFrame({"company_id": cid, "week_end": weeks, "x": x, "y": y}))
    return pd.concat(rows, ignore_index=True)


def monte_carlo_power(
    effects: list[float],
    *,
    n_entities: int = 32,
    n_weeks: int = 156,
    residual_sd: float = 0.05,
    rho: float = 0.1,
    n_sims: int = 200,
    seed: int = 11,
) -> pd.DataFrame:
    rows = []
    for j, effect in enumerate(effects):
        rejections = 0
        valid = 0
        for s in range(n_sims):
            d = simulate_panel(n_entities, n_weeks, effect, residual_sd=residual_sd, rho=rho, seed=seed + 1000 * j + s)
            r = fe_regression(d, "y", "x")
            if np.isfinite(r.get("p", np.nan)):
                valid += 1
                rejections += int(r["p"] < 0.05)
        rows.append({
            "effect_weekly": effect,
            "effect_annualized": effect * 52,
            "power": rejections / valid if valid else np.nan,
            "n_sims": valid,
        })
        log.info("MC power effect=%.3f -> %.2f", effect, rows[-1]["power"])
    return pd.DataFrame(rows)


def actual_panel_power(panel: pd.DataFrame) -> dict:
    d = panel.dropna(subset=["fwd_abn_1w", "rating_z"]).copy()
    if d.empty:
        return {}
    d["_y"] = d["fwd_abn_1w"]
    dd = _two_way_demean(d, ["_y"])
    resid = dd["_y"]
    residual_sd = float(resid.std(ddof=1))
    # within-company AR(1) of the demeaned outcome
    acs = []
    for _, g in resid.groupby(d["company_id"]):
        g = g.dropna()
        if len(g) > 10:
            acs.append(float(g.autocorr(1)))
    rho = float(np.nanmean(acs)) if acs else 0.1
    weeks = d.groupby("company_id")["week_end"].nunique()
    n_entities = int((weeks >= 52).sum())
    n_weeks = int(weeks.median()) if len(weeks) else 0
    if n_entities < 2 or n_weeks < 8:
        return {"n_entities": n_entities, "n_weeks": n_weeks, "residual_sd": residual_sd, "rho": rho}
    r = analytic_mde(n_entities, n_weeks, residual_sd, rho=rho)
    r["sharpe_mde"] = sharpe_mde(n_weeks)
    return r
