"""Panel predictive regressions, long-short backtests and event studies.

All inference is deliberately conservative:
- two-way (company and week) fixed effects absorb level and calendar effects;
- standard errors are clustered by company;
- the long-short result gets a randomization-inference p-value that permutes
  signals across companies within each week;
- every signal/horizon pair is reported, with Benjamini-Hochberg q-values,
  rather than only the best one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("prpanel.backtest")

SIGNALS = ["rating_z", "rating_mom_z", "one_star_z", "sent_z", "vol_z", "composite"]
OUTCOMES = ["fwd_abn_1w", "fwd_abn_4w", "fwd_abn_1w_lag"]


def cs_z(panel: pd.DataFrame, col: str, clip: float = 4.0) -> pd.Series:
    g = panel.groupby("week_end")[col]
    mu = g.transform("mean")
    sd = g.transform(lambda x: x.std(ddof=0))
    z = (panel[col] - mu) / sd.replace(0, np.nan)
    return z.clip(-clip, clip)


def _two_way_demean(
    df: pd.DataFrame, cols: list[str], entity: str = "company_id", time: str = "week_end",
    tol: float = 1e-9, max_iter: int = 200,
) -> pd.DataFrame:
    x = df[cols].astype(float).copy()
    for _ in range(max_iter):
        prev = x.to_numpy(copy=True)
        x = x - x.groupby(df[entity]).transform("mean")
        x = x - x.groupby(df[time]).transform("mean")
        if np.nanmax(np.abs(x.to_numpy() - prev)) < tol:
            break
    return x


def fe_regression(panel: pd.DataFrame, y: str, x: str, *, cluster: str = "company_id") -> dict:
    cols = list(dict.fromkeys([y, x, "company_id", "week_end", cluster]))
    d = panel[cols].dropna().copy()
    if len(d) < 30 or d["company_id"].nunique() < 3 or d["week_end"].nunique() < 4:
        return {"n": len(d), "coef": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n_entities": d["company_id"].nunique(),
                "r2_within": np.nan, "error": "insufficient data"}
    d["_cluster"] = d[cluster]
    dd = _two_way_demean(d, [y, x])
    yv = dd[y].to_numpy()
    xv = dd[x].to_numpy()
    X = sm.add_constant(xv, has_constant="add")
    groups = d["_cluster"].to_numpy()
    model = sm.OLS(yv, X)
    res = model.fit()
    try:
        robust = res.get_robustcov_results(cov_type="cluster", groups=groups, use_correction=True)
        se = float(robust.bse[1])
        t = float(robust.tvalues[1])
        p = float(robust.pvalues[1])
        ci = robust.conf_int()[1]
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("cluster SE failed for %s/%s: %s", y, x, exc)
        se, t, p = float(res.bse[1]), float(res.tvalues[1]), float(res.pvalues[1])
        ci = res.conf_int()[1]
    ss_res = float(np.sum(res.resid ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    return {
        "n": int(len(d)),
        "n_entities": int(d["company_id"].nunique()),
        "n_weeks": int(d["week_end"].nunique()),
        "coef": float(res.params[1]),
        "se": se,
        "t": t,
        "p": p,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "r2_within": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def regression_table(panel: pd.DataFrame, signals: list[str] | None = None, outcomes: list[str] | None = None) -> pd.DataFrame:
    signals = signals or SIGNALS
    outcomes = outcomes or OUTCOMES
    p = panel.copy()
    for s in signals:
        p[f"_{s}_cs"] = cs_z(p, s)
    rows = []
    for s in signals:
        for o in outcomes:
            if o not in p.columns:
                continue
            # Fixed clip, not a full-sample quantile: avoids any look-ahead in
            # how the tails are treated.
            p[o + "_w"] = p[o].clip(-0.5, 0.5)
            r = fe_regression(p, o + "_w", f"_{s}_cs")
            r.update({"signal": s, "outcome": o})
            rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        ok = out["p"].notna()
        out["q_value"] = np.nan
        if ok.any():
            out.loc[ok, "q_value"] = multipletests(out.loc[ok, "p"], method="fdr_bh")[1]
        out["sign"] = np.sign(out["coef"])
    return out


@dataclass
class LSResult:
    signal: str
    outcome: str
    series: pd.DataFrame
    stats: dict
    null_sharpes: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def _ls_weights(panel: pd.DataFrame, signal: str, outcome: str, quantiles: int, min_names: int) -> pd.DataFrame:
    d = panel[["company_id", "week_end", signal, outcome]].dropna().copy()
    rows = []
    for week, g in d.groupby("week_end"):
        if len(g) < min_names:
            continue
        g = g.copy()
        g["rk"] = g[signal].rank(method="first", pct=True)
        k_long = max(1, int(np.ceil(len(g) / quantiles)))
        longs = set(g.nlargest(k_long, signal)["company_id"])
        shorts = set(g.nsmallest(k_long, signal)["company_id"])
        if not longs or not shorts:
            continue
        w = {}
        for cid in longs:
            w[cid] = 1.0 / len(longs)
        for cid in shorts:
            w[cid] = -1.0 / len(shorts)
        for cid, wi in w.items():
            rows.append({"week_end": week, "company_id": cid, "w": wi, "outcome": float(g.loc[g.company_id == cid, outcome].iloc[0])})
    if not rows:
        return pd.DataFrame()
    w = pd.DataFrame(rows)
    w["contrib"] = w["w"] * w["outcome"]
    ls = w.groupby("week_end").agg(ls_ret=("contrib", "sum"), n_names=("company_id", "nunique")).reset_index()
    return ls.sort_values("week_end").reset_index(drop=True)


def _turnover(weights: pd.DataFrame) -> float:
    if weights.empty:
        return np.nan
    piv = weights.pivot_table(index="week_end", columns="company_id", values="w", fill_value=0.0).sort_index()
    if len(piv) < 2:
        return np.nan
    delta = piv.diff().abs().sum(axis=1).iloc[1:]
    return float(delta.mean() / 2.0)


def _max_drawdown(ls: pd.Series) -> float:
    if ls.empty:
        return np.nan
    cum = (1.0 + ls.fillna(0.0)).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _hac_t(series: pd.Series, maxlags: int = 4) -> float:
    s = series.dropna()
    if len(s) < 8:
        return np.nan
    res = sm.OLS(s.to_numpy(), np.ones((len(s), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.tvalues[0])


def long_short(
    panel: pd.DataFrame,
    signal: str,
    outcome: str = "fwd_abn_1w",
    *,
    quantiles: int = 5,
    min_names: int = 12,
    cost_bps: float = 10.0,
    n_perm: int = 300,
    seed: int = 7,
) -> LSResult:
    w = _ls_weights(panel, signal, outcome, quantiles, min_names)
    if w.empty or len(w) < 20:
        return LSResult(signal, outcome, pd.DataFrame(), {"error": "insufficient weeks", "n_weeks": len(w)})
    ls = w.set_index("week_end")["ls_ret"]
    # Turnover from the actual weekly weights.
    d = panel[["company_id", "week_end", signal]].dropna().copy()
    wrows = []
    for week, g in d.groupby("week_end"):
        if len(g) < min_names:
            continue
        k_long = max(1, int(np.ceil(len(g) / quantiles)))
        longs = g.nlargest(k_long, signal)["company_id"].tolist()
        shorts = g.nsmallest(k_long, signal)["company_id"].tolist()
        for cid in longs:
            wrows.append({"week_end": week, "company_id": cid, "w": 1.0 / len(longs)})
        for cid in shorts:
            wrows.append({"week_end": week, "company_id": cid, "w": -1.0 / len(shorts)})
    turnover = _turnover(pd.DataFrame(wrows))

    ann_ret = float(ls.mean() * 52)
    ann_vol = float(ls.std(ddof=1) * np.sqrt(52))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    tstat = _hac_t(ls)
    gross = ls.copy()
    cost = turnover * 2.0 * cost_bps / 1e4 if pd.notna(turnover) else 0.0
    net = gross - cost
    net_sharpe = float(net.mean() * 52 / (net.std(ddof=1) * np.sqrt(52))) if net.std(ddof=1) > 0 else np.nan

    bench = panel.dropna(subset=[outcome]).groupby("week_end")[outcome].mean()
    bench = bench.reindex(ls.index).dropna()
    excess = (ls - bench).dropna()
    excess_sharpe = float(excess.mean() * 52 / (excess.std(ddof=1) * np.sqrt(52))) if excess.std(ddof=1) > 0 else np.nan

    null_sharpes = None
    p_perm = np.nan
    if n_perm and len(ls) >= 20:
        rng = np.random.default_rng(seed)
        null = []
        base = panel[["week_end", "company_id", signal, outcome]].dropna()
        for _ in range(n_perm):
            perm = base.copy()
            perm[signal] = perm.groupby("week_end")[signal].transform(lambda s: rng.permutation(s.to_numpy()))
            wp = _ls_weights(perm, signal, outcome, quantiles, min_names)
            if wp.empty or len(wp) < 20:
                continue
            r = wp.set_index("week_end")["ls_ret"]
            v = r.std(ddof=1)
            null.append(float(r.mean() * 52 / (v * np.sqrt(52))) if v > 0 else 0.0)
        null_sharpes = np.asarray(null)
        if len(null_sharpes):
            p_perm = float((1 + np.sum(np.abs(null_sharpes) >= abs(sharpe))) / (1 + len(null_sharpes)))

    stats = {
        "n_weeks": int(len(ls)),
        "ann_return_gross": ann_ret,
        "ann_vol": ann_vol,
        "sharpe_gross": sharpe,
        "sharpe_net": net_sharpe,
        "t_hac": tstat,
        "p_perm": p_perm,
        "max_drawdown": _max_drawdown(gross),
        "hit_rate": float((gross > 0).mean()),
        "turnover_weekly": turnover,
        "cost_drag_ann": float(cost * 52),
        "ann_return_net": float(net.mean() * 52),
        "bench_ann_return": float(bench.mean() * 52) if len(bench) else np.nan,
        "excess_sharpe": excess_sharpe,
        "avg_names": float(w["n_names"].mean()),
        "first_week": ls.index.min(),
        "last_week": ls.index.max(),
    }
    out = pd.DataFrame({"ls_ret": ls, "ls_net": net, "bench": bench.reindex(ls.index)})
    out.index.name = "week_end"
    return LSResult(signal, outcome, out.reset_index(), stats, null_sharpes)


def mean_test(series: pd.Series, groups: pd.Series) -> tuple[float, float]:
    d = pd.DataFrame({"y": series, "g": groups}).dropna()
    if len(d) < 5 or d["g"].nunique() < 2:
        return np.nan, np.nan
    res = sm.OLS(d["y"].to_numpy(), np.ones((len(d), 1))).fit(
        cov_type="cluster", cov_kwds={"groups": d["g"].to_numpy()}, use_correction=True
    )
    return float(res.tvalues[0]), float(res.pvalues[0])


def event_study(panel: pd.DataFrame, *, threshold: float = -1.5, min_prior_weeks: int = 20) -> dict:
    d = panel.dropna(subset=["rating_z", "fwd_abn_1w", "fwd_abn_4w"]).copy()
    d["n_prior"] = d.groupby("company_id").cumcount()
    d = d[d["n_prior"] >= min_prior_weeks]
    neg = d[d["rating_z"] <= threshold]
    pos = d[d["rating_z"] >= -threshold]
    control = d[(d["rating_z"] > threshold) & (d["rating_z"] < -threshold)]

    def mean_car(g: pd.DataFrame) -> dict:
        out = {"n": len(g)}
        for h in (1, 2, 4):
            out[f"car_{h}w"] = float(g[f"fwd_abn_{h}w"].mean()) if len(g) else np.nan
        out["t_4w"], out["p_4w"] = mean_test(g["fwd_abn_4w"], g["company_id"])
        return out

    neg_stats = mean_car(neg)
    pos_stats = mean_car(pos)
    ctrl_stats = mean_car(control)
    # Difference vs control, clustered by company.
    diff = {}
    if len(neg) >= 5 and len(control) >= 5:
        pooled = pd.concat([neg.assign(_shock=1.0), control.assign(_shock=0.0)], ignore_index=True)
        pooled["_y"] = pooled["fwd_abn_4w"].clip(-0.5, 0.5)
        r = fe_regression(pooled, "_y", "_shock", cluster="company_id")
        diff = {"coef_4w": r["coef"], "t": r["t"], "p": r["p"], "n": r["n"]}
    return {
        "threshold": threshold,
        "negative": neg_stats,
        "positive": pos_stats,
        "control": ctrl_stats,
        "diff_neg_minus_control": diff,
        "events": neg[["company_id", "week_end", "rating_z", "fwd_abn_1w", "fwd_abn_4w"]],
    }


def signal_correlations(panel: pd.DataFrame, signals: list[str] | None = None) -> pd.DataFrame:
    signals = signals or SIGNALS
    d = panel[signals + ["week_end"]].copy()
    out = pd.DataFrame(index=signals, columns=signals, dtype=float)
    for a in signals:
        for b in signals:
            if a == b:
                out.loc[a, b] = 1.0
                continue
            rs = []
            for _, g in d[["week_end", a, b]].dropna().groupby("week_end"):
                if len(g) >= 8:
                    rs.append(g[a].corr(g[b], method="spearman"))
            out.loc[a, b] = float(np.nanmean(rs)) if rs else np.nan
    return out
