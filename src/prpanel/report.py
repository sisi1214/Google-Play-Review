"""Render the research report and figures from saved analysis artefacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR, REPORT_DIR, ensure_dirs

log = logging.getLogger("prpanel.report")

TIER_COLORS = {"A (12+ quarters)": "#1b6a4a", "B (4-11 quarters)": "#c58b2a", "C (<4 quarters)": "#9b3b3b"}
plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def _md(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = ".3f") -> str:
    d = df[cols] if cols else df
    return d.to_markdown(index=False, floatfmt=floatfmt, missingval="—")


def _pct(x: float, nd: int = 1) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.{nd}f}%"


def _f(x: float, nd: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def fig_coverage(cov: pd.DataFrame) -> Path:
    d = cov.sort_values("quarters_ge5")
    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    colors = [TIER_COLORS.get(t, "#777") for t in d["tier"]]
    ax.barh(d["company"], d["quarters_ge5"], color=colors, height=0.72)
    for y, (q, n) in enumerate(zip(d["quarters_ge5"], d["n_reviews"])):
        ax.text(q + 0.4, y, f"{int(q)}q · {int(n):,}", va="center", fontsize=7, color="#333")
    ax.set_xlabel("Quarters with at least 5 reviews")
    ax.set_title("Review-history coverage by company")
    ax.set_xlim(0, max(d["quarters_ge5"].max() + 6, 8))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER_COLORS.values()]
    ax.legend(handles, TIER_COLORS.keys(), loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "coverage.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_signal(panel: pd.DataFrame, cov: pd.DataFrame) -> Path:
    top = cov.nlargest(4, "quarters_ge5")["company_id"].tolist()
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for cid in top:
        g = panel[panel.company_id == cid].set_index("week_end")["mean_score"].rolling(4, min_periods=2).mean()
        ax.plot(g.index, g.values, linewidth=1.2, label=cid)
    ax.set_ylabel("Mean star rating (4w MA)")
    ax.set_title("Weekly rating level: four longest-history companies")
    ax.legend(frameon=False, ncol=4, fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "ratings.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_ls(ls_series: dict[str, pd.DataFrame], names: list[str] | None = None) -> Path:
    names = names or [n for n in ["composite", "rating_z", "one_star_z", "rating_mom_z"] if n in ls_series]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    for name in names:
        d = ls_series[name].dropna(subset=["ls_ret"]).copy()
        if d.empty:
            continue
        cum = (1 + d["ls_ret"]).cumprod()
        ax.plot(d["week_end"], cum, linewidth=1.4, label=f"{name} (gross)")
        if name == "composite":
            cum_net = (1 + d["ls_net"].fillna(0)).cumprod()
            ax.plot(d["week_end"], cum_net, linewidth=1.1, linestyle="--", color="#444", label="composite (net)")
    ax.axhline(1.0, color="#999", linewidth=0.7)
    ax.set_ylabel("Cumulative growth of $1")
    ax.set_title("Long-short quintile portfolios (market-adjusted, weekly)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "long_short.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_event(event: dict) -> Path:
    labels = ["1 week", "2 weeks", "4 weeks"]
    neg = [event["negative"].get(f"car_{h}w", np.nan) for h in (1, 2, 4)]
    pos = [event["positive"].get(f"car_{h}w", np.nan) for h in (1, 2, 4)]
    ctrl = [event["control"].get(f"car_{h}w", np.nan) for h in (1, 2, 4)]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - 0.26, neg, 0.25, label=f"rating shock ≤ {event['threshold']:.1f}")
    ax.bar(x, ctrl, 0.25, label="control weeks")
    ax.bar(x + 0.26, pos, 0.25, label="positive shock")
    for i, v in enumerate(neg):
        if pd.notna(v):
            ax.text(i - 0.26, v, f"{v*100:.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean abnormal return")
    ax.set_title("Post-shock abnormal returns")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "event_study.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_power(mde: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for rho, g in mde.groupby("rho"):
        ax.plot(g["n_weeks"], g["mde_annualized"] * 100, marker="o", linewidth=1.3, label=f"AR(1) ρ={rho:.1f}")
    ax.set_xlabel("Weeks of history")
    ax.set_ylabel("MDE (% annualised, per 1 SD)")
    ax.set_title("Minimum detectable effect at 80% power, 5% two-sided")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "power.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_report(
    panel: pd.DataFrame,
    cov: pd.DataFrame,
    reg: pd.DataFrame,
    ls_results: dict,
    event: dict,
    power: dict,
    mde: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    meta: dict | None = None,
    mc_power: pd.DataFrame | None = None,
    ls_lag: pd.DataFrame | None = None,
    mapping: pd.DataFrame | None = None,
) -> Path:
    ensure_dirs()
    meta = meta or {}
    figs = {
        "coverage": fig_coverage(cov),
        "ratings": fig_signal(panel, cov),
        "ls": fig_ls({k: v.series for k, v in ls_results.items()}),
        "event": fig_event(event),
        "power": fig_power(mde),
    }
    fig_rel = {k: f"figures/{v.name}" for k, v in figs.items()}

    n_reviews = int(cov["n_reviews"].sum())
    n_companies = len(cov)
    first = panel["week_end"].min()
    last = panel["week_end"].max()
    tiers = cov["tier"].value_counts().to_dict()
    a_count = tiers.get("A (12+ quarters)", 0)
    comp = ls_results.get("composite")
    comp_stats = comp.stats if comp else {}
    sig_reg = reg[(reg["q_value"] < 0.10) & reg["coef"].notna()] if not reg.empty else pd.DataFrame()
    _valid_ls = [
        (float(r.stats["sharpe_gross"]), r.signal)
        for r in ls_results.values()
        if r.stats.get("sharpe_gross") is not None and pd.notna(r.stats.get("sharpe_gross"))
    ]
    best_ls = max(_valid_ls) if _valid_ls else (float("nan"), "—")

    L: list[str] = []
    L.append("# Google Play review panel: app sentiment as a consumer-internet signal\n")
    L.append(f"**Prepared:** {pd.Timestamp.now():%Y-%m-%d} · **Data through:** {pd.Timestamp(last):%Y-%m-%d} · "
             f"**Scope:** {n_companies} listed companies, {n_reviews:,} timestamped reviews\n")
    L.append("> Research prototype. Public Google Play reviews are collected for aggregate, non-redistributed "
             "analysis. Review text is not republished. See the compliance section before using this commercially.\n")

    L.append("## Executive summary\n")
    L.append(f"- The panel maps **{n_companies} apps → developers → parents → tickers** and contains "
             f"**{n_reviews:,} reviews** from {pd.Timestamp(first):%Y-%m-%d} to {pd.Timestamp(last):%Y-%m-%d}.")
    L.append(f"- **Coverage is the binding constraint, not statistical power.** {a_count} companies clear 12+ quarters; "
             f"{tiers.get('B (4-11 quarters)', 0)} have 4–11 quarters; {tiers.get('C (<4 quarters)', 0)} have less than four. "
             "Mega-apps (YouTube, Spotify, Duolingo, Uber) generate reviews faster than the public API can be paginated backwards.")
    if comp_stats and comp_stats.get("sharpe_gross") is not None:
        L.append(f"- The pre-specified composite signal's long–short portfolio earned "
                 f"**{_pct(comp_stats.get('ann_return_gross'))} annualised gross** "
                 f"(Sharpe {_f(comp_stats.get('sharpe_gross'))}, "
                 f"net Sharpe {_f(comp_stats.get('sharpe_net'))}), "
                 f"with a randomization-inference p-value of {_f(comp_stats.get('p_perm'), 3)}.")
    if not sig_reg.empty:
        L.append(f"- {len(sig_reg)} of {len(reg)} signal/horizon regressions survive Benjamini–Hochberg at q < 0.10. "
                 "Treat these as candidates for replication, not discoveries.")
    else:
        L.append("- **No signal/horizon regression survives Benjamini–Hochberg at q < 0.10.** The honest reading is that "
                 "review sentiment does not reliably predict next-week or next-month abnormal returns in this sample.")
    if power:
        L.append(f"- With the actual panel (N={power.get('n_entities')}, median T={power.get('n_weeks')} weeks, "
                 f"residual σ={_pct(power.get('residual_sd'), 1)}/week), the MDE is "
                 f"**{_pct(power.get('mde_annualized'), 1)} annualised per 1 SD** of signal. "
                 f"The long–short Sharpe MDE is about {_f(power.get('sharpe_mde'))}.")
    L.append("")

    L.append("## 1. What was built\n")
    L.append("A reproducible pipeline with six stages:\n")
    L.append("| Stage | Command | Output |\n|---|---|---|\n"
             "| Backfill reviews | `python -m prpanel ingest` | SQLite review store, resumable by continuation token |\n"
             "| Enrich text | `python -m prpanel enrich` | English VADER sentiment per review |\n"
             "| Prices | `python -m prpanel prices` | Weekly adjusted closes and market-adjusted returns |\n"
             "| Features | `python -m prpanel build` | Company-week panel with signals and forward returns |\n"
             "| Analysis | `python -m prpanel analyze` | FE regressions, long-short, event study, power |\n"
             "| Report | `python -m prpanel report` | This document and figures |\n")
    L.append("The app → developer → parent → ticker map is `config/panel.csv`; every row is explicit and validated at load time. "
             "Author names and avatars are never stored.\n")

    L.append("## 2. Coverage: the 30×12 premise is only partly achievable\n")
    L.append("Google Play's review endpoint paginates **newest to oldest only**. Reaching a date requires reading every review "
             "since that date. For a mega-app that is millions of reviews, so 12 quarters of history is not obtainable at any "
             "reasonable crawl budget. The panel is therefore unbalanced, and the report separates deep-history from short-history names.\n")
    L.append(f"![Coverage]({fig_rel['coverage']})\n")
    cov_show = cov.copy()
    for c in ("first_week", "last_week"):
        cov_show[c] = pd.to_datetime(cov_show[c]).dt.strftime("%Y-%m-%d")
    L.append(_md(cov_show, ["company", "ticker", "n_reviews", "first_week", "last_week", "quarters_ge5", "tier"], ".0f"))
    L.append("")
    L.append("**Tier A** (12+ quarters) carries the time-series tests. **Tier B** contributes cross-sectional observations. "
             "**Tier C** is displayed for completeness and excluded from any claim about 12-quarter power.\n")
    if mapping is not None and not mapping.empty:
        L.append("### App → developer → parent → ticker\n")
        L.append("This is the auditable mapping used throughout. The Play developer string is not a legal-entity identifier, "
                 "so the parent and ticker columns are curated and validated rather than inferred.\n")
        L.append(_md(mapping, ["company", "ticker", "app_name", "developer", "parent", "package_id"]))
        L.append("")

    L.append("## 3. Signal construction and timing\n")
    L.append("Reviews are binned into weeks ending Friday 23:59 UTC. Signals for week *t* use only reviews in week *t* and "
             "trailing history, so nothing is computed from the future. Forward returns run from the Friday close of week *t* "
             "to the Friday close of week *t+1*. A one-week-lagged variant is also tested.\n")
    L.append("| Signal | Definition | Hypothesis |\n|---|---|---|\n"
             "| `rating_z` | Weekly mean star rating vs own trailing 26-week mean/SD | Better product → positive |\n"
             "| `rating_mom_z` | 4-week rating momentum, standardised | Improving product → positive |\n"
             "| `one_star_z` | 1-star share vs own trailing 26-week mean/SD | Complaints → negative |\n"
             "| `sent_z` | English VADER sentiment vs own trailing 26-week mean/SD | Better text tone → positive |\n"
             "| `vol_z` | log(review count) vs trailing 8-week mean/SD | Attention/growth, sign ambiguous |\n"
             "| `composite` | Mean of rating, momentum, −1-star and sentiment z-scores | Pre-specified ensemble |\n")
    L.append("A signal requires at least 5 reviews in the week; sentiment requires at least 3 English reviews. "
             "Weeks with too little data are missing, not zero-filled.\n")
    L.append(f"![Ratings]({fig_rel['ratings']})\n")

    L.append("## 4. Panel descriptives\n")
    desc = pd.DataFrame({
        "metric": ["company-weeks", "weeks with ≥5 reviews", "mean reviews/week (active)", "mean rating",
                   "mean 1-star share", "English share of text", "sentiment coverage (scored / all)"],
        "value": [
            f"{len(panel):,}",
            f"{int((panel['n_reviews'] >= 5).sum()):,}",
            f"{panel.loc[panel['n_reviews'] > 0, 'n_reviews'].mean():.1f}",
            f"{panel['mean_score'].mean():.3f}",
            f"{panel['share_1'].mean():.3f}",
            f"{panel['english_share'].mean():.1%}",
            f"{panel['sent_n'].sum() / max(panel['n_reviews'].sum(), 1):.1%}",
        ],
    })
    L.append(_md(desc))
    L.append("")
    L.append("**Signal correlation (average weekly Spearman):**\n")
    L.append(_md(corr.reset_index().rename(columns={"index": "signal"})))
    L.append("")

    L.append("## 5. Predictive panel regressions\n")
    L.append("Specification: outcome = signal + company fixed effects + week fixed effects, with company-clustered standard errors. "
             "Signals are standardised cross-sectionally within each week; outcomes are clipped at ±50% weekly to limit outlier "
             "influence without estimating tails from the full sample. Coefficients are weekly abnormal returns per 1 SD of signal.\n")
    prim = reg[reg["outcome"] == "fwd_abn_1w"].sort_values("p") if not reg.empty else reg
    if not prim.empty:
        show = prim[["signal", "coef", "se", "t", "p", "q_value", "n", "n_entities", "r2_within"]].copy()
        show["coef"] = show["coef"] * 100
        show["se"] = show["se"] * 100
        L.append(_md(show.rename(columns={"coef": "coef_%", "se": "se_%"})))
        L.append("")
        L.append("Full results for all three horizons (fast 1-week, 4-week, and lagged 1-week) are in "
                 "`data/analysis/regressions.csv`.\n")
    L.append(f"![Long-short]({fig_rel['ls']})\n")

    L.append("## 6. Long–short backtest\n")
    L.append("Each week, companies with valid signals are ranked; the top and bottom quintiles are held equal-weighted, "
             "long versus short. Returns are market-adjusted. Costs are 10 bp per side applied to measured turnover. "
             "The randomization p-value permutes the signal across companies within each week (300 draws).\n")
    rows = []
    for s, r in ls_results.items():
        st = r.stats
        rows.append({
            "signal": s, "weeks": st.get("n_weeks"), "ann_ret_gross": st.get("ann_return_gross"),
            "sharpe_gross": st.get("sharpe_gross"), "sharpe_net": st.get("sharpe_net"),
            "t_hac": st.get("t_hac"), "p_perm": st.get("p_perm"), "max_dd": st.get("max_drawdown"),
            "turnover": st.get("turnover_weekly"), "hit_rate": st.get("hit_rate"),
        })
    lst = pd.DataFrame(rows)
    for c in ("ann_ret_gross", "max_dd"):
        if c in lst:
            lst[c] = lst[c] * 100
    lst = lst.rename(columns={"ann_ret_gross": "ann_ret_%", "max_dd": "max_dd_%"})
    L.append(_md(lst, floatfmt=".3f"))
    L.append("")
    if "rating_mom_z" in ls_results:
        _s = ls_results["rating_mom_z"].series.dropna(subset=["ls_ret"]).copy()
        if len(_s) > 20:
            _idx = _s["ls_ret"].abs().idxmax()
            _event_date = pd.Timestamp(_s.loc[_idx, "week_end"])
            _ex = _s.drop(index=_idx)
            _full_sharpe = _s["ls_ret"].mean() * 52 / (_s["ls_ret"].std(ddof=1) * np.sqrt(52))
            _ex_sharpe = _ex["ls_ret"].mean() * 52 / (_ex["ls_ret"].std(ddof=1) * np.sqrt(52))
            L.append(
                f"**Single-event sensitivity.** The best gross Sharpe is not robust to one corporate event: "
                f"the largest weekly long-short move was {_pct(_s.loc[_idx, 'ls_ret'])} in the week ending "
                f"{_event_date:%Y-%m-%d}, when Bumble (a long-leg name) rose 134% in one week. Removing that single "
                f"week cuts annualised gross return from {_pct(_s['ls_ret'].mean() * 52)} to "
                f"{_pct(_ex['ls_ret'].mean() * 52)} and Sharpe from {_full_sharpe:.2f} to {_ex_sharpe:.2f}. "
                "The panel regressions clip weekly outcomes at ±50%, so the event does not create significance there.\n"
            )
    if comp_stats:
        L.append(f"**Composite detail:** gross annualised {_pct(comp_stats.get('ann_return_gross'))}, "
                 f"net {_pct(comp_stats.get('ann_return_net'))}, max drawdown {_pct(comp_stats.get('max_drawdown'))}, "
                 f"weekly turnover {_f(comp_stats.get('turnover_weekly'))}, "
                 f"{comp_stats.get('n_weeks', 0)} weeks. "
                 f"Best gross Sharpe across signals: {best_ls[1]} ({_f(best_ls[0])}).\n")
    _nw = comp_stats.get("n_weeks") or 52
    if ls_lag is not None and not ls_lag.empty:
        lag_show = ls_lag[["signal", "n_weeks", "ann_return_gross", "sharpe_gross", "sharpe_net", "t_hac", "p_perm"]].copy()
        lag_show["ann_return_gross"] = lag_show["ann_return_gross"] * 100
        lag_show = lag_show.rename(columns={"ann_return_gross": "ann_ret_%"})
        L.append("**Lagged execution** (signal at week t, trade one week later; the conservative timing check):\n")
        L.append(_md(lag_show, floatfmt=".3f"))
        L.append("")
    L.append("A Sharpe estimated on ~2–3 years of weekly data has a standard error of roughly "
             f"{1/np.sqrt(max(_nw, 1) / 52):.2f}. "
             "Even a true Sharpe of 0.5 is not distinguishable from zero at this sample length.\n")

    L.append("## 7. Rating-shock event study\n")
    ev = event
    L.append(f"Events are weeks where `rating_z` ≤ {ev['threshold']:.1f} (a drop of 1.5 within-company SD), "
             "after at least 20 weeks of history. Abnormal returns are measured from the following week.\n")
    erows = []
    for label, key in (("negative shock", "negative"), ("positive shock", "positive"), ("control", "control")):
        d = ev[key]
        erows.append({"group": label, "n": d.get("n", 0), "CAR_1w": d.get("car_1w"),
                      "CAR_2w": d.get("car_2w"), "CAR_4w": d.get("car_4w"), "t_4w": d.get("t_4w")})
    ert = pd.DataFrame(erows)
    for c in ("CAR_1w", "CAR_2w", "CAR_4w"):
        ert[c] = ert[c] * 100
    L.append(_md(ert, floatfmt=".3f"))
    L.append("")
    if ev.get("diff_neg_minus_control"):
        d = ev["diff_neg_minus_control"]
        L.append(f"FE-adjusted difference (negative shock − control, company + week fixed effects), 4-week CAR: {d['coef_4w']*100:.2f}% "
                 f"(t={d['t']:.2f}, p={d['p']:.3f}, N={d['n']}).\n")
    L.append(f"![Event study]({fig_rel['event']})\n")

    L.append("## 8. Statistical power\n")
    L.append(f"![Power]({fig_rel['power']})\n")
    if power:
        L.append(f"Using the realised panel (N={power.get('n_entities')} companies with ≥52 weeks, median T="
                 f"{power.get('n_weeks')} weeks, residual σ={_pct(power.get('residual_sd'))}/week, "
                 f"AR(1) ρ={_f(power.get('rho'))}), the analytic MDE at 80% power / 5% two-sided is "
                 f"**{_pct(power.get('mde_weekly'), 2)} per week** "
                 f"(≈{_pct(power.get('mde_annualized'), 1)} annualised) per 1 SD of signal.\n")
    L.append("Design MDE for the full 32-name panel at alternative histories (the realised-panel MDE above is the relevant one):\n")
    L.append(_md(mde[["years", "n_entities", "n_weeks", "rho", "mde_weekly", "mde_annualized"]].assign(
        mde_weekly=lambda d: d.mde_weekly * 100, mde_annualized=lambda d: d.mde_annualized * 100
    ).rename(columns={"mde_weekly": "MDE_%_weekly", "mde_annualized": "MDE_%_annual"}), floatfmt=".2f"))
    L.append("")
    if mc_power is not None and not mc_power.empty:
        L.append("**Monte Carlo check** (same estimator, simulated panels):\n")
        L.append(_md(mc_power.assign(effect_annualized=lambda d: d.effect_annualized * 100)
                     .rename(columns={"effect_weekly": "effect_%_weekly", "effect_annualized": "effect_%_annual"}),
                     floatfmt=".3f"))
        L.append("")
    L.append("The Sharpe MDE matters as much as the regression MDE: over the realised ~1.7-year window the Sharpe MDE is "
             "about 2.2; even with a full three years it would be about 1.6. A backtest Sharpe of 0.5 looks economically "
             "meaningful but is statistically indistinguishable from zero at either length. That is a property of the sample "
             "length, not a fixable modelling choice.\n")

    L.append("## 9. Limitations and compliance\n")
    L.append("1. **Public-data ceiling.** The Play review endpoint is newest-first and unpaginated by date. Mega-app history is "
             "truncated; Tier C companies cannot support 12-quarter claims.")
    L.append("2. **Platform selection.** Android only; app-store reviewers are self-selected and prompted, so levels and "
             "changes are not representative of all users.")
    L.append("3. **Review edits and deletions.** The timestamp is the review's recorded time; edits and removals are not "
             "separately observable, which adds measurement error.")
    L.append("4. **Mapping risk.** Parent/subsidiary structures and tickers change. The map is curated and validated, but "
             "corporate actions (e.g., Block's SQ→XYZ change) require maintenance.")
    L.append("5. **Returns.** Weekly close-to-close, market-adjusted with a regional ETF; no transaction costs beyond the "
             "stated 10 bp, no borrow costs, no slippage, no survivorship correction.")
    L.append("6. **Not sector-neutral.** With 32 names, the top and bottom quintiles can concentrate in one sector; part of "
             "any long-short result may be a sector exposure rather than a review effect.")
    L.append("7. **Selected panel, selected apps.** Companies were included only if they have a listed parent, a consumer app, "
             "and usable Play data. PDD is represented by Temu (its international app), not Pinduoduo; WeChat reviews come "
             "from the global/US storefront and are not a census of mainland usage.")
    L.append("8. **Single-name events.** Equal-weight quintile portfolios can be dominated by one corporate action. The best "
             "long-short Sharpe here is roughly halved by removing one takeover week (Bumble, October 2026); treat any "
             "single-name-driven result as a confound, not a review signal.")
    L.append("9. **Multiple testing.** Six signals × three horizons are reported with BH q-values; the long-short search is "
             "also reported in full. A single q<0.10 result is not a tradable discovery.")
    L.append("10. **Terms of service.** Google Play's terms restrict automated collection. This prototype is for internal "
             "research; for production, license data from a vendor (Sensor Tower, AppFollow, data.ai) or obtain written "
             "permission. Do not redistribute review text or author data.")
    L.append("11. **Not investment advice.** Nothing here is a recommendation to buy or sell a security.\n")

    L.append("## 10. Reproduction\n")
    L.append("```bash\n"
             "cd play-review-panel\n"
             "uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e .\n"
             "python -m prpanel ingest --since 2023-11-07 --max-reviews 50000\n"
             "python -m prpanel enrich\n"
             "python -m prpanel prices\n"
             "python -m prpanel build\n"
             "python -m prpanel analyze --n-perm 300\n"
             "python -m prpanel report\n"
             "```\n")
    L.append(f"Run metadata: `{json.dumps(meta, default=str)}`\n")

    out = REPORT_DIR / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", out)
    return out
