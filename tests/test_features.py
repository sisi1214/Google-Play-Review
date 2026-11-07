import numpy as np
import pandas as pd
import pytest

from prpanel.backtest import cs_z
from prpanel.features import add_forward_returns, add_signals, aggregate_weekly, coverage_table


def make_reviews(n_weeks=60, companies=("a", "b"), per_week=10):
    rows = []
    start = pd.Timestamp("2024-01-05", tz="UTC")  # Friday
    for ci, cid in enumerate(companies):
        for w in range(n_weeks):
            week_start = start + pd.Timedelta(weeks=w)
            for j in range(per_week):
                score = 3 + ((w + j) % 3)
                rows.append({
                    "company_id": cid,
                    "package_id": f"pkg.{cid}",
                    "at_utc": week_start + pd.Timedelta(hours=j),
                    "score": score,
                    "thumbs_up": j,
                    "replied": 0.0,
                    "sentiment": 0.1 if score >= 4 else -0.4,
                    "is_english": 1,
                })
    return pd.DataFrame(rows)


def test_weekly_aggregation_and_friday_bins():
    weekly = aggregate_weekly(make_reviews(10))
    assert set(weekly["company_id"]) == {"a", "b"}
    assert len(weekly) == 20
    assert weekly["week_end"].dt.day_of_week.eq(4).all()
    assert weekly["n_reviews"].eq(10).all()
    assert weekly["mean_score"].between(1, 5).all()


def test_signals_use_only_trailing_data():
    weekly = aggregate_weekly(make_reviews(60))
    panel = add_signals(weekly)
    # First 8 weeks cannot have a 26-week trailing window.
    early = panel.sort_values(["company_id", "week_end"]).groupby("company_id").head(7)
    assert early["rating_z"].isna().all()
    assert panel["rating_z"].notna().sum() > 0
    # Hand-check the trailing window for one company/week.
    g = panel[panel.company_id == "a"].sort_values("week_end").reset_index(drop=True)
    t = 40
    prior = g.loc[t - 26:t - 1, "rating_obs"]
    expected = (g.loc[t, "rating_obs"] - prior.mean()) / prior.std(ddof=1)
    assert g.loc[t, "rating_z"] == pytest.approx(expected)


def test_forward_returns_offsets():
    weeks = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    panel = pd.DataFrame({
        "company_id": "a",
        "week_end": weeks,
        "ret": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "abn_ret": [0.00, 0.01, 0.02, 0.03, 0.04, 0.05],
    })
    out = add_forward_returns(panel)
    assert out.loc[0, "fwd_ret_1w"] == pytest.approx(0.02)
    assert out.loc[0, "fwd_ret_2w"] == pytest.approx(1.02 * 1.03 - 1)
    assert out.loc[0, "fwd_ret_1w_lag"] == pytest.approx(0.03)
    assert out.loc[0, "fwd_ret_2w_lag"] == pytest.approx(1.03 * 1.04 - 1)
    # The analysis layer relies on these exact column names.
    assert "fwd_abn_1w" in out.columns and "fwd_abn_4w_lag" in out.columns
    assert out.loc[0, "fwd_abn_1w"] == pytest.approx(0.01)


def test_cross_sectional_z():
    panel = pd.DataFrame({
        "week_end": pd.to_datetime(["2024-01-05"] * 4),
        "x": [1.0, 2.0, 3.0, 4.0],
    })
    z = cs_z(panel, "x")
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std(ddof=0) == pytest.approx(1.0)
