import numpy as np
import pandas as pd
import pytest

from prpanel.backtest import event_study, fe_regression, long_short, mean_test, regression_table


def synthetic_panel(n_companies=20, n_weeks=80, effect=0.05, seed=3):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2023-01-06", periods=n_weeks, freq="W-FRI")
    rows = []
    for i in range(n_companies):
        x = rng.normal(size=n_weeks)
        y = effect * x + rng.normal(0, 0.02, n_weeks)
        rows.append(pd.DataFrame({
            "company_id": f"c{i:02d}", "week_end": weeks, "signal": x,
            "fwd_abn_1w": y, "fwd_abn_2w": y, "fwd_abn_4w": y, "fwd_abn_1w_lag": y,
            "rating_z": x, "rating_mom_z": x, "one_star_z": x,
            "sent_z": x, "vol_z": x, "composite": x,
            "n_reviews": 20,
        }))
    return pd.concat(rows, ignore_index=True)


def test_fe_regression_recovers_effect():
    panel = synthetic_panel()
    r = fe_regression(panel, "fwd_abn_1w", "signal")
    assert 0.035 < r["coef"] < 0.065
    assert r["n"] == len(panel)
    assert r["p"] < 0.01


def test_regression_table_has_fdr():
    panel = synthetic_panel()
    table = regression_table(panel, signals=["rating_z", "sent_z"], outcomes=["fwd_abn_1w"])
    assert len(table) == 2
    assert table["q_value"].between(0, 1).all()


def test_long_short_positive_when_signal_predicts():
    panel = synthetic_panel(effect=0.08)
    r = long_short(panel, "signal", "fwd_abn_1w", min_names=10, n_perm=10, seed=1)
    assert r.stats["sharpe_gross"] > 0
    assert r.stats["n_weeks"] > 40
    assert "p_perm" in r.stats


def test_event_study_runs():
    panel = synthetic_panel()
    ev = event_study(panel, threshold=-0.5)
    assert "negative" in ev and "control" in ev
    assert ev["negative"]["n"] > 0


def test_mean_test():
    t, p = mean_test(pd.Series([0.01] * 30), pd.Series(["a", "b", "c"] * 10))
    assert np.isfinite(t)
    assert 0 <= p <= 1
