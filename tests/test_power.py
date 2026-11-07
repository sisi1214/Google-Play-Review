import numpy as np
import pandas as pd

from prpanel.power import analytic_mde, mde_table, sharpe_mde, simulate_panel


def test_mde_falls_with_more_data():
    a = analytic_mde(32, 52, 0.05)
    b = analytic_mde(32, 156, 0.05)
    c = analytic_mde(64, 156, 0.05)
    assert b["mde_weekly"] < a["mde_weekly"]
    assert c["mde_weekly"] < b["mde_weekly"]


def test_sharpe_mde_falls_with_time():
    assert sharpe_mde(156) < sharpe_mde(52)
    assert 1.0 < sharpe_mde(156) < 2.5


def test_mde_table_shape():
    t = mde_table(32)
    assert len(t) == 12
    assert set(t["rho"]) == {0.0, 0.2, 0.4}


def test_simulate_panel():
    d = simulate_panel(10, 40, 0.0, seed=2)
    assert len(d) == 400
    assert set(d.columns) >= {"company_id", "week_end", "x", "y"}
    assert d.groupby("company_id").size().eq(40).all()
