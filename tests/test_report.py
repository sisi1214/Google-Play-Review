from types import SimpleNamespace

import pandas as pd

from prpanel import report as report_mod


def test_report_generates(tmp_path, monkeypatch):
    monkeypatch.setattr(report_mod, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(report_mod, "FIG_DIR", tmp_path / "figures")
    (tmp_path / "figures").mkdir()

    weeks = pd.date_range("2024-01-05", periods=30, freq="W-FRI")
    panel = pd.DataFrame({
        "company_id": ["a"] * 30 + ["b"] * 30,
        "week_end": list(weeks) * 2,
        "n_reviews": 20,
        "mean_score": 4.2,
        "share_1": 0.1,
        "english_share": 0.8,
        "sent_n": 10,
        "rating_z": 0.1,
        "composite": 0.1,
        "fwd_abn_1w": 0.01,
        "fwd_abn_4w": 0.02,
        "fwd_abn_2w": 0.015,
    })
    cov = pd.DataFrame({
        "company_id": ["a", "b"], "company": ["A", "B"], "ticker": ["AAA", "BBB"],
        "n_reviews": [600, 600], "first_week": [weeks[0]] * 2, "last_week": [weeks[-1]] * 2,
        "quarters_ge5": [2, 2], "tier": ["C (<4 quarters)"] * 2,
    })
    reg = pd.DataFrame({
        "signal": ["composite"], "outcome": ["fwd_abn_1w"], "coef": [0.01], "se": [0.01],
        "t": [1.0], "p": [0.3], "q_value": [0.3], "n": [60], "n_entities": [2], "r2_within": [0.01],
    })
    series = pd.DataFrame({"week_end": weeks, "ls_ret": 0.001, "ls_net": 0.0, "bench": 0.0})
    ls = {"composite": SimpleNamespace(signal="composite", series=series,
                                       stats={"sharpe_gross": 0.5, "sharpe_net": 0.4, "p_perm": 0.4,
                                              "n_weeks": 30, "ann_return_gross": 0.02, "turnover_weekly": 0.5,
                                              "max_drawdown": -0.1})}
    ev = {"threshold": -1.5, "negative": {"n": 0, "car_1w": 0.0, "car_2w": 0.0, "car_4w": 0.0, "t_4w": 0.0},
          "positive": {"n": 0, "car_1w": 0.0, "car_2w": 0.0, "car_4w": 0.0, "t_4w": 0.0},
          "control": {"n": 0, "car_1w": 0.0, "car_2w": 0.0, "car_4w": 0.0, "t_4w": 0.0},
          "diff_neg_minus_control": {}}
    power = {"n_entities": 2, "n_weeks": 30, "residual_sd": 0.05, "rho": 0.1,
             "mde_weekly": 0.01, "mde_annualized": 0.5, "sharpe_mde": 2.0}
    mde = pd.DataFrame({"years": [1.0], "n_entities": [2], "n_weeks": [52], "rho": [0.1],
                        "mde_weekly": [0.01], "mde_annualized": [0.5]})
    corr = pd.DataFrame({"composite": [1.0]}, index=["composite"])
    mapping = pd.DataFrame({"company": ["A"], "ticker": ["AAA"], "app_name": ["App"],
                            "developer": ["Dev"], "parent": ["A"], "package_id": ["pkg.a"], "sector": ["S"]})
    out = report_mod.generate_report(panel, cov, reg, ls, ev, power, mde, corr, mapping=mapping)
    text = out.read_text()
    assert "Google Play review panel" in text
    assert "App → developer → parent → ticker" in text
    assert (tmp_path / "figures" / "coverage.png").exists()
