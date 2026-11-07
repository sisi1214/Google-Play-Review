import pandas as pd
import pytest

from prpanel.config import load_panel


def test_panel_loads_and_is_unique():
    rows = load_panel()
    assert 25 <= len(rows) <= 40
    assert len({r.company_id for r in rows}) == len(rows)
    assert len({r.ticker for r in rows}) == len(rows)
    assert len({r.package_id for r in rows}) == len(rows)
    assert all(r.benchmark_symbol for r in rows)


def test_named_mappings_are_explicit():
    by_id = {r.company_id: r for r in load_panel()}
    assert by_id["grab"].parent == "Grab Holdings Limited"
    assert by_id["grab"].ticker == "GRAB"
    assert by_id["pdd"].package_id == "com.einnovation.temu"
    assert by_id["pdd"].parent == "PDD Holdings Inc."
    assert by_id["tencent"].ticker == "0700.HK"
    assert by_id["sea"].package_id == "com.shopee.sg"


def test_duplicate_package_rejected(tmp_path):
    from prpanel.config import PANEL_CSV

    df = pd.read_csv(PANEL_CSV)
    df.loc[1, "package_id"] = df.loc[0, "package_id"]
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_panel(path)
