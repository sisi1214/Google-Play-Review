"""Panel definition and project paths.

The panel is intentionally a plain CSV so that the app -> developer -> parent ->
ticker chain is auditable by a human. Nothing here may be inferred at runtime:
every mapping is explicit and validated before ingestion.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
LOG_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "reviews.sqlite"
PANEL_CSV = CONFIG_DIR / "panel.csv"
PANEL_WEEKLY = DATA_DIR / "panel_weekly.parquet"
PRICE_DIR = DATA_DIR / "prices"

REQUIRED_COLUMNS = [
    "company_id",
    "company",
    "ticker",
    "yahoo_symbol",
    "exchange",
    "country",
    "sector",
    "app_name",
    "package_id",
    "developer",
    "parent",
    "app_role",
    "benchmark_symbol",
    "notes",
]

_PKG_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$")


@dataclass(frozen=True)
class PanelRow:
    company_id: str
    company: str
    ticker: str
    yahoo_symbol: str
    exchange: str
    country: str
    sector: str
    app_name: str
    package_id: str
    developer: str
    parent: str
    app_role: str
    benchmark_symbol: str
    notes: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def ensure_dirs() -> None:
    for d in (DATA_DIR, REPORT_DIR, FIG_DIR, LOG_DIR, PRICE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_panel(path: Path | None = None) -> list[PanelRow]:
    """Load and validate the panel CSV. Raises ValueError on any inconsistency."""
    path = path or PANEL_CSV
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"panel is missing columns: {missing}")
        rows = [PanelRow(**{k: (v or "").strip() for k, v in r.items()}) for r in reader]

    if not 25 <= len(rows) <= 40:
        raise ValueError(f"expected 25-40 companies, got {len(rows)}")

    errors: list[str] = []
    for attr in ("company_id", "ticker", "yahoo_symbol", "package_id"):
        vals = [getattr(r, attr) for r in rows]
        dupes = sorted({v for v in vals if vals.count(v) > 1})
        if dupes:
            errors.append(f"duplicate {attr}: {dupes}")

    for r in rows:
        if not _PKG_RE.match(r.package_id):
            errors.append(f"{r.company_id}: invalid package_id {r.package_id!r}")
        if r.app_role not in {"primary", "secondary"}:
            errors.append(f"{r.company_id}: app_role must be primary/secondary")
        if not r.benchmark_symbol:
            errors.append(f"{r.company_id}: missing benchmark_symbol")
        if not r.parent or not r.developer:
            errors.append(f"{r.company_id}: developer/parent must be explicit")
        # notes are optional; the app/developer/parent/ticker chain is the
        # enforceable part of the mapping.

    if errors:
        raise ValueError("panel validation failed:\n  " + "\n  ".join(errors))
    return rows


def panel_frame(rows: list[PanelRow] | None = None):
    import pandas as pd

    rows = rows or load_panel()
    return pd.DataFrame([r.as_dict() for r in rows])


def symbols(rows: list[PanelRow] | None = None) -> list[str]:
    rows = rows or load_panel()
    syms = {r.yahoo_symbol for r in rows} | {r.benchmark_symbol for r in rows} | {"URTH"}
    return sorted(syms)
