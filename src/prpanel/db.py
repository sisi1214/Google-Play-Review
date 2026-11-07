"""SQLite storage for review-level data.

Design choices:
- review_id is the primary key, so re-running an ingestion is idempotent.
- User names, avatars and other author identifiers are deliberately NOT stored.
- The continuation token is checkpointed after every committed batch, so a
  crashed or interrupted backfill resumes where it stopped.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .config import DB_PATH, PanelRow

SCHEMA = """
CREATE TABLE IF NOT EXISTS apps (
    package_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    app_name TEXT NOT NULL,
    developer TEXT NOT NULL,
    parent TEXT NOT NULL,
    ticker TEXT NOT NULL,
    metadata_json TEXT,
    first_seen_at TEXT,
    last_seen_at TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    package_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    score INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
    at_utc TEXT NOT NULL,
    content TEXT,
    thumbs_up INTEGER,
    app_version TEXT,
    reply_content TEXT,
    replied_at_utc TEXT,
    lang TEXT,
    country TEXT,
    ingested_at TEXT NOT NULL,
    sentiment REAL,
    is_english INTEGER
);

CREATE INDEX IF NOT EXISTS idx_reviews_company_at ON reviews (company_id, at_utc);
CREATE INDEX IF NOT EXISTS idx_reviews_package_at ON reviews (package_id, at_utc);
CREATE INDEX IF NOT EXISTS idx_reviews_at ON reviews (at_utc);

CREATE TABLE IF NOT EXISTS ingest_state (
    package_id TEXT PRIMARY KEY,
    token_json TEXT,
    oldest_at TEXT,
    newest_at TEXT,
    n_reviews INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    updated_at TEXT NOT NULL,
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    mode TEXT NOT NULL,
    apps_requested INTEGER,
    apps_ok INTEGER,
    apps_failed INTEGER,
    reviews_added INTEGER,
    notes TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(reviews)")}
    for name, ddl in (("sentiment", "REAL"), ("is_english", "INTEGER")):
        if name not in cols:
            conn.execute(f"ALTER TABLE reviews ADD COLUMN {name} {ddl}")
    conn.commit()


def upsert_apps(conn: sqlite3.Connection, rows: Sequence[PanelRow]) -> None:
    now = utc_now()
    conn.executemany(
        """
        INSERT INTO apps (package_id, company_id, app_name, developer, parent, ticker,
                          metadata_json, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
        ON CONFLICT(package_id) DO UPDATE SET
            company_id=excluded.company_id,
            app_name=excluded.app_name,
            developer=excluded.developer,
            parent=excluded.parent,
            ticker=excluded.ticker,
            last_seen_at=excluded.last_seen_at
        """,
        [(r.package_id, r.company_id, r.app_name, r.developer, r.parent, r.ticker, now, now) for r in rows],
    )
    conn.commit()


def set_app_metadata(conn: sqlite3.Connection, package_id: str, metadata: dict) -> None:
    keep = {
        k: metadata.get(k)
        for k in ("title", "developer", "developerId", "score", "ratings", "reviews",
                  "installs", "updated", "version", "released", "genre")
    }
    conn.execute(
        "UPDATE apps SET metadata_json=?, developer=COALESCE(NULLIF(?, ''), developer), "
        "app_name=COALESCE(NULLIF(?, ''), app_name) WHERE package_id=?",
        (json.dumps(keep, ensure_ascii=False), keep.get("developer"), keep.get("title"), package_id),
    )
    conn.commit()


def insert_reviews(conn: sqlite3.Connection, records: Iterable[dict]) -> int:
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO reviews
        (review_id, package_id, company_id, score, at_utc, content, thumbs_up,
         app_version, reply_content, replied_at_utc, lang, country, ingested_at)
        VALUES (:review_id, :package_id, :company_id, :score, :at_utc, :content, :thumbs_up,
                :app_version, :reply_content, :replied_at_utc, :lang, :country, :ingested_at)
        """,
        list(records),
    )
    conn.commit()
    return conn.total_changes - before


def get_state(conn: sqlite3.Connection, package_id: str) -> dict:
    row = conn.execute("SELECT * FROM ingest_state WHERE package_id=?", (package_id,)).fetchone()
    return dict(row) if row else {}


def set_state(
    conn: sqlite3.Connection,
    package_id: str,
    *,
    token_json: str | None,
    oldest_at: str | None,
    newest_at: str | None,
    n_reviews: int,
    status: str,
    last_error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO ingest_state (package_id, token_json, oldest_at, newest_at, n_reviews,
                                  status, updated_at, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(package_id) DO UPDATE SET
            token_json=excluded.token_json,
            oldest_at=excluded.oldest_at,
            newest_at=excluded.newest_at,
            n_reviews=excluded.n_reviews,
            status=excluded.status,
            updated_at=excluded.updated_at,
            last_error=excluded.last_error
        """,
        (package_id, token_json, oldest_at, newest_at, n_reviews, status, utc_now(), last_error),
    )
    conn.commit()


def start_run(conn: sqlite3.Connection, mode: str, n_apps: int) -> int:
    cur = conn.execute(
        "INSERT INTO ingest_runs (started_at, mode, apps_requested, reviews_added, notes) "
        "VALUES (?, ?, ?, 0, '')",
        (utc_now(), mode, n_apps),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection, run_id: int, *, apps_ok: int, apps_failed: int, reviews_added: int, notes: str
) -> None:
    conn.execute(
        "UPDATE ingest_runs SET finished_at=?, apps_ok=?, apps_failed=?, reviews_added=?, notes=? "
        "WHERE run_id=?",
        (utc_now(), apps_ok, apps_failed, reviews_added, notes, run_id),
    )
    conn.commit()


def counts(conn: sqlite3.Connection) -> dict:
    out = {}
    for table in ("apps", "reviews", "ingest_runs"):
        out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    row = conn.execute("SELECT MIN(at_utc), MAX(at_utc) FROM reviews").fetchone()
    out["min_at"], out["max_at"] = row[0], row[1]
    return out
