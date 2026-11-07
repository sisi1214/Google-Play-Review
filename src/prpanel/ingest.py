"""Resumable Google Play review backfill.

Only public review metadata is collected. Author names, avatars and other
identifiers are dropped at the boundary. Each committed batch checkpoints a
continuation token, so an interrupted run resumes instead of re-downloading.

The review endpoint is paginated from newest to oldest only. That is the
binding constraint on historical coverage for high-volume apps: reaching a
date requires reading every review since that date.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable

from google_play_scraper import app as play_app
from google_play_scraper import reviews as play_reviews
from google_play_scraper import Sort
from google_play_scraper.features.reviews import _ContinuationToken

from .config import PanelRow
from . import db

log = logging.getLogger("prpanel.ingest")

BATCH = 2000
MAX_BATCHES_HARD = 500
SLEEP_RANGE = (0.7, 1.2)
MAX_RETRIES = 3


@dataclass
class IngestResult:
    company_id: str
    package_id: str
    added: int
    total: int
    oldest: str | None
    status: str
    error: str | None = None


def _aware_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return _aware_utc(dt).isoformat().replace("+00:00", "Z") if dt else None


def _token_to_json(tok: _ContinuationToken | None) -> str | None:
    if tok is None:
        return None
    return json.dumps(
        {
            "token": tok.token,
            "lang": tok.lang,
            "country": tok.country,
            "sort": tok.sort,
            "count": tok.count,
            "filter_score_with": tok.filter_score_with,
            "filter_device_with": tok.filter_device_with,
        }
    )


def _token_from_json(raw: str | None) -> _ContinuationToken | None:
    if not raw:
        return None
    d = json.loads(raw)
    if not d.get("token"):
        return None
    d["sort"] = Sort(d.get("sort", Sort.NEWEST.value))
    return _ContinuationToken(**d)


def fetch_metadata(row: PanelRow) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            return play_app(row.package_id, lang="en", country="us")
        except Exception as exc:  # region-locked or transient
            log.warning("metadata failed for %s (%s), attempt %d: %s", row.company_id, row.package_id, attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    return None


def _prepare_record(item: dict, row: PanelRow, requested_lang: str, requested_country: str) -> dict | None:
    rid = item.get("reviewId")
    at = _aware_utc(item.get("at"))
    score = item.get("score")
    if not rid or at is None or not isinstance(score, int):
        return None
    content = (item.get("content") or "").strip()
    reply = (item.get("replyContent") or "").strip() or None
    return {
        "review_id": rid,
        "package_id": row.package_id,
        "company_id": row.company_id,
        "score": score,
        "at_utc": _iso(at),
        "content": content[:8000] if content else None,
        "thumbs_up": item.get("thumbsUpCount") or 0,
        "app_version": item.get("reviewCreatedVersion") or item.get("appVersion"),
        "reply_content": reply[:4000] if reply else None,
        "replied_at_utc": _iso(item.get("repliedAt")),
        "lang": requested_lang,
        "country": requested_country,
        "ingested_at": db.utc_now(),
    }


def _store_batch(conn, row: PanelRow, records: Iterable[dict]) -> int:
    """Insert, ignoring duplicates inside the batch itself."""
    uniq: dict[str, dict] = {}
    for rec in records:
        uniq[rec["review_id"]] = rec
    return db.insert_reviews(conn, uniq.values())


def ingest_app(
    conn,
    row: PanelRow,
    *,
    since: date,
    max_reviews: int = 50_000,
    batch: int = BATCH,
    mode: str = "backfill",
    restart: bool = False,
) -> IngestResult:
    state = db.get_state(conn, row.package_id)
    total = conn.execute("SELECT COUNT(*) FROM reviews WHERE package_id=?", (row.package_id,)).fetchone()[0]
    if total >= max_reviews:
        return IngestResult(row.company_id, row.package_id, 0, total, state.get("oldest_at"), "capped")

    token = None
    if mode == "backfill" and not restart:
        token = _token_from_json(state.get("token_json"))
    resumed_from_checkpoint = token is not None
    restarted_from_newest = False

    meta = fetch_metadata(row)
    if meta:
        db.set_app_metadata(conn, row.package_id, meta)
        log.info("metadata %s: %s | %s | %s ratings", row.company_id, meta.get("title"), meta.get("developer"), meta.get("ratings"))
    else:
        log.warning("no Play metadata for %s; keeping curated mapping", row.company_id)

    since_dt = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)
    added = 0
    oldest_dt: datetime | None = None
    newest_dt: datetime | None = None
    status = "running"
    error: str | None = None
    empty_batches = 0
    started = time.time()

    for batch_i in range(MAX_BATCHES_HARD):
        if total >= max_reviews:
            status = "capped"
            break
        try:
            items, token = play_reviews(
                row.package_id,
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=batch,
                continuation_token=token,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.error("review fetch failed for %s: %s", row.company_id, error)
            status = "error"
            break

        if not items:
            if resumed_from_checkpoint and not restarted_from_newest and total < max_reviews:
                log.warning(
                    "checkpoint for %s returned no items (expired?); restarting from newest",
                    row.company_id,
                )
                token = None
                restarted_from_newest = True
                continue
            status = "done_end"
            break

        recs = []
        for item in items:
            rec = _prepare_record(item, row, "en", "us")
            if rec is None:
                continue
            at = _aware_utc(item["at"])
            if at < since_dt:
                continue
            recs.append(rec)
            oldest_dt = at if oldest_dt is None or at < oldest_dt else oldest_dt
            newest_dt = at if newest_dt is None or at > newest_dt else newest_dt

        n_added = _store_batch(conn, row, recs)
        added += n_added
        total = conn.execute("SELECT COUNT(*) FROM reviews WHERE package_id=?", (row.package_id,)).fetchone()[0]
        empty_batches = empty_batches + 1 if n_added == 0 else 0

        oldest_in_batch = min((_aware_utc(i["at"]) for i in items if i.get("at")), default=None)
        log.info(
            "%-10s batch %3d | items=%5d new=%5d total=%6d oldest=%s (%.0fs)",
            row.company_id, batch_i + 1, len(items), n_added, total,
            oldest_in_batch.date() if oldest_in_batch else "?", time.time() - started,
        )

        db.set_state(
            conn, row.package_id,
            token_json=_token_to_json(token),
            oldest_at=_iso(oldest_dt), newest_at=_iso(newest_dt),
            n_reviews=total, status="running", last_error=None,
        )

        if oldest_in_batch is not None and oldest_in_batch < since_dt:
            status = "done_since"
            break
        if mode == "refresh" and empty_batches >= 2:
            status = "done_refresh"
            break
        if len(items) < batch:
            status = "done_end"
            break
        if token is None:
            status = "done_end"
            break

        time.sleep(random.uniform(*SLEEP_RANGE))
    else:
        status = "batch_limit"

    db.set_state(
        conn, row.package_id,
        token_json=_token_to_json(token),
        oldest_at=_iso(oldest_dt) or state.get("oldest_at"),
        newest_at=_iso(newest_dt) or state.get("newest_at"),
        n_reviews=total, status=status, last_error=error,
    )
    return IngestResult(
        row.company_id, row.package_id, added, total,
        _iso(oldest_dt) or state.get("oldest_at"), status, error,
    )


def ingest_all(
    conn,
    rows: list[PanelRow],
    *,
    since: date,
    max_reviews: int = 50_000,
    mode: str = "backfill",
    restart: bool = False,
    only: set[str] | None = None,
    batch: int = BATCH,
) -> list[IngestResult]:
    db.init_schema(conn)
    db.upsert_apps(conn, rows)
    run_id = db.start_run(conn, mode, len(rows))
    results: list[IngestResult] = []
    for i, row in enumerate(rows, 1):
        if only and row.company_id not in only:
            continue
        state = db.get_state(conn, row.package_id)
        terminal = state.get("status") in {"done_since", "done_end", "capped"}
        continue_capped = state.get("status") == "capped" and state.get("n_reviews", 0) < max_reviews
        if mode == "backfill" and not restart and terminal and not continue_capped:
            log.info("[%d/%d] %s already %s; skipping (use --restart or --mode refresh)", i, len(rows), row.company_id, state["status"])
            results.append(IngestResult(row.company_id, row.package_id, 0, state.get("n_reviews", 0), state.get("oldest_at"), state["status"]))
            continue
        log.info("[%d/%d] %s (%s)", i, len(rows), row.company_id, row.package_id)
        try:
            res = ingest_app(conn, row, since=since, max_reviews=max_reviews, mode=mode, restart=restart, batch=batch)
        except Exception as exc:
            log.exception("unhandled ingest failure for %s", row.company_id)
            res = IngestResult(row.company_id, row.package_id, 0, 0, None, "error", str(exc))
        results.append(res)
        log.info("-> %s %s added=%d total=%d oldest=%s", res.company_id, res.status, res.added, res.total, res.oldest)
    ok = sum(1 for r in results if r.status not in {"error"})
    failed = sum(1 for r in results if r.status == "error")
    db.finish_run(
        conn, run_id,
        apps_ok=ok, apps_failed=failed, reviews_added=sum(r.added for r in results),
        notes="; ".join(f"{r.company_id}:{r.status}" for r in results),
    )
    return results
