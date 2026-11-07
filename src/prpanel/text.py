"""English-only VADER sentiment enrichment.

Star ratings are language-independent and are the primary signal. Text
sentiment is a secondary signal and is only computed for reviews that look
English (ASCII letters dominate, no CJK). Non-English reviews keep a NULL
sentiment and are excluded from sentiment aggregates, not imputed.
"""

from __future__ import annotations

import logging
import re
import time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger("prpanel.text")

_CJK = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\u3040-\u30ff]")
_ANALYZER: SentimentIntensityAnalyzer | None = None


def _analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def looks_english(text: str | None) -> bool:
    if not text or len(text.strip()) < 8:
        return False
    if _CJK.search(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 5:
        return False
    ascii_ratio = sum(c.isascii() for c in letters) / len(letters)
    return ascii_ratio >= 0.8


def score_text(text: str | None) -> float | None:
    if not text or not looks_english(text):
        return None
    try:
        return float(_analyzer().polarity_scores(text[:4000])["compound"])
    except Exception:
        return None


def enrich_sentiment(conn, *, batch_size: int = 4000, limit: int | None = None) -> dict:
    """Fill sentiment/is_english for unprocessed rows. Resumable and idempotent."""
    total = 0
    t0 = time.time()
    processed = 0
    while True:
        if limit is not None and processed >= limit:
            break
        take = batch_size if limit is None else min(batch_size, limit - processed)
        rows = conn.execute(
            "SELECT rowid, content FROM reviews WHERE is_english IS NULL LIMIT ?", (take,)
        ).fetchall()
        if not rows:
            break
        updates = []
        for rowid, content in rows:
            s = score_text(content)
            updates.append((s, 1 if s is not None else 0, rowid))
        conn.executemany("UPDATE reviews SET sentiment=?, is_english=? WHERE rowid=?", updates)
        conn.commit()
        processed += len(rows)
        total += len(updates)
        if total % 20000 == 0 or len(rows) < take:
            log.info("sentiment: %d rows (%.0f rows/s)", total, total / max(time.time() - t0, 1e-9))
    log.info("sentiment done: %d rows in %.1fs", total, time.time() - t0)
    return {"processed": total, "seconds": round(time.time() - t0, 1)}
