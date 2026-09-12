"""Branch telemetry for the CRAG pipeline.

Lightweight SQLite event log (stdlib sqlite3, no infra) recording which
loop branch fires per query, end-to-end latency, refinement usage and
web-fallback usage. Best-effort: telemetry failures never break the
pipeline — they print and continue.

Crude production heuristic implemented by get_stats(): if the
incorrect/fallback branches trigger for > 25% of queries, the primary
retriever or chunking needs fixing (CRAG is a safety net, not a
substitute for index quality).
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import BASE_DIR, TELEMETRY_ENABLED

DB_PATH = Path(BASE_DIR / "data" / "telemetry.db")
_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,             -- 'decision' | 'query' | 'refined'
    branch TEXT,                    -- correct | ambiguous | rewrite | fallback
    question TEXT,
    attempts INTEGER,
    relevant_chunks INTEGER,
    total_chunks INTEGER,
    web_used INTEGER,
    latency_ms INTEGER
)
"""


def _connect() -> sqlite3.Connection:
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute(_SCHEMA)
    return conn


def log_event(kind: str, **fields) -> None:
    if not TELEMETRY_ENABLED:
        return

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "branch": fields.get("branch"),
        "question": (fields.get("question") or "")[:300],
        "attempts": fields.get("attempts"),
        "relevant_chunks": fields.get("relevant_chunks"),
        "total_chunks": fields.get("total_chunks"),
        "web_used": 1 if fields.get("web_used") else 0,
        "latency_ms": fields.get("latency_ms"),
    }

    try:
        with _lock, _connect() as conn:
            conn.execute(
                """
                INSERT INTO events (ts, kind, branch, question, attempts,
                                    relevant_chunks, total_chunks, web_used,
                                    latency_ms)
                VALUES (:ts, :kind, :branch, :question, :attempts,
                        :relevant_chunks, :total_chunks, :web_used, :latency_ms)
                """,
                row,
            )
    except Exception as e:  # telemetry must never break the pipeline
        print(f"[TELEMETRY] write failed: {e}")


def get_stats() -> dict:
    """Aggregate branch distribution and health signals. Best-effort."""

    try:
        with _connect() as conn:
            decisions = conn.execute(
                """
                SELECT branch, COUNT(*) FROM events
                WHERE kind = 'decision' AND branch IS NOT NULL
                GROUP BY branch
                """
            ).fetchall()

            queries = conn.execute(
                """
                SELECT COUNT(*), AVG(latency_ms), AVG(attempts),
                       AVG(web_used), AVG(relevant_chunks), AVG(total_chunks)
                FROM events WHERE kind = 'query'
                """
            ).fetchone()

            refined = conn.execute(
                """
                SELECT COUNT(*) FROM events WHERE kind = 'refined'
                """
            ).fetchone()[0]
    except Exception as e:
        print(f"[TELEMETRY] read failed: {e}")
        return {"error": str(e)}

    branch_counts = {branch: count for branch, count in decisions}
    total = sum(branch_counts.values())

    n_queries = queries[0] if queries else 0
    total_queries = max(total, 1)
    incorrect_rate = (
        branch_counts.get("rewrite", 0) + branch_counts.get("fallback", 0)
    ) / total_queries

    stats = {
        "total_queries": n_queries,
        "avg_latency_ms": round(queries[1], 1) if queries[1] else None,
        "avg_attempts": round(queries[2], 2) if queries[2] else None,
        "web_fallback_rate": round(queries[3], 3) if queries[3] else 0.0,
        "avg_relevant_chunks": round(queries[4], 2) if queries[4] is not None else None,
        "avg_total_chunks": round(queries[5], 2) if queries[5] is not None else None,
        "refinement_count": refined,
        "branch_counts": branch_counts,
        "correct_rate": round(branch_counts.get("correct", 0) / total_queries, 3),
    }

    # Production heuristic: persistent incorrect-retrieval = index problem
    stats["index_health"] = (
        ("needs_work" if incorrect_rate > 0.25 else "healthy")
        if total > 0 else "no_data"
    )

    return stats
