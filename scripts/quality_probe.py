"""Sampled, post-response compression-quality measurement.

The problem this solves
-----------------------
`quality_score` in `router_logs` is only ever written by the shadow path. The
shadow experiment is disabled, so the model that actually serves ~99% of
compression traffic is **unscored** - there is no quality evidence at all for
the cheap rung. Cost without quality is half a measurement: a cheaper model
that degrades summaries looks like a pure win.

Design constraints
------------------
* **Off the critical path.** Scoring runs after the response has been handed to
  the client, so it can never add latency to a user request. Measured at ~1 ms
  for a 37K-char source, so even un-sampled it is cheap; sampling keeps
  storage and noise bounded anyway.
* **Deterministic per request.** Sampling hashes the request id, so a request
  is never measured twice and never flaps - a random draw per attempt would
  bias the sample toward requests that retry.
* **Never scores a non-summarisation workload.** `fact_coverage_v1` compares
  numbers between a source and its summary; applied to ordinary chat it would
  produce meaningless numbers.
* **NULL means unmeasured.** An unscoreable attempt is recorded with
  ``measurable=0`` and a NULL score. It must never be written as 0.0, or
  "we could not measure it" would be averaged as "it scored zero".
* **Probing never breaks a request.** Every entry point is exception-safe.

This module writes to its own `quality_probe` table rather than updating
`router_logs.quality_score`: the probe sample is a measurement series (it may
carry several probes per model over time), and leaving the production row's
semantics untouched keeps "measured on the request path" distinct from
"measured offline". The two are joined on ``request_id`` when needed.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from typing import Any, Dict, List, Optional

from quality import METHOD, score_summary

# Share of eligible requests probed. Env-tunable so the rate is an operational
# dial, not a code change.
DEFAULT_RATE = float(os.environ.get("BIGGIE_QUALITY_PROBE_RATE", "0.05"))

# Workloads worth scoring: summarisation, where "did the facts survive?" is the
# question. Deliberately a narrow allow-list, not a deny-list.
MEASURABLE_WORKLOADS = ("session_compression",)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quality_probe (
    request_id   TEXT PRIMARY KEY,
    model        TEXT NOT NULL DEFAULT '',
    workload     TEXT NOT NULL DEFAULT '',
    score        REAL,
    coverage     REAL,
    method       TEXT NOT NULL DEFAULT '',
    source_chars INTEGER NOT NULL DEFAULT 0,
    summary_chars INTEGER NOT NULL DEFAULT 0,
    measurable   INTEGER NOT NULL DEFAULT 0,
    probed_at    TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA)
    conn.commit()


def is_measurable_workload(workload_type: str) -> bool:
    return workload_type in MEASURABLE_WORKLOADS


def should_measure(request_id: str, rate: Optional[float] = None) -> bool:
    """Deterministically decide whether this request is probed.

    Hashing the request id (rather than drawing a random number) makes the
    decision stable across retries and escalation attempts, so a request that
    fails once and succeeds on retry is measured exactly as often as any other.

    When ``rate`` is omitted the environment is read at CALL time, not at import
    time, so the operational dial actually takes effect on a running endpoint
    instead of requiring a restart to change the sampling rate.
    """
    if rate is None:
        try:
            r = float(os.environ.get("BIGGIE_QUALITY_PROBE_RATE", DEFAULT_RATE))
        except (TypeError, ValueError):
            r = DEFAULT_RATE
    else:
        r = float(rate)
    if r <= 0.0:
        return False
    if r >= 1.0:
        return True
    digest = hashlib.sha256(str(request_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return bucket < r


def _source_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(m.get("content") or "")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    )


def record_probe(
    conn: sqlite3.Connection,
    request_id: str,
    source: Any,
    summary: Any,
    model: str = "",
    workload: str = "",
) -> Optional[float]:
    """Score one summary against its source and store the measurement.

    Returns the score, or ``None`` when the attempt was not scoreable. Never
    raises: a measurement failure must not propagate into a served request.
    """
    try:
        migrate(conn)
        src = source if isinstance(source, str) else _source_text(source)
        smy = summary if isinstance(summary, str) else (summary or "")
        src = src or ""
        smy = smy or ""

        score: Optional[float] = None
        coverage: Optional[float] = None
        measurable = 0
        if src.strip() and smy.strip():
            qs = score_summary(src, smy)
            score = float(qs.score)
            coverage = float(qs.coverage)
            measurable = 1

        conn.execute(
            """
            INSERT INTO quality_probe
                (request_id, model, workload, score, coverage, method,
                 source_chars, summary_chars, measurable, probed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(request_id) DO UPDATE SET
                model=excluded.model,
                workload=excluded.workload,
                score=excluded.score,
                coverage=excluded.coverage,
                method=excluded.method,
                source_chars=excluded.source_chars,
                summary_chars=excluded.summary_chars,
                measurable=excluded.measurable,
                probed_at=excluded.probed_at
            """,
            (request_id, model or "", workload or "", score, coverage,
             METHOD if measurable else "", len(src), len(smy), measurable),
        )
        conn.commit()
        return score
    except Exception:
        # Measurement is best-effort by contract. A probe must never be the
        # reason a request that already succeeded is reported as failed.
        try:
            conn.rollback()
        except Exception:
            pass
        return None


def summary_metrics(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Aggregate the probe series, separating measured from unmeasured."""
    migrate(conn)
    row = conn.execute(
        """
        SELECT COUNT(*) FILTER (WHERE measurable = 1) AS measured,
               COUNT(*) FILTER (WHERE measurable = 0) AS unmeasured,
               AVG(score) FILTER (WHERE measurable = 1) AS avg_score,
               AVG(coverage) FILTER (WHERE measurable = 1) AS avg_coverage
        FROM quality_probe
        """
    ).fetchone()
    measured = int(row[0] or 0)
    return {
        "measured": measured,
        "unmeasured": int(row[1] or 0),
        "avg_score": float(row[2]) if row[2] is not None else None,
        "avg_coverage": float(row[3]) if row[3] is not None else None,
    }


def by_model(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Per-model measured quality, for cost/quality comparisons."""
    migrate(conn)
    rows = conn.execute(
        """
        SELECT model,
               COUNT(*) FILTER (WHERE measurable = 1) AS measured,
               COUNT(*) FILTER (WHERE measurable = 0) AS unmeasured,
               AVG(score) FILTER (WHERE measurable = 1) AS avg_score,
               AVG(coverage) FILTER (WHERE measurable = 1) AS avg_coverage
        FROM quality_probe
        GROUP BY model
        ORDER BY avg_score DESC NULLS LAST
        """
    ).fetchall()
    return [
        {
            "model": r[0],
            "measured": int(r[1] or 0),
            "unmeasured": int(r[2] or 0),
            "avg_score": float(r[3]) if r[3] is not None else None,
            "avg_coverage": float(r[4]) if r[4] is not None else None,
        }
        for r in rows
    ]


def format_metrics(conn: sqlite3.Connection) -> str:
    m = summary_metrics(conn)
    score_txt = f"{m['avg_score']:.3f}" if m['avg_score'] is not None else "n/a — nothing measured"
    out = [
        "Compression quality probes (sampled, post-response)",
        f"  measured   : {m['measured']}",
        f"  unmeasured : {m['unmeasured']}",
        f"  avg score  : {score_txt}",
    ]
    rows = by_model(conn)
    if rows:
        out.append("")
        out.append(
            f"  {'model':30s} {'measured':>8s} {'unmeas':>7s} "
            f"{'score':>7s} {'coverage':>9s}"
        )
        for r in rows:
            s = f"{r['avg_score']:.3f}" if r['avg_score'] is not None else "n/a"
            c = f"{r['avg_coverage']:.3f}" if r['avg_coverage'] is not None else "n/a"
            out.append(
                f"  {r['model'][:30]:30s} {r['measured']:8d} {r['unmeasured']:7d} "
                f"{s:>7s} {c:>9s}"
            )
    return "\n".join(out)
