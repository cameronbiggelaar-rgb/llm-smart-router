"""Unit cost accounting for models, call types, volumes and response quality.

Design decisions (see ``references/self-optimising-router-plan.md``):

* **Real USD, not relative compute units.** The pre-existing ``model_costs``
  table held Phase-1 ratios (deepseek-v4-flash ``0.5/1.5``), which can *order*
  models but cannot answer "what did this week cost". ``model_pricing`` holds
  actual USD per 1M tokens.
* **Versioned pricing.** Prices change; a call logged in September must not be
  re-priced at October rates. ``price_for(..., at=...)`` resolves the price in
  force at the call time, so historical rows stay correct.
* **Decimal, not float.** These are money values that get summed over hundreds
  of thousands of rows; binary float drift is unacceptable.
* **Unknown is not free.** An unpriced model returns ``None`` so callers can
  flag the row (``cost_unknown=1``) rather than silently accounting $0.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List, Optional

from rollup import BILLABLE_ROW_SQL

# Provider suffixes that may appear in logged model names.
_PROVIDER_SUFFIXES = (":cloud", ":local", ":ollama")

# Quality values are 0..1 scores; below this many samples an average is noise.
MIN_QUALITY_SAMPLES = 1


@dataclass(frozen=True)
class Price:
    """A dated USD price for one model."""

    model: str
    provider: str
    input_usd_per_1m: Decimal
    output_usd_per_1m: Decimal
    effective_from: str
    source: str = ""
    # Discounted rate for tokens served from the provider's prompt cache.
    # ``None`` means "no cache rate on record" and the input rate is used, which
    # preserves the old behaviour for uncached/unpriced rows. Providers charge
    # ~1/50th of the input rate for a hit (verified: deepseek-v4.1-flash
    # 0.15 -> 0.003), so leaving this unset overstates spend on cached traffic.
    cached_input_usd_per_1m: Optional[Decimal] = None


@dataclass(frozen=True)
class UnitCostRow:
    """Aggregated cost + volume + quality for one (model, call_type) pair."""

    model: str
    call_type: str
    provider: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_per_call: float
    cost_per_1k_in: float
    quality_avg: Optional[float]
    quality_n: int
    cost_per_quality_point: Optional[float]
    cost_unknown_calls: int


def strip_provider_suffix(model: str) -> str:
    """'deepseek-v4.1-flash:cloud' -> 'deepseek-v4.1-flash'."""
    for suffix in _PROVIDER_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def record_price(conn: sqlite3.Connection, price: Price) -> None:
    """Insert or replace one dated price.

    Tolerates a database whose ``model_pricing`` predates the cached-rate
    column: the ALTER runs first, so a writer never has to care which schema
    generation it is talking to.
    """
    _ensure_cached_rate_column(conn)
    cached = price.cached_input_usd_per_1m
    conn.execute(
        "INSERT OR REPLACE INTO model_pricing "
        "(model, provider, input_usd_per_1m, output_usd_per_1m, "
        " cached_input_usd_per_1m, effective_from, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            strip_provider_suffix(price.model),
            price.provider,
            float(price.input_usd_per_1m),
            float(price.output_usd_per_1m),
            (float(cached) if cached is not None else float(price.input_usd_per_1m)),
            price.effective_from,
            price.source,
        ),
    )
    conn.commit()


def _has_cached_rate_column(conn: sqlite3.Connection) -> bool:
    """True when ``model_pricing`` carries the cached-input rate column."""
    try:
        return any(
            r[1] == "cached_input_usd_per_1m"
            for r in conn.execute("PRAGMA table_info(model_pricing)")
        )
    except Exception:                                        # pragma: no cover
        return False


def _ensure_cached_rate_column(conn: sqlite3.Connection) -> None:
    """Add ``cached_input_usd_per_1m`` to an existing ``model_pricing`` table.

    Existing rows are backfilled with their own input rate, i.e. "no cache
    discount known", which is the conservative (non-understating) default. The
    correct discounted rates are then applied by ``record_price`` callers.

    Never raises: a schema probe must not be able to break cost capture.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(model_pricing)")}
        if not cols:
            return
        if "cached_input_usd_per_1m" not in cols:
            conn.execute(
                "ALTER TABLE model_pricing ADD COLUMN cached_input_usd_per_1m REAL"
            )
            conn.execute(
                "UPDATE model_pricing SET cached_input_usd_per_1m = input_usd_per_1m "
                "WHERE cached_input_usd_per_1m IS NULL"
            )
            conn.commit()
    except Exception:                                        # pragma: no cover
        pass


def seed_prices(conn: sqlite3.Connection) -> int:
    """Seed ``model_pricing`` from ``MODEL_REGISTRY`` (single source of truth).

    ``MODEL_REGISTRY`` already carries real USD per 1M tokens, so this keeps the
    two from drifting. Idempotent: returns the number of NEW prices written.

    Returns 0 on a second run.
    """
    try:
        from models import MODEL_REGISTRY
    except Exception:  # pragma: no cover - import guard
        return 0

    written = 0
    for name, cfg in MODEL_REGISTRY.items():
        base = strip_provider_suffix(name)
        effective = cfg.get("date", "") or "1970-01-01"
        exists = conn.execute(
            "SELECT 1 FROM model_pricing WHERE model=? AND effective_from=?",
            (base, effective),
        ).fetchone()
        if exists:
            continue
        try:
            in_usd = Decimal(str(cfg.get("input", 0) or 0))
            out_usd = Decimal(str(cfg.get("output", 0) or 0))
        except Exception:
            continue
        cached_usd = cfg.get("cached")
        try:
            cached_usd = Decimal(str(cached_usd)) if cached_usd is not None else None
        except Exception:
            cached_usd = None
        record_price(
            conn,
            Price(
                model=base,
                provider=cfg.get("provider", ""),
                input_usd_per_1m=in_usd,
                output_usd_per_1m=out_usd,
                effective_from=effective,
                source="MODEL_REGISTRY",
                cached_input_usd_per_1m=cached_usd,
            ),
        )
        written += 1
    return written


def price_for(
    model: str, at: Optional[str] = None, conn: Optional[sqlite3.Connection] = None
) -> Optional[Price]:
    """Resolve the price in force for ``model`` at ISO time ``at``.

    With ``at=None`` the newest known price is used. With ``at`` set, the newest
    price with ``effective_from <= at`` is used; if the model has prices but all
    are newer than ``at``, the answer is ``None`` (unknown at that time) rather
    than silently the future price.
    """
    if conn is None:
        conn = _default_conn()
    base = strip_provider_suffix(model)
    # A reader must not assume the cached-rate column exists: any DB created
    # before this column (or by a test fixture) is still perfectly readable, and
    # a missing column simply means "no cache discount on record".
    cached_col = (
        "cached_input_usd_per_1m" if _has_cached_rate_column(conn) else "NULL"
    )
    cols = (
        "model, provider, input_usd_per_1m, output_usd_per_1m, "
        f"effective_from, source, {cached_col} "
    )
    if at is None:
        row = conn.execute(
            f"SELECT {cols} FROM model_pricing WHERE model=? "
            "ORDER BY effective_from DESC LIMIT 1",
            (base,),
        ).fetchone()
    else:
        # ISO-8601 timestamps sort lexicographically, so a string compare is a
        # correct date compare for same-format values; substr() keeps a date
        # like '2026-09-15' comparable against '2026-09-15T10:00:00+00:00'.
        row = conn.execute(
            f"SELECT {cols} FROM model_pricing WHERE model=? "
            "AND substr(?, 1, 10) >= substr(effective_from, 1, 10) "
            "ORDER BY effective_from DESC LIMIT 1",
            (base, at),
        ).fetchone()
    if not row:
        return None
    cached = None
    try:
        if len(row) > 6 and row[6] is not None:
            cached = Decimal(str(row[6]))
    except Exception:                                        # pragma: no cover
        cached = None
    return Price(
        model=row[0],
        provider=row[1],
        input_usd_per_1m=Decimal(str(row[2])),
        output_usd_per_1m=Decimal(str(row[3])),
        effective_from=row[4],
        source=row[5],
        cached_input_usd_per_1m=cached,
    )


def cost_of_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    at: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Decimal]:
    """Exact USD cost of one call, or ``None`` when the model is unpriced.

    ``cached_input_tokens`` are billed at ``cached_input_usd_per_1m`` when the
    price row carries one, else at the input rate (conservative: never
    understates). Verified live that a cache hit really is ~50x cheaper
    (deepseek-v4.1-flash 0.15 -> 0.003 per 1M), so billing hits at the full
    rate made the reported spend a large overstatement of the invoice.
    """
    if conn is None:
        conn = _default_conn()
    price = price_for(model, at=at, conn=conn)
    if price is None:
        # A caller-supplied DB may not be seeded yet. Seed once and retry, so
        # an un-seeded connection reports a real price instead of "unknown".
        try:
            seed_prices(conn)
        except Exception:                                   # pragma: no cover
            return None
        price = price_for(model, at=at, conn=conn)
    if price is None:
        return None
    try:
        cached_tok = int(cached_input_tokens or 0)
    except (TypeError, ValueError):
        cached_tok = 0
    total_in = int(input_tokens or 0)
    if cached_tok < 0:
        cached_tok = 0
    # A provider (or a buggy caller) can report more cached than total tokens.
    # Clamping keeps the arithmetic sane and the cost non-negative.
    if cached_tok > total_in:
        cached_tok = total_in
    billable_in = total_in - cached_tok
    cached_rate = price.cached_input_usd_per_1m
    if cached_rate is None:
        cached_rate = price.input_usd_per_1m
    million = Decimal(1_000_000)
    return (
        (Decimal(billable_in) / million) * price.input_usd_per_1m
        + (Decimal(cached_tok) / million) * cached_rate
        + (Decimal(output_tokens or 0) / million) * price.output_usd_per_1m
    )


_DEFAULT_CONN: Optional[sqlite3.Connection] = None


def _default_conn() -> sqlite3.Connection:
    """Lazily open the production router DB for module-level convenience calls."""
    global _DEFAULT_CONN
    if _DEFAULT_CONN is None:
        from pathlib import Path

        db_path = Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db"
        _DEFAULT_CONN = sqlite3.connect(str(db_path), check_same_thread=False)
        seed_prices(_DEFAULT_CONN)
    return _DEFAULT_CONN


def unit_cost(
    conn: sqlite3.Connection,
    since: str,
    until: Optional[str] = None,
    include_shadow: bool = False,
) -> List[UnitCostRow]:
    """Aggregate cost, volume and quality per (model, call_type) in a window.

    ``call_type`` is the logged ``workload_type`` (normal_chat,
    session_compression, ...). Timestamps are ISO strings, compared on the date
    portion so a window of ``since='2026-09-01', until='2026-09-02'`` covers
    exactly that one day.

    Shadow rows are excluded by default. A shadow call spends real money, but
    it is experiment spend rather than what the router spends to serve
    production — adding the two reports a cost the router never incurred for
    the work it did. Pass ``include_shadow=True`` for total experiment spend.
    """
    sql = """
        SELECT
            model_used,
            COALESCE(NULLIF(workload_type, ''), 'unknown') AS call_type,
            provider,
            COUNT(*)                                  AS calls,
            COALESCE(SUM(input_tokens), 0)            AS in_tok,
            COALESCE(SUM(output_tokens), 0)           AS out_tok,
            COALESCE(SUM(cost_usd), 0)                AS cost_usd,
            COALESCE(SUM(cost_unknown), 0)            AS unknown_calls,
            AVG(CASE WHEN quality_method = 'fact_fidelity_v2' THEN quality_score END) AS quality_avg,
            COUNT(CASE WHEN quality_method = 'fact_fidelity_v2' THEN quality_score END) AS quality_n
        FROM router_logs
        WHERE substr(timestamp, 1, 10) >= substr(?, 1, 10)
          AND {billable}
        """.format(billable=BILLABLE_ROW_SQL)
    params: List[Any] = [since]
    if not include_shadow:
        sql += " AND COALESCE(is_shadow, 0) = 0"
    if until is not None:
        sql += " AND substr(timestamp, 1, 10) <= substr(?, 1, 10)"
        params.append(until)
    sql += " GROUP BY model_used, call_type, provider ORDER BY cost_usd DESC, calls DESC"

    rows: List[UnitCostRow] = []
    for r in conn.execute(sql, params):
        (model, call_type, provider, calls, in_tok, out_tok, cost_usd,
         unknown_calls, quality_avg, quality_n) = r
        calls = int(calls or 0)
        in_tok = int(in_tok or 0)
        cost = float(cost_usd or 0.0)
        cost_per_call = (cost / calls) if calls else 0.0
        cost_per_1k_in = (cost / (in_tok / 1000.0)) if in_tok else 0.0
        q_avg = float(quality_avg) if quality_avg is not None else None
        q_n = int(quality_n or 0)
        cqp = None
        if q_avg is not None and q_avg > 0 and q_n >= MIN_QUALITY_SAMPLES:
            cqp = cost_per_call / q_avg
        rows.append(
            UnitCostRow(
                model=model,
                call_type=call_type,
                provider=provider or "",
                calls=calls,
                input_tokens=in_tok,
                output_tokens=int(out_tok or 0),
                cost_usd=cost,
                cost_per_call=cost_per_call,
                cost_per_1k_in=cost_per_1k_in,
                quality_avg=q_avg,
                quality_n=q_n,
                cost_per_quality_point=cqp,
                cost_unknown_calls=int(unknown_calls or 0),
            )
        )
    return rows

# cost backfill 

BACKFILL_VERSION_PREFIX = "backfill"


def backfill_costs(
    conn: sqlite3.Connection,
    dry_run: bool = False,
    batch: int = 20_000,
) -> dict:
    """Reconstruct ``cost_usd`` for rows logged before cost capture existed.

    231k historical rows carry ``cost_unknown=1`` and ``cost_usd=0``, but their
    model names and token counts survive — so the spend is recoverable.

    Two honesty rules make this safe to run:

    * Reconstructed rows are stamped ``pricing_version='backfill:<date>'`` so a
      recovered figure is never mistaken for a measured one. Reporting can then
      separate "what we logged" from "what we worked out".
    * A model with no price on record stays ``cost_unknown=1`` with ``cost_usd=0``.
      Backfill must never invent a price to make a total look complete.

    Output-token cost is understated for streaming rows (the endpoint did not
    record ``output_tokens`` for them); ``rows_missing_output_tokens`` reports
    how many, so the resulting figure carries its own caveat.
    """
    from datetime import date

    version = f"{BACKFILL_VERSION_PREFIX}:{date.today().isoformat()}"
    stats = {
        "updated": 0,
        "would_update": 0,
        "skipped_unpriced": 0,
        "rows_missing_output_tokens": 0,
        "pricing_version": version,
        "dry_run": bool(dry_run),
    }

    total = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE cost_unknown = 1"
    ).fetchone()[0]
    stats["rows_missing_output_tokens"] = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE cost_unknown = 1 AND output_tokens = 0"
    ).fetchone()[0]

    # Resolve each distinct model once: pricing a million rows should not mean
    # a million price lookups.
    models = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT model_used FROM router_logs WHERE cost_unknown = 1"
        ).fetchall()
    ]
    priced = {}
    for m in models:
        if m == "":
            continue
        pr = price_for(m, conn=conn)
        if pr is not None:
            priced[m] = pr
    stats["skipped_unpriced"] = sum(1 for m in models if m not in priced)

    if dry_run:
        for m, pr in priced.items():
            n = conn.execute(
                "SELECT COUNT(*) FROM router_logs WHERE cost_unknown = 1 AND model_used = ?",
                (m,),
            ).fetchone()[0]
            stats["would_update"] += n
        return stats

    # Batched so a 231k-row rewrite never holds a long write lock.
    for model, pr in priced.items():
        while True:
            ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM router_logs WHERE cost_unknown = 1 AND model_used = ? LIMIT ?",
                    (model, batch),
                ).fetchall()
            ]
            if not ids:
                break
            for row_id in ids:
                in_tok, out_tok = conn.execute(
                    "SELECT input_tokens, output_tokens FROM router_logs WHERE id = ?",
                    (row_id,),
                ).fetchone()
                amount = (
                    Decimal(int(in_tok or 0)) / Decimal(1_000_000) * pr.input_usd_per_1m
                    + Decimal(int(out_tok or 0)) / Decimal(1_000_000) * pr.output_usd_per_1m
                )
                conn.execute(
                    "UPDATE router_logs SET cost_usd = ?, cost_unknown = 0, "
                    "pricing_version = ? WHERE id = ?",
                    (float(amount), version, row_id),
                )
            conn.commit()
            stats["updated"] += len(ids)

    stats["unpriced_after"] = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE cost_unknown = 1"
    ).fetchone()[0]
    stats["rows_total"] = total
    return stats
