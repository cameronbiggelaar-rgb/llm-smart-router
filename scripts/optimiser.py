"""The self-optimising call tree — propose cheaper routing without losing quality.

Reads the durable ``daily_findings`` rollup and, for each workload class, ranks
models by **cost per quality point** subject to a quality floor and a minimum
sample size, then proposes a cheaper incumbent where one exists.

**The optimiser proposes; a human promotes.** It never edits
``routing_table.yaml``, never flips an experiment live, and never applies a
change. The reasoning (ADR-5): an auto-applied routing change that regresses
quality is far worse than a stale-but-good routing table, and quality is
measured on a sampled basis — so the evidence is never strong enough to justify
unattended promotion.

The intended loop is:

    daily findings -> propose() -> candidate_routing.yaml
        -> shadow experiment -> real-load evidence -> human edits routing table

Quality enters as a **constraint**, not a tiebreak: a 5x cheaper model that
scores below ``quality_floor`` is excluded entirely, rather than ranked lower.
A model with no measured quality is likewise not eligible — absence of evidence
is not evidence of parity.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

# A model needs at least this many calls in the window before it may be
# promoted. Below this, differences are noise.
MIN_SAMPLES = 30

# Default quality floor: a candidate must be at least this good.
DEFAULT_QUALITY_FLOOR = 0.80


@dataclass(frozen=True)
class RankedModel:
    """One model's cost/quality standing for a workload."""

    model: str
    workload: str
    calls: int
    cost_usd: float
    cost_per_call: float
    quality_avg: float
    cost_per_quality_point: float
    days_with_data: int = 1


@dataclass
class Candidate:
    """A proposed change to one workload's routing."""

    workload: str
    incumbent_model: str
    recommended_model: str
    incumbent_cost_per_call: float
    candidate_cost_per_call: float
    savings_pct: float
    incumbent_quality: float
    candidate_quality: float
    quality_delta: float
    est_weekly_delta_usd: float
    experiment: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class Proposal:
    """The full set of proposals. Never auto-applied."""

    candidates: List[Candidate] = field(default_factory=list)
    days: int = 14
    quality_floor: float = DEFAULT_QUALITY_FLOOR
    min_samples: int = MIN_SAMPLES
    applied: bool = False
    generated: str = ""
    notes: str = ""


def _window(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def rank_models(
    conn: sqlite3.Connection,
    workload: str,
    days: int = 14,
    quality_floor: float = DEFAULT_QUALITY_FLOOR,
    min_samples: int = MIN_SAMPLES,
    include_shadow: bool = False,
) -> List[RankedModel]:
    """Rank eligible models for ``workload``, cheapest cost-per-quality first.

    Eligibility requires: >= ``min_samples`` calls, measured quality, and
    ``quality_avg >= quality_floor``.

    Shadow rows are excluded by default: a candidate under observation is not
    serving production, so ranking it as if it were would propose a routing
    change backed by traffic that never arrived. Promotion is a separate,
    explicit decision — pass ``include_shadow=True`` to evaluate it.
    """
    since = _window(days)
    rows = conn.execute(
        """
        SELECT model,
               SUM(calls)            AS calls,
               SUM(cost_usd)         AS cost,
               AVG(quality_avg)      AS quality_avg,
               SUM(quality_n)        AS quality_n,
               COUNT(DISTINCT day)   AS days_with_data
        FROM daily_findings
        WHERE workload_type = ? AND day >= ?
        """ + ("" if include_shadow else " AND COALESCE(is_shadow, 0) = 0") + """
        GROUP BY model
        """,
        (workload, since),
        ).fetchall()

    ranked: List[RankedModel] = []
    for model, calls, cost, quality_avg, quality_n, days_with_data in rows:
        calls = int(calls or 0)
        if calls < min_samples:
            continue
        if not quality_n or quality_avg is None:
            continue          # unmeasured -> not eligible
        q = float(quality_avg)
        if q < quality_floor:
            continue
        cost = float(cost or 0.0)
        cpc = cost / calls
        ranked.append(
            RankedModel(
                model=model,
                workload=workload,
                calls=calls,
                cost_usd=cost,
                cost_per_call=cpc,
                quality_avg=q,
                cost_per_quality_point=(cpc / q) if q > 0 else float("inf"),
                days_with_data=int(days_with_data or 1),
            )
        )

    ranked.sort(key=lambda r: r.cost_per_quality_point)
    return ranked


def _workloads(conn: sqlite3.Connection, days: int) -> List[str]:
    since = _window(days)
    return [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT workload_type FROM daily_findings WHERE day >= ? ORDER BY workload_type",
            (since,),
        )
    ]


def propose(
    conn: sqlite3.Connection,
    days: int = 14,
    quality_floor: float = DEFAULT_QUALITY_FLOOR,
    min_samples: int = MIN_SAMPLES,
) -> Proposal:
    """Propose cheaper qualified routing per workload. Read-only.

    For each workload, the *observed* incumbent is the model carrying the most
    calls (what production actually routes to), and the recommendation is the
    cheapest eligible model. A candidate is only emitted when it is genuinely
    cheaper — otherwise the incumbent is already right and no change is
    proposed.
    """
    from datetime import datetime, timezone

    prop = Proposal(
        days=days,
        quality_floor=quality_floor,
        min_samples=min_samples,
        generated=datetime.now(timezone.utc).isoformat(),
    )

    for workload in _workloads(conn, days):
        ranked = rank_models(
            conn, workload, days=days, quality_floor=quality_floor, min_samples=min_samples
        )
        if not ranked:
            continue

        # Incumbent = the model actually taking the most traffic. Deterministic
        # tie-break so a proposal never depends on dict/row ordering: most
        # calls, then most evidence, then costliest (the thing we want to fix).
        incumbent = max(
            ranked, key=lambda r: (r.calls, r.days_with_data, r.cost_per_call)
        )
        # Recommendation = cheapest eligible (ranked[0] is already cheapest).
        best = ranked[0]

        if best.model == incumbent.model:
            continue                       # already optimal; propose nothing
        if best.cost_per_call >= incumbent.cost_per_call:
            continue                       # not actually cheaper

        savings_pct = (
            (incumbent.cost_per_call - best.cost_per_call) / incumbent.cost_per_call * 100.0
            if incumbent.cost_per_call > 0
            else 0.0
        )
        # Project onto observed volume. Divide by days that actually have data:
        # a 3650-day window holding one day of rows must not drown the weekly
        # projection by a factor of ~3650.
        window_days = max(incumbent.days_with_data, 1)
        per_day = (incumbent.cost_per_call - best.cost_per_call) * (incumbent.calls / window_days)
        # Negative == saving.
        weekly = -per_day * 7.0

        prop.candidates.append(
            Candidate(
                workload=workload,
                incumbent_model=incumbent.model,
                recommended_model=best.model,
                incumbent_cost_per_call=incumbent.cost_per_call,
                candidate_cost_per_call=best.cost_per_call,
                savings_pct=round(savings_pct, 2),
                incumbent_quality=incumbent.quality_avg,
                candidate_quality=best.quality_avg,
                quality_delta=round(best.quality_avg - incumbent.quality_avg, 4),
                est_weekly_delta_usd=round(weekly, 2),
                experiment={
                    "name": f"{best.model}-{workload}-shadow".replace(".", "").replace(":", "-"),
                    "enabled": False,
                    "model": best.model,
                    "mode": "shadow",
                    "percent": 0,
                    "match": {"workload": [workload]},
                    "notes": (
                        f"Validate {best.model} against {incumbent.model} on real "
                        f"{workload} traffic before promoting."
                    ),
                },
                rationale=(
                    f"{best.model} costs {savings_pct:.1f}% less per call than "
                    f"{incumbent.model} ({best.cost_per_call:.5f} vs "
                    f"{incumbent.cost_per_call:.5f}) at quality "
                    f"{best.quality_avg:.3f} vs {incumbent.quality_avg:.3f} "
                    f"(delta {best.quality_avg - incumbent.quality_avg:+.4f}), "
                    f"over {best.calls} calls."
                ),
            )
        )

    prop.candidates.sort(key=lambda c: c.est_weekly_delta_usd)
    return prop


def write_candidate(prop: Proposal, path: str) -> None:
    """Write the proposal as a clearly-labelled YAML candidate file.

    Deliberately a distinct filename from ``routing_table.yaml`` and marked
    ``applied: false`` so it can never be mistaken for live config.
    """
    import yaml

    doc = {
        "proposal": {
            "generated": prop.generated,
            "days": prop.days,
            "quality_floor": prop.quality_floor,
            "min_samples": prop.min_samples,
            "applied": False,
            "note": (
                "Proposal only. NOT applied. Promotion requires a shadow "
                "experiment on real traffic and an explicit human edit of "
                "routing_table.yaml."
            ),
        },
        "candidates": [
            {
                "workload": c.workload,
                "incumbent": c.incumbent_model,
                "recommended": c.recommended_model,
                "cost_per_call": {
                    "incumbent": round(c.incumbent_cost_per_call, 6),
                    "candidate": round(c.candidate_cost_per_call, 6),
                },
                "savings_pct": c.savings_pct,
                "quality": {
                    "incumbent": round(c.incumbent_quality, 4),
                    "candidate": round(c.candidate_quality, 4),
                    "delta": c.quality_delta,
                },
                "est_weekly_delta_usd": c.est_weekly_delta_usd,
                "experiment": c.experiment,
                "rationale": c.rationale,
            }
            for c in prop.candidates
        ],
    }
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)


def format_proposal(prop: Proposal) -> str:
    """Human-readable rendering for reports and CLI output."""
    lines = [
        f"Routing proposal (last {prop.days}d, quality floor {prop.quality_floor}, "
        f"min {prop.min_samples} calls) — NOT APPLIED",
        "",
    ]
    if not prop.candidates:
        lines.append("  No cheaper qualified routing identified — incumbent is optimal.")
        return "\n".join(lines)
    for c in prop.candidates:
        lines.append(f"  [{c.workload}] {c.incumbent_model} -> {c.recommended_model}")
        lines.append(
            f"      {c.savings_pct:.1f}% cheaper/call "
            f"({c.incumbent_cost_per_call:.5f} -> {c.candidate_cost_per_call:.5f}), "
            f"quality {c.incumbent_quality:.3f} -> {c.candidate_quality:.3f} "
            f"({c.quality_delta:+.4f})"
        )
        lines.append(f"      est {c.est_weekly_delta_usd:+.2f} USD/week")
        lines.append(f"      validate via shadow experiment: {c.experiment['name']}")
    return "\n".join(lines)
