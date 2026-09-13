"""Allowance-burn reporting against a fixed plan's included usage.

Why this module exists
----------------------
The router ledger prices every token at metered list rate. On a fixed plan with
an included allowance that is the wrong currency: in-allowance usage is not
billed at all, and only consumption *beyond* the allowance produces a charge.
Measured gap on one week of the same traffic: ledger $3,977 vs vendor dashboard
$48.26 (~80x). So the ledger is a **routing-mix proxy**; the vendor dashboard is
**billing ground truth**.

This module therefore calibrates against recorded dashboard readings rather than
list prices, using a marginal rate:

    $/token = dashboard_$ / (avg_input_tokens_per_request x dashboard_requests)

Units: ``$ / (tokens/request x requests) = $/token``. Because it is a marginal
rate (only the billed tail is counted) it lands well below list price - that is
expected, not a bug.

Honesty rules this module enforces
----------------------------------
* No recorded snapshot => ``status='no_snapshot'`` and **no projection**. It is
  never acceptable to fall back to list prices and call the result a bill.
* A projected figure is always labelled ``basis='projected_from_snapshot'`` and
  carries the snapshot it was calibrated from, so it can never be mistaken for
  an actual reading.
* An actual dashboard figure is never overwritten by a projection.

The included allowance differs by tier (the vendor page advertises $300/mo of
usage credits on the $100 tier while older notes assume the $100 itself is the
allowance), so it is an input - ``DEFAULT_ALLOWANCE_USD`` /
``BIGGIE_ALLOWANCE_USD`` - and the report always states which value it used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Warn once this share of the allowance is projected to be consumed. Warning
# before the allowance is gone is the entire purpose of the report.
WARN_FRACTION = 0.8

# Fraction of the period used to warn on pace, independent of total burn.
WARN_PACE_FRACTION = 0.9

DEFAULT_ALLOWANCE_USD = float(os.environ.get("BIGGIE_ALLOWANCE_USD", "100"))

DEFAULT_STORE = (
    Path(__file__).resolve().parents[1] / "data" / "allowance_snapshots.json"
)


@dataclass
class Snapshot:
    """One recorded vendor-dashboard reading.

    These numbers are *given* by the vendor. Nothing here may be synthesised:
    a missing field stays missing rather than defaulting to a plausible value.
    """

    at: str
    week_to_date_usd: Optional[float] = None
    credit_billed_requests: Optional[int] = None
    balance_usd: Optional[float] = None
    per_model: List[Dict[str, Any]] = field(default_factory=list)

    def per_model_total_usd(self) -> float:
        return sum(float(r.get("usd") or 0.0) for r in self.per_model)

    def per_model_requests(self) -> int:
        return sum(int(r.get("requests") or 0) for r in self.per_model)

    def per_model_is_complete(self) -> bool:
        """True only when the per-model split accounts for the whole bill.

        A partial split must never be treated as the total, or the report would
        understate spend by exactly the omitted models.
        """
        if not self.per_model or self.week_to_date_usd is None:
            return False
        return abs(self.per_model_total_usd() - self.week_to_date_usd) < 0.01


class SnapshotStore:
    """Append-only JSON list of dashboard readings.

    Deliberately dumb: it stores and returns what was recorded. A corrupt file
    degrades to "no snapshots" rather than raising, because a report that
    cannot calibrate must say so, not crash or invent.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_STORE

    def load(self) -> List[Snapshot]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        out: List[Snapshot] = []
        for item in raw:
            if not isinstance(item, dict) or "at" not in item:
                continue
            out.append(
                Snapshot(
                    at=str(item.get("at")),
                    week_to_date_usd=_opt_float(item.get("week_to_date_usd")),
                    credit_billed_requests=_opt_int(item.get("credit_billed_requests")),
                    balance_usd=_opt_float(item.get("balance_usd")),
                    per_model=list(item.get("per_model") or []),
                )
            )
        return out

    def append(self, snap: Snapshot) -> None:
        rows = self.load()
        rows.append(snap)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
        )

    def latest(self) -> Optional[Snapshot]:
        rows = self.load()
        return max(rows, key=lambda s: s.at) if rows else None


def _opt_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _opt_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def marginal_rate_usd_per_token(
    snap: Snapshot, avg_input_tokens: float
) -> Optional[float]:
    """The operator's reconciliation method: the marginal $/token of billed usage.

    ``dashboard_$ / (avg_input_tokens_per_request x dashboard_requests)``

    Returns ``None`` when the snapshot lacks the inputs to calibrate, rather
    than substituting list price - a wrong rate silently mis-projects every
    warning that follows.
    """
    if snap.week_to_date_usd is None or not snap.credit_billed_requests:
        return None
    if avg_input_tokens is None or avg_input_tokens <= 0:
        return None
    denom = float(avg_input_tokens) * float(snap.credit_billed_requests)
    if denom <= 0:
        return None
    return float(snap.week_to_date_usd) / denom


@dataclass
class BurnReport:
    """Projected allowance consumption for a period."""

    status: str  # no_snapshot | ok | warning | over
    allowance_usd: float
    actual_spend_usd: Optional[float] = None
    projected_spend_usd: Optional[float] = None
    percent_of_allowance: Optional[float] = None
    marginal_usd_per_token: Optional[float] = None
    marginal_usd_per_1m: Optional[float] = None
    snapshot_at: Optional[str] = None
    basis: str = "none"
    notes: List[str] = field(default_factory=list)

    def warns_before_overage(self) -> bool:
        return self.status in {"warning", "over"}


def burn_report(
    snapshots: List[Snapshot],
    ledger_avg_input_tokens: float,
    period_tokens: float,
    allowance_usd: Optional[float] = None,
    ledger_requests: Optional[int] = None,
) -> BurnReport:
    """Project this period's billed spend and warn before the allowance is gone.

    Scaling rule (the part that keeps this honest): only a minority of requests
    are credit-billed at all - measured at 8.5% on one week (2,960 of ~34,700) -
    because in-allowance usage is free. Projecting the period's *entire* token
    burn at the marginal rate therefore overstates spend by roughly an order of
    magnitude and would cry wolf.

    So the projection scales the snapshot's own billed spend by how the period's
    *billable* token volume compares to the snapshot's billed volume:

        billable_share   = snapshot.credit_billed_requests / ledger_requests
        billable_tokens  = period_tokens x billable_share
        projected        = snapshot_$ x (billable_tokens / snapshot_billed_tokens)

    Equal billable volume therefore reproduces the snapshot exactly - a
    projection that cannot reconcile with the reading it came from is not a
    projection, it is a guess.
    """
    allowance = float(allowance_usd if allowance_usd is not None else DEFAULT_ALLOWANCE_USD)

    if not snapshots:
        return BurnReport(
            status="no_snapshot",
            allowance_usd=allowance,
            notes=[
                "No vendor-dashboard snapshot recorded, so no marginal rate can "
                "be calibrated. Refusing to project: list prices are not the "
                "operator's bill on a fixed plan.",
                f"Record one with: router_ops.py allowance --record "
                f"--spend <week_to_date_usd> --requests <credit_billed_requests>",
            ],
        )

    snap = max(snapshots, key=lambda s: s.at)
    rate = marginal_rate_usd_per_token(snap, ledger_avg_input_tokens)

    if rate is None:
        return BurnReport(
            status="no_snapshot",
            allowance_usd=allowance,
            snapshot_at=snap.at,
            actual_spend_usd=snap.week_to_date_usd,
            notes=[
                "Snapshot is missing week_to_date_usd, credit_billed_requests or "
                "the ledger average input size, so the marginal rate cannot be "
                "calibrated.",
            ],
        )

    # Projection extends the snapshot's billed spend by the period's *billable*
    # token volume at the calibrated marginal rate. Labelled, never presented
    # as an actual.
    snapshot_billed_tokens = float(ledger_avg_input_tokens) * float(
        snap.credit_billed_requests or 0
    )
    if ledger_requests:
        billable_share = min(
            1.0, float(snap.credit_billed_requests or 0) / float(ledger_requests)
        )
    else:
        billable_share = 1.0
    billable_tokens = float(period_tokens) * billable_share
    if snapshot_billed_tokens > 0:
        projected = float(snap.week_to_date_usd or 0.0) * (
            max(0.0, billable_tokens) / snapshot_billed_tokens
        )
    else:
        projected = float(snap.week_to_date_usd or 0.0)

    percent = (projected / allowance * 100.0) if allowance > 0 else None
    if percent is None:
        status = "ok"
    elif projected > allowance:
        status = "over"
    elif percent >= WARN_FRACTION * 100:
        status = "warning"
    else:
        status = "ok"

    notes: List[str] = []
    if not snap.per_model_is_complete():
        notes.append(
            "Per-model split is absent or partial; the report does not "
            "attribute spend to models beyond what the dashboard gave."
        )
    notes.append(
        f"Projected from the dashboard reading at {snap.at}; the actual "
        f"reading is ${snap.week_to_date_usd:.2f}. Projection is not an invoice."
    )

    return BurnReport(
        status=status,
        allowance_usd=allowance,
        actual_spend_usd=snap.week_to_date_usd,
        projected_spend_usd=projected,
        percent_of_allowance=percent,
        marginal_usd_per_token=rate,
        marginal_usd_per_1m=rate * 1_000_000.0,
        snapshot_at=snap.at,
        basis="projected_from_snapshot",
        notes=notes,
    )


def format_report(r: BurnReport) -> str:
    """Render a report for a terminal. Every figure carries its provenance."""
    out = [f"Allowance burn — status: {r.status.upper()}"]
    out.append(f"  allowance            ${r.allowance_usd:,.2f} (plan included usage)")
    out.append(
        f"  dashboard actual     "
        + (f"${r.actual_spend_usd:,.2f} at {r.snapshot_at}" if r.actual_spend_usd is not None else "n/a")
    )
    out.append(
        f"  projected spend      "
        + (f"${r.projected_spend_usd:,.2f}  [{r.basis}]" if r.projected_spend_usd is not None else "n/a — not projected")
    )
    if r.percent_of_allowance is not None:
        out.append(f"  percent of allowance {r.percent_of_allowance:.1f}%")
    if r.marginal_usd_per_1m is not None:
        out.append(
            f"  marginal rate        ${r.marginal_usd_per_1m:,.3f}/1M input "
            f"(calibrated, not list)"
        )
    for n in r.notes:
        out.append(f"  note: {n}")
    return "\n".join(out)
