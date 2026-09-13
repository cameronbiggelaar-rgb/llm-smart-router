"""Batch 1 (RED) — the allowance-burn report must warn before overage.

The operator is on a fixed Ollama plan whose included usage is consumed by
credit-billed requests beyond the allowance. List-price ledger dollars are NOT
that currency: the ledger priced one glm-heavy week at $3,977 while the
vendor's dashboard showed $48.26 week-to-date for the same traffic (~80x).
Only the dashboard is billing ground truth.

So the report must be built on the operator-supplied reconciliation method:
`$/ (avg_input x dashboard_requests)` - a marginal rate calibrated against a
real recorded snapshot, never a list price.

These tests pin the arithmetic, and - more importantly - the honesty rules:
no snapshot means NO projection, and a projected figure is never presented as
an actual one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import allowance as a  # noqa: E402

# The operator's real dashboard readings (see references/ollama-unit-economics.md).
REAL_SNAPSHOTS = [
    {"at": "2026-09-11T20:02", "balance_usd": 16.28, "week_to_date_usd": 23.73,
     "credit_billed_requests": 1749},
    {"at": "2026-09-11T23:14", "balance_usd": 21.88, "week_to_date_usd": 38.11,
     "credit_billed_requests": 2578},
    {"at": "2026-09-12T06:10", "balance_usd": 11.74, "week_to_date_usd": 48.26,
     "credit_billed_requests": 2960},
]


def test_marginal_rate_matches_the_dashboard_arithmetic():
    """$/token = dashboard_$ / (avg_input_per_request x dashboard_requests)."""
    avg_input = 52_000.0
    snap = a.Snapshot(
        at="2026-09-12T06:10", week_to_date_usd=48.26,
        credit_billed_requests=2960,
    )
    rate = a.marginal_rate_usd_per_token(snap, avg_input_tokens=avg_input)
    assert rate == pytest.approx(48.26 / (52_000.0 * 2960), rel=1e-9)


def test_marginal_rate_is_below_list_price_for_a_real_snapshot():
    """In-allowance usage is not billed, so the marginal rate must be well below list."""
    snap = a.Snapshot(
        at="2026-09-12T06:10", week_to_date_usd=48.26,
        credit_billed_requests=2960,
    )
    rate = a.marginal_rate_usd_per_token(snap, avg_input_tokens=52_000.0)
    # expressed per 1M tokens, compare against the cheapest list input price
    per_1m = rate * 1_000_000
    assert per_1m < 1.40, f"marginal $/1M {per_1m} should be below glm list 1.40"


def test_calibration_uses_per_model_split_when_present():
    """Per-model rows calibrate per model; the totals row must still reconcile."""
    snap = a.Snapshot(
        at="2026-09-12T06:10", week_to_date_usd=48.26,
        credit_billed_requests=2960,
        per_model=[
            {"model": "glm-5.3", "requests": 431, "usd": 24.91},
            {"model": "deepseek-v4.1-flash", "requests": 406, "usd": 1.80},
        ],
    )
    assert snap.per_model_total_usd() == pytest.approx(26.71)
    # the split is partial, so it must not be mistaken for the whole bill
    assert snap.per_model_is_complete() is False


def test_no_snapshot_means_no_projection():
    """Without a recorded dashboard reading there is nothing to calibrate on."""
    report = a.burn_report(snapshots=[], ledger_avg_input_tokens=52_000.0,
                           period_tokens=100_000_000, allowance_usd=100.0)
    assert report.projected_spend_usd is None
    assert report.status == "no_snapshot"
    assert report.warns_before_overage() is False


def test_projection_is_labelled_and_never_reported_as_actual():
    """A projection must be distinguishable from a dashboard reading."""
    snap = a.Snapshot(at="2026-09-12T06:10", week_to_date_usd=48.26,
                      credit_billed_requests=2960)
    report = a.burn_report(snapshots=[snap], ledger_avg_input_tokens=52_000.0,
                           period_tokens=200_000_000, allowance_usd=100.0)
    assert report.projected_spend_usd is not None
    assert report.basis == "projected_from_snapshot"
    assert report.snapshot_at == "2026-09-12T06:10"
    assert report.actual_spend_usd == pytest.approx(48.26)
    assert report.projected_spend_usd != report.actual_spend_usd


def test_warning_fires_before_the_allowance_is_consumed():
    """The whole point: warn while there is still allowance left."""
    snap = a.Snapshot(at="2026-09-12T06:10", week_to_date_usd=48.26,
                      credit_billed_requests=2960)
    # burn roughly a full allowance again
    report = a.burn_report(snapshots=[snap], ledger_avg_input_tokens=52_000.0,
                           period_tokens=300_000_000, allowance_usd=100.0)
    assert report.status in {"warning", "over"}
    assert report.percent_of_allowance is not None
    assert report.percent_of_allowance >= a.WARN_FRACTION * 100


def test_allowance_used_is_reported_and_configurable():
    """The tier's included allowance is an input, not a hardcoded guess."""
    snap = a.Snapshot(at="2026-09-12T06:10", week_to_date_usd=10.0,
                      credit_billed_requests=100)
    report = a.burn_report(snapshots=[snap], ledger_avg_input_tokens=52_000.0,
                           period_tokens=10_000_000, allowance_usd=300.0)
    assert report.allowance_usd == 300.0


def test_snapshots_round_trip_through_the_store(tmp_path):
    """Readings accumulate on disk; the store never invents a number."""
    path = tmp_path / "allowance_snapshots.json"
    store = a.SnapshotStore(path)
    assert store.load() == []
    store.append(a.Snapshot(at="2026-09-12T06:10", week_to_date_usd=48.26,
                            credit_billed_requests=2960))
    reloaded = a.SnapshotStore(path).load()
    assert len(reloaded) == 1
    assert reloaded[0].week_to_date_usd == pytest.approx(48.26)


def test_latest_snapshot_is_used_for_calibration(tmp_path):
    store = a.SnapshotStore(tmp_path / "s.json")
    for s in REAL_SNAPSHOTS:
        store.append(a.Snapshot(**s))
    latest = store.latest()
    assert latest.at == "2026-09-12T06:10"
    assert latest.week_to_date_usd == pytest.approx(48.26)


def test_malformed_store_does_not_invent_a_projection(tmp_path):
    """A corrupt/unparseable store must degrade to 'no snapshot', not guess."""
    path = tmp_path / "s.json"
    path.write_text("{not json", encoding="utf-8")
    assert a.SnapshotStore(path).load() == []


def test_projection_reconciles_with_the_reading_it_came_from():
    """Equal billable volume must reproduce the snapshot, not exceed it.

    The first cut of this projected the period's ENTIRE token burn at the
    marginal rate and reported $568 against a dashboard reading of $48.26 -
    a ~12x cry-wolf, because only 8.5% of requests are credit-billed at all.
    A projection that cannot reconcile with its own snapshot is a guess.
    """
    snap = a.Snapshot(at="2026-09-12T06:10", week_to_date_usd=48.26,
                      credit_billed_requests=2960)
    # the same shape the snapshot was taken over: ~34,700 requests, of which
    # 2,960 were billed, avg 76.5K input tokens
    ledger_requests = 34_733
    avg_in = 76_504.0
    report = a.burn_report(
        snapshots=[snap],
        ledger_avg_input_tokens=avg_in,
        period_tokens=avg_in * ledger_requests,
        ledger_requests=ledger_requests,
        allowance_usd=100.0,
    )
    # billable share = 2960/34733; billable tokens = period x share; and the
    # snapshot's billed volume = avg_in x 2960 -> the period is ~8x the
    # snapshot's billable volume, so ~8x its spend. The point of the test is
    # that it stays in that ratio instead of counting all 34,733 requests.
    assert report.projected_spend_usd == pytest.approx(
        48.26 * (avg_in * ledger_requests * (2960 / ledger_requests)) / (avg_in * 2960),
        rel=1e-9,
    )
    assert report.projected_spend_usd < 800.0, "must not count unbilled volume"


def test_billable_share_shrinks_the_projection_monotonically():
    """More total requests for the same billable count cannot raise the bill."""
    snap = a.Snapshot(at="2026-09-12T06:10", week_to_date_usd=48.26,
                      credit_billed_requests=2960)
    avg_in, period = 76_504.0, 76_504.0 * 34_733
    small = a.burn_report([snap], avg_in, period, 100.0, ledger_requests=34_733)
    large = a.burn_report([snap], avg_in, period, 100.0, ledger_requests=347_330)
    assert large.projected_spend_usd < small.projected_spend_usd


def test_report_states_that_a_projection_is_not_an_invoice():
    snap = a.Snapshot(at="2026-09-12T06:10", week_to_date_usd=48.26,
                      credit_billed_requests=2960)
    report = a.burn_report([snap], 76_504.0, 76_504.0 * 34_733, 100.0,
                           ledger_requests=34_733)
    rendered = a.format_report(report)
    assert "not an invoice" in rendered
    assert "projected_from_snapshot" in rendered
