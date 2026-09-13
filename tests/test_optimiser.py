"""B6 — the self-optimising call tree.

The requirement: look at new and existing models and optimise the call tree for
cost without losing quality.

The load-bearing safety property: the optimiser **proposes**, it never applies.
An auto-applied routing change that regresses quality is worse than a stale
routing table, so every proposal must be expressed as a candidate plus the
experiment that would validate it.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from optimiser import (  # noqa: E402
    MIN_SAMPLES,
    Candidate,
    Proposal,
    propose,
    rank_models,
    write_candidate,
)
from rollup import migrate, rollup_day  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "router_logs.db"))
    migrate(conn)
    return conn


def _day(conn, day, model, workload, calls, cost_each, quality=None, latency=1.0,
        in_tok=1000, out_tok=100, quality_method="fact_fidelity_v2"):
    for _ in range(calls):
        conn.execute(
            "INSERT INTO router_logs (timestamp, session_id, model_used, workload_type, "
            "input_tokens, output_tokens, cost_usd, latency_seconds, success, "
            "quality_score, quality_method) "
            "VALUES (?,'s',?,?,?,?,?,?,1,?,?)",
            (f"{day}T10:00:00+00:00", model, workload, in_tok, out_tok, cost_each,
             latency, quality, quality_method),
        )
    conn.commit()
    rollup_day(conn, day)


# ── ranking ───────────────────────────────────────────────────────────────────

def test_rank_models_orders_by_cost_per_quality_point(db):
    """Cheaper at equal quality ranks first."""
    _day(db, "2026-09-01", "expensive", "session_compression", 40, 0.10, quality=0.9)
    _day(db, "2026-09-01", "cheap", "session_compression", 40, 0.05, quality=0.9)
    ranked = rank_models(db, workload="session_compression", days=3650)
    assert [r.model for r in ranked][:2] == ["cheap", "expensive"]


def test_rank_models_excludes_below_min_samples(db):
    """A model with too few calls must not be rankable — 3 calls prove nothing."""
    _day(db, "2026-09-01", "tried-once", "session_compression", MIN_SAMPLES - 1, 0.01, quality=1.0)
    _day(db, "2026-09-01", "proven", "session_compression", MIN_SAMPLES + 10, 0.10, quality=0.9)
    ranked = rank_models(db, workload="session_compression", days=3650)
    assert "tried-once" not in {r.model for r in ranked}
    assert "proven" in {r.model for r in ranked}


def test_rank_models_excludes_below_quality_floor(db):
    """Cheap-but-bad must not win; quality is a constraint, not a tiebreak."""
    _day(db, "2026-09-01", "cheap-bad", "session_compression", 40, 0.01, quality=0.3)
    _day(db, "2026-09-01", "dear-good", "session_compression", 40, 0.10, quality=0.95)
    ranked = rank_models(db, workload="session_compression", days=3650, quality_floor=0.8)
    assert [r.model for r in ranked] == ["dear-good"]


def test_rank_models_unmeasured_quality_is_not_eligible(db):
    """No quality evidence means we cannot claim it is safe to promote."""
    _day(db, "2026-09-01", "unmeasured", "session_compression", 40, 0.01, quality=None)
    ranked = rank_models(db, workload="session_compression", days=3650)
    assert ranked == []


def test_rank_models_sorted_cheapest_first_within_floor(db):
    _day(db, "2026-09-01", "a", "session_compression", 40, 0.20, quality=0.85)
    _day(db, "2026-09-01", "b", "session_compression", 40, 0.05, quality=0.82)
    _day(db, "2026-09-01", "c", "session_compression", 40, 0.10, quality=0.95)
    ranked = rank_models(db, workload="session_compression", days=3650)
    assert [r.model for r in ranked] == ["b", "c", "a"]


# ── proposing ─────────────────────────────────────────────────────────────────

def test_propose_suggests_cheaper_qualified_model(db):
    _day(db, "2026-09-01", "incumbent", "session_compression", 40, 0.20, quality=0.9)
    _day(db, "2026-09-01", "challenger", "session_compression", 40, 0.05, quality=0.88)
    p = propose(db, days=3650)
    cands = [c for c in p.candidates if c.workload == "session_compression"]
    assert cands, "expected a candidate for session_compression"
    c = cands[0]
    assert c.recommended_model == "challenger"
    # 0.05 vs 0.20 -> 75% cheaper
    assert c.savings_pct == pytest.approx(75.0)
    assert c.quality_delta == pytest.approx(-0.02, abs=0.001)


def test_propose_never_applies_anything(db):
    """propose() must be read-only: no routing table written, no flag flipped."""
    _day(db, "2026-09-01", "incumbent", "session_compression", 40, 0.20, quality=0.9)
    _day(db, "2026-09-01", "challenger", "session_compression", 40, 0.05, quality=0.88)
    before = db.execute("SELECT COUNT(*) FROM daily_findings").fetchone()[0]
    p = propose(db, days=3650)
    after = db.execute("SELECT COUNT(*) FROM daily_findings").fetchone()[0]
    assert before == after
    assert p.applied is False


def test_propose_reports_no_change_when_incumbent_is_cheapest(db):
    # The cheapest model also carries the most traffic -> nothing to propose.
    _day(db, "2026-09-01", "best", "session_compression", 100, 0.01, quality=0.99)
    _day(db, "2026-09-01", "worse", "session_compression", 40, 0.50, quality=0.90)
    p = propose(db, days=3650)
    cands = [c for c in p.candidates if c.workload == "session_compression"]
    assert cands == []


def test_propose_attaches_an_experiment_to_each_candidate(db):
    """Every proposal must name the experiment that would validate it."""
    _day(db, "2026-09-01", "incumbent", "session_compression", 40, 0.20, quality=0.9)
    _day(db, "2026-09-01", "challenger", "session_compression", 40, 0.05, quality=0.88)
    p = propose(db, days=3650)
    c = [x for x in p.candidates if x.workload == "session_compression"][0]
    assert c.experiment["mode"] == "shadow"        # always observe first
    assert c.experiment["model"] == "challenger"
    assert c.experiment["match"]["workload"] == ["session_compression"]


def test_propose_empty_db_yields_empty_proposal(db):
    p = propose(db, days=3650)
    assert p.candidates == []
    assert p.applied is False


def test_proposal_estimates_weekly_delta(db):
    """A saving per call must be projected onto observed call volume."""
    _day(db, "2026-09-01", "incumbent", "session_compression", 40, 0.20, quality=0.9)
    _day(db, "2026-09-01", "challenger", "session_compression", 40, 0.05, quality=0.88)
    p = propose(db, days=3650)
    c = [x for x in p.candidates if x.workload == "session_compression"][0]
    # 40 calls/day * 0.15 saved = 6.00/day -> 42.00/week
    assert c.est_weekly_delta_usd == pytest.approx(-42.0, rel=0.01)


# ── writing a candidate ───────────────────────────────────────────────────────

def test_write_candidate_emits_yaml_never_live_config(db, tmp_path):
    _day(db, "2026-09-01", "incumbent", "session_compression", 40, 0.20, quality=0.9)
    _day(db, "2026-09-01", "challenger", "session_compression", 40, 0.05, quality=0.88)
    p = propose(db, days=3650)
    out = tmp_path / "candidate_routing.yaml"
    write_candidate(p, str(out))
    text = out.read_text()
    assert "challenger" in text
    assert out.exists()
    # it must be clearly labelled a proposal, not a live table
    assert "candidate" in text.lower()
    assert "not applied" in text.lower() or "proposal" in text.lower()


def test_write_candidate_is_importable_yaml(db, tmp_path):
    import yaml

    _day(db, "2026-09-01", "incumbent", "session_compression", 40, 0.20, quality=0.9)
    _day(db, "2026-09-01", "challenger", "session_compression", 40, 0.05, quality=0.88)
    p = propose(db, days=3650)
    out = tmp_path / "candidate_routing.yaml"
    write_candidate(p, str(out))
    parsed = yaml.safe_load(out.read_text())
    assert isinstance(parsed, dict)
    assert parsed["proposal"]["applied"] is False
