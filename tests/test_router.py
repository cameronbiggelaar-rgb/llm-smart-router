"""Tests for the LLM Smart Router — Phase 1: Data Collection.

Test strategy:
- Unit tests for classify_task, extract_features, estimate_cost
- Integration tests for RouterCollector against temp SQLite DBs
- Integration tests for report generation
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add scripts dir to path
import sys
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from models import (
    RouterLog,
    ModelCost,
    CostSummary,
    DEFAULT_MODEL_COSTS,
    MODEL_COST_ORDER,
    init_db,
    insert_router_log,
    get_model_cost,
    estimate_cost,
    get_collection_state,
    set_collection_state,
)
from feature_extractor import (
    classify_task,
    contains_code_blocks,
    has_router_keywords,
    extract_features,
    score_complexity,
    count_instructions,
    has_format_constraint,
    has_niche_references,
    is_correction_message,
    count_corrections,
    compute_model_fit,
    CODING_KEYWORDS,
    PLANNING_KEYWORDS,
    RESEARCH_KEYWORDS,
    DEBUGGING_KEYWORDS,
)
from collector import RouterCollector
from report import get_cost_summary, generate_report


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — classify_task
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyTask:
    def test_coding_prompt(self):
        assert classify_task("Write a function to sort a list") == "coding"
        assert classify_task("def hello_world():") == "coding"
        assert classify_task("Refactor this class to use async") == "coding"
        assert classify_task("Debug this traceback error") == "debugging"  # debugging > coding

    def test_debugging_prompt(self):
        assert classify_task("Why is this not working?") == "debugging"
        assert classify_task("This code is broken, fix it") == "debugging"
        assert classify_task("Root cause of crash") == "debugging"

    def test_planning_prompt(self):
        assert classify_task("Plan the architecture for a new service") == "planning"
        assert classify_task("Design a database schema") == "planning"
        assert classify_task("What's the best strategy for migration?") == "planning"

    def test_research_prompt(self):
        assert classify_task("What is the capital of France?") == "research"
        assert classify_task("Explain how quantum computing works") == "research"
        assert classify_task("Research the latest AI trends") == "research"

    def test_qa_prompt(self):
        assert classify_task("Hello, how are you?") == "qa"
        assert classify_task("What time is it?") == "qa"
        assert classify_task("Thanks for your help") == "qa"

    def test_high_tool_count_defaults_to_coding(self):
        assert classify_task("Can you help me?", tool_call_count=5) == "coding"

    def test_empty_prompt(self):
        assert classify_task("") == "qa"
        assert classify_task(None) == "qa"  # type: ignore


class TestContainsCodeBlocks:
    def test_code_block(self):
        assert contains_code_blocks("Here is some ```python\nprint('hello')\n``` code")
        assert contains_code_blocks("Use `os.path.join()` for paths")

    def test_no_code(self):
        assert not contains_code_blocks("Just plain text")
        assert not contains_code_blocks("")
        assert not contains_code_blocks(None)  # type: ignore


class TestHasRouterKeywords:
    def test_with_keywords(self):
        assert has_router_keywords("Refactor this module")
        assert has_router_keywords("Design the architecture")
        assert has_router_keywords("Optimize the query")

    def test_without_keywords(self):
        assert not has_router_keywords("Hello world")
        assert not has_router_keywords("")
        assert not has_router_keywords(None)  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — extract_features
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractFeatures:
    def test_basic_extraction(self):
        session = {
            "id": "test-session-1",
            "model": "deepseek-v4-flash",
            "billing_provider": "ollama-cloud",
            "tool_call_count": 3,
            "message_count": 10,
            "started_at": 1000.0,
            "ended_at": 1015.0,
            "end_reason": "completed",
        }
        usage = [{
            "model": "deepseek-v4-flash",
            "billing_provider": "ollama-cloud",
            "input_tokens": 500,
            "output_tokens": 200,
            "estimated_cost_usd": 0.00055,
        }]
        first_msg = "Write a function to parse JSON data"

        log = extract_features(session, usage, first_msg)
        assert log is not None
        assert log.session_id == "test-session-1"
        assert log.model_used == "deepseek-v4-flash"
        assert log.provider == "ollama-cloud"
        assert log.task_type == "coding"
        assert log.tool_call_count == 3
        assert log.input_tokens == 500
        assert log.output_tokens == 200
        assert log.latency_seconds == 15.0
        assert log.success is True
        assert log.estimated_cost_usd == 0.00055

    def test_no_usage_returns_none(self):
        session = {"id": "test-session-2", "model": "gpt-5.5"}
        log = extract_features(session, [], "")
        assert log is None

    def test_error_session(self):
        session = {
            "id": "test-session-3",
            "model": "gpt-5.5",
            "end_reason": "error",
            "started_at": 2000.0,
            "ended_at": 2010.0,
        }
        usage = [{"model": "gpt-5.5", "input_tokens": 100, "output_tokens": 50}]
        log = extract_features(session, usage, "")
        assert log is not None
        assert log.success is False
        assert log.error_type == "error"

    def test_research_task(self):
        session = {
            "id": "test-session-4",
            "model": "glm-5.2",
            "started_at": 3000.0,
            "ended_at": 3005.0,
        }
        usage = [{"model": "glm-5.2", "input_tokens": 200, "output_tokens": 100}]
        log = extract_features(session, usage, "What is the latest research on LLMs?")
        assert log is not None
        assert log.task_type == "research"

    def test_code_blocks_detected(self):
        session = {
            "id": "test-session-5",
            "model": "gpt-5.5",
            "started_at": 4000.0,
            "ended_at": 4010.0,
        }
        usage = [{"model": "gpt-5.5", "input_tokens": 50, "output_tokens": 25}]
        log = extract_features(session, usage, "```python\nprint('hello')\n```")
        assert log is not None
        assert log.contains_code_blocks is True


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — models
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelCost:
    def test_default_costs_have_all_models(self):
        models = {mc.model for mc in DEFAULT_MODEL_COSTS}
        assert "gpt-5.5" in models
        assert "deepseek-v4-flash" in models
        assert "glm-5.2" in models
        assert "llama3.1:8b" in models

    def test_cost_order_cheapest_first(self):
        assert MODEL_COST_ORDER[0] == "llama3.1:8b"
        assert MODEL_COST_ORDER[-1] == "gpt-5.5"

    def test_router_log_to_dict(self):
        log = RouterLog(
            session_id="s1",
            model_used="deepseek-v4-flash",
            task_type="coding",
            input_tokens=100,
            output_tokens=50,
        )
        d = log.to_dict()
        assert d["session_id"] == "s1"
        assert d["model_used"] == "deepseek-v4-flash"
        assert d["input_tokens"] == 100

    def test_router_log_from_dict(self):
        d = {
            "session_id": "s1",
            "model_used": "deepseek-v4-flash",
            "task_type": "coding",
            "input_tokens": 100,
            "output_tokens": 50,
        }
        log = RouterLog.from_dict(d)
        assert log.session_id == "s1"
        assert log.model_used == "deepseek-v4-flash"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — SQLite DB
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def router_db():
    """Create a temporary router database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    conn = init_db(db_path)
    conn.close()
    yield db_path
    os.unlink(db_path)


class TestRouterDB:
    def test_init_creates_tables(self, router_db):
        conn = sqlite3.connect(router_db)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        assert "router_logs" in tables
        assert "model_costs" in tables
        assert "collection_state" in tables

    def test_init_seeds_model_costs(self, router_db):
        conn = sqlite3.connect(router_db)
        count = conn.execute("SELECT COUNT(*) FROM model_costs").fetchone()[0]
        conn.close()
        assert count == len(DEFAULT_MODEL_COSTS)

    def test_insert_and_query_router_log(self, router_db):
        conn = sqlite3.connect(router_db)
        log = RouterLog(
            session_id="test-session",
            model_used="deepseek-v4-flash",
            provider="ollama-cloud",
            task_type="coding",
            prompt_length=50,
            input_tokens=100,
            output_tokens=50,
            estimated_cost_usd=0.0001,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        log_id = insert_router_log(conn, log)
        assert log_id is not None

        row = conn.execute("SELECT * FROM router_logs WHERE id = ?", (log_id,)).fetchone()
        assert row is not None
        assert row[2] == "test-session"  # session_id
        assert row[3] == "deepseek-v4-flash"  # model_used
        conn.close()

    def test_get_model_cost(self, router_db):
        conn = sqlite3.connect(router_db)
        mc = get_model_cost("deepseek-v4-flash", conn)
        assert mc is not None
        assert mc.model == "deepseek-v4-flash"
        assert mc.input_cost_per_1m == 0.50
        assert mc.output_cost_per_1m == 1.50

        unknown = get_model_cost("nonexistent-model", conn)
        assert unknown is None
        conn.close()

    def test_estimate_cost(self, router_db):
        conn = sqlite3.connect(router_db)
        cost = estimate_cost("deepseek-v4-flash", 1_000_000, 500_000, conn)
        # 1M input * $0.50/1M = $0.50, 500K output * $1.50/1M = $0.75
        assert cost == pytest.approx(1.25, rel=0.01)

        # Unknown model = free
        cost = estimate_cost("nonexistent", 1000, 500, conn)
        assert cost == 0.0
        conn.close()

    def test_collection_state(self, router_db):
        conn = sqlite3.connect(router_db)
        assert get_collection_state(conn, "last_collected_at") == ""

        set_collection_state(conn, "last_collected_at", "2026-07-01T00:00:00")
        assert get_collection_state(conn, "last_collected_at") == "2026-07-01T00:00:00"
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — RouterCollector
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def hermes_db():
    """Create a temporary Hermes sessions database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT 'telegram',
            model TEXT,
            billing_provider TEXT DEFAULT '',
            billing_base_url TEXT DEFAULT '',
            tool_call_count INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0,
            started_at REAL,
            ended_at REAL,
            end_reason TEXT,
            parent_session_id TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS session_model_usage (
            session_id TEXT NOT NULL,
            model TEXT NOT NULL,
            billing_provider TEXT NOT NULL DEFAULT '',
            billing_base_url TEXT NOT NULL DEFAULT '',
            billing_mode TEXT NOT NULL DEFAULT '',
            task TEXT NOT NULL DEFAULT '',
            api_call_count INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT
        );
    """)
    conn.close()
    yield db_path
    os.unlink(db_path)


class TestRouterCollector:
    def test_collect_empty_db(self, router_db, hermes_db):
        collector = RouterCollector(
            hermes_home=Path(hermes_db).parent,
            sessions_db=Path(hermes_db).name,
            router_db=router_db,
        )
        count = collector.collect()
        assert count == 0  # No sessions

    def test_collect_with_data(self, router_db, hermes_db):
        # Insert a session
        conn = sqlite3.connect(hermes_db)
        conn.execute(
            "INSERT INTO sessions (id, source, model, billing_provider, tool_call_count, message_count, started_at, ended_at, end_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("test-session-1", "telegram", "deepseek-v4-flash", "ollama-cloud",
             3, 10, 1000.0, 1015.0, "completed"),
        )
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model, billing_provider, input_tokens, output_tokens, estimated_cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test-session-1", "deepseek-v4-flash", "ollama-cloud", 500, 200, 0.00055),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)",
            ("test-session-1", "Write a function to parse JSON data"),
        )
        conn.commit()
        conn.close()

        collector = RouterCollector(
            hermes_home=Path(hermes_db).parent,
            sessions_db=Path(hermes_db).name,
            router_db=router_db,
        )
        count = collector.collect()
        assert count == 1

        # Verify data in router DB
        rconn = sqlite3.connect(router_db)
        row = rconn.execute("SELECT * FROM router_logs").fetchone()
        assert row is not None
        assert row[2] == "test-session-1"  # session_id
        assert row[3] == "deepseek-v4-flash"  # model_used
        assert row[4] == "ollama-cloud"  # provider
        assert row[5] == "coding"  # task_type
        rconn.close()

    def test_collect_tracks_state(self, router_db, hermes_db):
        conn = sqlite3.connect(hermes_db)
        conn.execute(
            "INSERT INTO sessions (id, source, model, started_at, ended_at, end_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test-session-2", "telegram", "gpt-5.5", 2000.0, 2010.0, "completed"),
        )
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model, input_tokens, output_tokens) "
            "VALUES (?, ?, ?, ?)",
            ("test-session-2", "gpt-5.5", 100, 50),
        )
        conn.commit()
        conn.close()

        collector = RouterCollector(
            hermes_home=Path(hermes_db).parent,
            sessions_db=Path(hermes_db).name,
            router_db=router_db,
        )
        count = collector.collect()
        assert count == 1

        # Check state was saved
        rconn = sqlite3.connect(router_db)
        last_at = get_collection_state(rconn, "last_collected_at")
        assert last_at != ""
        last_count = get_collection_state(rconn, "last_collect_count")
        assert last_count == "1"
        rconn.close()

    def test_collect_incremental(self, router_db, hermes_db):
        """Second collect should only pick up new sessions."""
        now = time.time()
        conn = sqlite3.connect(hermes_db)
        conn.execute(
            "INSERT INTO sessions (id, source, model, started_at, ended_at, end_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("session-old", "telegram", "deepseek-v4-flash", now - 3600, now - 3500, "completed"),
        )
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model, input_tokens, output_tokens) "
            "VALUES (?, ?, ?, ?)",
            ("session-old", "deepseek-v4-flash", 100, 50),
        )
        conn.commit()

        collector = RouterCollector(
            hermes_home=Path(hermes_db).parent,
            sessions_db=Path(hermes_db).name,
            router_db=router_db,
        )
        count1 = collector.collect()
        assert count1 == 1

        # Add a new session (after the first collect)
        conn.execute(
            "INSERT INTO sessions (id, source, model, started_at, ended_at, end_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("session-new", "telegram", "gpt-5.5", now + 1, now + 15, "completed"),
        )
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model, input_tokens, output_tokens) "
            "VALUES (?, ?, ?, ?)",
            ("session-new", "gpt-5.5", 200, 100),
        )
        conn.commit()
        conn.close()

        # Second collect should only pick up the new session
        count2 = collector.collect()
        assert count2 == 1

        rconn = sqlite3.connect(router_db)
        total = rconn.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0]
        assert total == 2
        rconn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — Report
# ═══════════════════════════════════════════════════════════════════════════════

class TestReport:
    def test_empty_report(self, router_db):
        summary = get_cost_summary(router_db, days=30)
        assert summary.total_calls == 0
        assert summary.total_cost_usd == 0.0

    def test_report_with_data(self, router_db):
        conn = sqlite3.connect(router_db)
        now = datetime.now(timezone.utc).isoformat()
        logs = [
            RouterLog(session_id="s1", model_used="deepseek-v4-flash", provider="ollama-cloud",
                      task_type="coding", input_tokens=1000, output_tokens=500,
                      estimated_cost_usd=0.00125, timestamp=now),
            RouterLog(session_id="s2", model_used="gpt-5.5", provider="openai-codex",
                      task_type="qa", input_tokens=500, output_tokens=200,
                      estimated_cost_usd=0.0195, timestamp=now),
            RouterLog(session_id="s3", model_used="deepseek-v4-flash", provider="ollama-cloud",
                      task_type="research", input_tokens=2000, output_tokens=1000,
                      estimated_cost_usd=0.0025, timestamp=now),
        ]
        for log in logs:
            insert_router_log(conn, log)
        conn.commit()
        conn.close()

        summary = get_cost_summary(router_db, days=30)
        assert summary.total_calls == 3
        assert summary.total_cost_usd == pytest.approx(0.02325, rel=0.01)
        assert "deepseek-v4-flash" in summary.by_model
        assert "gpt-5.5" in summary.by_model
        assert "coding" in summary.by_task
        assert "qa" in summary.by_task

    def test_report_string_output(self, router_db):
        conn = sqlite3.connect(router_db)
        now = datetime.now(timezone.utc).isoformat()
        insert_router_log(conn, RouterLog(
            session_id="s1", model_used="deepseek-v4-flash",
            input_tokens=100, output_tokens=50,
            estimated_cost_usd=0.0001, timestamp=now,
        ))
        conn.commit()
        conn.close()

        report = generate_report(router_db, days=30)
        assert "LLM Smart Router" in report
        assert "Compute by Model" in report
        assert "deepseek-v4-flash" in report
        assert "Total compute units" in report


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — Model fit signals
# ═══════════════════════════════════════════════════════════════════════════════

class TestComplexityScoring:
    def test_trivial_prompt(self):
        assert score_complexity("Hello", 0, 1) < 0.2

    def test_long_prompt_scores_higher(self):
        short = score_complexity("Hi", 0, 1)
        long = score_complexity("Write a function that parses JSON, then validates the schema, then formats the output as a table with columns for name, age, and email. Make sure to handle edge cases like null values and empty arrays." * 3, 0, 1)
        assert long > short

    def test_multi_step_instructions(self):
        score = score_complexity("First do X, then do Y, after that do Z", 0, 1)
        assert score > 0.1

    def test_high_tool_count(self):
        score = score_complexity("Do something", 15, 5)
        assert score >= 0.2

    def test_long_session(self):
        score = score_complexity("Hi", 0, 60)
        assert score >= 0.15

    def test_code_blocks_add_complexity(self):
        score = score_complexity("```python\nprint('hello')\n```", 0, 1)
        assert score >= 0.15

    def test_empty_prompt(self):
        assert score_complexity("", 0, 0) == 0.0


class TestInstructionCounting:
    def test_count_instructions(self):
        assert count_instructions("Do X. Do Y. Do Z.") >= 1

    def test_no_instructions(self):
        assert count_instructions("Hello") == 0

    def test_empty(self):
        assert count_instructions("") == 0


class TestFormatConstraint:
    def test_json_format(self):
        assert has_format_constraint("Return the result in JSON")
        assert has_format_constraint("Format as a table with columns")

    def test_no_constraint(self):
        assert not has_format_constraint("Hello world")
        assert not has_format_constraint("")


class TestNicheReferences:
    def test_niche_refs(self):
        assert has_niche_references("Deploy to kubernetes")
        assert has_niche_references("Using pytorch for training")
        assert has_niche_references("Configure oauth2")

    def test_no_refs(self):
        assert not has_niche_references("Hello world")
        assert not has_niche_references("")


class TestCorrectionDetection:
    def test_correction_patterns(self):
        assert is_correction_message("No, that's not what I meant")
        assert is_correction_message("That's wrong")
        assert is_correction_message("Actually, I need it differently")
        assert is_correction_message("My ask was to do X")
        assert is_correction_message("Let me rephrase")

    def test_not_a_correction(self):
        assert not is_correction_message("That looks great, thanks!")
        assert not is_correction_message("Hello, how are you?")
        assert not is_correction_message("")

    def test_count_corrections(self):
        messages = [
            {"role": "user", "content": "Write a function"},
            {"role": "assistant", "content": "Here's the code"},
            {"role": "tool", "content": "output"},
            {"role": "assistant", "content": "Done"},
            {"role": "user", "content": "No, that's not what I wanted"},
            {"role": "assistant", "content": "OK, let me fix it"},
            {"role": "user", "content": "That's still wrong"},
        ]
        assert count_corrections(messages) == 2

    def test_no_corrections(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Thanks"},
        ]
        assert count_corrections(messages) == 0

    def test_short_message_list(self):
        assert count_corrections([]) == 0
        assert count_corrections([{"role": "user", "content": "Hi"}]) == 0


class TestModelFit:
    def test_simple_task_on_cheap_model(self):
        fit = compute_model_fit("deepseek-v4-flash", 0.1, 0, True)
        assert not fit["overkill"]
        assert not fit["underpowered"]

    def test_simple_task_on_expensive_model(self):
        fit = compute_model_fit("gpt-5.5", 0.1, 0, True)
        assert fit["overkill"]
        assert not fit["underpowered"]

    def test_complex_task_on_cheap_model(self):
        fit = compute_model_fit("deepseek-v4-flash", 0.9, 2, True)
        assert fit["underpowered"]

    def test_corrections_increase_min_tier(self):
        fit_no_corr = compute_model_fit("deepseek-v4-flash", 0.3, 0, True)
        fit_with_corr = compute_model_fit("deepseek-v4-flash", 0.3, 2, True)
        assert fit_with_corr["min_tier_needed"] > fit_no_corr["min_tier_needed"]


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — extract_features (fit signals)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractFeaturesFit:
    def test_subagent_detection(self):
        session = {"id": "sub-session", "model": "gpt-5.5", "parent_session_id": "parent-1",
                   "started_at": 1000.0, "ended_at": 1010.0}
        usage = [{"model": "gpt-5.5", "input_tokens": 100, "output_tokens": 50}]
        parent = {"model": "deepseek-v4-flash"}
        log = extract_features(session, usage, "Do something", parent_session=parent, delegation_depth=2)
        assert log is not None
        assert log.is_subagent is True
        assert log.parent_model == "deepseek-v4-flash"
        assert log.delegation_depth == 2

    def test_complexity_scored(self):
        session = {"id": "cpx-session", "model": "gpt-5.5",
                   "tool_call_count": 8, "message_count": 30,
                   "started_at": 1000.0, "ended_at": 1010.0}
        usage = [{"model": "gpt-5.5", "input_tokens": 100, "output_tokens": 50}]
        log = extract_features(session, usage, "First do X, then do Y, make sure to handle Z")
        assert log is not None
        assert log.complexity_score > 0.1
        assert log.instruction_count > 0

    def test_format_constraint_detected(self):
        session = {"id": "fmt-session", "model": "deepseek-v4-flash",
                   "started_at": 1000.0, "ended_at": 1010.0}
        usage = [{"model": "deepseek-v4-flash", "input_tokens": 100, "output_tokens": 50}]
        log = extract_features(session, usage, "Return the result in JSON format")
        assert log is not None
        assert log.has_format_constraint is True

    def test_niche_references_detected(self):
        session = {"id": "niche-session", "model": "gpt-5.5",
                   "started_at": 1000.0, "ended_at": 1010.0}
        usage = [{"model": "gpt-5.5", "input_tokens": 100, "output_tokens": 50}]
        log = extract_features(session, usage, "Deploy to kubernetes with helm charts")
        assert log is not None
        assert log.has_niche_references is True

    def test_correction_count(self):
        session = {"id": "corr-session", "model": "gpt-5.5",
                   "started_at": 1000.0, "ended_at": 1010.0}
        usage = [{"model": "gpt-5.5", "input_tokens": 100, "output_tokens": 50}]
        messages = [
            {"role": "user", "content": "Write code"},
            {"role": "assistant", "content": "Here"},
            {"role": "user", "content": "No, that's not what I wanted"},
        ]
        log = extract_features(session, usage, "Write code", messages=messages)
        assert log is not None
        assert log.user_correction_count == 1
        assert log.user_corrected is True

    def test_model_switch_detected(self):
        session = {"id": "switch-session", "model": "gpt-5.5",
                   "started_at": 1000.0, "ended_at": 1010.0}
        usage = [
            {"model": "deepseek-v4-flash", "input_tokens": 50, "output_tokens": 25},
            {"model": "gpt-5.5", "input_tokens": 100, "output_tokens": 50},
        ]
        log = extract_features(session, usage, "Do something")
        assert log is not None
        assert log.model_switched is True


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_missing_sessions_db(self, router_db):
        collector = RouterCollector(
            hermes_home=Path("/nonexistent"),
            sessions_db="missing.db",
            router_db=router_db,
        )
        count = collector.collect()
        assert count == 0  # Graceful degradation

    def test_partial_session_data(self, router_db, hermes_db):
        """Session with missing optional fields should still work."""
        conn = sqlite3.connect(hermes_db)
        conn.execute(
            "INSERT INTO sessions (id, source, model, started_at, ended_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("partial-session", "telegram", "deepseek-v4-flash", 1000.0, 1010.0),
        )
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model, input_tokens, output_tokens) "
            "VALUES (?, ?, ?, ?)",
            ("partial-session", "deepseek-v4-flash", 100, 50),
        )
        conn.commit()
        conn.close()

        collector = RouterCollector(
            hermes_home=Path(hermes_db).parent,
            sessions_db=Path(hermes_db).name,
            router_db=router_db,
        )
        count = collector.collect()
        assert count == 1

    def test_unknown_model_cost(self, router_db):
        """Unknown models should estimate cost as 0."""
        conn = sqlite3.connect(router_db)
        cost = estimate_cost("some-unknown-model", 1000, 500, conn)
        assert cost == 0.0
        conn.close()
