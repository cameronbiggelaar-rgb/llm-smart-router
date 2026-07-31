"""Data models and SQLite schema for the LLM Smart Router.

This module defines the dataclasses used throughout the router pipeline
and provides the schema initialization for the router SQLite database.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class RouterLog:
    """One row in router_logs — a single model call observation.

    All fields are extracted from Hermes session data (sessions.db) and
    enriched with task classification, complexity scoring, and fit signals.
    """

    session_id: str
    model_used: str
    provider: str = ""
    task_type: str = "other"
    prompt_length: int = 0
    context_length: int = 0
    tool_call_count: int = 0
    contains_code_blocks: bool = False
    has_keywords: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    success: bool = True
    retry_count: int = 0
    escalated: bool = False
    user_corrected: bool = False
    error_type: Optional[str] = None
    timestamp: str = ""

    # ── Model fit signals (Phase 2) ──
    complexity_score: float = 0.0       # 0.0 (trivial) to 1.0 (very complex)
    instruction_count: int = 0           # number of explicit instructions/requirements
    has_format_constraint: bool = False  # "in JSON", "as a table", "formatted as"
    has_niche_references: bool = False   # references to specific libs/frameworks/APIs
    is_subagent: bool = False            # this is a subagent session
    parent_model: str = ""               # model used by parent session (if subagent)
    delegation_depth: int = 0            # how deep in the delegation chain
    user_correction_count: int = 0       # number of user corrections in session
    model_switched: bool = False         # user manually switched models mid-session
    session_message_count: int = 0       # total messages in session

    # ── Model fit probe (Phase 2) ──
    cheaper_model_would_work: bool = False  # router would have picked a cheaper model
    recommended_model: str = ""            # what the router would have selected

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RouterLog:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclasses.dataclass
class ModelCost:
    """Known model cost per 1M tokens (USD). Source: provider pricing."""

    model: str
    provider: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    effective_date: str  # ISO date when pricing was verified

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CostSummary:
    """Aggregated cost/compute summary for a report period."""

    total_cost_usd: float = 0.0
    cost_if_expensive_usd: float = 0.0
    savings_usd: float = 0.0
    savings_pct: float = 0.0
    total_calls: int = 0
    by_model: Dict[str, float] = dataclasses.field(default_factory=dict)
    by_task: Dict[str, float] = dataclasses.field(default_factory=dict)
    period_start: str = ""
    period_end: str = ""


@dataclasses.dataclass
class FitSummary:
    """Aggregated model fit summary for a report period."""

    total_sessions: int = 0
    subagent_sessions: int = 0
    avg_complexity: float = 0.0
    correction_rate: float = 0.0          # sessions with corrections / total
    avg_corrections_per_session: float = 0.0
    model_switch_rate: float = 0.0        # sessions where model was switched
    regret_rate: float = 0.0              # sessions where cheaper would have failed
    by_model: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    by_task: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    period_start: str = ""
    period_end: str = ""


# ── Known model pricing (Phase 1) ────────────────────────────────────────────
#
# Pricing model:
#   All models are flat-subscription or free:
#     - gpt-5.5 (ChatGPT OAuth): $20/mo flat — no per-token cost
#     - Ollama Cloud: $100/mo flat — no per-token cost
#     - Local: free
#
# The "cost" values below are RELATIVE COMPUTE UNITS — a dimensionless
# measure of how much of the shared subscription budget each model call
# consumes. This lets the router prefer cheaper (less capable) models for
# simple tasks, preserving expensive model capacity for complex work.
#
# Baseline: deepseek-v4-flash = 1.0 unit per 1M tokens
#
# Relative compute ratios:
#   deepseek-v4-flash      1.0x  — small, fast, cheap
#   minimax-m2.7:cloud     2.0x  — mid-size
#   glm-5                  2.0x  — mid-size
#   glm-5.1                2.5x  — slightly larger
#   glm-5.2                3.0x  — larger context, more compute
#   deepseek-v4-pro        4.0x  — premium tier
#   deepseek-v3.1:671b    10.0x  — massive 671B MoE, most expensive cloud
#   gpt-5.5               30.0x  — ChatGPT $20/mo, rate-limited, most capable

DEFAULT_MODEL_COSTS: List[ModelCost] = [
    # ── Local (free, unlimited) ──
    ModelCost("llama3.1:8b", "local", 0.00, 0.00, "2026-07-01"),
    ModelCost("qwen3:14b", "local", 0.00, 0.00, "2026-07-01"),

    # ── Ollama Cloud ($100/mo flat — relative compute units) ──
    ModelCost("deepseek-v4-flash", "ollama-cloud", 0.50, 1.50, "2026-07-01"),
    ModelCost("minimax-m2.7:cloud", "ollama-cloud", 1.00, 3.00, "2026-07-01"),
    ModelCost("glm-5", "ollama-cloud", 1.00, 3.00, "2026-07-01"),
    ModelCost("glm-5.1", "ollama-cloud", 1.25, 3.75, "2026-07-01"),
    ModelCost("glm-5.2", "ollama-cloud", 1.50, 4.50, "2026-07-01"),
    ModelCost("deepseek-v4-pro", "ollama-cloud", 2.00, 6.00, "2026-07-01"),
    ModelCost("deepseek-v3.1:671b", "ollama-cloud", 5.00, 15.00, "2026-07-01"),

    # ── ChatGPT OAuth ($20/mo flat — rate-limited, most capable) ──
    ModelCost("gpt-5.5", "openai-codex", 15.00, 60.00, "2026-07-01"),
]

# Models ordered by cost (cheapest first) for escalation chain
MODEL_COST_ORDER = [
    "llama3.1:8b",          # 0.0 — local, free
    "qwen3:14b",            # 0.0 — local, free
    "deepseek-v4-flash",    # 1.0x — Ollama Cloud baseline
    "minimax-m2.7:cloud",   # 2.0x — Ollama Cloud
    "glm-5",                # 2.0x — Ollama Cloud
    "glm-5.1",              # 2.5x — Ollama Cloud
    "glm-5.2",              # 3.0x — Ollama Cloud
    "deepseek-v4-pro",      # 4.0x — Ollama Cloud
    "deepseek-v3.1:671b",   # 10.0x — Ollama Cloud (most expensive cloud)
    "gpt-5.5",              # 30.0x — ChatGPT $20/mo (most capable, rate-limited)
]

# Model capability tiers (for fit scoring)
MODEL_CAPABILITY_TIERS = {
    "llama3.1:8b": 1,
    "qwen3:14b": 2,
    "deepseek-v4-flash": 3,
    "minimax-m2.7:cloud": 4,
    "glm-5": 4,
    "glm-5.1": 5,
    "glm-5.2": 6,
    "deepseek-v4-pro": 7,
    "deepseek-v3.1:671b": 8,
    "gpt-5.5": 10,
}


# ── SQLite Schema ─────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS router_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    model_used TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT 'other',
    prompt_length INTEGER NOT NULL DEFAULT 0,
    context_length INTEGER NOT NULL DEFAULT 0,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    contains_code_blocks INTEGER NOT NULL DEFAULT 0,
    has_keywords INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_seconds REAL NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1,
    retry_count INTEGER NOT NULL DEFAULT 0,
    escalated INTEGER NOT NULL DEFAULT 0,
    user_corrected INTEGER NOT NULL DEFAULT 0,
    error_type TEXT,

    -- Model fit signals (Phase 2)
    complexity_score REAL NOT NULL DEFAULT 0.0,
    instruction_count INTEGER NOT NULL DEFAULT 0,
    has_format_constraint INTEGER NOT NULL DEFAULT 0,
    has_niche_references INTEGER NOT NULL DEFAULT 0,
    is_subagent INTEGER NOT NULL DEFAULT 0,
    parent_model TEXT NOT NULL DEFAULT '',
    delegation_depth INTEGER NOT NULL DEFAULT 0,
    user_correction_count INTEGER NOT NULL DEFAULT 0,
    model_switched INTEGER NOT NULL DEFAULT 0,
    session_message_count INTEGER NOT NULL DEFAULT 0,

    -- Model fit probe (Phase 2)
    cheaper_model_would_work INTEGER NOT NULL DEFAULT 0,
    recommended_model TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_router_logs_timestamp ON router_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_router_logs_model ON router_logs(model_used);
CREATE INDEX IF NOT EXISTS idx_router_logs_task ON router_logs(task_type);
CREATE INDEX IF NOT EXISTS idx_router_logs_session ON router_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_router_logs_subagent ON router_logs(is_subagent);
CREATE INDEX IF NOT EXISTS idx_router_logs_complexity ON router_logs(complexity_score);

CREATE TABLE IF NOT EXISTS model_costs (
    model TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    input_cost_per_1m REAL NOT NULL,
    output_cost_per_1m REAL NOT NULL,
    effective_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ── DB helpers ────────────────────────────────────────────────────────────────

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize the router database with schema and default pricing.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection with WAL mode enabled.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA_SQL)

    # Seed default model costs if table is empty
    existing = conn.execute("SELECT COUNT(*) FROM model_costs").fetchone()[0]
    if existing == 0:
        for mc in DEFAULT_MODEL_COSTS:
            conn.execute(
                "INSERT OR IGNORE INTO model_costs (model, provider, input_cost_per_1m, output_cost_per_1m, effective_date) "
                "VALUES (?, ?, ?, ?, ?)",
                (mc.model, mc.provider, mc.input_cost_per_1m, mc.output_cost_per_1m, mc.effective_date),
            )
        conn.commit()

    return conn


def get_collection_state(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    """Get a value from the collection_state table."""
    row = conn.execute("SELECT value FROM collection_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_collection_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set a value in the collection_state table."""
    conn.execute(
        "INSERT OR REPLACE INTO collection_state (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()


def insert_router_log(conn: sqlite3.Connection, log: RouterLog) -> int:
    """Insert a RouterLog row and return its ID."""
    cur = conn.execute(
        """INSERT INTO router_logs (
            timestamp, session_id, model_used, provider, task_type,
            prompt_length, context_length, tool_call_count,
            contains_code_blocks, has_keywords,
            input_tokens, output_tokens, latency_seconds,
            estimated_cost_usd, success, retry_count,
            escalated, user_corrected, error_type,
            complexity_score, instruction_count,
            has_format_constraint, has_niche_references,
            is_subagent, parent_model, delegation_depth,
            user_correction_count, model_switched, session_message_count,
            cheaper_model_would_work, recommended_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            log.timestamp,
            log.session_id,
            log.model_used,
            log.provider,
            log.task_type,
            log.prompt_length,
            log.context_length,
            log.tool_call_count,
            1 if log.contains_code_blocks else 0,
            1 if log.has_keywords else 0,
            log.input_tokens,
            log.output_tokens,
            log.latency_seconds,
            log.estimated_cost_usd,
            1 if log.success else 0,
            log.retry_count,
            1 if log.escalated else 0,
            1 if log.user_corrected else 0,
            log.error_type,
            log.complexity_score,
            log.instruction_count,
            1 if log.has_format_constraint else 0,
            1 if log.has_niche_references else 0,
            1 if log.is_subagent else 0,
            log.parent_model,
            log.delegation_depth,
            log.user_correction_count,
            1 if log.model_switched else 0,
            log.session_message_count,
            1 if log.cheaper_model_would_work else 0,
            log.recommended_model,
        ),
    )
    return cur.lastrowid


def get_model_cost(model: str, conn: sqlite3.Connection) -> Optional[ModelCost]:
    """Look up pricing for a model. Returns None if unknown."""
    row = conn.execute(
        "SELECT model, provider, input_cost_per_1m, output_cost_per_1m, effective_date "
        "FROM model_costs WHERE model = ?",
        (model,),
    ).fetchone()
    if row:
        return ModelCost(*row)
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int, conn: sqlite3.Connection) -> float:
    """Estimate cost in USD for a model call using known pricing.

    Returns 0.0 for unknown models (local/free).
    """
    mc = get_model_cost(model, conn)
    if mc is None:
        return 0.0
    return (input_tokens / 1_000_000 * mc.input_cost_per_1m) + (output_tokens / 1_000_000 * mc.output_cost_per_1m)
