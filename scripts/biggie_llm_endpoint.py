"""Biggie LLM Endpoint — reads Hermes config, routes to the cheapest capable model.

Hermes points to this as a custom provider. The endpoint:
1. Reads Hermes config.yaml to discover available models/providers
2. Identifies itself and skips itself (no routing loops)
3. Extracts features from the prompt
4. Routes to the cheapest capable model
5. Proxies the request to the chosen backend
6. Handles rate limits, circuit breakers, limp-home mode

Configurable routing profile:
  - cheap:      prefer cheapest model that can handle the task
  - goldilocks: balanced — prefer mid-tier, escalate only when needed
  - expensive:  prefer most capable model (current behaviour)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Load .env file for API keys (systemd EnvironmentFile may not work with ProtectHome)
_env_path = Path.home() / ".hermes" / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                if _key not in os.environ:
                    os.environ[_key] = _val

# Add the router module to path
sys.path.insert(0, str(Path(__file__).parent))

from router import (
    route_task,
    escalate_on_failure,
    is_limp_home,
    get_limp_home_message,
    check_limp_home_status,
    get_recovery_summary,
    mark_rate_limited,
    mark_credential_expired,
    log_recovery_event,
    mark_available,
    RoutingDecision,
    MODEL_CAPABILITY_TIERS,
    TOOL_CAPABLE_MODELS,
)
from feature_extractor import (
    classify_task,
    score_complexity,
    count_instructions,
    has_format_constraint,
    has_niche_references,
    contains_code_blocks,
)
from compression import compress_messages
from compression_sampler import capture_compression_sample
from unit_economics import cost_of_call
from traffic_split import (
    Experiment,
    load_experiments,
    pick_experiment,
    select_arm as _select_arm,
)

logger = logging.getLogger("biggie-llm-endpoint")

# Reasoning-capable summariser models can spend several thousand tokens on their
# hidden reasoning trace before emitting visible summary content. A low caller
# budget therefore creates an empty completion even at small context sizes. Give
# native session-compression requests enough room for reasoning plus the summary;
# models may still stop earlier, so this is a ceiling rather than reserved spend.
SESSION_COMPRESSION_MIN_MAX_TOKENS = int(
    os.environ.get("BIGGIE_SESSION_COMPRESSION_MIN_MAX_TOKENS", "16384")
)


def apply_session_compression_output_floor(
    body: Dict[str, Any], workload_type: str
) -> bool:
    """Ensure native compactions have room to emit content after reasoning.

    Mutates ``body`` and returns whether the value changed. Explicit caller
    budgets above the floor are preserved.
    """
    if workload_type != "session_compression":
        return False
    current = body.get("max_tokens")
    try:
        current = int(current) if current is not None else None
    except (TypeError, ValueError):
        current = None
    if current is not None and current >= SESSION_COMPRESSION_MIN_MAX_TOKENS:
        return False
    body["max_tokens"] = SESSION_COMPRESSION_MIN_MAX_TOKENS
    return True


# ── Cached state ───────────────────────────────────────────────────────────────

# Cache for discovered backends — refreshed every 60s or on demand
_backends_cache: Dict[str, Any] = {}
_backends_cache_time: float = 0
_BACKENDS_CACHE_TTL = 60  # seconds

# Persistent httpx client for connection pooling
_httpx_client: Optional[httpx.AsyncClient] = None

# Persistent SQLite connection for request logging
_sqlite_conn: Optional[Any] = None
_sqlite_lock: Any = None  # will be threading.Lock

# Extra observability columns for streaming requests (FIX 3). Added at runtime
# via ALTER TABLE so existing databases migrate without dropping data.
_STREAM_OBS_COLUMNS = {
    "request_id": "TEXT NOT NULL DEFAULT ''",
    "requested_model": "TEXT NOT NULL DEFAULT ''",
    "streaming": "INTEGER NOT NULL DEFAULT 0",
    "workload_type": "TEXT NOT NULL DEFAULT ''",
    "requires_tools": "INTEGER NOT NULL DEFAULT 0",
    "context_tokens": "INTEGER NOT NULL DEFAULT 0",
    "empty_stream": "INTEGER NOT NULL DEFAULT 0",
    "saw_content": "INTEGER NOT NULL DEFAULT 0",
    "saw_tool_calls": "INTEGER NOT NULL DEFAULT 0",
    "final_model": "TEXT NOT NULL DEFAULT ''",
    "routing_reason": "TEXT NOT NULL DEFAULT ''",
}

# Cost / quality / experiment columns (self-optimising router). Kept identical
# in intent to rollup.NEW_LOG_COLUMNS so the endpoint's own migration and the
# offline migrate() agree; rollup.migrate() is the authoritative source.
_COST_OBS_COLUMNS = {
    "compression_level": "TEXT NOT NULL DEFAULT ''",
    "compression_savings_pct": "REAL NOT NULL DEFAULT 0",
    "compression_time_ms": "REAL NOT NULL DEFAULT 0",
    "cost_usd": "REAL NOT NULL DEFAULT 0",
    "cost_unknown": "INTEGER NOT NULL DEFAULT 1",
    "quality_score": "REAL",
    "quality_method": "TEXT NOT NULL DEFAULT ''",
    "pricing_version": "TEXT NOT NULL DEFAULT ''",
    "experiment": "TEXT NOT NULL DEFAULT ''",
    "experiment_arm": "TEXT NOT NULL DEFAULT ''",
    "is_shadow": "INTEGER NOT NULL DEFAULT 0",
}

# Re-auth hint surfaced when a provider credential expires (HTTP 401/403).
_CODEX_REAUTH_HINT = "run: hermes auth add openai-codex --type oauth"

# Experiment config (experiments.yaml) is re-read at most this often, so an
# operator can change traffic split without restarting the service.
_EXPERIMENTS_TTL = 30.0
_experiments_cache: Optional[List[Experiment]] = None
_experiments_cache_time: float = 0.0


def _handle_backend_status(
    provider: str, backend_model: str, status: int, error_text: str = ""
) -> None:
    """Classify a non-2xx backend response and record the failure state.

    - 429 -> mark_rate_limited (transient; recovers on its own)
    - 401/403 -> mark_credential_expired on the codex provider (needs human
      re-auth; surfaces as a distinct 'credential_expired' event instead of a
      generic 'error' / silent circuit trip)
    - everything else -> left to normal escalation / circuit breaker
    """
    if status == 429:
        mark_rate_limited(backend_model)
    elif status in (401, 403) and provider == "openai-codex":
        mark_credential_expired(
            backend_model,
            f"{provider} credential rejected (HTTP {status}) — {_CODEX_REAUTH_HINT}",
        )


# Access tokens are short-lived; warn well before they expire so a dead
# credential surfaces proactively instead of as a mid-request 401.
_CODEX_TOKEN_AGE_WARN_DAYS = 6   # OpenAI codex access tokens typically live ~1 week
_CODEX_TOKEN_AGE_ALERT_DAYS = 14 # beyond this, treat as expired outright
_last_token_age_warn_at: float = 0.0


def _check_codex_token_age() -> None:
    """Warn if the openai-codex credential pool's newest token is old.

    Reads the same auth.json pool the backend loader uses, computes the age of
    the most recent ``last_refresh`` across entries, and surfaces a distinct,
    actionable warning — WITHOUT taking the model out of rotation. A token near
    expiry is a routine maintenance signal, not a failure; the request path
    still handles an actual 401 via ``_handle_backend_status``.

    Rate-limited to one warning per 6 hours so it doesn't spam logs/DB on the
    60s backend-cache refresh.
    """
    global _last_token_age_warn_at
    now = time.time()
    if now - _last_token_age_warn_at < 6 * 3600:
        return

    newest_refresh_ts: Optional[float] = None
    try:
        _auth_path = Path.home() / ".hermes" / "auth.json"
        if not _auth_path.exists():
            return
        with open(_auth_path) as _af:
            _auth_data = json.load(_af)
        for _entry in (_auth_data.get("credential_pool", {}).get("openai-codex", []) or []):
            if not isinstance(_entry, dict):
                continue
            _lr = _entry.get("last_refresh")
            if _lr:
                try:
                    _ts = datetime.fromisoformat(_lr.replace("Z", "+00:00")).timestamp()
                except (TypeError, ValueError):
                    continue
                if newest_refresh_ts is None or _ts > newest_refresh_ts:
                    newest_refresh_ts = _ts
    except Exception:
        return  # auth.json unreadable — request path already handles failure

    if newest_refresh_ts is None:
        return

    age_min = (now - newest_refresh_ts) / 60.0
    if age_min >= _CODEX_TOKEN_AGE_ALERT_DAYS * 24 * 60:
        level = "ALERT"
        msg = (f"openai-codex token {age_min / 1440:.1f} days old — almost "
               f"certainly expired. {_CODEX_REAUTH_HINT}")
    elif age_min >= _CODEX_TOKEN_AGE_WARN_DAYS * 24 * 60:
        level = "WARN"
        msg = (f"openai-codex token {age_min / 1440:.1f} days old — near expiry. "
               f"Refresh proactively: {_CODEX_REAUTH_HINT}")
    else:
        return  # young enough — no warning

    _last_token_age_warn_at = now
    logger.warning("[token-age %s] %s", level, msg)
    log_recovery_event(
        "openai-codex",
        "token_age_warn" if level == "WARN" else "token_age_alert",
        msg,
    )


def _ensure_log_columns(db: Any) -> None:
    """Add cost/quality/experiment + streaming-observability columns if missing."""
    try:
        existing = {r[1] for r in db.execute("PRAGMA table_info(router_logs)").fetchall()}
        for col, ddl in {**_STREAM_OBS_COLUMNS, **_COST_OBS_COLUMNS}.items():
            if col not in existing:
                db.execute(f"ALTER TABLE router_logs ADD COLUMN {col} {ddl}")
        db.commit()
    except Exception as e:
        logger.warning("Failed to migrate router_logs columns: %s", e)


def compute_cost_fields(
    model: str,
    input_tokens: Any,
    output_tokens: Any,
    conn: Any = None,
) -> Tuple[float, int]:
    """Return ``(cost_usd, cost_unknown)`` for one call.

    ``cost_unknown`` is 1 when the model has no price on record. That flag is
    the whole point: before this existed every row in router_logs carried
    ``estimated_cost_usd = 0``, so an unpriced model was indistinguishable from
    a free one and the spend number was fiction.

    Never raises — cost capture is diagnostic and must not break a request.
    """
    try:
        in_tok = int(input_tokens or 0)
        out_tok = int(output_tokens or 0)
    except (TypeError, ValueError):
        return 0.0, 1
    if in_tok == 0 and out_tok == 0:
        return 0.0, 0
    try:
        if conn is None:
            conn = _get_db_connection()
        amount = cost_of_call(model, in_tok, out_tok, conn=conn)
    except Exception as e:                                  # pragma: no cover
        logger.debug("cost_of_call failed for %s: %s", model, e)
        return 0.0, 1
    if amount is None:
        return 0.0, 1
    return float(amount), 0


def _pricing_version() -> str:
    """Date stamp of the pricing table in force, for auditing a cost figure.

    A cost number is only meaningful alongside the prices that produced it.
    """
    try:
        row = _get_db_connection().execute(
            "SELECT MAX(effective_from) FROM model_pricing"
        ).fetchone()
        return str(row[0]) if row and row[0] else ""
    except Exception:
        return ""


def _experiments_cached() -> List[Experiment]:
    """Load experiments.yaml, cached briefly so config edits are picked up
    without a restart but not re-read on every request."""
    global _experiments_cache, _experiments_cache_time
    now = time.time()
    if _experiments_cache is not None and (now - _experiments_cache_time) < _EXPERIMENTS_TTL:
        return _experiments_cache
    try:
        _experiments_cache = load_experiments()
    except Exception as e:                                  # pragma: no cover
        logger.warning("Failed to load experiments.yaml: %s", e)
        _experiments_cache = []
    _experiments_cache_time = now
    return _experiments_cache


def apply_experiment(
    selected_model: str,
    experiment: Any,
    workload_type: str = "",
    request_id: str = "",
    session_id: str = "",
    tier: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Decide whether an experiment changes who serves this request.

    Returns ``(model_to_serve, observability_fields)``.

    * **split** — the treatment arm genuinely serves the candidate, so the
      request is the real-load A/B test.
    * **shadow** — the incumbent still serves the user; the candidate is only
      recorded for observation. Shadow must never change what the user gets.
    """
    obs: Dict[str, Any] = {
        "experiment": "",
        "experiment_arm": "",
        "is_shadow": False,
        "shadow_model": "",
    }
    if experiment is None or not experiment.enabled:
        return selected_model, obs

    if workload_type and not _experiment_matches(experiment, workload_type, tier):
        return selected_model, obs

    arm = _select_arm(experiment, request_id=request_id, session_id=session_id)
    obs["experiment"] = experiment.name
    obs["experiment_arm"] = "shadow" if experiment.is_shadow else arm

    if experiment.is_shadow:
        obs["is_shadow"] = True
        obs["shadow_model"] = experiment.model
        return selected_model, obs

    if arm == "treatment":
        return experiment.model, obs
    return selected_model, obs


# ── Shadow execution ──────────────────────────────────────────────────────────
# A shadow experiment is only worth anything if the candidate is ACTUALLY
# called. Before this existed, "shadow" merely stamped rows with is_shadow=1
# while the candidate was never exercised — which would have produced tens of
# thousands of rows claiming observational coverage of a model that never ran.
# Shadow calls are real spend, so they are bounded and labelled.

_SHADOW_MAX_CONCURRENCY = 2
_SHADOW_TIMEOUT_SECONDS = 90.0
_shadow_semaphore = None


def _get_shadow_semaphore():
    """Bound concurrent shadow calls so an experiment cannot starve production.

    Built lazily inside the running loop: constructing a Semaphore at import
    time binds it to the wrong event loop and the first await then fails.
    """
    global _shadow_semaphore
    if _shadow_semaphore is None:
        _shadow_semaphore = asyncio.Semaphore(_SHADOW_MAX_CONCURRENCY)
    return _shadow_semaphore


def _resolve_candidate_backend(
    model_name: str,
    provider: str = "",
) -> Optional[Dict[str, Any]]:
    """Find the backend config for an experiment's candidate model.

    Resolution order:
      1. the experiment's own ``provider`` (preferred — the candidate usually
         is not in production routing, and adding it there to make an
         experiment work could start serving real traffic);
      2. the discovered routing backends, tolerating the ":cloud"/":local"
         suffix difference between the routing name and the config key.
    """
    if provider:
        try:
            hermes = load_hermes_config()
        except Exception as e:                              # pragma: no cover
            logger.warning("Shadow: could not load config for provider %s: %s", provider, e)
            hermes = {}
        pconf = (hermes.get("providers") or {}).get(provider) or {}
        api_key_env = pconf.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if not api_key:
            key_env = BUILTIN_PROVIDER_KEYS.get(provider, "")
            if key_env:
                api_key = os.environ.get(key_env, "")
        base_url = pconf.get("base_url") or BUILTIN_PROVIDER_URLS.get(provider, "")
        if base_url:
            return {
                "provider": provider,
                "base_url": base_url,
                "api_key": api_key,
                "backend_model": re.sub(r":(cloud|local|ollama)$", "", model_name),
            }
        logger.warning("Shadow: provider %s has no base_url; falling back", provider)

    try:
        backends = discover_backends()
    except Exception as e:                                  # pragma: no cover
        logger.warning("Shadow: could not discover backends: %s", e)
        return None
    if model_name in backends:
        return backends[model_name]
    stem = re.sub(r":(cloud|local|ollama)$", "", model_name)
    for key, info in backends.items():
        if re.sub(r":(cloud|local|ollama)$", "", key) == stem:
            return info
    return None


def _shadow_default_row(row: Dict[str, Any]) -> None:
    """Persist a shadow observation. Never raises."""
    try:
        _log_request_to_db(**row)
    except Exception as e:                                  # pragma: no cover
        logger.warning("Shadow: failed to log observation: %s", e)


async def run_shadow_experiment(
    experiment: Any,
    backend: Optional[Dict[str, Any]] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    request_body: Optional[Dict[str, Any]] = None,
    request_id: str = "",
    session_id: str = "",
    incumbent_model: str = "",
    on_row: Any = None,
    conn: Any = None,
) -> None:
    """Call the experiment's candidate on this request and record the result.

    Returns ``None`` — always. The candidate's output is deliberately
    discarded: shadow must never change what the user receives, which is the
    only property that makes it safe to run against live traffic.

    Guarantees, each covered by a test:
      * the candidate is genuinely invoked (otherwise the data is fiction);
      * the call is forced non-streaming (a stream we discard would hang);
      * cost is attributed to the candidate model, not the incumbent;
      * a candidate failure is recorded as a failure and never propagates.
    """
    if experiment is None or not getattr(experiment, "enabled", False):
        return
    if not getattr(experiment, "is_shadow", False):
        return

    if backend is None:
        backend = _resolve_candidate_backend(
            experiment.model, getattr(experiment, "provider", "")
        )
        if backend is None:
            logger.warning(
                "Shadow: no backend found for candidate %s — skipping", experiment.model
            )
            return

    candidate_model = experiment.model
    body = dict(request_body or {})
    # Force a non-streaming call: we discard the output, and an unconsumed
    # stream would hold the connection until timeout.
    body["stream"] = False
    body.pop("stream_options", None)
    if messages is not None:
        body["messages"] = messages

    sem = _get_shadow_semaphore()
    t0 = time.time()
    content = ""
    in_tok = out_tok = 0
    success = True
    error_type = ""

    async with sem:
        try:
            result = await asyncio.wait_for(
                proxy_to_backend(backend, list(messages or []), body),
                timeout=_SHADOW_TIMEOUT_SECONDS,
            )
            try:
                usage = (result or {}).get("usage", {}) or {}
                in_tok = int(usage.get("prompt_tokens") or 0)
                out_tok = int(usage.get("completion_tokens") or 0)
                choice = (((result or {}).get("choices") or [{}])[0] or {})
                msg = choice.get("message") or {}
                content = str(msg.get("content") or "")
                if not content:
                    success = False
                    error_type = "shadow_empty"
            except Exception:
                success = False
                error_type = "shadow_unparseable"
        except asyncio.TimeoutError:
            success = False
            error_type = "shadow_timeout"
        except Exception as e:
            success = False
            error_type = f"shadow_error:{type(e).__name__}"

    latency = time.time() - t0

    # Quality is scored only when we have both a source and a candidate answer;
    # an unscored observation must stay NULL rather than default to a number.
    quality_score = None
    quality_method = ""
    if content and messages:
        try:
            from quality import score_summary

            source = "\n".join(
                str(m.get("content") or "")
                for m in messages
                if isinstance(m, dict) and m.get("role") == "user"
            )
            if source.strip():
                qs = score_summary(source, content)
                quality_score = float(qs.score)
                quality_method = qs.method
        except Exception as e:                              # pragma: no cover
            logger.debug("Shadow: quality scoring failed: %s", e)

    cost_usd, cost_unknown = compute_cost_fields(candidate_model, in_tok, out_tok, conn=conn)

    row: Dict[str, Any] = {
        "model_used": candidate_model,
        "provider": backend.get("provider", ""),
        "task_type": "shadow",
        "complexity_score": 0.0,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_seconds": latency,
        "routing_time_ms": 0,
        "success": success,
        "escalated": False,
        "error_type": error_type,
        "request_id": request_id,
        "requested_model": incumbent_model,
        "streaming": False,
        "workload_type": "shadow",
        "final_model": candidate_model,
        "routing_reason": f"shadow:{experiment.name}",
        "cost_usd": cost_usd,
        "cost_unknown": cost_unknown,
        "quality_score": quality_score,
        "quality_method": quality_method,
        "pricing_version": _pricing_version(),
        "experiment": getattr(experiment, "name", ""),
        "experiment_arm": "shadow",
        "is_shadow": True,
    }

    try:
        (on_row or _shadow_default_row)(row)
    except Exception as e:                                  # pragma: no cover
        logger.warning("Shadow: on_row callback failed: %s", e)

    return None


def _experiment_matches(experiment: Experiment, workload_type: str, tier: int = 0) -> bool:
    """Re-exported match check so the endpoint and traffic_split agree."""
    from traffic_split import pick_experiment

    picked = pick_experiment([experiment], workload=workload_type, tier=tier)
    return picked is not None


def _find_abandoned_streams(db: Any, older_than_seconds: int = 300) -> List[str]:
    """Return request_ids whose streaming start row has no completion row.

    Streaming requests are logged twice: a ``streaming_in_progress`` start row,
    then a completion/failure row with the same request_id. This helper makes
    observability actionable by separating genuinely unpaired/abandoned streams
    from normal in-flight rows. It does not mutate history.
    """
    from datetime import datetime, timezone, timedelta

    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)).isoformat()
    try:
        rows = db.execute(
            """
            SELECT s.request_id
            FROM router_logs s
            WHERE s.streaming = 1
              AND s.error_type = 'streaming_in_progress'
              AND s.request_id != ''
              AND s.timestamp < ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM router_logs c
                  WHERE c.request_id = s.request_id
                    AND c.id != s.id
                    AND c.error_type != 'streaming_in_progress'
              )
            ORDER BY s.timestamp ASC
            """,
            (cutoff,),
        ).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logger.warning("Failed to query abandoned streams: %s", e)
        return []


def _get_httpx_client() -> httpx.AsyncClient:
    """Get or create a shared httpx client with connection pooling."""
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=300.0)
    return _httpx_client


def _get_db_connection() -> Any:
    """Get or create a persistent SQLite connection."""
    global _sqlite_conn, _sqlite_lock
    if _sqlite_conn is None:
        import sqlite3
        import threading
        _sqlite_lock = threading.Lock()
        db_path = Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db"
        _sqlite_conn = sqlite3.connect(str(db_path))
        _ensure_log_columns(_sqlite_conn)
    return _sqlite_conn


def _get_sqlite_lock() -> Any:
    """Return the connection mutex, creating it if the connection was injected.

    Tests replace ``_get_db_connection`` with a fixture connection, which skips
    the lazy ``_sqlite_lock`` creation above. Without this the logging path
    raises "'NoneType' object does not support the context manager protocol"
    and silently drops every row.
    """
    global _sqlite_lock
    if _sqlite_lock is None:
        import threading
        _sqlite_lock = threading.Lock()
    return _sqlite_lock


def _invalidate_backends_cache():
    """Force refresh of the backends cache on next request."""
    global _backends_cache_time
    _backends_cache_time = 0

# ── Self-identification ───────────────────────────────────────────────────────

# The endpoint's own identity — used to skip itself in routing
BIGGIE_PROVIDER_NAME = "biggie-llm"
BIGGIE_MODEL_NAMES = {"biggie-router", "biggie-llm"}

# ── Configuration ─────────────────────────────────────────────────────────────

# Built-in provider base URLs (Hermes knows these internally)
BUILTIN_PROVIDER_URLS = {
    "ollama-cloud": "https://ollama.com/v1",
    "openai-codex": "https://chatgpt.com/backend-api/codex",
    "openrouter": "https://openrouter.ai/api/v1",
    "minimax": "https://api.minimax.chat/v1",
    "bedrock": "",  # AWS IAM — no URL
}

# Built-in provider API key env vars
BUILTIN_PROVIDER_KEYS = {
    "ollama-cloud": "OLLAMA_API_KEY",
    "openai-codex": "OPENAI_CODEX_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}

# Routing profile: cheap, goldilocks, expensive
ROUTING_PROFILE = os.environ.get("BIGGIE_ROUTING_PROFILE", "goldilocks").lower()

# Compression settings
COMPRESSION_LEVEL = os.environ.get("BIGGIE_COMPRESSION", "standard").lower()
if COMPRESSION_LEVEL not in ("off", "lite", "standard", "structural", "aggressive"):
    COMPRESSION_LEVEL = "standard"

# Path to Hermes config
HERMES_CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"

# Port and host
PORT = int(os.environ.get("BIGGIE_LLM_PORT", "8080"))
HOST = os.environ.get("BIGGIE_LLM_HOST", "127.0.0.1")


def load_hermes_config() -> Dict[str, Any]:
    """Load Hermes config.yaml and extract model/provider info.

    Returns a dict with:
    - providers: {name: {base_url, api_key_env, type}}
    - fallback_chain: [{provider, model}, ...]
    - default_model: str
    - default_provider: str
    """
    if not HERMES_CONFIG_PATH.exists():
        logger.warning("Hermes config not found at %s", HERMES_CONFIG_PATH)
        return {"providers": {}, "fallback_chain": [], "default_model": "", "default_provider": ""}

    with open(HERMES_CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}

    result: Dict[str, Any] = {
        "providers": {},
        "fallback_chain": [],
        "default_model": "",
        "default_provider": "",
    }

    # Default model
    model_section = config.get("model", {})
    if isinstance(model_section, dict):
        result["default_model"] = model_section.get("default", "")
        result["default_provider"] = model_section.get("provider", "")
    elif isinstance(model_section, str):
        result["default_model"] = model_section

    # Provider definitions
    providers = config.get("providers", {})
    for name, pconf in providers.items():
        if not isinstance(pconf, dict):
            continue
        # Skip ourselves
        if name == BIGGIE_PROVIDER_NAME:
            continue

        base_url = pconf.get("base_url", "")
        api_key_env = ""
        provider_type = pconf.get("type", "")

        # Extract API key env var from type or args
        if provider_type == "openai-codex":
            api_key_env = "OPENAI_CODEX_API_KEY"
        elif provider_type == "openrouter":
            api_key_env = "OPENROUTER_API_KEY"
        elif provider_type == "minimax":
            api_key_env = "MINIMAX_API_KEY"
        elif provider_type == "bedrock":
            api_key_env = ""  # AWS IAM
        elif provider_type == "custom":
            # Custom providers might have an api_key field
            api_key_env = pconf.get("api_key_env", "")

        result["providers"][name] = {
            "base_url": base_url,
            "api_key_env": api_key_env,
            "type": provider_type,
        }

    # Fallback chain
    fallbacks = config.get("fallback_providers", [])
    if isinstance(fallbacks, list):
        for fb in fallbacks:
            if isinstance(fb, dict):
                provider = fb.get("provider", "")
                model = fb.get("model", "")
                # Skip ourselves
                if provider == BIGGIE_PROVIDER_NAME:
                    continue
                result["fallback_chain"].append({
                    "provider": provider,
                    "model": model,
                })

    return result


def discover_backends() -> Dict[str, Dict[str, Any]]:
    """Discover available backends from Hermes config.

    Returns {model_name: {provider, base_url, api_key, backend_model}}
    Results are cached for 60 seconds to avoid re-parsing config on every request.
    """
    global _backends_cache, _backends_cache_time
    now = time.time()
    if _backends_cache and (now - _backends_cache_time) < _BACKENDS_CACHE_TTL:
        return _backends_cache

    # Proactive credential-health check: warn (durably) if the openai-codex
    # access token is approaching expiry. Runs on this 60s refresh, rate-limited
    # internally to one warning per 6h.
    _check_codex_token_age()

    hermes = load_hermes_config()
    backends: Dict[str, Dict[str, Any]] = {}

    # Map Hermes provider names to model names
    # The fallback chain tells us which models are available on which providers
    for fb in hermes.get("fallback_chain", []):
        provider = fb.get("provider", "")
        model = fb.get("model", "")

        if not model or not provider:
            continue

        # Skip ourselves
        if provider == BIGGIE_PROVIDER_NAME:
            continue

        _add_backend(backends, hermes, provider, model)

    # Also discover the primary model/provider from the model section
    # (e.g. gpt-5.5 on openai-codex, which may not be in the fallback chain)
    default_model = hermes.get("default_model", "")
    default_provider = hermes.get("default_provider", "")
    if default_model and default_provider and default_provider != BIGGIE_PROVIDER_NAME:
        # Check if it's already been added via the fallback chain
        if default_model not in backends:
            _add_backend(backends, hermes, default_provider, default_model)

    # Also add local models if they're not already in the chain
    local_models = ["llama3.1:8b", "dolphin3"]
    for m in local_models:
        if m not in backends:
            backends[m] = {
                "provider": "local",
                "hermes_provider": "local-ollama",
                "base_url": "http://127.0.0.1:11434",
                "api_key": "",
                "backend_model": m,
            }

    _backends_cache = backends
    _backends_cache_time = time.time()
    return backends


def _add_backend(
    backends: Dict[str, Dict[str, Any]],
    hermes: Dict[str, Any],
    provider: str,
    model: str,
) -> None:
    """Add a backend to the backends dict if not already present."""
    if model in backends:
        return

    pconf = hermes.get("providers", {}).get(provider, {})
    base_url = pconf.get("base_url", "")
    api_key_env = pconf.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    # Determine the backend model name
    # Hermes model names sometimes have provider suffixes (e.g. "deepseek-v4-flash:cloud")
    # Strip those for the backend call, but keep model tags like ":8b" or ":14b"
    backend_model = re.sub(r":(cloud|local|ollama)$", "", model)

    # Determine the provider type for the backend URL
    if provider in ("local-ollama", "mac-ollama"):
        backend_provider = "local"
    elif provider == "openai-codex":
        backend_provider = "openai-codex"
    else:
        backend_provider = provider

    # If no base_url from config, use built-in URL
    if not base_url:
        base_url = BUILTIN_PROVIDER_URLS.get(provider, "")

    # If no api_key from config, try built-in env var
    if not api_key:
        key_env = BUILTIN_PROVIDER_KEYS.get(provider, "")
        if key_env:
            api_key = os.environ.get(key_env, "")

    # For openai-codex, try to get the OAuth access token from auth.json
    if not api_key and provider == "openai-codex":
        try:
            _auth_path = Path.home() / ".hermes" / "auth.json"
            if _auth_path.exists():
                with open(_auth_path) as _af:
                    _auth_data = json.load(_af)
                _pool = _auth_data.get("credential_pool", {})
                _entries = _pool.get("openai-codex", [])
                for _entry in _entries:
                    if isinstance(_entry, dict):
                        _status = _entry.get("last_status")
                        _error = _entry.get("last_error_code")
                        _token = _entry.get("access_token", "")
                        if _status == "exhausted" and _error == 429:
                            continue
                        if _token:
                            api_key = _token
                            break
                if not api_key:
                    for _entry in _entries:
                        if isinstance(_entry, dict):
                            _token = _entry.get("access_token", "")
                            if _token:
                                api_key = _token
                                break
        except Exception:
            pass

    backends[model] = {
        "provider": backend_provider,
        "hermes_provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "backend_model": backend_model,
    }


# ── Routing profile ───────────────────────────────────────────────────────────

def apply_routing_profile(decision: RoutingDecision, features: Dict[str, Any]) -> RoutingDecision:
    """Adjust the routing decision based on the configured profile.

    Profiles:
      - cheap:      prefer cheapest model, even if it's slightly underpowered
      - goldilocks: balanced — prefer mid-tier, escalate only when needed
      - expensive:  prefer most capable model (current behaviour)
    """
    if ROUTING_PROFILE == "cheap":
        # Bias toward cheaper models — reduce min_tier by 1-2
        # Already handled by route_task() with default params
        return decision

    elif ROUTING_PROFILE == "expensive":
        # Bias toward more capable models — increase min_tier by 1-2
        # Re-route with a higher complexity score
        boosted_features = dict(features)
        boosted_features["complexity_score"] = min(features["complexity_score"] + 0.2, 1.0)
        boosted_features["instruction_count"] = features["instruction_count"] + 1

        return route_task(
            complexity_score=boosted_features["complexity_score"],
            task_type=boosted_features["task_type"],
            has_niche_references=boosted_features["has_niche_references"],
            has_format_constraint=boosted_features["has_format_constraint"],
            instruction_count=boosted_features["instruction_count"],
        )

    # goldilocks — default, no adjustment
    return decision


# ── Feature extraction from chat messages ─────────────────────────────────────

def extract_features_from_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract routing features from a chat completion request's messages.

    Uses the last 4-10 user messages for complexity scoring — captures the
    recent direction of the conversation without being dominated by the very
    first message from hours ago. Also uses the full message count and tool
    call count for session length context.
    """
    # Collect all user messages and count tool calls in a single pass
    user_messages = []
    tool_call_count = 0
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_messages.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_messages.append(part.get("text", ""))
                        break
        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_call_count += len(msg["tool_calls"])

    # Use the last 8 user messages (or all if fewer) for complexity scoring
    recent_window = 8
    recent_user_msgs = user_messages[-recent_window:] if len(user_messages) > recent_window else user_messages
    combined_prompt = "\n".join(recent_user_msgs)

    # Use the last user message for task classification (most recent context)
    last_prompt = user_messages[-1] if user_messages else ""

    message_count = len(messages)

    complexity = score_complexity(combined_prompt, tool_call_count, message_count)
    task_type = classify_task(last_prompt, tool_call_count)
    niche = has_niche_references(combined_prompt)
    fmt = has_format_constraint(combined_prompt)
    instr_count = count_instructions(combined_prompt)

    return {
        "complexity_score": complexity,
        "task_type": task_type,
        "has_niche_references": niche,
        "has_format_constraint": fmt,
        "instruction_count": instr_count,
        "prompt_text": combined_prompt,
        "tool_call_count": tool_call_count,
        "message_count": message_count,
        "context_tokens": _estimate_context_tokens(messages),
    }


def _message_text(content: Any) -> str:
    """Return text from OpenAI-style message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content") or part.get("input_text") or part.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _estimate_context_tokens(messages: List[Dict[str, Any]]) -> int:
    """Cheap token estimate for routing policy decisions."""
    chars = 0
    for msg in messages:
        chars += len(_message_text(msg.get("content", "")))
        # Account for OpenAI message framing and tool-call metadata roughly.
        chars += 16
        if msg.get("tool_calls"):
            chars += len(json.dumps(msg.get("tool_calls"), ensure_ascii=False))
    return max(1, chars // 4)


def detect_workload_type(
    messages: List[Dict[str, Any]],
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Detect first-class Biggie workload type.

    Hermes auxiliary session compression should be routed natively, not treated
    as arbitrary high-complexity chat. Prefer explicit metadata/headers when
    present; fall back to stable Hermes compaction prompt/status phrases.
    """
    headers_l = {str(k).lower(): str(v).strip().lower() for k, v in (headers or {}).items()}
    explicit = (
        headers_l.get("x-hermes-auxiliary-task")
        or headers_l.get("x-biggie-workload")
        or headers_l.get("x-hermes-task")
    )
    raw_metadata = body.get("metadata")
    metadata: Dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    explicit = explicit or str(
        metadata.get("hermes_task")
        or metadata.get("auxiliary_task")
        or metadata.get("workload_type")
        or metadata.get("task")
        or ""
    ).strip().lower()

    if explicit in {"compression", "context_compression", "session_compression", "compaction"}:
        return "session_compression"

    prompt = "\n".join(_message_text(m.get("content", "")) for m in messages).lower()
    compression_markers = (
        "context compaction",
        "context compression",
        "compacting context",
        "summarizing earlier conversation",
        "summarising earlier conversation",
        "summarize earlier conversation",
        "summarise earlier conversation",
        "conversation summary",
        "compressed summary",
        "compress the conversation",
        "summarize the conversation so i can continue",
        "summarise the conversation so i can continue",
        "preserve facts, decisions",
        "preserve key facts",
    )
    if any(marker in prompt for marker in compression_markers):
        return "session_compression"

    return "normal_chat"


def compression_level_for_workload(
    request: Request,
    workload_type: str,
    context_tokens: int = 0,
    requires_tools: bool = False,
) -> str:
    """Choose Biggie request-compression level for the detected workload.

    When a request carries OpenAI tool schemas (``requires_tools=True``) we
    SKIP compression for NORMAL workloads ("off"). Compressing tool-calling
    traffic costs tokens up front (to compress) without benefit — the
    downstream model still needs the full tool schema + history to execute
    calls. This keeps tool requests lean and avoids paying to compress before
    calling.

    EXCEPTION: a giant session_compaction (>=50k context) carrying tools must
    STILL get structural repetition collapse. Such a payload is too large to
    forward raw to a paid summariser model — the structural pass is what makes
    the compaction affordable. The tool-skip rule only applies to normal
    workloads, not to session compactions that would otherwise blow the paid
    context budget.
    """
    header_level = request.headers.get("X-Compression-Level")
    if header_level in ("off", "lite", "standard", "structural", "aggressive"):
        return header_level
    if workload_type == "session_compression":
        if COMPRESSION_LEVEL == "off":
            return "off"
        # Small/normal compactions stay conservative, but giant session payloads
        # need structural repetition collapse before spending paid model context.
        # This applies even when the payload carries tool schemas — a >=50k
        # compaction is too large to forward raw.
        if context_tokens >= 50_000:
            return "structural"
        # Small session compactions carrying tools stay lean (skip compression).
        if requires_tools:
            return "off"
        return "lite"
    if requires_tools:
        return "off"
    return COMPRESSION_LEVEL


# ── Request logging ────────────────────────────────────────────────────────────

def _get_usage_tokens(response: dict, direction: str) -> int:
    """Extract token counts from an OpenAI-compatible response."""
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if direction == "input":
        return usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
    return usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0


def _response_has_empty_content(response: Any) -> bool:
    """Return True if a chat-completion response has empty assistant content.

    A backend that returns HTTP 200 with an empty assistant message is a
    degenerate success — the model produced no output. Treating it as a
    success lets empty turns flow through to the client, which then wastes
    its turn budget self-healing them. This helper lets the router escalate
    on empty content exactly as it does on HTTPException.

    A response that carries structured ``tool_calls`` is NOT empty even when
    ``content`` is blank — the model produced a real action for the client to
    execute. Such responses must not be escalated.
    """
    if not isinstance(response, dict):
        return False
    choices = response.get("choices") or []
    if not choices:
        return False
    choice = choices[0]
    if not isinstance(choice, dict):
        return False
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        return False
    # Structured tool calls = meaningful action, never "empty".
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    # Content can be a list of parts (e.g. Codex normalization)
    if isinstance(content, list):
        return all(
            (part.get("text") or "").strip() == ""
            for part in content
            if isinstance(part, dict)
        )
    return False


def _parse_tool_arguments(arguments: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse a tool-call arguments payload, returning (args, error).

    OpenAI-compatible backends should return ``function.arguments`` as a JSON
    object string. Cheap/fallback models sometimes emit XML/attribute-like tool
    syntax inside the JSON key, e.g. ``{"command=\"echo hi\" timeout=\"10\"":""}``.
    That parses as JSON but has no required ``command`` field, so Hermes later
    calls terminal(command=None). Treat that as a model-output failure so the
    router can escalate before the malformed tool call reaches Hermes.
    """
    if isinstance(arguments, dict):
        return arguments, None
    if arguments in (None, ""):
        return {}, None
    if not isinstance(arguments, str):
        return None, f"arguments must be a JSON object string, got {type(arguments).__name__}"
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as e:
        return None, f"arguments are not valid JSON: {e.msg}"
    if not isinstance(parsed, dict):
        return None, f"arguments must decode to object, got {type(parsed).__name__}"
    return parsed, None


def _tool_required_arg_error(function_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Return a validation error when tool args are structurally unusable."""
    if function_name == "terminal":
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            # Explicitly call out the common leaked-attribute shape to make logs useful.
            weird_keys = [k for k in args.keys() if isinstance(k, str) and "command=" in k]
            suffix = f" (saw leaked attribute key {weird_keys[0]!r})" if weird_keys else ""
            return f"terminal tool_call missing non-empty string 'command'{suffix}"
    return None


# Tool names are simple identifiers: letters, digits, underscores, hyphens.
# Anything else (parentheses, quotes, equals, spaces) means the model leaked
# inline call syntax into the function name field, e.g. terminal(command="gh.
_TOOL_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


def _malformed_function_name_error(function_name: str) -> Optional[str]:
    """Return an error string if a function name is structurally invalid.

    Cloud fallback models sometimes emit the entire tool invocation as the
    ``function.name`` field, e.g. ``terminal(command="gh repo list ...")``.
    That is not a valid tool name — it contains parentheses, quotes, etc. —
    and Hermes cannot dispatch it. Treat it as a model-output failure so the
    router escalates.
    """
    if not function_name or not isinstance(function_name, str):
        return None  # absent name handled elsewhere; don't double-report
    if _TOOL_NAME_RE.match(function_name):
        return None
    return (
        f"malformed function name {function_name!r}: contains characters outside "
        f"[a-zA-Z0-9_-] — model likely leaked inline tool-call syntax into name field"
    )


def _normalize_tool_call(tc: Any) -> Tuple[str, str]:
    """Return (name, normalized_arguments) for a tool call, or empty on junk."""
    if not isinstance(tc, dict):
        return ("", "")
    fn = tc.get("function") or {}
    if not isinstance(fn, dict):
        return ("", "")
    name = fn.get("name") or ""
    args = fn.get("arguments")
    if not isinstance(name, str):
        name = ""
    if isinstance(args, dict):
        try:
            args = json.dumps(args, sort_keys=True)
        except Exception:
            args = ""
    elif isinstance(args, str):
        # Canonicalize JSON-string args (key order can vary between otherwise
        # identical tool calls — a semantic duplicate the model may emit as
        # part of a degeneration loop). Non-JSON strings are compared verbatim.
        try:
            parsed = json.loads(args)
            if isinstance(parsed, (dict, list)):
                args = json.dumps(parsed, sort_keys=True)
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        args = ""
    return (name, args)


def _degeneration_error(response: Any) -> Optional[str]:
    """Return an error string if a chat-completion response shows degeneration.

    Degeneration is the failure class where the model is NOT erroring (non-empty
    content, well-formed tool calls) but is stuck in a repetition loop — emitting
    the same tool call twice in one turn, or repeatedly spitting the same tokens.

    ``None`` means the response is safe to forward. Any string means the selected
    model degenerated and the router should escalate exactly as it does for
    empty-content/malformed-tool-call failures, so corrupt output never reaches
    the client (Hermes) where it would waste turn budget or break downstream
    parsing (e.g. failing build assertions).

    Detects two distinct signatures:
      1. Duplicate tool calls — the same (name, arguments) emitted 2+ times in
         one response. The agent logs "Removed duplicate tool call" for these,
         but by then the degenerate turn has already consumed a model call and
         returned corrupt output; better to catch and escalate server-side.
      2. Runaway text repetition — a word/token repeated many times in a row,
         the classic degeneration loop (e.g. ``turning_offturning_off...``).
    """
    if not isinstance(response, dict):
        return None
    choices = response.get("choices") or []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue

        # 1) Duplicate tool calls in a single turn.
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) >= 2:
            seen: Dict[Tuple[str, str], int] = {}
            for tc in tool_calls:
                key = _normalize_tool_call(tc)
                if not key[0]:
                    continue
                seen[key] = seen.get(key, 0) + 1
                if seen[key] >= 2:
                    return (
                        f"degenerate tool calls: {key[0]!r} emitted {seen[key]} times "
                        f"with identical arguments in a single response"
                    )

        # 2) Runaway text repetition.
        content = message.get("content")
        if isinstance(content, str) and len(content) >= 12:
            rep_err = _repetition_error(content)
            if rep_err:
                return f"degenerate content repetition: {rep_err}"

    return None


_DEGEN_WORD_RE = None  # compiled lazily below (module import order safety)


def _repetition_error(text: str) -> Optional[str]:
    """Return a description if ``text`` shows runaway repetition, else None.

    Signature: a short token or word (>=2 chars, <=20) repeated 8+ times in a
    row, with or without intervening whitespace (e.g. ``turning_offturning_off``
    or ``response response response``). 8+ identical consecutive units is far
    beyond any legitimate prose and reliably marks a degeneration loop.
    """
    global _DEGEN_WORD_RE
    if _DEGEN_WORD_RE is None:
        # Word: 2-20 word-chars, no boundary requirement (degens repeat inside
        # runs too). Repeated 8+ times, back-to-back or space-separated.
        _DEGEN_WORD_RE = re.compile(
            r"(\b[a-zA-Z_]{2,20}\b)(?:\1|[ \t]+(?:\1)){7,}",
            re.IGNORECASE,
        )
    # Also catch concatenated no-space repeats of common short tokens like
    # "turning_offturning_off" (>=2 chars, 8+ times directly adjacent).
    m = _DEGEN_WORD_RE.search(text)
    if m:
        return f"{m.group(1)!r} repeated {m.group(0).count(m.group(1)) + 1} times consecutively"
    # Concatenated no-space runs of a short word-like token (e.g.
    # "turning_offturning_off"). Require the token to contain at least 2
    # DISTINCT characters so runs of identical symbols (=====, aaaa, -----)
    # — legitimate in code/plain text — are never flagged.
    concat = re.search(
        r"([a-zA-Z_]{3,40}?[a-zA-Z0-9_-]*[a-zA-Z_]){0}([a-zA-Z_]{2,40}[-_]?[a-zA-Z0-9_]*)\2{7,}",
        text,
    )
    if concat:
        tok = concat.group(2)
        if len(set(tok)) >= 2:
            return f"{tok!r} repeated in a concatenated loop"
    return None


def _malformed_tool_call_error(response: Any) -> Optional[str]:
    """Return an error string if a chat-completion response has bad tool args.

    ``None`` means the response is safe to forward. Any string means the selected
    model emitted a malformed tool call and the router should escalate exactly as
    it does for empty-content/HTTP failures.
    """
    if not isinstance(response, dict):
        return None
    choices = response.get("choices") or []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name") or ""
            # Check function name first — a malformed name means the model
            # leaked inline syntax (e.g. terminal(command="gh) and the args
            # will be garbage too.
            name_err = _malformed_function_name_error(name)
            if name_err:
                return name_err
            args, parse_err = _parse_tool_arguments(fn.get("arguments", "{}"))
            if parse_err:
                return f"malformed tool_call arguments for {name or '<unknown>'}: {parse_err}"
            if args is None:
                continue
            arg_err = _tool_required_arg_error(name, args)
            if arg_err:
                return arg_err
    return None


def _malformed_tool_call_delta_error(data: str) -> Optional[str]:
    """Return an error string if a streaming SSE delta has bad tool args."""
    if data == "[DONE]":
        return None
    try:
        evt = json.loads(data)
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(evt, dict) or evt.get("error"):
        return None
    choices = evt.get("choices") or []
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        delta = ch.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        tool_calls = delta.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name") or ""
            # Check function name (malformed names are detectable even before
            # arguments arrive in the stream).
            name_err = _malformed_function_name_error(name)
            if name_err:
                return name_err
            # Streaming APIs may send the function name first and arguments later;
            # absent/empty arguments are therefore not malformed yet. But when a
            # non-empty arguments string is present, it must be structurally usable.
            if "arguments" not in fn or fn.get("arguments") in (None, ""):
                continue
            args, parse_err = _parse_tool_arguments(fn.get("arguments"))
            if parse_err:
                return f"malformed tool_call arguments for {name or '<unknown>'}: {parse_err}"
            if args is None:
                continue
            arg_err = _tool_required_arg_error(name, args)
            if arg_err:
                return arg_err
    return None


def _log_request_to_db(
    model_used: str,
    provider: str,
    task_type: str,
    complexity_score: float,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    routing_time_ms: int,
    success: bool = True,
    escalated: bool = False,
    error_type: str = "",
    compression_level: str = "off",
    compression_savings_pct: float = 0.0,
    compression_time_ms: float = 0.0,
    request_id: str = "",
    requested_model: str = "",
    streaming: bool = False,
    workload_type: str = "normal_chat",
    requires_tools: bool = False,
    context_tokens: int = 0,
    empty_stream: bool = False,
    saw_content: bool = True,
    saw_tool_calls: bool = False,
    final_model: str = "",
    routing_reason: str = "",
    cost_usd: float = 0.0,
    cost_unknown: int = 1,
    quality_score: Optional[float] = None,
    quality_method: str = "",
    pricing_version: str = "",
    experiment: str = "",
    experiment_arm: str = "",
    is_shadow: bool = False,
):
    """Log a single request to the router_logs DB for analysis.

    Uses a persistent SQLite connection to avoid open/close overhead.
    Streaming requests are logged at both route-start and completion so the
    router's behaviour on streaming is observable (FIX 3). No prompt bodies or
    secrets are stored.

    ``cost_unknown`` defaults to 1: an un-priced call is recorded as *unknown*,
    never as a confident zero, so cost rollups can exclude it rather than
    understate spend.
    """
    try:
        from datetime import datetime, timezone

        db = _get_db_connection()
        with _get_sqlite_lock():
            db.execute(
                """INSERT INTO router_logs (
                timestamp, session_id, model_used, provider, task_type,
                input_tokens, output_tokens, latency_seconds, complexity_score,
                success, escalated, error_type,
                compression_level, compression_savings_pct, compression_time_ms,
                request_id, requested_model, streaming, workload_type,
                requires_tools, context_tokens, empty_stream, saw_content,
                saw_tool_calls, final_model, routing_reason,
                cost_usd, cost_unknown, quality_score, quality_method,
                pricing_version, experiment, experiment_arm, is_shadow
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    "",  # session_id — not available at endpoint level
                    model_used,
                    provider,
                    task_type,
                    input_tokens,
                    output_tokens,
                    latency_seconds,
                    complexity_score,
                    1 if success else 0,
                    1 if escalated else 0,
                    error_type,
                    compression_level,
                    compression_savings_pct,
                    compression_time_ms,
                    request_id,
                    requested_model,
                    1 if streaming else 0,
                    workload_type,
                    1 if requires_tools else 0,
                    context_tokens,
                    1 if empty_stream else 0,
                    1 if saw_content else 0,
                    1 if saw_tool_calls else 0,
                    final_model or model_used,
                    routing_reason,
                    float(cost_usd or 0.0),
                    1 if cost_unknown else 0,
                    quality_score,
                    quality_method,
                    pricing_version,
                    experiment,
                    experiment_arm,
                    1 if is_shadow else 0,
                ),
            )
        db.commit()
    except Exception as e:
        logger.warning("Failed to log request to DB: %s", e)


# ── Backend proxy ─────────────────────────────────────────────────────────────

def _codex_input_from_responses_items(responses_input: list) -> list:
    """Convert Hermes' Responses input items into the Codex /responses body.

    ``_chat_messages_to_responses_input`` returns a MIX of item types for a
    tool conversation:
      - {"role": "user"/"assistant", "content": ...}  -> message items
      - {"type": "function_call", "call_id", "name", "arguments"}
      - {"type": "function_call_output", "call_id", "output"}

    Only the plain message dicts must be wrapped as "message" items. The
    function_call / function_call_output items MUST be passed through verbatim
    — wrapping them as empty user messages (role defaults to "user", content
    "") drops the tool result and the assistant's function_call from the
    upstream request, so gpt-5.5 never sees the tool output and repeats the
    same call every turn (the repeated tool-call loop).
    """
    codex_input = []
    for item in responses_input:
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("function_call", "function_call_output"):
            # Pass through verbatim — preserves call_id pairing and tool
            # result binding across turns.
            codex_input.append(item)
            continue
        role = item.get("role", "user")
        content = item.get("content", "")
        text_type = "output_text" if role == "assistant" else "input_text"
        codex_input.append({
            "type": "message",
            "role": role,
            "content": [{"type": text_type, "text": content}],
        })
    return codex_input


async def proxy_to_backend(
    backend: Dict[str, Any],
    messages: List[Dict[str, Any]],
    request_body: Dict[str, Any],
) -> Any:
    """Proxy a chat completion request to the chosen backend.

    Uses the backend's native API format:
    - Local Ollama: /api/chat
    - OpenAI-compatible: /v1/chat/completions
    - OpenAI Codex: /responses (via Hermes' own adapter)
    """
    base_url = backend.get("base_url", "")
    api_key = backend.get("api_key", "")
    backend_model = backend.get("backend_model", "")
    provider = backend.get("provider", "")

    if not base_url:
        raise HTTPException(status_code=502, detail=f"No base_url for backend")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # ── OpenAI Codex (Responses API) ──────────────────────────────────────
    if provider == "openai-codex":
        # Use Hermes' own adapter to convert chat messages to Responses format
        import sys
        _hermes_path = str(Path.home() / ".hermes" / "hermes-agent")
        if _hermes_path not in sys.path:
            sys.path.insert(0, _hermes_path)

        from agent.codex_responses_adapter import (
            _chat_messages_to_responses_input,
            _normalize_codex_response,
        )

        responses_input = _chat_messages_to_responses_input(messages)
        codex_input = _codex_input_from_responses_items(responses_input)

        body = {
            "model": backend_model,
            "input": codex_input,
            "store": False,
            "stream": True,
        }

        for param in ("temperature", "top_p", "stop", "frequency_penalty", "presence_penalty"):
            if param in request_body:
                body[param] = request_body[param]

        # Forward tool schemas so the model can call terminal/file tools.
        # Convert chat-completions tool schemas to Responses function-tool
        # schemas using Hermes' own adapter (mirrors the codex transport).
        tools = request_body.get("tools")
        if tools:
            try:
                from agent.codex_responses_adapter import _responses_tools
                converted = _responses_tools(tools)
                if converted:
                    body["tools"] = converted
            except Exception as e:
                logger.warning("Failed to convert tools for codex backend: %s", e)

        url = f"{base_url.rstrip('/')}/responses"

        try:
            client = _get_httpx_client()
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    detail = f"Backend error: {error_text[:500].decode()}"
                    logger.error("Backend %s returned %d: %s", provider, resp.status_code, detail)
                    _handle_backend_status(provider, backend_model, resp.status_code)
                    raise HTTPException(status_code=502, detail=detail)

                # Collect SSE events — assemble the response from stream events
                responses_data = None
                collected_output = []
                current_event = None
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            if current_event == "response.completed":
                                responses_data = event.get("response") or event
                                if collected_output:
                                    responses_data["output"] = collected_output
                                break
                            elif current_event == "response.output_item.added":
                                item = event.get("item") or event
                                if isinstance(item, dict) and item.get("type") in ("message", "function_call"):
                                    collected_output.append(item)
                            elif current_event == "response.output_item.done":
                                # The completed item carries the final arguments
                                # for a function_call (the .added event has empty
                                # arguments). Replace the placeholder with it.
                                item = event.get("item") or event
                                if isinstance(item, dict) and item.get("type") == "function_call":
                                    for i, existing in enumerate(collected_output):
                                        if existing.get("id") == item.get("id"):
                                            collected_output[i] = item
                                            break
                                    else:
                                        collected_output.append(item)
                            elif current_event == "response.content_part.done":
                                part = event.get("part", {})
                                item_id = event.get("item_id", "")
                                if part.get("type") == "output_text":
                                    for msg in collected_output:
                                        if msg.get("id") == item_id:
                                            msg["content"] = [part]
                                            break
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = f"Backend error: {e.response.text[:500]}"
            logger.error("Backend %s returned %d: %s", provider, status, detail)
            _handle_backend_status(provider, backend_model, status)
            raise HTTPException(status_code=502, detail=detail)
        except httpx.RequestError as e:
            logger.error("Backend %s request failed: %s", provider, e)
            raise HTTPException(status_code=502, detail=f"Backend request failed: {e}")

        if not responses_data:
            raise HTTPException(status_code=502, detail="No response data from Codex API")

        # Convert Responses format back to Chat Completions format
        try:
            from types import SimpleNamespace

            def _dict_to_obj(d):
                if isinstance(d, dict):
                    return SimpleNamespace(**{k: _dict_to_obj(v) for k, v in d.items()})
                elif isinstance(d, list):
                    return [_dict_to_obj(v) for v in d]
                return d

            response_obj = _dict_to_obj(responses_data)
            normalized = _normalize_codex_response(response_obj)
            # normalized is (assistant_message, finish_reason)
            if isinstance(normalized, tuple):
                msg, reason = normalized
            else:
                msg, reason = normalized, "stop"

            content = ""
            tool_calls = None
            if isinstance(msg, dict):
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls")
            elif hasattr(msg, "content"):
                content = msg.content
                tool_calls = getattr(msg, "tool_calls", None)

            message = {
                "role": "assistant",
                "content": content or "",
            }
            if tool_calls:
                # Normalize SimpleNamespace tool_calls to plain dicts for JSON
                message["tool_calls"] = [
                    {
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(tc.function, "name", "") if hasattr(tc, "function") else "",
                            "arguments": getattr(tc.function, "arguments", "") if hasattr(tc, "function") else "",
                        },
                    }
                    for tc in tool_calls
                ]

            return {
                "id": responses_data.get("id", ""),
                "object": "chat.completion",
                "created": responses_data.get("created_at", 0),
                "model": responses_data.get("model", backend_model),
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": reason or "stop",
                }],
                "usage": responses_data.get("usage", {}),
            }
        except Exception as e:
            logger.error("Failed to normalize Codex response: %s", e)
            raise HTTPException(status_code=502, detail=f"Codex response normalization failed: {e}")

    # ── Local Ollama ───────────────────────────────────────────────────────
    if provider == "local":
        body = {
            "model": backend_model,
            "messages": messages,
            "stream": False,
        }
        for param in ("temperature", "top_p", "max_tokens", "stop", "frequency_penalty", "presence_penalty"):
            if param in request_body:
                body[param] = request_body[param]

        url = f"{base_url.rstrip('/')}/api/chat"

        try:
            client = _get_httpx_client()
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            ollama_data = resp.json()
            # Convert Ollama format to OpenAI Chat Completions format
            import time
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": ollama_data.get("model", backend_model),
                "choices": [{
                    "index": 0,
                    "message": ollama_data.get("message", {"role": "assistant", "content": ""}),
                    "finish_reason": ollama_data.get("done_reason", "stop"),
                }],
                "usage": {
                    "prompt_tokens": ollama_data.get("prompt_eval_count", 0),
                    "completion_tokens": ollama_data.get("eval_count", 0),
                    "total_tokens": (ollama_data.get("prompt_eval_count", 0) or 0) + (ollama_data.get("eval_count", 0) or 0),
                },
            }
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = f"Backend error: {e.response.text[:500]}"
            logger.error("Backend %s returned %d: %s", provider, status, detail)
            _handle_backend_status(provider, backend_model, status)
            raise HTTPException(status_code=502, detail=detail)
        except httpx.RequestError as e:
            logger.error("Backend %s request failed: %s", provider, e)
            raise HTTPException(status_code=502, detail=f"Backend request failed: {e}")

    # ── OpenAI-compatible (Ollama Cloud, etc.) ─────────────────────────────
    body = {
        "model": backend_model,
        "messages": messages,
        "stream": False,
    }
    for param in ("temperature", "top_p", "max_tokens", "stop", "frequency_penalty", "presence_penalty"):
        if param in request_body:
            body[param] = request_body[param]

    # Forward tool schemas so the model can emit structured tool_calls.
    # Ollama Cloud's /chat/completions accepts standard OpenAI tool format
    # natively — no conversion needed (unlike Codex Responses API).
    if request_body.get("tools"):
        body["tools"] = request_body["tools"]
    if request_body.get("tool_choice"):
        body["tool_choice"] = request_body["tool_choice"]

    url = f"{base_url.rstrip('/')}/chat/completions"

    try:
        client = _get_httpx_client()
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        detail = f"Backend error: {e.response.text[:500]}"
        logger.error("Backend %s returned %d: %s", provider, status, detail)
        _handle_backend_status(provider, backend_model, status)
        raise HTTPException(status_code=502, detail=detail)
    except httpx.RequestError as e:
        logger.error("Backend %s request failed: %s", provider, e)
        raise HTTPException(status_code=502, detail=f"Backend request failed: {e}")


@dataclass
class _StreamPreflight:
    """Result of a bounded stream preflight, before committing a downstream response.

    The endpoint must not return a StreamingResponse until it knows whether the
    upstream will produce meaningful content. This lets server-side escalation
    happen for empty/error streams instead of surfacing a degenerate success to
    the client (which would then retry the SAME failing backend).

    The probe connection is KEPT OPEN after first content. The downstream stream
    continues iterating the same response — no fresh connection is opened. This
    avoids token-split misalignment and the 10+ second latency of a new round-trip.
    """

    status: str                          # "ok" | "empty" | "error"
    backend_model: str = ""
    provider: str = ""
    error: str = ""
    buffered: List[str] = field(default_factory=list)  # raw SSE data lines to replay
    saw_content: bool = False
    saw_tool_calls: bool = False
    response: Any = None                 # the still-open httpx streaming response
    iterator: Any = None                 # the SAME aiter_lines() iterator to resume
    accumulated: str = ""                # raw streamed content tail, for mid-stream
                                         # degeneration alerting (observability only)


def _sse_delta_parts(data: str) -> Tuple[bool, bool, bool]:
    """Inspect one SSE ``data:`` payload.

    Returns (saw_content, saw_tool_call, is_terminal) where is_terminal is True
    for ``[DONE]`` or an upstream error payload.
    """
    if data == "[DONE]":
        return (False, False, True)
    try:
        evt = json.loads(data)
    except (json.JSONDecodeError, AttributeError):
        return (False, False, False)
    if isinstance(evt, dict) and evt.get("error"):
        return (False, False, True)
    choices = evt.get("choices") or []
    saw_content = False
    saw_tool = False
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        delta = ch.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        c = delta.get("content")
        if isinstance(c, str) and c.strip():
            saw_content = True
        if delta.get("tool_calls"):
            saw_tool = True
    return (saw_content, saw_tool, False)


def _sse_content_text(data: str) -> str:
    """Extract the text content from one SSE ``data:`` payload.

    Returns only the streamed content delta(s) as string, concatenated across
    choices. Tool-call deltas and non-content fields are ignored. Empty string
    when the payload carries no content (or is unparsable / terminal). This is
    the raw accumulation source for mid-stream degeneration detection — it must
    NOT be fed back into the client path.
    """
    if data == "[DONE]":
        return ""
    try:
        evt = json.loads(data)
    except (json.JSONDecodeError, AttributeError):
        return ""
    if not isinstance(evt, dict):
        return ""
    parts: List[str] = []
    for ch in evt.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        delta = ch.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        c = delta.get("content")
        if isinstance(c, str) and c:
            parts.append(c)
    return "".join(parts)


def _alert_stream_degeneration(
    provider: str,
    backend_model: str,
    pattern: str,
    output_tail: str,
) -> None:
    """Emit a durable, full-context ALERT for mid-stream degeneration.

    Called when the streaming tail accumulates text that the repetition guard
    flags as a degeneration loop (the glm-5.3 optooloop failure class). Writes a
    recovery_log row (event ``stream_degeneration_alert``) and logs at ERROR
    with the exact repeated pattern plus a bounded tail of the emitted output,
    so future occurrences are greppable and debuggable without re-routing the
    request. This is observability-first: it does NOT abort or re-route the
    in-flight stream.
    """
    try:
        log_recovery_event(
            backend_model,
            "stream_degeneration_alert",
            (
                f"provider={provider} pattern={pattern!r} "
                f"output_tail={output_tail!r}"
            ),
        )
    except Exception as exc:  # never let alerting break the stream path
        logger.warning("Failed to log stream degeneration alert: %s", exc)
    logger.error(
        "STREAM DEGENERATION provider=%s model=%s pattern=%r output_tail=%r",
        provider,
        backend_model,
        pattern,
        output_tail,
    )


def _scan_stream_content(accumulated: str, minimum: int = 64) -> Optional[str]:
    """Run the degeneration detector over accumulated streamed text.

    Accumulating per-delta and running a regex each time is wasteful and would
    raise latency; this helper runs the same ``_repetition_error`` detector the
    non-streaming guard uses, only once enough content has accumulated to be
    meaningful. Returns the pattern description on a hit, else None.
    """
    if len(accumulated) < minimum:
        return None
    return _repetition_error(accumulated)


async def _preflight_openai_stream(
    backend: Dict[str, Any],
    messages: List[Dict[str, Any]],
    body: Dict[str, Any],
) -> _StreamPreflight:
    """Probe a streaming backend connection and buffer until a decision.

    Buffers upstream SSE events until one of:
      A. first meaningful content delta      -> status "ok"
      B. first structured tool-call delta    -> status "ok"
      C. upstream error                      -> status "error"
      D. [DONE] with no prior content        -> status "empty"

    The probe connection is CLOSED in all cases. For a healthy stream (A/B) the
    caller re-opens a FRESH connection for the actual downstream stream (httpx
    responses can only be consumed once). For C/D the caller escalates
    server-side.
    """
    base_url = backend.get("base_url", "")
    api_key = backend.get("api_key", "")
    backend_model = backend.get("backend_model", "")
    provider = backend.get("provider", "")
    if not base_url:
        return _StreamPreflight(
            status="error", backend_model=backend_model, provider=provider,
            error="no base_url for backend",
        )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    out = {
        "model": backend_model,
        "messages": messages,
        "stream": True,
    }
    for param in ("temperature", "top_p", "max_tokens", "stop", "frequency_penalty", "presence_penalty"):
        if param in body:
            out[param] = body[param]

    # Forward tool schemas so the model can emit structured tool_calls.
    if body.get("tools"):
        out["tools"] = body["tools"]
    if body.get("tool_choice"):
        out["tool_choice"] = body["tool_choice"]

    url = f"{base_url.rstrip('/')}/chat/completions"
    client = _get_httpx_client()
    try:
        req = client.build_request("POST", url, json=out, headers=headers)
        resp = await client.send(req, stream=True)
    except httpx.RequestError as e:
        logger.error("Backend %s stream request failed during preflight: %s", provider, e)
        return _StreamPreflight(
            status="error", backend_model=backend_model, provider=provider,
            error=f"Backend request failed: {e}",
        )
    if resp.status_code != 200:
        try:
            error_text = (await resp.aread()).decode()
        except Exception:
            error_text = ""
        await resp.aclose()
        _handle_backend_status(provider, backend_model, resp.status_code)
        logger.error("Backend %s returned %d during preflight: %s", provider, resp.status_code, error_text[:500])
        return _StreamPreflight(
            status="error", backend_model=backend_model, provider=provider,
            error=f"Backend error: {error_text[:500]}",
        )
    buffered: List[str] = []
    pf = _StreamPreflight(
        status="empty", backend_model=backend_model, provider=provider,
    )
    try:
        # Capture the iterator ONCE. httpx streaming responses are single-use —
        # a second aiter_lines() call on the same response yields nothing. The
        # downstream stream must resume THIS iterator, never re-create it.
        pf.iterator = resp.aiter_lines()
        async for line in pf.iterator:
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                buffered.append(f"data: {data}")
                malformed_tool_error = _malformed_tool_call_delta_error(data)
                if malformed_tool_error:
                    pf.status = "error"
                    pf.error = f"Backend {backend_model} emitted malformed tool_call: {malformed_tool_error}"
                    pf.buffered = buffered
                    await resp.aclose()
                    return pf
                saw_content, saw_tool, is_terminal = _sse_delta_parts(data)
                if saw_content:
                    pf.saw_content = True
                if saw_tool:
                    pf.saw_tool_calls = True
                if saw_content or saw_tool:
                    pf.status = "ok"
                    pf.buffered = buffered
                    pf.response = resp
                    return pf
                if is_terminal:
                    if data == "[DONE]":
                        pf.status = "empty"
                    else:
                        pf.status = "error"
                        try:
                            pf.error = json.loads(data).get("error", {}).get("message", "upstream error")
                        except Exception:
                            pf.error = "upstream error"
                    pf.buffered = buffered
                    await resp.aclose()
                    return pf
    except httpx.RequestError as e:
        await resp.aclose()
        return _StreamPreflight(
            status="error", backend_model=backend_model, provider=provider,
            error=f"Backend request failed: {e}",
        )
    # Stream ended without content — treat as empty.
    await resp.aclose()
    pf.status = "empty"
    pf.buffered = buffered
    return pf


async def _open_fresh_stream(
    backend: Dict[str, Any],
    messages: List[Dict[str, Any]],
    request_body: Dict[str, Any],
) -> Tuple[Any, str]:
    """Open a fresh streaming backend connection for the downstream stream.

    Returns (httpx response, url). Raises HTTPException on non-200/request error.
    The response must be closed by the caller.
    """
    base_url = backend.get("base_url", "")
    api_key = backend.get("api_key", "")
    backend_model = backend.get("backend_model", "")
    provider = backend.get("provider", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    out = {
        "model": backend_model,
        "messages": messages,
        "stream": True,
    }
    for param in ("temperature", "top_p", "max_tokens", "stop", "frequency_penalty", "presence_penalty"):
        if param in request_body:
            out[param] = request_body[param]

    # Forward tool schemas so the model can emit structured tool_calls.
    if request_body.get("tools"):
        out["tools"] = request_body["tools"]
    if request_body.get("tool_choice"):
        out["tool_choice"] = request_body["tool_choice"]

    url = f"{base_url.rstrip('/')}/chat/completions"
    client = _get_httpx_client()
    try:
        req = client.build_request("POST", url, json=out, headers=headers)
        resp = await client.send(req, stream=True)
    except httpx.RequestError as e:
        logger.error("Backend %s stream request failed: %s", provider, e)
        raise HTTPException(status_code=502, detail=f"Backend request failed: {e}")
    if resp.status_code != 200:
        try:
            error_text = (await resp.aread()).decode()
        except Exception:
            error_text = ""
        await resp.aclose()
        _handle_backend_status(provider, backend_model, resp.status_code)
        logger.error("Backend %s returned %d: %s", provider, resp.status_code, error_text[:500])
        raise HTTPException(status_code=502, detail=f"Backend error: {error_text[:500]}")
    return resp, url


async def _resume_stream(
    pf: _StreamPreflight,
    backend: Dict[str, Any],
    messages: List[Dict[str, Any]],
    request_body: Dict[str, Any],
    on_complete: Optional[Callable[[], None]] = None,
) -> AsyncIterator[str]:
    """Replay buffered preflight events, then continue the SAME backend connection.

    The preflight probe is KEPT OPEN. This replays the buffered events (so no
    content is lost) then continues iterating the same response — no fresh
    connection, no regeneration, no token-split misalignment, no extra latency.
    Never emits a clean ``[DONE]`` for an empty preflight — empty streams are
    escalated by the caller, not returned as degenerate success.
    """
    resp = pf.response
    try:
        # Replay buffered events from the preflight
        for evt in pf.buffered:
            yield f"{evt}\n\n"
        if pf.buffered and pf.buffered[-1] == "data: [DONE]":
            return
        # Mid-stream degeneration alert: accumulate the raw streamed content and
        # scan it with the same repetition guard the non-streaming path uses.
        # The preflight stopped at the first content delta, so the degenerate
        # tail typically lands here. This is observability-only — the stream is
        # forwarded unchanged and the alert (recovery_log + ERROR log) captures
        # the full context for future debugging.
        accumulated = pf.accumulated
        for evt in pf.buffered:
            if evt.startswith("data: "):
                accumulated += _sse_content_text(evt[6:])
        # Continue the SAME iterator captured during preflight — never call
        # resp.aiter_lines() again (httpx responses are single-use; a second
        # call silently drops the rest of the stream).
        if pf.iterator is not None:
            async for line in pf.iterator:
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield f"data: {data}\n\n"
                        break
                    yield f"data: {data}\n\n"
                    accumulated += _sse_content_text(data)
            await resp.aclose()
        # Scan once at the end of the stream. Repeatedly re-regexing the growing
        # buffer per delta is pointless latency; a single scan after stream close
        # still catches the degenerate output that reached the client.
        pattern = _scan_stream_content(accumulated)
        if pattern:
            _alert_stream_degeneration(
                pf.provider, pf.backend_model, pattern, output_tail=accumulated[-400:],
            )
    finally:
        if resp is not None:
            try:
                await resp.aclose()
            except Exception:
                pass
        if on_complete:
            on_complete()


def _response_delta_parts(response: Any) -> Tuple[bool, bool]:
    """Return (has_content, has_tool_calls) for a chat completion response."""
    if not isinstance(response, dict):
        return (False, False)
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return (False, False)
    msg = choices[0].get("message") or {}
    if not isinstance(msg, dict):
        return (False, False)
    content = msg.get("content")
    has_content = isinstance(content, str) and bool(content.strip())
    has_tool_calls = bool(msg.get("tool_calls"))
    return (has_content, has_tool_calls)


def _wrap_non_streaming(
    provider: str,
    backend_model: str,
    result: Any,
    on_complete: Optional[Callable[[], None]] = None,
):
    """Build an async generator wrapping a non-streaming backend result as SSE.

    Content must be non-empty — empty results are escalated by the caller before
    this is reached. This is also used as the final degradation fallback for
    flaky upstream streaming: stream preflight failure -> one server-side
    escalation -> non-streaming retry -> synthesized SSE.
    """
    content = ""
    tool_calls = None
    finish_reason = "stop"
    if isinstance(result, dict):
        choices = result.get("choices", [])
        if choices:
            msg = (choices[0].get("message", {}) or {})
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls")
            fr = choices[0].get("finish_reason")
            if isinstance(fr, str) and fr:
                finish_reason = fr

    async def wrap():
        try:
            delta = {"role": "assistant"}
            if content:
                delta["content"] = content
            if tool_calls:
                delta["tool_calls"] = tool_calls
            chunk = {
                "id": result.get("id", ""),
                "object": "chat.completion.chunk",
                "created": result.get("created", 0),
                "model": result.get("model", backend_model),
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            finish = {
                "id": result.get("id", ""),
                "object": "chat.completion.chunk",
                "created": result.get("created", 0),
                "model": result.get("model", backend_model),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
            }
            yield f"data: {json.dumps(finish)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if on_complete:
                on_complete()

    return wrap()


async def _non_streaming_fallback_to_sse(
    backend: Dict[str, Any],
    messages: List[Dict[str, Any]],
    request_body: Dict[str, Any],
    on_complete: Optional[Callable[[], None]] = None,
    stats: Optional[Dict[str, Any]] = None,
    reason: str = "stream_preflight_failed",
    alternative_backends: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> StreamingResponse:
    """Retry without streaming and synthesize SSE from the first useful result.

    This is the final degradation fallback after the streaming preflight path has
    failed for the initial backend and the one allowed server-side escalation.
    It first tries the escalated backend non-streaming, then a bounded list of
    reachable alternatives. This keeps client semantics streaming-shaped while
    avoiding user-visible 502s when a cloud model can answer non-streaming but
    emits empty SSE.
    """
    fallback_body = dict(request_body)
    fallback_body["stream"] = False
    candidates: List[Tuple[str, Dict[str, Any]]] = [(backend.get("backend_model", ""), backend)]
    if alternative_backends:
        candidates.extend(alternative_backends)

    seen = set()
    last_detail = "streaming fallback exhausted"
    for candidate_name, candidate_backend in candidates:
        backend_model = candidate_backend.get("backend_model", "") or candidate_name
        provider = candidate_backend.get("provider", "")
        dedupe_key = (provider, backend_model)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        logger.warning(
            "Streaming fallback: retrying %s/%s non-streaming and synthesizing SSE (%s)",
            provider,
            backend_model,
            reason,
        )
        try:
            result = await proxy_to_backend(candidate_backend, messages, fallback_body)
        except HTTPException as e:
            last_detail = str(e.detail)
            logger.warning(
                "Streaming fallback non-streaming retry failed for %s/%s: %s",
                provider,
                backend_model,
                e.detail,
            )
            continue
        if _response_has_empty_content(result):
            last_detail = f"Backend {backend_model} returned empty content after streaming fallback"
            logger.warning(
                "Streaming fallback non-streaming retry returned empty content for %s/%s",
                provider,
                backend_model,
            )
            continue
        malformed_tool_error = _malformed_tool_call_error(result)
        if malformed_tool_error:
            last_detail = f"Backend {backend_model} emitted malformed tool_call: {malformed_tool_error}"
            logger.warning(
                "Streaming fallback non-streaming retry returned malformed tool_call for %s/%s: %s",
                provider,
                backend_model,
                malformed_tool_error,
            )
            continue

        saw_content, saw_tool_calls = _response_delta_parts(result)
        if stats is not None:
            stats["saw_content"] = saw_content
            stats["saw_tool_calls"] = saw_tool_calls
            stats["degraded_to_non_streaming"] = True
            stats["final_model"] = candidate_name or backend_model
            stats["final_provider"] = provider
        return StreamingResponse(
            _wrap_non_streaming(provider, backend_model, result, on_complete=on_complete),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Biggie-Stream-Fallback": "non-streaming-sse",
            },
        )

    raise HTTPException(status_code=502, detail=last_detail)


async def proxy_to_backend_streaming(
    backend: Dict[str, Any],
    messages: List[Dict[str, Any]],
    request_body: Dict[str, Any],
    on_complete: Optional[Callable[[], None]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Any:
    """Proxy a chat completion request to the chosen backend with SSE streaming.

    Honors the incoming ``stream`` flag. For OpenAI-compatible backends
    (Ollama Cloud, etc.) this performs a bounded preflight before committing the
    downstream response: if the upstream produces no content or errors, an
    HTTPException(502) is raised so the caller escalates server-side instead of
    returning a degenerate empty-success to the client.

    ``on_complete`` is invoked (once) after the downstream stream has finished or
    been closed, so the caller can persist a completion observability record.
    If ``stats`` is provided it is filled with saw_content/saw_tool_calls from
    the preflight so the caller can log stream observability.

    Raises:
        HTTPException(502): when the backend stream is empty or errors.
    """
    base_url = backend.get("base_url", "")
    backend_model = backend.get("backend_model", "")
    provider = backend.get("provider", "")

    if not base_url:
        raise HTTPException(status_code=502, detail=f"No base_url for backend")

    # Local Ollama and Codex do not support true SSE streaming here — call the
    # non-streaming path and wrap. Empty results escalate (no degenerate success).
    if provider in ("local", "openai-codex"):
        result = await proxy_to_backend(backend, messages, request_body)
        if _response_has_empty_content(result):
            logger.warning(
                "Backend %s returned empty content for %s, escalating",
                provider,
                backend_model,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Backend {backend_model} returned empty content",
            )
        saw_content, saw_tool_calls = _response_delta_parts(result)
        if stats is not None:
            stats["saw_content"] = saw_content
            stats["saw_tool_calls"] = saw_tool_calls
        return StreamingResponse(
            _wrap_non_streaming(provider, backend_model, result, on_complete=on_complete),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # OpenAI-compatible streaming path: bounded preflight, then commit.
    pf = await _preflight_openai_stream(backend, messages, request_body)

    if pf.status == "error":
        logger.error(
            "Backend %s stream error during preflight for %s: %s",
            pf.provider, pf.backend_model, pf.error,
        )
        raise HTTPException(status_code=502, detail=pf.error)

    if pf.status == "empty":
        logger.warning(
            "Backend %s streamed empty content for %s (preflight), escalating",
            pf.provider,
            pf.backend_model,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Backend {pf.backend_model} returned empty content",
        )

    # Healthy — commit the downstream streaming response.
    if stats is not None:
        stats["saw_content"] = pf.saw_content
        stats["saw_tool_calls"] = pf.saw_tool_calls
    return StreamingResponse(
        _resume_stream(pf, backend, messages, request_body, on_complete=on_complete),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Biggie LLM Endpoint",
    description="Smart model router — reads Hermes config, routes to cheapest capable model",
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    info = check_limp_home_status(needs_llm=False)
    return {
        "status": "ok",
        "limp_home": info["active"],
        "routing_profile": ROUTING_PROFILE,
        "compression": COMPRESSION_LEVEL,
        "version": "1.0.0",
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    backends = discover_backends()
    models = []
    for model_name, info in backends.items():
        models.append({
            "id": model_name,
            "object": "model",
            "owned_by": info.get("provider", "unknown"),
        })
    return {"object": "list", "data": models}


@app.get("/config")
async def show_config():
    """Show the discovered Hermes config and available backends."""
    hermes = load_hermes_config()
    backends = discover_backends()
    return {
        "routing_profile": ROUTING_PROFILE,
        "hermes_config": {
            "default_model": hermes.get("default_model"),
            "default_provider": hermes.get("default_provider"),
            "providers": {k: {kk: vv for kk, vv in v.items() if kk != "api_key"}
                         for k, v in hermes.get("providers", {}).items()},
            "fallback_chain": hermes.get("fallback_chain"),
        },
        "discovered_backends": {k: {kk: vv for kk, vv in v.items() if kk != "api_key"}
                                for k, v in backends.items()},
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Main chat completion endpoint — routes to the best model.

    Reads Hermes config to discover backends. Routes based on
    prompt features, circuit breaker state, and routing profile.
    """
    body = await request.json()
    messages = body.get("messages", [])
    # Hermes sends its configured default model (e.g. "biggie-router") on every
    # request. That is the router's own sentinel — it must NOT be treated as a
    # force_model override, or capability-based routing is always short-circuited.
    # Only honor an explicit model override for genuine specialist requests
    # (e.g. force_model="gpt-5.5").
    requested_model = body.get("model", "")
    force_model = "" if requested_model in ("", "biggie-router", "biggie-llm") else requested_model
    want_stream = bool(body.get("stream", False))
    # A request that carries OpenAI tool schemas requires a proven tool-capable
    # backend — this must dominate prompt classification / complexity routing.
    requires_tools = bool(body.get("tools"))

    # Track timing
    t0 = time.time()
    request_id = uuid.uuid4().hex[:12]

    # Discover available backends from Hermes config
    backends = discover_backends()

    # Extract features from the prompt and classify first-class workload.
    features = extract_features_from_messages(messages)
    workload_type = detect_workload_type(messages, body, dict(request.headers))
    routed_task_type = "session_compression" if workload_type == "session_compression" else features["task_type"]

    # Check limp-home
    if is_limp_home():
        logger.info("Limp-home active — routing to local model")

    # Route the task
    decision = route_task(
        complexity_score=features["complexity_score"],
        task_type=routed_task_type,
        has_niche_references=features["has_niche_references"],
        has_format_constraint=features["has_format_constraint"],
        instruction_count=features["instruction_count"],
        force_model=force_model if force_model else "",
        prompt=features["prompt_text"],
        workload_type=workload_type,
        context_tokens=features.get("context_tokens", 0),
        requires_tools=requires_tools,
    )

    # Apply routing profile (normal workloads only; session_compression has its
    # own native summarisation policy so profile boosting doesn't overroute it).
    # Also skip profile boosting for tool-required requests — they have already
    # been routed to a proven tool-capable model and profile re-routing must not
    # bypass the tool-capability gate.
    if workload_type != "session_compression" and not requires_tools:
        decision = apply_routing_profile(decision, features)

    # Record routing time
    routing_time = time.time() - t0

    # If no model is available, return a clear error
    if not decision.selected_model:
        limp_msg = get_limp_home_message()
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": limp_msg or "All models are currently unavailable due to rate limits. Try again in a few minutes.",
                    "type": "service_unavailable",
                    "code": "all_models_exhausted",
                }
            },
        )

    # Find the backend for the selected model
    backend = backends.get(decision.selected_model)
    if not backend:
        # Router returns names without provider suffix (e.g. "deepseek-v4-flash")
        # but backends are keyed with suffix (e.g. "deepseek-v4-flash:cloud")
        # Try all known provider suffixes
        for suffix in [":cloud", ":local", ":ollama"]:
            with_suffix = decision.selected_model + suffix
            if with_suffix in backends:
                backend = backends[with_suffix]
                decision.selected_model = with_suffix
                break

    if not backend:
        # Selected model isn't a discovered backend (e.g. routing table references
        # a model not in the config's fallback chain). Fall back to the CHEAPEST
        # reachable backend that still meets the needed tier; if none does, use the
        # most capable reachable. Otherwise heavy work silently drops to flash.
        #
        # For tool-required requests, the reachable-backend fallback must NOT
        # downgrade below the tool-capability gate — only proven tool-capable
        # backends may serve them.
        logger.warning("Selected model %s not in discovered backends, falling back to best reachable", decision.selected_model)
        needed_tier = MODEL_CAPABILITY_TIERS.get(decision.selected_model, 0)
        best_name = None
        best_tier = -1
        cheapest_meeting = None
        cheapest_meeting_tier = 99
        for model_name, bk in backends.items():
            base = re.sub(r":(cloud|local|ollama)$", "", model_name)
            if requires_tools and base not in TOOL_CAPABLE_MODELS:
                continue
            tier = MODEL_CAPABILITY_TIERS.get(base, 0)
            if tier >= needed_tier and tier < cheapest_meeting_tier:
                cheapest_meeting = model_name
                cheapest_meeting_tier = tier
            if tier > best_tier:
                best_tier = tier
                best_name = model_name
        # Prefer cheapest that meets the tier; else most capable
        chosen = cheapest_meeting or best_name
        if chosen:
            backend = backends[chosen]
            decision.selected_model = chosen
            logger.info("Fell back to reachable backend: %s (tier %d, needed %d)", chosen, cheapest_meeting_tier if cheapest_meeting else best_tier, needed_tier)
        elif requires_tools:
            # No reachable tool-capable backend — fail closed.
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": (
                            "Request requires structured tool calling, but no "
                            "proven tool-capable backend is reachable. "
                            "Tool-capable models: " + ", ".join(sorted(TOOL_CAPABLE_MODELS))
                        ),
                        "type": "service_unavailable",
                        "code": "tool_capability_unavailable",
                    }
                },
            )

    if not backend:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"Selected model '{decision.selected_model}' has no configured backend. Check Hermes config.",
                    "type": "configuration_error",
                    "code": "no_backend",
                }
            },
        )

    logger.info(
        "Routing: %s → %s/%s (profile=%s, cpx=%.2f, task=%s, workload=%s%s)",
        features["prompt_text"][:60],
        backend.get("provider", "?"),
        decision.selected_model,
        ROUTING_PROFILE,
        features["complexity_score"],
        routed_task_type,
        workload_type,
        " LIMP" if decision.limp_home else "",
    )

    # ── Compression ─────────────────────────────────────────────────────────
    compression_level = compression_level_for_workload(
        request,
        workload_type,
        context_tokens=features.get("context_tokens", 0),
        requires_tools=requires_tools,
    )
    if compression_level not in ("off", "lite", "standard", "structural", "aggressive"):
        compression_level = COMPRESSION_LEVEL

    if compression_level != "off":
        compressed_messages, compression_stats = compress_messages(
            messages,
            compression_level,
            workload_type=workload_type,
            context_tokens=features.get("context_tokens", 0),
        )
        logger.info(
            "Compression: %s — %s chars → %s chars (%.1f%%) in %.2fms",
            compression_level,
            compression_stats["input_chars"],
            compression_stats["output_chars"],
            compression_stats["savings_pct"],
            compression_stats["compression_time_ms"],
        )
        sample_path = capture_compression_sample(
            request_id=request_id,
            messages=messages,
            workload_type=workload_type,
            context_tokens=features.get("context_tokens", 0),
            requested_model=requested_model,
            selected_model=decision.selected_model,
            compression_level=compression_level,
            compression_stats=compression_stats,
            route_metadata={
                "provider": backend.get("provider", ""),
                "routed_task_type": routed_task_type,
                "routing_profile": ROUTING_PROFILE,
                "complexity_score": features["complexity_score"],
            },
            headers=dict(request.headers),
        )
        if sample_path:
            logger.info("Captured compression sample: %s", sample_path)
    else:
        compressed_messages = messages
        compression_stats = {
            "level": "off",
            "input_chars": 0,
            "output_chars": 0,
            "savings_pct": 0.0,
            "compression_time_ms": 0.0,
        }

    # Session compression is output-budget sensitive: low max_tokens caused
    # reasoning-only empty completions even at 16K context. Enforce the floor
    # after any in-request compression so both the initial upstream call and
    # escalation/fallback attempts inherit it through `body`.
    if apply_session_compression_output_floor(body, workload_type):
        logger.info(
            "Session compression max_tokens floor applied: %s",
            SESSION_COMPRESSION_MIN_MAX_TOKENS,
        )

    # Proxy to the backend
    _obs = {
        "request_id": request_id,
        "requested_model": requested_model,
        "streaming": want_stream,
        "workload_type": workload_type,
        "requires_tools": requires_tools,
        "context_tokens": features.get("context_tokens", 0),
    }

    # Self-optimising router: optional experiment routes a share of production
    # to a candidate model, and every call is priced so spend is measured
    # rather than inferred. Both are observability-first: a shadow experiment
    # never changes what the user receives.
    _experiment = pick_experiment(
        _experiments_cached(),
        workload=workload_type,
        tier=MODEL_CAPABILITY_TIERS.get(decision.selected_model, 0),
    )
    decision.selected_model, _exp_obs = apply_experiment(
        decision.selected_model,
        _experiment,
        workload_type=workload_type,
        request_id=request_id,
    )
    _obs.update({k: v for k, v in _exp_obs.items() if k != "shadow_model"})
    if _exp_obs.get("experiment_arm") == "treatment":
        # The candidate may live under a provider-suffixed backend key.
        _cand_backend = backends.get(decision.selected_model)
        if _cand_backend is None:
            for _suffix in (":cloud", ":local", ":ollama"):
                if decision.selected_model + _suffix in backends:
                    _cand_backend = backends[decision.selected_model + _suffix]
                    decision.selected_model = decision.selected_model + _suffix
                    break
        if _cand_backend is not None:
            backend = _cand_backend
            logger.info(
                "Experiment %s: serving %s instead of %s",
                _exp_obs["experiment"], decision.selected_model,
                _exp_obs.get("experiment", "?"),
            )
        else:
            # Candidate is not reachable — stay on the incumbent rather than
            # failing the request. Log the arm as control so the experiment
            # does not silently report treatment outcomes it never served.
            logger.warning(
                "Experiment %s wants %s but no backend is configured; staying on %s",
                _exp_obs.get("experiment"), decision.selected_model, decision.selected_model,
            )
            _obs["experiment_arm"] = "control"

    # ── Shadow execution ─────────────────────────────────────────────────────
    # Fire the candidate call for real, discard its output, and record it as its
    # own row. The user's response is untouched — that is the only property that
    # makes running an experiment against live traffic safe. Concurrency and
    # timeout are bounded so a slow candidate cannot stall production.
    if _experiment is not None and getattr(_experiment, "is_shadow", False):
        try:
            asyncio.create_task(
                run_shadow_experiment(
                    _experiment,
                    messages=compressed_messages,
                    request_body=body,
                    request_id=request_id,
                    session_id=features.get("session_id", ""),
                    incumbent_model=decision.selected_model,
                )
            )
        except RuntimeError:                                # pragma: no cover
            # No running loop — shadow is strictly best-effort and must never
            # break a request.
            logger.debug("Shadow: no event loop available; skipping")

    try:
        if want_stream:
            # Streaming requests are observable too: record the route at start,
            # then the outcome is logged once the stream completes (FIX 3).
            _log_request_to_db(
                model_used=decision.selected_model,
                provider=backend.get("provider", ""),
                task_type=routed_task_type,
                complexity_score=features["complexity_score"],
                input_tokens=features.get("context_tokens", 0),
                output_tokens=0,
                latency_seconds=0.0,
                routing_time_ms=round(routing_time * 1000),
                success=False,
                escalated=False,
                error_type="streaming_in_progress",
                compression_level=compression_stats["level"],
                compression_savings_pct=compression_stats["savings_pct"],
                compression_time_ms=compression_stats["compression_time_ms"],
                routing_reason=decision.reason,
                **_obs,
            )
            _stream_stats: Dict[str, Any] = {"saw_content": False, "saw_tool_calls": False}
            started = {"t": time.time()}

            def _log_stream_complete():
                elapsed = time.time() - started["t"]
                _log_request_to_db(
                    model_used=decision.selected_model,
                    provider=backend.get("provider", ""),
                    task_type=routed_task_type,
                    complexity_score=features["complexity_score"],
                    input_tokens=features.get("context_tokens", 0),
                    output_tokens=0,
                    latency_seconds=elapsed,
                    routing_time_ms=round(routing_time * 1000),
                    success=True,
                    escalated=False,
                    error_type="",
                    compression_level=compression_stats["level"],
                    compression_savings_pct=compression_stats["savings_pct"],
                    compression_time_ms=compression_stats["compression_time_ms"],
                    saw_content=_stream_stats.get("saw_content", False),
                    saw_tool_calls=_stream_stats.get("saw_tool_calls", False),
                    routing_reason=decision.reason,
                    **_obs,
                )

            result = await proxy_to_backend_streaming(
                backend,
                compressed_messages,
                body,
                on_complete=_log_stream_complete,
                stats=_stream_stats,
            )
        else:
            result = await proxy_to_backend(backend, compressed_messages, body)
        # A 200 with empty assistant content is a degenerate success —
        # the model produced no output. Escalate exactly as on HTTPException
        # so the client never receives an empty turn that would waste its
        # turn budget self-healing.
        if _response_has_empty_content(result):
            logger.warning(
                "Backend %s returned empty content for %s, escalating...",
                backend.get("provider"),
                decision.selected_model,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Backend {decision.selected_model} returned empty content",
            )
        malformed_tool_error = _malformed_tool_call_error(result)
        if malformed_tool_error:
            logger.warning(
                "Backend %s/%s emitted malformed tool call (%s), escalating...",
                backend.get("provider"),
                decision.selected_model,
                malformed_tool_error,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Backend {decision.selected_model} emitted malformed tool_call: {malformed_tool_error}",
            )
        # Degeneration guard: catch the model stuck in a repetition loop —
        # duplicate tool calls or runaway text repetition. Non-empty and
        # well-formed, so it slips past the checks above, but corrupted output
        # would waste turn budget and break downstream parsing (build failures).
        # Escalate to a stronger model exactly as for malformed/empty.
        degeneration_error = _degeneration_error(result)
        if degeneration_error:
            logger.warning(
                "Backend %s/%s degenerated (%s), escalating...",
                backend.get("provider"),
                decision.selected_model,
                degeneration_error,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Backend {decision.selected_model} degenerated: {degeneration_error}",
            )
        total_time = time.time() - t0
        llm_time = total_time - routing_time
        # Log the request to DB (skip for streaming — usage comes from stream)
        if not want_stream:
            _in_tok = _get_usage_tokens(result, "input") or features.get("prompt_text", "").count(" ")
            _out_tok = _get_usage_tokens(result, "output") or 0
            _cost, _cost_unknown = compute_cost_fields(
                decision.selected_model, _in_tok, _out_tok
            )
            _log_request_to_db(
                model_used=decision.selected_model,
                provider=backend.get("provider", ""),
                task_type=routed_task_type,
                complexity_score=features["complexity_score"],
                input_tokens=_in_tok,
                output_tokens=_out_tok,
                latency_seconds=total_time,
                routing_time_ms=round(routing_time * 1000),
                success=True,
                escalated=False,
                compression_level=compression_stats["level"],
                compression_savings_pct=compression_stats["savings_pct"],
                compression_time_ms=compression_stats["compression_time_ms"],
                routing_reason=decision.reason,
                cost_usd=_cost,
                cost_unknown=_cost_unknown,
                pricing_version=_pricing_version(),
                **_obs,
            )
        return result
    except HTTPException as http_exc:
        # Backend failed — try escalation. Classify the failure so model-output
        # quality issues (malformed tool call, empty content) escalate WITHOUT
        # tripping the circuit breaker, while genuine provider/availability
        # errors count toward the breaker as before.
        detail = str(http_exc.detail)
        if "malformed tool_call" in detail or "malformed function name" in detail:
            fail_type = "malformed_tool_call"
        elif "degenerated" in detail:
            fail_type = "degeneration"
        elif "empty content" in detail:
            fail_type = "empty_content"
        else:
            fail_type = "error"
        logger.warning("Backend %s failed (%s), escalating...", decision.selected_model, fail_type)
        escalation = escalate_on_failure(
            failed_model=decision.selected_model,
            complexity_score=features["complexity_score"],
            error_type=fail_type,
            requires_tools=requires_tools,
        )

        if not escalation.selected_model:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "All models failed. Try again later.",
                        "type": "service_unavailable",
                        "code": "all_models_failed",
                    }
                },
            )

        # Try the escalated model
        backend = backends.get(escalation.selected_model)
        if not backend:
            # Try with provider suffixes
            for suffix in [":cloud", ":local", ":ollama"]:
                with_suffix = escalation.selected_model + suffix
                if with_suffix in backends:
                    backend = backends[with_suffix]
                    escalation.selected_model = with_suffix
                    break

        if not backend:
            # Escalated model isn't a discovered backend — fall back to the cheapest
            # reachable backend that meets the needed tier, else the most capable.
            # For tool-required work, only proven tool-capable backends may serve.
            logger.warning("Escalated model %s not in discovered backends, falling back to best reachable", escalation.selected_model)
            needed_tier = MODEL_CAPABILITY_TIERS.get(escalation.selected_model, 0)
            best_name = None
            best_tier = -1
            cheapest_meeting = None
            cheapest_meeting_tier = 99
            for model_name, bk in backends.items():
                base = re.sub(r":(cloud|local|ollama)$", "", model_name)
                if requires_tools and base not in TOOL_CAPABLE_MODELS:
                    continue
                tier = MODEL_CAPABILITY_TIERS.get(base, 0)
                if tier >= needed_tier and tier < cheapest_meeting_tier:
                    cheapest_meeting = model_name
                    cheapest_meeting_tier = tier
                if tier > best_tier:
                    best_tier = tier
                    best_name = model_name
            chosen = cheapest_meeting or best_name
            if chosen:
                backend = backends[chosen]
                escalation.selected_model = chosen
                logger.info("Escalation fell back to reachable backend: %s (tier %d, needed %d)", chosen, cheapest_meeting_tier if cheapest_meeting else best_tier, needed_tier)
            elif requires_tools:
                # No reachable tool-capable backend — fail closed.
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": (
                                "Escalation: request requires structured tool calling, "
                                "but no proven tool-capable backend is reachable."
                            ),
                            "type": "service_unavailable",
                            "code": "tool_capability_unavailable",
                        }
                    },
                )

        if not backend:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": f"Escalated model '{escalation.selected_model}' has no configured backend.",
                        "type": "configuration_error",
                        "code": "no_backend",
                    }
                },
            )

        logger.info("Escalated to: %s/%s", backend.get("provider"), escalation.selected_model)
        if want_stream:
            _stream_stats: Dict[str, Any] = {"saw_content": False, "saw_tool_calls": False}
            started = {"t": time.time()}

            def _log_escalated_stream_complete():
                elapsed = time.time() - started["t"]
                _log_request_to_db(
                    model_used=_stream_stats.get("final_model", escalation.selected_model),
                    provider=_stream_stats.get("final_provider", backend.get("provider", "")),
                    task_type=routed_task_type,
                    complexity_score=features["complexity_score"],
                    input_tokens=features.get("context_tokens", 0),
                    output_tokens=0,
                    latency_seconds=elapsed,
                    routing_time_ms=round(routing_time * 1000),
                    success=True,
                    escalated=True,
                    error_type="streaming_degraded_to_non_streaming" if _stream_stats.get("degraded_to_non_streaming") else fail_type,
                    compression_level=compression_stats["level"],
                    compression_savings_pct=compression_stats["savings_pct"],
                    compression_time_ms=compression_stats["compression_time_ms"],
                    saw_content=_stream_stats.get("saw_content", False),
                    saw_tool_calls=_stream_stats.get("saw_tool_calls", False),
                    final_model=_stream_stats.get("final_model", escalation.selected_model),
                    routing_reason=escalation.reason,
                    **_obs,
                )

            try:
                result = await proxy_to_backend_streaming(
                    backend,
                    compressed_messages,
                    body,
                    on_complete=_log_escalated_stream_complete,
                    stats=_stream_stats,
                )
            except HTTPException as stream_exc:
                # The initial streaming backend already failed, and the one allowed
                # server-side stream escalation also failed. As a final graceful
                # degradation, retry the final backend once without streaming and
                # synthesize SSE from the non-streaming response. If that also
                # returns empty/error, fail closed as before.
                logger.warning(
                    "Escalated streaming backend %s failed (%s); trying non-streaming SSE fallback",
                    escalation.selected_model,
                    stream_exc.detail,
                )
                alternative_backends: List[Tuple[str, Dict[str, Any]]] = []
                for alt_name, alt_backend in sorted(
                    backends.items(),
                    key=lambda item: MODEL_CAPABILITY_TIERS.get(
                        re.sub(r":(cloud|local|ollama)$", "", item[0]),
                        0,
                    ),
                ):
                    alt_base = re.sub(r":(cloud|local|ollama)$", "", alt_name)
                    if alt_name == escalation.selected_model or alt_name == decision.selected_model:
                        continue
                    if requires_tools and alt_base not in TOOL_CAPABLE_MODELS:
                        continue
                    alternative_backends.append((alt_name, alt_backend))

                result = await _non_streaming_fallback_to_sse(
                    backend,
                    compressed_messages,
                    body,
                    on_complete=_log_escalated_stream_complete,
                    stats=_stream_stats,
                    reason=str(stream_exc.detail),
                    alternative_backends=alternative_backends,
                )
        else:
            result = await proxy_to_backend(backend, compressed_messages, body)
        total_time = time.time() - t0
        if not want_stream:
            _log_request_to_db(
                model_used=escalation.selected_model,
                provider=backend.get("provider", ""),
                task_type=routed_task_type,
                complexity_score=features["complexity_score"],
                input_tokens=_get_usage_tokens(result, "input") or features.get("prompt_text", "").count(" "),
                output_tokens=_get_usage_tokens(result, "output") or 0,
                latency_seconds=total_time,
                routing_time_ms=round(routing_time * 1000),
                success=True,
                escalated=True,
                error_type=fail_type,
                compression_level=compression_stats["level"],
                compression_savings_pct=compression_stats["savings_pct"],
                compression_time_ms=compression_stats["compression_time_ms"],
                routing_reason=escalation.reason,
            )
        return result


@app.get("/status")
async def status():
    """Get router status — model health, limp-home, routing stats."""
    backends = discover_backends()
    return {
        "limp_home": is_limp_home(),
        "limp_home_message": get_limp_home_message(),
        "routing_profile": ROUTING_PROFILE,
        "compression": COMPRESSION_LEVEL,
        "model_health": get_recovery_summary(),
        "discovered_backends": list(backends.keys()),
    }


@app.get("/compression")
async def compression_report():
    """Get compression effectiveness report from recent requests."""
    try:
        import sqlite3
        db = sqlite3.connect(str(Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db"))
        rows = db.execute("""
            SELECT compression_level, COUNT(*) as calls,
                   ROUND(AVG(compression_savings_pct), 1) as avg_savings,
                   ROUND(AVG(compression_time_ms), 2) as avg_time_ms,
                   ROUND(AVG(input_tokens), 0) as avg_input_tokens
            FROM router_logs
            WHERE compression_level != ''
              AND timestamp > datetime('now', '-24 hours')
            GROUP BY compression_level
            ORDER BY compression_level
        """).fetchall()
        db.close()
        return {
            "period": "last_24h",
            "levels": [
                {
                    "level": r[0],
                    "calls": r[1],
                    "avg_savings_pct": r[2],
                    "avg_compression_time_ms": r[3],
                    "avg_input_tokens": r[4],
                }
                for r in rows
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Start the Biggie LLM Endpoint server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Biggie LLM Endpoint")
    logger.info("=" * 60)
    logger.info("Host: %s:%s", HOST, PORT)
    logger.info("Routing profile: %s", ROUTING_PROFILE)
    logger.info("Hermes config: %s", HERMES_CONFIG_PATH)

    # Discover backends
    backends = discover_backends()
    logger.info("Discovered %d backends:", len(backends))
    for name, info in backends.items():
        logger.info("  %s → %s (%s)", name, info.get("provider"), info.get("base_url"))

    logger.info("Limp-home: %s", "ACTIVE" if is_limp_home() else "inactive")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
