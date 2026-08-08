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

import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
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
    mark_available,
    RoutingDecision,
    MODEL_CAPABILITY_TIERS,
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

logger = logging.getLogger("biggie-llm-endpoint")

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
}


def _ensure_log_columns(db: Any) -> None:
    """Add streaming-observability columns to router_logs if missing."""
    try:
        existing = {r[1] for r in db.execute("PRAGMA table_info(router_logs)").fetchall()}
        for col, ddl in _STREAM_OBS_COLUMNS.items():
            if col not in existing:
                db.execute(f"ALTER TABLE router_logs ADD COLUMN {col} {ddl}")
        db.commit()
    except Exception as e:
        logger.warning("Failed to migrate router_logs columns: %s", e)


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
if COMPRESSION_LEVEL not in ("off", "lite", "standard", "aggressive"):
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


def compression_level_for_workload(request: Request, workload_type: str) -> str:
    """Choose Biggie request-compression level for the detected workload."""
    header_level = request.headers.get("X-Compression-Level")
    if header_level in ("off", "lite", "standard", "aggressive"):
        return header_level
    if workload_type == "session_compression":
        # Hermes is already doing semantic summarisation. Keep Biggie's request
        # compression conservative so it does not strip details before summary.
        return "lite" if COMPRESSION_LEVEL != "off" else "off"
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
):
    """Log a single request to the router_logs DB for analysis.

    Uses a persistent SQLite connection to avoid open/close overhead.
    Streaming requests are logged at both route-start and completion so the
    router's behaviour on streaming is observable (FIX 3). No prompt bodies or
    secrets are stored.
    """
    try:
        from datetime import datetime, timezone

        db = _get_db_connection()
        with _sqlite_lock:
            db.execute(
                """INSERT INTO router_logs (
                    timestamp, session_id, model_used, provider, task_type,
                    input_tokens, output_tokens, latency_seconds, complexity_score,
                    success, escalated, error_type,
                    compression_level, compression_savings_pct, compression_time_ms,
                    request_id, requested_model, streaming, workload_type,
                    requires_tools, context_tokens, empty_stream, saw_content,
                    saw_tool_calls, final_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    if resp.status_code == 429:
                        mark_rate_limited(backend_model)
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
            if status == 429:
                mark_rate_limited(backend_model)
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
            if status == 429:
                mark_rate_limited(backend_model)
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
        if status == 429:
            mark_rate_limited(backend_model)
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

    NOTE: the probe connection used for preflight is CLOSED after first content.
    The actual downstream stream issues a FRESH request (httpx responses can only
    be consumed once, so the probe stream cannot be resumed in place).
    """

    status: str                          # "ok" | "empty" | "error"
    backend_model: str = ""
    provider: str = ""
    error: str = ""
    buffered: List[str] = field(default_factory=list)  # raw SSE data lines to replay
    saw_content: bool = False
    saw_tool_calls: bool = False


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
        if resp.status_code == 429:
            mark_rate_limited(backend_model)
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
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                buffered.append(f"data: {data}")
                saw_content, saw_tool, is_terminal = _sse_delta_parts(data)
                if saw_content:
                    pf.saw_content = True
                if saw_tool:
                    pf.saw_tool_calls = True
                if saw_content or saw_tool:
                    pf.status = "ok"
                    pf.buffered = buffered
                    await resp.aclose()
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
        if resp.status_code == 429:
            mark_rate_limited(backend_model)
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
    """Replay preflighted events, then stream a FRESH backend connection.

    The preflight probe is closed; this opens a new connection and replays the
    buffered events (so no content is lost) before continuing live. Never emits
    a clean ``[DONE]`` for an empty preflight — empty streams are escalated by
    the caller, not returned as degenerate success.
    """
    try:
        for evt in pf.buffered:
            yield f"{evt}\n\n"
        if pf.buffered and pf.buffered[-1] == "data: [DONE]":
            return
        resp, _url = await _open_fresh_stream(backend, messages, request_body)
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                yield f"data: {data}\n\n"
                if data == "[DONE]":
                    break
        await resp.aclose()
    finally:
        if on_complete:
            on_complete()


def _wrap_non_streaming(provider: str, backend_model: str, result: Any):
    """Build an async generator wrapping a non-streaming backend result as SSE.

    Content must be non-empty — empty results are escalated by the caller before
    this is reached.
    """
    content = ""
    tool_calls = None
    if isinstance(result, dict):
        choices = result.get("choices", [])
        if choices:
            msg = (choices[0].get("message", {}) or {})
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls")

    async def wrap():
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
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(finish)}\n\n"
        yield "data: [DONE]\n\n"

    return wrap()


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
        return StreamingResponse(
            _wrap_non_streaming(provider, backend_model, result),
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
        from router import MODEL_CAPABILITY_TIERS, TOOL_CAPABLE_MODELS
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
    compression_level = compression_level_for_workload(request, workload_type)
    if compression_level not in ("off", "lite", "standard", "aggressive"):
        compression_level = COMPRESSION_LEVEL

    if compression_level != "off":
        compressed_messages, compression_stats = compress_messages(
            messages, compression_level
        )
        logger.info(
            "Compression: %s — %s chars → %s chars (%.1f%%) in %.2fms",
            compression_level,
            compression_stats["input_chars"],
            compression_stats["output_chars"],
            compression_stats["savings_pct"],
            compression_stats["compression_time_ms"],
        )
    else:
        compressed_messages = messages
        compression_stats = {
            "level": "off",
            "input_chars": 0,
            "output_chars": 0,
            "savings_pct": 0.0,
            "compression_time_ms": 0.0,
        }

    # Proxy to the backend
    _obs = {
        "request_id": request_id,
        "requested_model": requested_model,
        "streaming": want_stream,
        "workload_type": workload_type,
        "requires_tools": requires_tools,
        "context_tokens": features.get("context_tokens", 0),
    }
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
        total_time = time.time() - t0
        llm_time = total_time - routing_time
        # Log the request to DB (skip for streaming — usage comes from stream)
        if not want_stream:
            _log_request_to_db(
                model_used=decision.selected_model,
                provider=backend.get("provider", ""),
                task_type=routed_task_type,
                complexity_score=features["complexity_score"],
                input_tokens=_get_usage_tokens(result, "input") or features.get("prompt_text", "").count(" "),
                output_tokens=_get_usage_tokens(result, "output") or 0,
                latency_seconds=total_time,
                routing_time_ms=round(routing_time * 1000),
                success=True,
                escalated=False,
                compression_level=compression_stats["level"],
                compression_savings_pct=compression_stats["savings_pct"],
                compression_time_ms=compression_stats["compression_time_ms"],
                **_obs,
            )
        return result
    except HTTPException:
        # Backend failed — try escalation
        logger.warning("Backend %s failed, escalating...", decision.selected_model)
        escalation = escalate_on_failure(
            failed_model=decision.selected_model,
            complexity_score=features["complexity_score"],
            error_type="error",
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
            from router import MODEL_CAPABILITY_TIERS, TOOL_CAPABLE_MODELS
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
            result = await proxy_to_backend_streaming(backend, compressed_messages, body)
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
                compression_level=compression_stats["level"],
                compression_savings_pct=compression_stats["savings_pct"],
                compression_time_ms=compression_stats["compression_time_ms"],
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
