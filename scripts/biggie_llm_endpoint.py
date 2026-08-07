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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    }


# ── Request logging ────────────────────────────────────────────────────────────

def _get_usage_tokens(response: dict, direction: str) -> int:
    """Extract token counts from an OpenAI-compatible response."""
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if direction == "input":
        return usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
    return usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0


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
):
    """Log a single request to the router_logs DB for analysis.

    Uses a persistent SQLite connection to avoid open/close overhead.
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
                    compression_level, compression_savings_pct, compression_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                ),
            )
            db.commit()
    except Exception as e:
        logger.warning("Failed to log request to DB: %s", e)


# ── Backend proxy ─────────────────────────────────────────────────────────────

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
        # Codex backend uses "message" type items, not "input_text"/"output_text"
        codex_input = []
        for item in responses_input:
            role = item.get("role", "user")
            content = item.get("content", "")
            codex_input.append({
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            })

        body = {
            "model": backend_model,
            "input": codex_input,
            "store": False,
            "stream": True,
        }

        for param in ("temperature", "top_p", "stop", "frequency_penalty", "presence_penalty"):
            if param in request_body:
                body[param] = request_body[param]

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
                                if isinstance(item, dict) and item.get("type") == "message":
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
            if isinstance(msg, dict):
                content = msg.get("content", "")
            elif hasattr(msg, "content"):
                content = msg.content

            return {
                "id": responses_data.get("id", ""),
                "object": "chat.completion",
                "created": responses_data.get("created_at", 0),
                "model": responses_data.get("model", backend_model),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content or "",
                    },
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


async def proxy_to_backend_streaming(
    backend: Dict[str, Any],
    messages: List[Dict[str, Any]],
    request_body: Dict[str, Any],
) -> Any:
    """Proxy a chat completion request to the chosen backend with SSE streaming.

    Honors the incoming ``stream`` flag. For OpenAI-compatible backends
    (Ollama Cloud, etc.) this streams SSE ``data:`` events back to the client
    so that streaming clients (e.g. the code-editing agent) receive tokens as
    they are generated instead of a single JSON body.
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

    # Only OpenAI-compatible backends support SSE streaming here.
    # Local Ollama and Codex use their own paths (non-streaming is fine).
    if provider in ("local", "openai-codex"):
        # Fall back to non-streaming for these providers.
        return await proxy_to_backend(backend, messages, request_body)

    body = {
        "model": backend_model,
        "messages": messages,
        "stream": True,
    }
    for param in ("temperature", "top_p", "max_tokens", "stop", "frequency_penalty", "presence_penalty"):
        if param in request_body:
            body[param] = request_body[param]

    url = f"{base_url.rstrip('/')}/chat/completions"

    async def event_generator():
        try:
            client = _get_httpx_client()
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    detail = f"Backend error: {error_text[:500].decode()}"
                    logger.error("Backend %s returned %d: %s", provider, resp.status_code, detail)
                    if resp.status_code == 429:
                        mark_rate_limited(backend_model)
                    yield f"data: {json.dumps({'error': {'message': detail, 'type': 'backend_error', 'code': 'backend_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        # Pass through the SSE event unchanged
                        yield f"data: {data}\n\n"
        except httpx.RequestError as e:
            logger.error("Backend %s stream request failed: %s", provider, e)
            yield f"data: {json.dumps({'error': {'message': f'Backend request failed: {e}', 'type': 'backend_error', 'code': 'backend_error'}})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
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
    force_model = body.get("model", "")
    want_stream = bool(body.get("stream", False))

    # Track timing
    t0 = time.time()

    # Discover available backends from Hermes config
    backends = discover_backends()

    # Extract features from the prompt
    features = extract_features_from_messages(messages)

    # Check limp-home
    if is_limp_home():
        logger.info("Limp-home active — routing to local model")

    # Route the task
    decision = route_task(
        complexity_score=features["complexity_score"],
        task_type=features["task_type"],
        has_niche_references=features["has_niche_references"],
        has_format_constraint=features["has_format_constraint"],
        instruction_count=features["instruction_count"],
        force_model=force_model if force_model else "",
        prompt=features["prompt_text"],
    )

    # Apply routing profile
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
        # Try to find a fallback — any model on any provider
        logger.warning("Selected model %s not in discovered backends, trying fallbacks", decision.selected_model)
        for model_name, bk in backends.items():
            backend = bk
            decision.selected_model = model_name
            break

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
        "Routing: %s → %s/%s (profile=%s, cpx=%.2f, task=%s%s)",
        features["prompt_text"][:60],
        backend.get("provider", "?"),
        decision.selected_model,
        ROUTING_PROFILE,
        features["complexity_score"],
        features["task_type"],
        " LIMP" if decision.limp_home else "",
    )

    # ── Compression ─────────────────────────────────────────────────────────
    # Determine compression level: request header > env var > default
    compression_level = request.headers.get(
        "X-Compression-Level", COMPRESSION_LEVEL
    )
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
    try:
        if want_stream:
            result = await proxy_to_backend_streaming(backend, compressed_messages, body)
        else:
            result = await proxy_to_backend(backend, compressed_messages, body)
        total_time = time.time() - t0
        llm_time = total_time - routing_time
        # Log the request to DB (skip for streaming — usage comes from stream)
        if not want_stream:
            _log_request_to_db(
                model_used=decision.selected_model,
                provider=backend.get("provider", ""),
                task_type=features["task_type"],
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
            )
        return result
    except HTTPException:
        # Backend failed — try escalation
        logger.warning("Backend %s failed, escalating...", decision.selected_model)
        escalation = escalate_on_failure(
            failed_model=decision.selected_model,
            complexity_score=features["complexity_score"],
            error_type="error",
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
                task_type=features["task_type"],
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
