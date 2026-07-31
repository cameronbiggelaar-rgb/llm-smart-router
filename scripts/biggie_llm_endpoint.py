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
from fastapi.responses import JSONResponse

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

logger = logging.getLogger("biggie-llm-endpoint")

# ── Self-identification ───────────────────────────────────────────────────────

# The endpoint's own identity — used to skip itself in routing
BIGGIE_PROVIDER_NAME = "biggie-llm"
BIGGIE_MODEL_NAMES = {"biggie-router", "biggie-llm"}

# ── Configuration ─────────────────────────────────────────────────────────────

# Built-in provider base URLs (Hermes knows these internally)
BUILTIN_PROVIDER_URLS = {
    "ollama-cloud": "https://api.ollama.cloud/v1",
    "openai-codex": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "minimax": "https://api.minimax.chat/v1",
    "bedrock": "",  # AWS IAM — no URL
}

# Built-in provider API key env vars
BUILTIN_PROVIDER_KEYS = {
    "ollama-cloud": "OLLAMA_CLOUD_API_KEY",
    "openai-codex": "OPENAI_CODEX_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}

# Routing profile: cheap, goldilocks, expensive
ROUTING_PROFILE = os.environ.get("BIGGIE_ROUTING_PROFILE", "goldilocks").lower()

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
    """
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

        # Get provider config
        pconf = hermes.get("providers", {}).get(provider, {})
        base_url = pconf.get("base_url", "")
        api_key_env = pconf.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""

        # Determine the backend model name
        # Hermes model names sometimes have provider prefixes (e.g. "deepseek-v4-flash:cloud")
        # Strip those for the backend call
        backend_model = re.sub(r":\w+$", "", model)

        # Determine the provider type for the backend URL
        if provider == "local-ollama" or provider == "mac-ollama":
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

        backends[model] = {
            "provider": backend_provider,
            "hermes_provider": provider,
            "base_url": base_url,
            "api_key": api_key,
            "backend_model": backend_model,
        }

    # Also add local models if they're not already in the chain
    local_models = ["llama3.1:8b", "qwen3:14b", "dolphin3"]
    for m in local_models:
        if m not in backends:
            backends[m] = {
                "provider": "local",
                "hermes_provider": "local-ollama",
                "base_url": "http://127.0.0.1:11434",
                "api_key": "",
                "backend_model": m,
            }

    return backends


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
    # Collect all user messages in order
    user_messages = []
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

    # Use the last 8 user messages (or all if fewer) for complexity scoring
    recent_window = 8
    recent_user_msgs = user_messages[-recent_window:] if len(user_messages) > recent_window else user_messages
    combined_prompt = "\n".join(recent_user_msgs)

    # Use the last user message for task classification (most recent context)
    last_prompt = user_messages[-1] if user_messages else ""

    message_count = len(messages)
    tool_call_count = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_call_count += len(msg["tool_calls"])

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
    """
    base_url = backend.get("base_url", "")
    api_key = backend.get("api_key", "")
    backend_model = backend.get("backend_model", "")
    provider = backend.get("provider", "")

    if not base_url:
        raise HTTPException(status_code=502, detail=f"No base_url for backend")

    # Build the request body
    body = {
        "model": backend_model,
        "messages": messages,
        "stream": False,
    }

    for param in ("temperature", "top_p", "max_tokens", "stop", "frequency_penalty", "presence_penalty"):
        if param in request_body:
            body[param] = request_body[param]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Local Ollama uses /api/chat
    if provider == "local":
        url = f"{base_url.rstrip('/')}/api/chat"
    else:
        url = f"{base_url.rstrip('/')}/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
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
    )

    # Apply routing profile
    decision = apply_routing_profile(decision, features)

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

    # Proxy to the backend
    try:
        return await proxy_to_backend(backend, messages, body)
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
        return await proxy_to_backend(backend, messages, body)


@app.get("/status")
async def status():
    """Get router status — model health, limp-home, routing stats."""
    backends = discover_backends()
    return {
        "limp_home": is_limp_home(),
        "limp_home_message": get_limp_home_message(),
        "routing_profile": ROUTING_PROFILE,
        "model_health": get_recovery_summary(),
        "discovered_backends": list(backends.keys()),
    }


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
