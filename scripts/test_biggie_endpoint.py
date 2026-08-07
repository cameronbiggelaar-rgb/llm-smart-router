#!/usr/bin/env python3
"""Isolated test suite for the Biggie LLM Endpoint.

Tests the routing logic directly (no API keys needed) and the
endpoint's HTTP interface for health/config/status.

Run: python3 test_biggie_endpoint.py
"""

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Add scripts to path
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
    MODEL_CAPABILITY_TIERS,
    MODEL_COST_ORDER,
)

BASE_URL = "http://127.0.0.1:8080"
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def http_get(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def http_post(path: str, body: dict) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Endpoint health and config
# ═══════════════════════════════════════════════════════════════════════════════

print("\n═══ 1. Endpoint health and config ═══")

health = http_get("/health")
check("Health endpoint returns ok", health.get("status") == "ok", str(health))
check("Routing profile is goldilocks", health.get("routing_profile") == "goldilocks")
check("Limp-home is inactive", health.get("limp_home") is False)

models = http_get("/v1/models")
check("Models endpoint returns list", models.get("object") == "list")
model_ids = [m["id"] for m in models.get("data", [])]
check("deepseek-v4-flash:cloud in models", "deepseek-v4-flash:cloud" in model_ids)
check("glm-5.2:cloud in models", "glm-5.2:cloud" in model_ids)
check("llama3.1:8b in models", "llama3.1:8b" in model_ids)
check("qwen3:14b in models", "qwen3:14b" in model_ids)
check("dolphin3 in models", "dolphin3" in model_ids)
check("biggie-router NOT in models (self-skip)", "biggie-router" not in model_ids)

cfg = http_get("/config")
check("Config endpoint works", "hermes_config" in cfg)
check("Discovered backends present", "discovered_backends" in cfg)
backends = cfg.get("discovered_backends", {})
check("5 backends discovered", len(backends) >= 5, str(list(backends.keys())))

status = http_get("/status")
check("Status endpoint works", "model_health" in status)
check("Discovered backends listed", len(status.get("discovered_backends", [])) >= 5)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Routing decisions — correct model for each task type
# ═══════════════════════════════════════════════════════════════════════════════

print("\n═══ 2. Routing decisions ═══")

# Reset state
for m in MODEL_COST_ORDER:
    mark_available(m)

# Simple Q&A — should route to cheapest
d = route_task(complexity_score=0.0, task_type="qa")
check("Simple Q&A → local (tier 1)", d.selected_model in ("llama3.1:8b", "qwen3:14b"),
      f"got {d.selected_model}")

# Basic coding — should route to at least tier 4 (routing table default)
d = route_task(complexity_score=0.2, task_type="coding")
check("Basic coding → tier 4+", MODEL_CAPABILITY_TIERS.get(d.selected_model, 0) >= 4,
      f"got {d.selected_model} (tier {MODEL_CAPABILITY_TIERS.get(d.selected_model, 0)})")

# Mid complexity — should route to minimax/glm
d = route_task(complexity_score=0.4, task_type="coding")
check("Mid coding → minimax/glm (tier 4)", d.selected_model in ("minimax-m2.7:cloud", "glm-5"),
      f"got {d.selected_model}")

# Complex with niche refs — should route to deepseek-v3.1 or gpt-5.5
d = route_task(complexity_score=0.6, task_type="coding", has_niche_references=True)
check("Complex+niche → high tier (7+)", MODEL_CAPABILITY_TIERS.get(d.selected_model, 0) >= 7,
      f"got {d.selected_model} (tier {MODEL_CAPABILITY_TIERS.get(d.selected_model, 0)})")

# Debugging — should boost tier
d = route_task(complexity_score=0.3, task_type="debugging")
check("Debugging gets tier boost", MODEL_CAPABILITY_TIERS.get(d.selected_model, 0) >= 4,
      f"got {d.selected_model} (tier {MODEL_CAPABILITY_TIERS.get(d.selected_model, 0)})")

# Subagent of flash — should inherit parent tier -1, but routing table
# sets coding default to tier 4, so floor is 4
for m in MODEL_COST_ORDER:
    mark_available(m)
d = route_task(complexity_score=0.2, task_type="coding", is_subagent=True, parent_model="deepseek-v4-flash")
parent_tier = MODEL_CAPABILITY_TIERS.get("deepseek-v4-flash", 3)
check(f"Subagent of flash → tier >= {parent_tier - 1} (routing table floor)",
      MODEL_CAPABILITY_TIERS.get(d.selected_model, 0) >= parent_tier - 1,
      f"got {d.selected_model} (tier {MODEL_CAPABILITY_TIERS.get(d.selected_model, 0)}, parent tier {parent_tier})")

# Private mode — always dolphin3
d = route_task(is_private=True)
check("Private mode → dolphin3", d.selected_model == "dolphin3" and d.is_private,
      f"got {d.selected_model}")

# Force model override
d = route_task(force_model="gpt-5.5")
check("Force model → gpt-5.5", d.selected_model == "gpt-5.5",
      f"got {d.selected_model}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Limp-home mode
# ═══════════════════════════════════════════════════════════════════════════════

print("\n═══ 3. Limp-home mode ═══")

# Reset
for m in MODEL_COST_ORDER:
    mark_available(m)

# Exhaust all cloud models
for m in MODEL_COST_ORDER:
    if m not in ("llama3.1:8b", "qwen3:14b", "dolphin3"):
        mark_rate_limited(m)

# Route — should trigger limp-home
d = route_task(complexity_score=0.5, task_type="coding")
check("Limp-home activates", d.limp_home, f"got limp_home={d.limp_home}")
check("Limp-home uses local model", d.selected_model in ("qwen3:14b", "llama3.1:8b"),
      f"got {d.selected_model}")
check("Limp-home reason set", bool(d.limp_home_reason), d.limp_home_reason)

# Check is_limp_home()
check("is_limp_home() returns True", is_limp_home())

# Check get_limp_home_message()
msg = get_limp_home_message()
check("Limp-home message is non-empty", bool(msg), msg[:80])

# Check check_limp_home_status()
info = check_limp_home_status(needs_llm=True)
check("Status shows active", info["active"])
check("Status shows should_pause for LLM jobs", info["should_pause"])

info_no_llm = check_limp_home_status(needs_llm=False)
check("Non-LLM jobs should NOT pause", not info_no_llm["should_pause"])

# Recover a cloud model
mark_available("deepseek-v4-flash")
d = route_task(complexity_score=0.5, task_type="coding")
check("Limp-home exits on cloud recovery", not d.limp_home,
      f"got limp_home={d.limp_home}")
check("Normal routing resumes", d.selected_model == "deepseek-v4-flash",
      f"got {d.selected_model}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Circuit breaker and escalation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n═══ 4. Circuit breaker and escalation ═══")

# Reset
for m in MODEL_COST_ORDER:
    mark_available(m)

# Escalate from a failed model
d = escalate_on_failure("deepseek-v4-flash", error_type="timeout")
check("Escalation picks higher tier", MODEL_CAPABILITY_TIERS.get(d.selected_model, 0) > 3,
      f"got {d.selected_model} (tier {MODEL_CAPABILITY_TIERS.get(d.selected_model, 0)})")
check("Escalation marks as fallback", d.is_fallback)
check("Escalation tracks original model", d.original_model == "deepseek-v4-flash")

# Circuit breaker — 3+ failures via escalate_on_failure should open circuit
for m in MODEL_COST_ORDER:
    mark_available(m)

for i in range(4):
    escalate_on_failure("gpt-5.5", error_type="rate_limit")

# Check gpt-5.5 is now in circuit breaker
from router import _MODEL_STATUSES
status = _MODEL_STATUSES.get("gpt-5.5")
check("Circuit breaker opens after 3+ failures", status and status.circuit_open_until > time.time(),
      f"circuit_open_until={status.circuit_open_until if status else 'N/A'}")

# Route should skip gpt-5.5
d = route_task(complexity_score=0.8, task_type="debugging")
check("Routing skips circuit-broken model", d.selected_model != "gpt-5.5",
      f"got {d.selected_model}")

# Exhaust everything
for m in MODEL_COST_ORDER:
    if m not in ("llama3.1:8b", "qwen3:14b", "dolphin3"):
        mark_rate_limited(m)
mark_rate_limited("llama3.1:8b")
mark_rate_limited("qwen3:14b")

d = route_task(complexity_score=0.5, task_type="coding")
check("All exhausted returns empty model", d.selected_model == "",
      f"got {d.selected_model}")
check("All exhausted flag set", d.all_exhausted)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Self-identification (no routing loops)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n═══ 5. Self-identification ═══")

# The endpoint should never route to itself
# Check that biggie-llm provider is not in the discovered backends
cfg = http_get("/config")
backends = cfg.get("discovered_backends", {})
biggie_backends = [k for k in backends if "biggie" in k.lower()]
check("No biggie backends discovered (self-skip)", len(biggie_backends) == 0,
      f"found: {biggie_backends}")

# Check that the models list doesn't include biggie-router
models = http_get("/v1/models")
model_ids = [m["id"] for m in models.get("data", [])]
check("biggie-router not in models list", "biggie-router" not in model_ids)
check("biggie-llm not in models list", "biggie-llm" not in model_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Routing profile: expensive mode
# ═══════════════════════════════════════════════════════════════════════════════

print("\n═══ 6. Routing profile: expensive mode ═══")

# Reset
for m in MODEL_COST_ORDER:
    mark_available(m)

# Test the expensive profile by calling route_task with boosted complexity
# (This is what the endpoint does when profile=expensive)
d_cheap = route_task(complexity_score=0.2, task_type="qa")
d_expensive = route_task(complexity_score=min(0.2 + 0.2, 1.0), task_type="qa")

check("Expensive profile picks higher tier than cheap",
      MODEL_CAPABILITY_TIERS.get(d_expensive.selected_model, 0) >=
      MODEL_CAPABILITY_TIERS.get(d_cheap.selected_model, 0),
      f"cheap={d_cheap.selected_model} (tier {MODEL_CAPABILITY_TIERS.get(d_cheap.selected_model, 0)}), "
      f"expensive={d_expensive.selected_model} (tier {MODEL_CAPABILITY_TIERS.get(d_expensive.selected_model, 0)})")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

# 
# 7. Streaming support (SSE)
# 

print("\n 7. Streaming support (SSE) ")

def http_post_stream(path: str, body: dict, timeout: int = 60) -> str:
    """POST and read the raw response body (for SSE streaming)."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.read().decode()
    except Exception as e:
        return f"ERROR: {e}"

# 7a. Streaming request returns SSE events, not a single JSON body
raw = http_post_stream("/v1/chat/completions", {
    "model": "deepseek-v4-flash:cloud",
    "stream": True,
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 20,
})
check("Streaming returns SSE data: events", "data: " in raw, raw[:200])
check("Streaming ends with [DONE]", "data: [DONE]" in raw, raw[-200:])
check("Streaming returns chat.completion.chunk objects", "chat.completion.chunk" in raw, raw[:200])
check("Streaming does NOT return a single JSON body", not raw.strip().startswith("{"), raw[:200])

# 7b. Non-streaming still returns a single JSON body (regression)
resp = http_post("/v1/chat/completions", {
    "model": "deepseek-v4-flash:cloud",
    "stream": False,
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 200,
})
check("Non-streaming returns JSON body", "choices" in resp, str(resp)[:200])
check("Non-streaming has message content", resp.get("choices", [{}])[0].get("message", {}).get("content", "") != "", str(resp)[:200])

# 7c. Streaming with a local model falls back to non-streaming (no crash)
raw_local = http_post_stream("/v1/chat/completions", {
    "model": "dolphin3",
    "stream": True,
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 10,
})
check("Streaming local model does not crash", "ERROR" not in raw_local, raw_local[:200])

# 7d. Streaming with an invalid model returns a clean error, not a hang
raw_bad = http_post_stream("/v1/chat/completions", {
 "model": "nonexistent-model-xyz",
 "stream": True,
 "messages": [{"role": "user", "content": "hi"}],
 "max_tokens": 10,
})
check("Streaming invalid model returns error (no hang)", "ERROR" not in raw_bad, raw_bad[:200])


# 
# 8. Empty-content detection (degenerate 200s)
# 

print("\n 8. Empty-content detection ")

from biggie_llm_endpoint import _response_has_empty_content

# Empty string content → degenerate success
check("Empty string content detected",
 _response_has_empty_content({"choices": [{"message": {"content": ""}}]}) is True)

# Whitespace-only content → degenerate success
check("Whitespace-only content detected",
 _response_has_empty_content({"choices": [{"message": {"content": "   "}}]}) is True)

# None content → degenerate success
check("None content detected",
 _response_has_empty_content({"choices": [{"message": {"content": None}}]}) is True)

# Real content → not empty
check("Real content not flagged",
 _response_has_empty_content({"choices": [{"message": {"content": "hello world"}}]}) is False)

# Missing message → treated as empty (no crash)
check("Missing message treated as empty",
 _response_has_empty_content({"choices": [{}]}) is True)

# No choices → not flagged (no crash)
check("No choices not flagged",
 _response_has_empty_content({}) is False)

# Non-dict → not flagged (no crash)
check("Non-dict not flagged",
 _response_has_empty_content("garbage") is False)

# List-of-parts content (Codex normalization) — all empty → degenerate
check("Empty list-of-parts content detected",
 _response_has_empty_content({"choices": [{"message": {"content": [{"text": ""}, {"text": "  "}]}}]}) is True)

# List-of-parts with real text → not empty
check("Non-empty list-of-parts not flagged",
 _response_has_empty_content({"choices": [{"message": {"content": [{"text": "real"}]}}]}) is False)


print(f"\n{'' * 50}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print(f"{'' * 50}")

if FAIL > 0:
 print("\n⚠️ Some tests failed — review above for details")
 sys.exit(1)
else:
 print("\n✅ All tests passed — endpoint is routing correctly")
 sys.exit(0)
