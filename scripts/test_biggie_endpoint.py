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
from biggie_llm_endpoint import (
    detect_workload_type,
    compression_level_for_workload,
    COMPRESSION_LEVEL,
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

# Native Hermes session compression — huge context should be routed as a
# summarisation workload, not over-escalated to the biggest reasoning model.
for m in MODEL_COST_ORDER:
    mark_available(m)
d = route_task(
    complexity_score=0.99,
    task_type="session_compression",
    workload_type="session_compression",
    context_tokens=240_000,
    prompt="Context compaction: summarize earlier conversation and preserve key facts.",
)
check("Session compression → cheap cloud summariser",
      d.selected_model == "deepseek-v4-flash",
      f"got {d.selected_model} ({d.reason})")
check("Session compression does not default to gpt-5.5",
      d.selected_model != "gpt-5.5",
      f"got {d.selected_model}")

# If flash is unavailable, native compression escalates within the router policy.
mark_rate_limited("deepseek-v4-flash")
d = route_task(
    complexity_score=0.99,
    task_type="session_compression",
    workload_type="session_compression",
    context_tokens=240_000,
)
check("Session compression falls back to next summariser",
      d.selected_model in ("glm-5.2", "deepseek-v4-pro", "deepseek-v3.1:671b", "gpt-5.5"),
      f"got {d.selected_model}")
mark_available("deepseek-v4-flash")

# Workload detection — explicit metadata/header is preferred; Hermes prompt
# markers are a fallback for current auxiliary clients that do not send headers.
compression_messages = [{
    "role": "user",
    "content": "Context compaction: summarize earlier conversation so I can continue. Preserve key facts, decisions, and open tasks.",
}]
check("Detect session compression from metadata",
      detect_workload_type(compression_messages, {"metadata": {"hermes_task": "compression"}}, {}) == "session_compression")
check("Detect session compression from prompt markers",
      detect_workload_type(compression_messages, {}, {}) == "session_compression")
check("Normal prompt is normal_chat workload",
      detect_workload_type([{"role": "user", "content": "Say hi"}], {}, {}) == "normal_chat")

class _FakeRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}

check("Session compression uses conservative Biggie compression by default",
      compression_level_for_workload(_FakeRequest(), "session_compression") in ("lite", "off"))
check("Header overrides workload compression level",
      compression_level_for_workload(_FakeRequest({"X-Compression-Level": "off"}), "session_compression") == "off")
check("Normal workload uses configured Biggie compression",
      compression_level_for_workload(_FakeRequest(), "normal_chat") == COMPRESSION_LEVEL)


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


# ── Codex tool-conversation state preservation (repeated tool-call loop fix) ──
from biggie_llm_endpoint import _codex_input_from_responses_items

def _codex_items(messages):
    """Run the full Hermes adapter + router conversion on a message list."""
    import sys as _sys
    _hermes = str(Path.home() / ".hermes" / "hermes-agent")
    if _hermes not in _sys.path:
        _sys.path.insert(0, _hermes)
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    return _codex_input_from_responses_items(_chat_messages_to_responses_input(messages))

# TEST 1 — sequential commands: tool result must be present upstream after FIRST
_seq = [
    {"role": "user", "content": "Run echo FIRST then echo SECOND."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\": \"echo FIRST\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "FIRST"},
]
_seq_items = _codex_items(_seq)
check("TEST1: function_call item preserved upstream",
 any(i.get("type") == "function_call" and i.get("call_id") == "call_1" for i in _seq_items))
check("TEST1: function_call_output (tool result) preserved upstream",
 any(i.get("type") == "function_call_output" and i.get("call_id") == "call_1" and i.get("output") == "FIRST" for i in _seq_items))
check("TEST1: no empty user message wrapping tool items",
 not any(i.get("type") == "message" and i.get("role") == "user" and i.get("content") == [{"type": "input_text", "text": ""}] for i in _seq_items))

# TEST 2 — no duplicate loop: next request must NOT reconstruct pre-tool state.
# The tool result must be present; the original user prompt must remain once.
_user_msgs = [i for i in _seq_items if i.get("type") == "message" and i.get("role") == "user"]
check("TEST2: user prompt present exactly once",
 len(_user_msgs) == 1 and "echo FIRST" in _user_msgs[0]["content"][0]["text"])
check("TEST2: tool result present (not dropped to pre-tool state)",
 any(i.get("type") == "function_call_output" for i in _seq_items))

# TEST 3 — call identity: function_call call_id and function_call_output call_id paired
_fc = [i for i in _seq_items if i.get("type") == "function_call"]
_fco = [i for i in _seq_items if i.get("type") == "function_call_output"]
check("TEST3: function_call and output call_ids paired",
 len(_fc) == 1 and len(_fco) == 1 and _fc[0]["call_id"] == _fco[0]["call_id"] == "call_1")

# TEST 4 — multi-turn tool sequence: three distinct calls all preserved
_multi = [{"role": "user", "content": "Run FIRST, SECOND, THIRD in order."}]
for i, cmd in enumerate(["FIRST", "SECOND", "THIRD"]):
    _multi.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": "terminal", "arguments": f'{{"command": "echo {cmd}"}}'}}
    ]})
    _multi.append({"role": "tool", "tool_call_id": f"call_{i}", "content": cmd})
_multi_items = _codex_items(_multi)
_fcos = [i for i in _multi_items if i.get("type") == "function_call_output"]
check("TEST4: three distinct tool results preserved",
 len(_fcos) == 3 and [i["output"] for i in _fcos] == ["FIRST", "SECOND", "THIRD"])
check("TEST4: three distinct function_calls preserved",
 len([i for i in _multi_items if i.get("type") == "function_call"]) == 3)

# TEST 5 — existing routes unchanged: plain (non-tool) conversation still wraps as messages
_plain = _codex_items([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}])
check("TEST5: plain user message wrapped correctly",
 _plain[0] == {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]})
check("TEST5: plain assistant message wrapped correctly",
 _plain[1] == {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]})


print(f"\n{'' * 50}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print(f"{'' * 50}")

if FAIL > 0:
 print("\n⚠️ Some tests failed — review above for details")
 sys.exit(1)
else:
 print("\n✅ All tests passed — endpoint is routing correctly")
 sys.exit(0)
