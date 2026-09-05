"""Probe the three gpt-5.6 models for clean structured tool-calling.

Mirrors the deepseek-v4-pro verification: each model must emit a real
function call (tool_calls) with a clean finish_reason, not fake tool-call
syntax as prose. Uses the same OAuth path the router uses
(chatgpt.com/backend-api/codex/responses, stream:true).

Usage: python3 scripts/probe_tool_calling.py
"""
import json
import sys
from pathlib import Path

import httpx

MODELS = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
BASE_URL = "https://chatgpt.com/backend-api/codex/responses"

TOOL_SCHEMA = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
}


def get_codex_token() -> str:
    auth_path = Path.home() / ".hermes" / "auth.json"
    with open(auth_path) as f:
        data = json.load(f)
    pool = data.get("credential_pool", {})
    entries = pool.get("openai-codex", [])
    for entry in entries:
        if isinstance(entry, dict):
            status = entry.get("last_status")
            err = entry.get("last_error_code")
            token = entry.get("access_token", "")
            if status == "exhausted" and err == 429:
                continue
            if token:
                return token
    for entry in entries:
        if isinstance(entry, dict) and entry.get("access_token"):
            return entry["access_token"]
    raise RuntimeError("no openai-codex token found in auth.json")


def probe(model: str, token: str) -> dict:
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": "What is the weather in Sydney? Use the get_weather tool.",
            }
        ],
        "tools": [{"type": "function", "name": "get_weather", "description": TOOL_SCHEMA["description"], "parameters": TOOL_SCHEMA["parameters"]}],
        "store": False,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    with httpx.Client(timeout=60) as client:
        with client.stream("POST", BASE_URL, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                return {"model": model, "status": resp.status_code, "error": resp.text[:300]}
            # Collect SSE events
            output = []
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    try:
                        evt = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    output.append(evt)
    return {"model": model, "status": 200, "events": output}


def analyze(model: str, result: dict) -> dict:
    if result.get("status") != 200:
        return {"model": model, "ok": False, "reason": f"HTTP {result['status']}: {result.get('error')}"}
    events = result.get("events", [])
    # Look for function_call output items
    tool_calls = []
    finish_reason = None
    for evt in events:
        if evt.get("type") == "response.output_item.added":
            item = evt.get("item", {})
            if item.get("type") == "function_call":
                tool_calls.append(item.get("name"))
        if evt.get("type") == "response.completed":
            finish_reason = evt.get("response", {}).get("status")
    clean = bool(tool_calls) and finish_reason in ("completed", "in_progress")
    return {
        "model": model,
        "ok": clean,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "event_count": len(events),
    }


def main() -> int:
    token = get_codex_token()
    all_ok = True
    for model in MODELS:
        result = probe(model, token)
        verdict = analyze(model, result)
        print(json.dumps(verdict))
        if not verdict["ok"]:
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
