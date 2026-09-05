#!/usr/bin/env python3
"""Probe gpt-6-astra DIRECTLY on the codex /responses endpoint, with reasoning.

The router proxies codex calls but hardcodes the response body and does NOT
forward a `reasoning` param (no effort control). For a large review you'll want
to hit chatgpt.com/backend-api/codex/responses directly so you can pass
reasoning effort. This verifies that path works for astra.

Usage:
  python3 probe_astra_direct.py [--no-reasoning] [--stream]
"""
import json
import sys
from pathlib import Path

import httpx

BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
MODEL = "gpt-6-astra"
PROMPT = "Reply with exactly: ASTRA_DIRECT_OK"


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


def main() -> int:
    use_reasoning = "--no-reasoning" not in sys.argv
    stream = "--stream" in sys.argv or True
    effort = "low"
    if "--effort" in sys.argv:
        i = sys.argv.index("--effort")
        effort = sys.argv[i + 1]

    body = {
        "model": MODEL,
        "input": [{"role": "user", "content": PROMPT}],
        "store": False,
        "stream": stream,
    }
    if use_reasoning:
        body["reasoning"] = {"effort": effort}

    token = get_codex_token()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    print(f"POST {BASE_URL}  model={MODEL}  reasoning={'yes' if use_reasoning else 'no'}  effort={effort}  stream={stream}")
    try:
        with httpx.Client(timeout=120) as client:
            if stream:
                with client.stream("POST", BASE_URL, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        print("HTTP ERROR", resp.status_code, resp.text[:500])
                        return 1
                    text = []
                    usage = None
                    for line in resp.iter_lines():
                        line = line.strip()
                        if line.startswith("data: "):
                            try:
                                evt = json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue
                            if evt.get("type") == "response.output_text.delta":
                                text.append(evt.get("delta", ""))
                            elif evt.get("type") == "response.completed":
                                usage = evt.get("response", {}).get("usage")
                    if usage:
                        print("USAGE:", json.dumps(usage))
                    print("OUTPUT:", "".join(text).strip() or "(none extracted)")
            else:
                resp = client.post(BASE_URL, json=body, headers=headers)
                if resp.status_code != 200:
                    print("HTTP ERROR", resp.status_code, resp.text[:500])
                    return 1
                data = resp.json()
                print("USAGE:", json.dumps(data.get("usage", {})))
                out = []
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                out.append(c.get("text", ""))
                print("OUTPUT:", "".join(out).strip() or "(none)")
    except Exception as e:  # noqa: BLE001
        print("REQUEST FAILED:", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
