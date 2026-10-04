#!/usr/bin/env python3
"""Focused native tool-calling probe: deepseek-v4.1-flash vs deepseek-v4-pro.

Answers the question "can simple tool-bearing traffic go to flash?" with real
measurement, mirroring the existing probe_glm53_flash.py conventions (direct
provider call, key from .env, never printed).

Test matrix (T1-T3 from the review):
  T1  single tool, non-stream (read-only get_weather)
  T2  single tool, stream
  T3  multi-tool, stream
Each with a REAL instruction (no synthetic no-task prompt, which provokes
reasoning runaway and false "empty" — see shadow harness lesson).

Also records finish_reason, content length, reasoning length, valid-json args,
and empty-content rate.

Usage: python3 scripts/probe_tool_calling_ollama.py
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

ENV = Path.home() / ".hermes" / ".env"
OUT_DIR = Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data"
BASE = "https://ollama.com/v1/chat/completions"

MODELS = ["deepseek-v4.1-flash", "deepseek-v4-pro"]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file at a path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        },
    },
}
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}


def api_key() -> str:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("OLLAMA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OLLAMA_API_KEY not found in .env")


def call(model, messages, tools=None, max_tokens=4096, stream=False, timeout=120):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": stream}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    out = {"model": model, "stream": stream, "error": None, "content": "",
           "reasoning": "", "finish_reason": None, "tool_calls": [], "usage": {}}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not stream:
                data = json.load(r)
                ch = (data.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                out["content"] = msg.get("content") or ""
                out["reasoning"] = msg.get("reasoning") or ""
                out["finish_reason"] = ch.get("finish_reason")
                out["tool_calls"] = msg.get("tool_calls") or []
                out["usage"] = data.get("usage") or {}
            else:
                tc_acc = {}
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("usage"):
                        out["usage"] = ev["usage"]
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        if d.get("content"):
                            out["content"] += d["content"]
                        if d.get("reasoning"):
                            out["reasoning"] += d["reasoning"]
                        for tc in d.get("tool_calls") or []:
                            i = tc.get("index", 0)
                            slot = tc_acc.setdefault(i, {"id": "", "function": {"name": "", "arguments": ""}})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]
                        if ch.get("finish_reason"):
                            out["finish_reason"] = ch["finish_reason"]
                out["tool_calls"] = [tc_acc[i] for i in sorted(tc_acc)]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["latency_s"] = round(time.time() - t0, 2)
    return out


def classify(res):
    tcs = res["tool_calls"]
    if not tcs:
        prose = "tool" in (res["content"] or "").lower() or "get_weather" in (res["content"] or "")
        return "prose-tool-syntax" if prose else "none"
    for tc in tcs:
        args = (tc.get("function") or {}).get("arguments") or ""
        try:
            json.loads(args or "{}")
        except json.JSONDecodeError:
            return "invalid-json-args"
    return "structured"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="reps per case per model")
    args = ap.parse_args()

    global KEY
    KEY = api_key()

    cases = [
        ("T1_single_nonstream", dict(
            messages=[{"role": "user", "content":
                       "What is the current weather in Melbourne? Use the get_weather tool."}],
            tools=[WEATHER_TOOL], max_tokens=4096, stream=False)),
        ("T2_single_stream", dict(
            messages=[{"role": "user", "content":
                       "Read the file at /tmp/example.txt and tell me what it says. Use the read_file tool."}],
            tools=[READ_TOOL], max_tokens=4096, stream=True)),
        ("T3_multi_stream", dict(
            messages=[{"role": "user", "content":
                       "Get the weather for Sydney, then add 17 and 25. Use the available tools."}],
            tools=[WEATHER_TOOL, CALC_TOOL], max_tokens=4096, stream=True)),
    ]

    rows = []
    for model in MODELS:
        print(f"\n-- {model}")
        for name, kw in cases:
            for r in range(args.reps):
                res = call(model, **kw)
                res["test"] = name
                rows.append(res)
                cls = classify(res)
                usage = res.get("usage") or {}
                print(f"  {name:20s} rep={r} cls={cls:20s} fin={str(res['finish_reason']):12s} "
                      f"content={len(res['content'] or ''):>5} reason={len(res['reasoning'] or ''):>5} "
                      f"in={usage.get('prompt_tokens',0):>6} out={usage.get('completion_tokens',0):>5} "
                      f"err={res['error']}")
                time.sleep(0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "probe_tool_calling_ollama.json"
    out.write_text(json.dumps(rows, indent=2))

    print("\n=== AGGREGATE (structured tool-call rate) ===")
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        structured = sum(1 for r in sub if classify(r) == "structured")
        prose = sum(1 for r in sub if classify(r) == "prose-tool-syntax")
        none = sum(1 for r in sub if classify(r) == "none")
        invalid = sum(1 for r in sub if classify(r) == "invalid-json-args")
        errs = sum(1 for r in sub if r["error"])
        print(f"{model:22s} n={len(sub)} structured={structured} prose={prose} "
              f"none={none} invalid={invalid} errors={errs} "
              f"rate={structured/len(sub)*100:.0f}%")
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
