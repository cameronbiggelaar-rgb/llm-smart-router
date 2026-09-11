#!/usr/bin/env python3
"""Basic capability tests for glm-5.3-flash:cloud (Phase 1) + head-to-head vs
the deepseek flash models (Phase 2).

Calls the provider (https://ollama.com/v1) DIRECTLY: the router overrides the
client model for session_compression and discovery ignores custom_providers, so
a direct provider call is the only honest way to test a not-yet-wired model.

Credential is read from .env at runtime and never printed.

Usage:
  python3 scripts/probe_glm53_flash.py --phase sanity
  python3 scripts/probe_glm53_flash.py --phase h2h
  python3 scripts/probe_glm53_flash.py --phase both
"""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

ENV = Path.home() / ".hermes" / ".env"
OUT_DIR = Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data"
BASE = "https://ollama.com/v1/chat/completions"

GLM = "glm-5.3-flash"
FLASH_MODELS = ["glm-5.3-flash", "deepseek-v4.1-flash", "deepseek-v4-flash"]

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


KEY = None


def call(model, messages, tools=None, max_tokens=2048, stream=False, timeout=300):
    """Return a normalized result dict."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": stream}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    out = {"model": model, "stream": stream, "error": None,
           "content": "", "reasoning": "", "finish_reason": None,
           "tool_calls": [], "usage": {}, "latency_s": 0.0}
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
                    if ev.get("model"):
                        out["served_model"] = ev["model"]
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


def tool_shape(res):
    """Classify tool-call output: structured / invalid-json / none."""
    if not res["tool_calls"]:
        prose = "tool" in res["content"].lower() or "get_weather" in res["content"]
        return "prose-tool-syntax" if prose else "none"
    for tc in res["tool_calls"]:
        args = (tc.get("function") or {}).get("arguments") or ""
        try:
            json.loads(args or "{}")
        except json.JSONDecodeError:
            return "invalid-json-args"
    return "structured"


def filler_prompt(target_tokens, salt):
    """Numeric filler calibrates ~2.5 chars/token on the ollama backend."""
    chunk = " ".join(str((i * 7919 + salt) % 100000) for i in range(1, 801))
    return (chunk + " ") * max(1, int(target_tokens * 2.5 / len(chunk)))


# ── Test cases ────────────────────────────────────────────────────────────────
def sanity_cases():
    return [
        ("compliance", dict(messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                            max_tokens=512)),
        ("tool_single_nonstream", dict(
            messages=[{"role": "user", "content": "What is the weather in Melbourne right now? Use the tool."}],
            tools=[WEATHER_TOOL], max_tokens=2048)),
        ("tool_single_stream", dict(
            messages=[{"role": "user", "content": "What is the weather in Melbourne right now? Use the tool."}],
            tools=[WEATHER_TOOL], max_tokens=2048, stream=True)),
        ("tool_multi_stream", dict(
            messages=[{"role": "user", "content": "Get the weather for Sydney and also add 17 and 25. Use the tools."}],
            tools=[WEATHER_TOOL, CALC_TOOL], max_tokens=2048, stream=True)),
        ("structured_json", dict(
            messages=[{"role": "user", "content":
                       'Return ONLY compact JSON: {"status":"ok","count":3,"items":["a","b","c"]}'}],
            max_tokens=1024)),
        ("reasoning_arithmetic", dict(
            messages=[{"role": "user", "content":
                       "A train travels 240 km in 2.5 hours, then 90 km in 1.5 hours. "
                       "What is the average speed for the whole trip in km/h? Answer with the number only."}],
            max_tokens=4096)),
        ("output_budget_64", dict(
            messages=[{"role": "user", "content":
                       "Write a two-sentence summary of what a SQLite WAL file is."}],
            max_tokens=64)),
        ("context_30k", dict(
            messages=[{"role": "system", "content": "The user message contains numeric filler followed by a question."},
                       {"role": "user",
                        "content": filler_prompt(30000, 11) + "\n\nQUESTION: What single word appears immediately after the filler above?"}],
            max_tokens=2048)),
    ]


def h2h_cases():
    """Same payloads for every model so differences are the model, not the input."""
    return [
        ("compliance", dict(messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                            max_tokens=512)),
        ("tool_single_nonstream", dict(
            messages=[{"role": "user", "content": "What is the weather in Melbourne right now? Use the tool."}],
            tools=[WEATHER_TOOL], max_tokens=2048)),
        ("tool_single_stream", dict(
            messages=[{"role": "user", "content": "What is the weather in Melbourne right now? Use the tool."}],
            tools=[WEATHER_TOOL], max_tokens=2048, stream=True)),
        ("structured_json", dict(
            messages=[{"role": "user", "content":
                       'Return ONLY compact JSON: {"status":"ok","count":3,"items":["a","b","c"]}'}],
            max_tokens=1024)),
        ("reasoning_arithmetic", dict(
            messages=[{"role": "user", "content":
                       "A train travels 240 km in 2.5 hours, then 90 km in 1.5 hours. "
                       "What is the average speed for the whole trip in km/h? Answer with the number only."}],
            max_tokens=4096)),
        ("context_30k", dict(
            messages=[{"role": "system", "content": "The user message contains numeric filler followed by a question."},
                       {"role": "user",
                        "content": filler_prompt(30000, 11) + "\n\nQUESTION: What single word appears immediately after the filler above?"}],
            max_tokens=2048)),
    ]


def summarize(name, res, extra=""):
    content = (res["content"] or "").strip().replace("\n", " ")[:140]
    usage = res.get("usage") or {}
    return (f"  {name:26s} fin={str(res['finish_reason']):12s} "
            f"lat={res['latency_s']:7.2f}s "
            f"in={usage.get('prompt_tokens', 0):>7} out={usage.get('completion_tokens', 0):>5} "
            f"reason_chars={len(res['reasoning'] or ''):>6} "
            f"content={len(res['content'] or ''):>5} "
            f"tools={tool_shape(res):18s} {extra}"
            f"err={res['error']}")


def run(models, cases, label):
    rows = []
    print(f"\n=== {label} ===")
    for model in models:
        print(f"-- {model}")
        for name, kw in cases:
            res = call(model, **kw)
            rows.append({"test": name, **res})
            print(summarize(name, res))
            time.sleep(0.3)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sanity", "h2h", "both"], default="sanity")
    args = ap.parse_args()

    global KEY
    KEY = api_key()

    rows = []
    if args.phase in ("sanity", "both"):
        rows += run([GLM], sanity_cases(), "PHASE 1 — glm-5.3-flash basic sanity")
    if args.phase in ("h2h", "both"):
        rows += run(FLASH_MODELS, h2h_cases(), "PHASE 2 — head-to-head vs deepseek flash")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"probe_glm53_flash_{args.phase}.json"
    out.write_text(json.dumps(rows, indent=2))

    # Aggregate per model
    print("\n=== AGGREGATE ===")
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        lats = [r["latency_s"] for r in sub]
        tool_shape_ok = sum(1 for r in sub if tool_shape(r) == "structured")
        empties = sum(1 for r in sub
                      if not (r["content"] or "").strip() and not r["tool_calls"]
                      and not r["error"])
        errs = sum(1 for r in sub if r["error"])
        in_tok = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in sub)
        out_tok = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in sub)
        print(f"{model:22s} n={len(sub)} tool_structured={tool_shape_ok} "
              f"empty={empties} errors={errs} "
              f"lat_med={statistics.median(lats):.2f}s in={in_tok} out={out_tok}")
    print(f"\nraw → {out}")


if __name__ == "__main__":
    main()
