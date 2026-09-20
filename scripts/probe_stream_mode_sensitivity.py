#!/usr/bin/env python3
"""Stream-mode sensitivity probe: does flash's tool CHOICE change with stream mode?

The live shadow forces flash to stream=False while pro streams, and the audit
flagged that as a real experimental-control confound: the ~44% tool-choice
divergence (flash picks `terminal` where pro picks `read_file`) might be partly
a decoding artifact rather than a stable model trait.

This probe takes the divergence class (file-inspection tasks where the right
tool is read_file/search_files, and flash was observed reaching for terminal)
and runs the SAME prompts through deepseek-v4.1-flash in BOTH stream and
non-stream modes, with a realistic tool catalog. If flash picks `read_file`
when streamed but `terminal` when non-streamed, the harness is inflating the
divergence. If it picks `terminal` in both modes, the preference is real.

No synthetic no-task prompts (those provoke reasoning runaway — see shadow
harness lesson). Every prompt is a concrete file-inspection task.

Usage: python3 scripts/probe_stream_mode_sensitivity.py [--reps N]
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

ENV = Path.home() / ".hermes" / ".env"
OUT_DIR = Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data"
BASE = "https://ollama.com/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"


def _tool(name, desc, props, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


# Realistic Hermes-style catalog, read_file listed BEFORE terminal (as the
# agent's own tool listing does — read-only file tools are preferred first).
TOOLS = [
    _tool("read_file", "Read the contents of a text file at a given path.",
          {"path": {"type": "string", "description": "File path"}}, ["path"]),
    _tool("search_files", "Search file contents or find files by name.",
          {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    _tool("terminal", "Run a shell command and return its output.",
          {"command": {"type": "string", "description": "The shell command"}}, ["command"]),
    _tool("write_file", "Write content to a file.",
          {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
]

# File-inspection prompts: the divergence class. Each should select read_file
# or search_files; terminal is the "blunt instrument" flash was seen reaching for.
PROMPTS = [
    "Read the contents of the file at /home/hermes/config.yaml and show me the first 20 lines.",
    "What does the file /etc/hostname contain? Show me its contents.",
    "Look at the file ~/.hermes/config.yaml and tell me which models are configured there.",
    "Find every Python file under src/ that imports 'requests'.",
    "Show me the contents of /home/hermes/notes/todo.md.",
    "Read the README at the repo root and summarise the setup instructions.",
    "Check what's in /home/hermes/.hermes/skills/llm-smart-router/scripts/experiments.yaml.",
    "Search the file /var/log/syslog for lines containing the word 'error'.",
]


def api_key() -> str:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("OLLAMA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OLLAMA_API_KEY not found in .env")


def call(stream, messages, tools, max_tokens=512, timeout=120):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "stream": stream, "tools": tools}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    out = {"stream": stream, "finish_reason": None, "tool_names": [], "error": None}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not stream:
                data = json.load(r)
                ch = (data.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                out["finish_reason"] = ch.get("finish_reason")
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        out["tool_names"].append(fn["name"])
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
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        for tc in d.get("tool_calls") or []:
                            i = tc.get("index", 0)
                            slot = tc_acc.setdefault(i, "")
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot = fn["name"]
                            tc_acc[i] = slot
                        if ch.get("finish_reason"):
                            out["finish_reason"] = ch["finish_reason"]
                out["tool_names"] = [tc_acc[i] for i in sorted(tc_acc)]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["latency_s"] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="reps per prompt per mode")
    args = ap.parse_args()
    global KEY
    KEY = api_key()

    rows = []
    for pi, prompt in enumerate(PROMPTS):
        for stream in (False, True):
            for r in range(args.reps):
                res = call(stream, [{"role": "user", "content": prompt}], TOOLS)
                res["prompt_idx"] = pi
                res["prompt"] = prompt
                rows.append(res)
                names = ",".join(res["tool_names"]) or "(none)"
                print(f"p{pi:02d} stream={int(stream)} rep={r} "
                      f"tools=[{names}] fin={res['finish_reason']} err={res['error']}")
                time.sleep(0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "probe_stream_mode_sensitivity.json"
    out.write_text(json.dumps(rows, indent=2))

    # Aggregate: for each prompt, does tool choice differ by stream mode?
    print("\n=== TOOL CHOICE BY STREAM MODE (flash only) ===")
    for pi, prompt in enumerate(PROMPTS):
        ns = [tuple(sorted(r["tool_names"])) for r in rows
              if r["prompt_idx"] == pi and not r["stream"] and not r["error"]]
        s = [tuple(sorted(r["tool_names"])) for r in rows
             if r["prompt_idx"] == pi and r["stream"] and not r["error"]]
        ns_set = set(ns); s_set = set(s)
        tag = "SAME" if ns_set == s_set else "DIFFERS"
        print(f"p{pi:02d} [{tag}] nonstream={sorted(ns_set)} stream={sorted(s_set)}")

    # Count read_file/terminal picks per mode
    def pick_count(stream):
        counts = {}
        for r in rows:
            if r["stream"] != stream or r["error"]:
                continue
            for n in r["tool_names"]:
                counts[n] = counts.get(n, 0) + 1
        return counts
    print("\nnon-stream picks:", pick_count(False))
    print("stream picks:    ", pick_count(True))
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
