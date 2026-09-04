#!/usr/bin/env python3
"""
flash_ceiling_headtohead.py
===========================
Head-to-head test: does deepseek-v4-flash succeed on 105-145K-token contexts,
or does it empty-stream (which is the rationale for the BIGGIE_FLASH_MAX_CONTEXT_TOKENS
bypass that starts compression at glm-5.3 instead of flash)?

Method: For each target context size, build a realistic payload from REAL
Hermes session content (system prompt + tool results + user turns), then POST
it to the LIVE biggie-llm router (:8080) forcing the model. The router's own
escalation logic reports which model actually produced content:
  - response.model == flash + content present  -> flash succeeded
  - response.model != flash (escalated)        -> flash failed/empty-stream

No direct ollama.com calls. Read-only on state.db. Live router untouched
(config/cap not modified).

Usage:
  python3 flash_ceiling_headtohead.py --sizes 105000 125000 145000 --runs 2
"""
import argparse
import json
import sqlite3
import time
import urllib.request
from pathlib import Path

ROUTER_URL = "http://127.0.0.1:8080/v1/chat/completions"
STATE_DB = Path.home() / ".hermes" / "state.db"

MODELS = ["deepseek-v4-flash:cloud", "glm-5.3:cloud"]
# Note: endpoint treats "model" not in ("","biggie-router","biggie-llm") as force_model

TARGETS = [105_000, 115_000, 125_000, 135_000, 145_000]

# Rough token estimate: ~4 chars/token for mixed content.
CHARS_PER_TOKEN = 4.0


def load_real_context(sid: str = "20260612_202206_c89a7e"):
    """Pull the real system prompt + an ordered message list from a large session."""
    conn = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    sp = conn.execute(
        "SELECT system_prompt FROM sessions WHERE id=?", (sid,)
    ).fetchone()[0]
    rows = conn.execute(
        """SELECT role, content FROM messages
           WHERE session_id=? AND active=1
           ORDER BY timestamp ASC, id ASC""",
        (sid,),
    ).fetchall()
    conn.close()
    sys_prompt = sp or "You are a helpful assistant."
    messages = [{"role": r, "content": c or ""} for r, c in rows if c]
    return sys_prompt, messages


def build_payload(sys_prompt, real_msgs, target_tokens):
    """Build a chat payload whose visible content approximates target_tokens."""
    # Seed with the real conversation, then pad with realistic filler (tool-style
    # content) to reach the target size. Filler is derived from real content.
    system_msg = {"role": "system", "content": sys_prompt}
    base_chars = int(target_tokens * CHARS_PER_TOKEN)
    current_chars = len(sys_prompt) + sum(len(m["content"]) for m in real_msgs)

    messages = [system_msg] + list(real_msgs)

    if current_chars >= base_chars:
        return messages, current_chars

    # Pad using realistic long tool-result style filler.
    filler_block = (
        "=== tool_result: web_search ===\n"
        "Query: enterprise-architecture australia health systems compliance 2026. "
        "Returned 20 results. Each result below includes title, url, snippet, and "
        "relevance score relative to the query. Relevance is scored 0-100 by cosine "
        "similarity against the embedding of the full user query and is capped at "
        "100 to avoid overflow. Documents are ordered by relevance descending, ties "
        "broken by recency. Full-text extraction was performed on each candidate "
        "URL; boilerplate (nav, footer, cookie banners) was stripped before "
        "chunking. Chunks are 512 tokens with 64-token overlap to preserve "
        "paragraph boundaries across the split.\n\n"
        "[1] https://example.gov.au/standard — Australian Digital Health Agency "
        "standards document covering interoperability, security controls, audit "
        "logging, and data retention. Key findings summarised in the following "
        "paragraphs. The agency mandates TLS 1.2 minimum, FIPS-validated crypto "
        "modules, and quarterly penetration testing on all production endpoints. "
        "Access to patient data requires dual-role separation and is logged to an "
        "immutable audit trail retained for seven years. Incident response "
        "timelines: P1 < 15 min, P2 < 1 hr, P3 < 24 hr.\n\n"
    )
    filler_chars = len(filler_block)
    blocks_needed = (base_chars - current_chars) // filler_chars + 1
    filler = filler_block * blocks_needed
    messages.append({"role": "tool", "tool_name": "web_search", "content": filler})
    total = current_chars + len(filler)
    return messages, total


def call_router(model, messages, timeout=300, max_tokens=2048):
    """POST to live router forcing `model`. Returns parsed JSON or error string."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        ROUTER_URL, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            elapsed = time.time() - t0
            parsed = json.loads(raw)
            parsed["_elapsed_s"] = round(elapsed, 2)
            return parsed
    except Exception as e:
        elapsed = time.time() - t0
        return {"error": str(e), "_elapsed_s": round(elapsed, 2)}


def summarize(model, resp, size):
    if "error" in resp and "choices" not in resp:
        return {
            "size": size, "forced": model, "error": resp["error"],
            "latency_s": resp.get("_elapsed_s"),
        }
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "")
    reason = msg.get("reasoning") or msg.get("reasoning_content") or ""
    used_model = resp.get("model", "")
    finish = choice.get("finish_reason")
    usage = resp.get("usage", {})
    succeeded = bool(content and content.strip())
    escalated = used_model != model.split(":")[0]  # rough
    return {
        "size": size,
        "forced": model,
        "used_model": used_model,
        "finish": finish,
        "content_len": len(content),
        "has_reasoning_only": bool(not content.strip() and reason),
        "succeeded": succeeded,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_s": resp.get("_elapsed_s"),
        "escalated_away": escalated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=TARGETS)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    sys_prompt, real_msgs = load_real_context()
    print(f"real sys_prompt: {len(sys_prompt)} chars, {len(real_msgs)} messages")

    results = []
    for size in args.sizes:
        payload, est_chars = build_payload(sys_prompt, real_msgs, size)
        est_tokens = int(est_chars / CHARS_PER_TOKEN)
        print(f"\n=== target {size} tokens | built payload ~{est_tokens} est | "
              f"{len(payload)} messages ===")
        for model in MODELS:
            for run in range(1, args.runs + 1):
                print(f"  [{run}] {model} ...", flush=True)
                resp = call_router(model, payload)
                summary = summarize(model, resp, size)
                summary["est_tokens"] = est_tokens
                summary["run"] = run
                results.append(summary)
                print(f"      -> used={summary.get('used_model')} "
                      f"finish={summary.get('finish')} "
                      f"content_len={summary.get('content_len')} "
                      f"succeeded={summary.get('succeeded')} "
                      f"lat={summary.get('latency_s')}s "
                      f"err={summary.get('error','')[:60]}")

    out = Path(__file__).parent / "flash_ceiling_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {out}")

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'size':>8} {'model':<20} {'succ':>4} {'content_len':>11} {'lat':>6}")
    for r in results:
        print(f"{r['size']:>8} {r['forced']:<20} "
              f"{'Y' if r.get('succeeded') else 'N':>4} "
              f"{r.get('content_len',0):>11} {r.get('latency_s','-'):>6}")


if __name__ == "__main__":
    main()
