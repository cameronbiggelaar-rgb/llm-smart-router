#!/usr/bin/env python3
"""
ceiling_band_test.py
====================
Decisive test of the BIGGIE_FLASH_MAX_CONTEXT_TOKENS=100000 ceiling.

Sizes straddle the ceiling (95K below / 110K, 130K above) and max_tokens is
raised to 8192 so a reasoning model's thinking trace cannot consume the whole
budget and masquerade as an empty stream (the flaw that made the earlier
max_tokens=256/4096 runs unreadable).

For each forced model we record whether the router SERVED that model
(used_model == forced, real content) or escalated away. Escalation away from a
flash model at a size ABOVE the ceiling, while it serves itself BELOW, is the
evidence that the ceiling is justified.

Read-only: request header only, no config or service changes.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flash_ceiling_headtohead import load_real_context  # noqa: E402

ROUTER_URL = "http://127.0.0.1:8080/v1/chat/completions"
SIZES = [95_000, 110_000, 130_000]
MODELS = ["deepseek-v4.1-flash:cloud", "glm-5.3:cloud"]
MAX_TOKENS = 8192
RUNS = 2


def unique_filler(n_chars: int) -> str:
    parts, total, i = [], 0, 0
    while total < n_chars:
        line = (
            f"[{i}] doc-{i:07d} :: record {i*7919 % 999983} :: "
            f"hash {i*2654435761 % 4294967291} :: value {i*104729 % 1000003}\n"
        )
        parts.append(line)
        total += len(line)
        i += 1
    return "".join(parts)


def empty_events(since: str = "5 min ago") -> int:
    try:
        out = subprocess.run(
            ["sudo", "journalctl", "-u", "biggie-llm-endpoint", "--since", since,
             "--no-pager"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return -1
    return sum(1 for ln in out.splitlines() if "empty_content" in ln)


def probe(model, payload):
    body = {"model": model, "messages": payload, "max_tokens": MAX_TOKENS,
            "temperature": 0.0, "stream": False}
    req = urllib.request.Request(
        ROUTER_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Compression-Level": "off"},
    )
    b = empty_events()
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            p = json.loads(resp.read().decode())
    except Exception as e:
        return {"forced": model, "error": f"{type(e).__name__}: {e}",
                "latency_s": round(time.time() - t0, 1)}
    a = empty_events()
    ch = (p.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reason = msg.get("reasoning") or msg.get("reasoning_content") or ""
    u = p.get("usage") or {}
    used = p.get("model") or ""
    return {
        "forced": model, "used_model": used,
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "finish": ch.get("finish_reason"),
        "content_len": len(content), "reasoning_len": len(reason),
        "empty_content": not content.strip(),
        "escalated_away": used.split(":")[0] != model.split(":")[0],
        "router_empty_events": max(0, a - b) if a >= 0 else None,
        "latency_s": round(time.time() - t0, 1),
    }


def main():
    sys_prompt, real_msgs = load_real_context()
    seed = [{"role": "system", "content": sys_prompt}] + list(real_msgs)
    seed_chars = sum(len(m.get("content") or "") for m in seed)
    # Calibrated from the earlier run: 2.33 chars/token.
    cpt = 2.33
    print(f"seed_chars={seed_chars:,} cpt={cpt} max_tokens={MAX_TOKENS} runs={RUNS}\n")

    rows = []
    for target in SIZES:
        need = max(0, int(target * cpt) - seed_chars)
        payload = list(seed) + [{"role": "user", "content": unique_filler(need)}]
        print(f"=== target {target:,} (need {need:,} filler chars) ===")
        for model in MODELS:
            for run in range(1, RUNS + 1):
                r = probe(model, payload)
                r.update(target_tokens=target, run=run)
                rows.append(r)
                print(f"  [r{run}] {model:<26} used={str(r.get('used_model')):<14} "
                      f"pt={r.get('prompt_tokens') or '-':>7} "
                      f"fin={str(r.get('finish')):<8} content={r.get('content_len','-')!s:<6} "
                      f"reason={r.get('reasoning_len','-')!s:<6} "
                      f"esc={r.get('escalated_away')} "
                      f"empty_evts={r.get('router_empty_events')} "
                      f"{r.get('latency_s')}s {r.get('error','')}", flush=True)

    out = Path(__file__).resolve().parent / "ceiling_band_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows -> {out}")

    print("\n=== VERDICT ===")
    for model in MODELS:
        for target in SIZES:
            rs = [r for r in rows if r["forced"] == model and r.get("target_tokens") == target]
            served = sum(1 for r in rs
                         if not r.get("escalated_away") and not r.get("empty_content"))
            print(f"  {model:<26} {target:>7,}: served_itself={served}/{len(rs)}")


if __name__ == "__main__":
    main()
