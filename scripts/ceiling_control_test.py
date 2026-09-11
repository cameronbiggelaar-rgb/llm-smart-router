#!/usr/bin/env python3
"""
ceiling_control_test.py
=======================
Removes the two confounds in the v2 ceiling run:

  CONFOUND 1 (token budget): v2 sent max_tokens=256. glm-5.3's reasoning trace
    consumed the whole budget and returned finish=length with 0 content — so
    "0 content" could not be attributed to the empty-stream bug. Fix: send a
    realistic max_tokens (4096), matching what a real compaction asks for.

  CONFOUND 2 (no control): v2 only tested 105-145K. If flash empty-streams on
    SMALL contexts too, the failure is not a large-context ceiling problem at
    all and the BIGGIE_FLASH_MAX_CONTEXT_TOKENS rationale is unproven. Fix:
    include small controls (20K, 60K) below the ceiling.

Outcome per leg is read from the ROUTER's own logs (empty_content) plus the
returned model, so we distinguish:
  - flash served itself with real content   -> ceiling NOT needed at that size
  - flash empty-streamed -> escalated       -> ceiling justified at that size

Read-only: no config or service changes. Only the documented request header.
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
SIZES = [20_000, 60_000, 105_000, 125_000, 145_000]
MODELS = ["deepseek-v4-flash:cloud", "deepseek-v4.1-flash:cloud"]
MAX_TOKENS = 4096


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


def journal_failures(since: str = "3 min ago") -> int:
    """Count router-logged empty_content failures in the window."""
    try:
        out = subprocess.run(
            ["sudo", "journalctl", "-u", "biggie-llm-endpoint", "--since", since,
             "--no-pager"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return -1
    return sum(1 for ln in out.splitlines() if "empty_content" in ln)


def probe(model: str, payload, cpt: float) -> dict:
    body = {
        "model": model,
        "messages": payload,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        ROUTER_URL, data=data,
        headers={"Content-Type": "application/json", "X-Compression-Level": "off"},
    )
    before = journal_failures()
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            parsed = json.loads(resp.read().decode())
    except Exception as e:
        return {"forced": model, "error": f"{type(e).__name__}: {e}",
                "latency_s": round(time.time() - t0, 1)}
    after = journal_failures()
    ch = (parsed.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reason = msg.get("reasoning") or msg.get("reasoning_content") or ""
    usage = parsed.get("usage") or {}
    used = parsed.get("model") or ""
    return {
        "forced": model,
        "used_model": used,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": ch.get("finish_reason"),
        "content_len": len(content),
        "reasoning_len": len(reason),
        "empty_content": not content.strip(),
        "escalated_away": used.split(":")[0] != model.split(":")[0],
        "router_empty_content_events": max(0, after - before) if after >= 0 else None,
        "latency_s": round(time.time() - t0, 1),
        "content_head": content[:100].replace("\n", " "),
    }


def main():
    cpt = float(sys.argv[1]) if len(sys.argv) > 1 else 2.33
    sys_prompt, real_msgs = load_real_context()
    seed = [{"role": "system", "content": sys_prompt}] + list(real_msgs)
    seed_chars = sum(len(m.get("content") or "") for m in seed)
    print(f"seed_chars={seed_chars:,}  chars/token={cpt}  max_tokens={MAX_TOKENS}\n")

    rows = []
    for target in SIZES:
        need = max(0, int(target * cpt) - seed_chars)
        payload = list(seed) + [{"role": "user", "content": unique_filler(need)}]
        print(f"=== target {target:,} ===")
        for model in MODELS:
            r = probe(model, payload, cpt)
            r["target_tokens"] = target
            rows.append(r)
            print(
                f"  {model:<26} used={str(r.get('used_model')):<14} "
                f"pt={r.get('prompt_tokens') or '-':>7} "
                f"finish={str(r.get('finish')):<8} "
                f"content={r.get('content_len','-')!s:<6} "
                f"reason={r.get('reasoning_len','-')!s:<6} "
                f"esc={r.get('escalated_away')} "
                f"router_empty_evts={r.get('router_empty_content_events')} "
                f"{r.get('latency_s')}s {r.get('error','')}"
            )

    out = Path(__file__).resolve().parent / "ceiling_control_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows -> {out}")

    print("\n=== VERDICT: did the forced flash model serve itself? ===")
    for model in MODELS:
        for target in SIZES:
            rs = [r for r in rows if r["forced"] == model and r["target_tokens"] == target]
            served = sum(1 for r in rs if not r.get("escalated_away") and not r.get("empty_content"))
            print(f"  {model:<26} {target:>7,} : served_itself={served}/{len(rs)}")


if __name__ == "__main__":
    main()
