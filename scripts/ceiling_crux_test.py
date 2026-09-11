#!/usr/bin/env python3
"""
ceiling_crux_test.py
====================
Settles two questions the earlier runs left confounded.

Q1 (CAUSE): is the router's empty_content escalation caused by CONTEXT SIZE, or
    by the OUTPUT BUDGET being too small for a reasoning model's thinking trace?
    Test: force v4.1-flash at a SMALL (20K) context with max_tokens=256.
      - If it empty-streams at 20K, the failure is budget-induced, NOT a
        large-context problem, and BIGGIE_FLASH_MAX_CONTEXT_TOKENS is
        mis-justified.
      - If it serves fine at 20K with 256, the failure really is size-related.

Q2 (HEADROOM): does v4.1-flash reliably serve ~140K real prompt_tokens with a
    realistic output budget? Adaptive calibration hits the band by measuring
    actual prompt_tokens and correcting (the fixed 2.33 chars/token guess
    drifted to ~2.77 and undershot every target).

Read-only: request header only, no config or service change.
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
FLASH = "deepseek-v4.1-flash:cloud"
GLM = "glm-5.3:cloud"


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


def empty_events(since="6 min ago"):
    try:
        out = subprocess.run(
            ["sudo", "journalctl", "-u", "biggie-llm-endpoint", "--since", since,
             "--no-pager"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return -1
    return sum(1 for ln in out.splitlines() if "empty_content" in ln)


def probe(model, payload, max_tokens):
    body = {"model": model, "messages": payload, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False}
    req = urllib.request.Request(
        ROUTER_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Compression-Level": "off"})
    b = empty_events()
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            p = json.loads(resp.read().decode())
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_s": round(time.time()-t0, 1)}
    a = empty_events()
    ch = (p.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reason = msg.get("reasoning") or msg.get("reasoning_content") or ""
    u = p.get("usage") or {}
    used = p.get("model") or ""
    return {
        "used_model": used, "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"), "finish": ch.get("finish_reason"),
        "content_len": len(content), "reasoning_len": len(reason),
        "empty_content": not content.strip(),
        "escalated_away": used.split(":")[0] != model.split(":")[0],
        "router_empty_events": max(0, a-b) if a >= 0 else None,
        "latency_s": round(time.time()-t0, 1),
    }


def main():
    sys_prompt, real_msgs = load_real_context()
    seed = [{"role": "system", "content": sys_prompt}] + list(real_msgs)
    seed_chars = sum(len(m.get("content") or "") for m in seed)
    rows = []

    # ---- Q1: small context, small output budget -------------------------
    print(f"seed_chars={seed_chars:,}\n=== Q1: SMALL context (20K) x SMALL budget (256) ===")
    need = max(0, int(20_000 * 2.77) - seed_chars)
    small = list(seed) + [{"role": "user", "content": unique_filler(need)}]
    for model in (FLASH, GLM):
        for mx in (256, 8192):
            r = probe(model, small, mx)
            r.update(target_tokens=20_000, max_tokens=mx, model=model, scenario="Q1")
            rows.append(r)
            print(f"  {model:<26} max_tokens={mx:<5} used={str(r.get('used_model')):<14} "
                  f"pt={r.get('prompt_tokens') or '-':>6} fin={str(r.get('finish')):<8} "
                  f"content={r.get('content_len','-')!s:<6} reason={r.get('reasoning_len','-')!s:<6} "
                  f"esc={r.get('escalated_away')} empty_evts={r.get('router_empty_events')} "
                  f"{r.get('latency_s')}s", flush=True)

    # ---- Q2: real 140K band, adaptive sizing ----------------------------
    print("\n=== Q2: ~140K real prompt_tokens on flash, max_tokens=8192, 3 runs ===")
    cpt = 2.77
    for attempt in range(3):
        need = max(0, int(140_000 * cpt) - seed_chars)
        payload = list(seed) + [{"role": "user", "content": unique_filler(need)}]
        r = probe(FLASH, payload, 8192)
        pt = r.get("prompt_tokens") or 0
        print(f"  [calib {attempt+1}] need={need:,} chars -> pt={pt:,} "
              f"(used={r.get('used_model')})", flush=True)
        if pt:
            total_chars = seed_chars + need
            cpt = total_chars / pt
        if 135_000 <= pt <= 150_000:
            break

    for run in range(1, 4):
        r = probe(FLASH, payload, 8192)
        r.update(target_tokens=140_000, model=FLASH, scenario="Q2", run=run)
        rows.append(r)
        print(f"  [r{run}] {FLASH:<26} used={str(r.get('used_model')):<14} "
              f"pt={r.get('prompt_tokens') or '-':>7} fin={str(r.get('finish')):<8} "
              f"content={r.get('content_len','-')!s:<6} reason={r.get('reasoning_len','-')!s:<6} "
              f"esc={r.get('escalated_away')} empty_evts={r.get('router_empty_events')} "
              f"{r.get('latency_s')}s", flush=True)

    out = Path(__file__).resolve().parent / "ceiling_crux_results.json"
    out.write_text(json.dumps(rows, indent=2))
    q2 = [r for r in rows if r.get("scenario") == "Q2"]
    ok = sum(1 for r in q2 if not r.get("escalated_away") and not r.get("empty_content"))
    print(f"\nQ2 verdict: flash served itself {ok}/{len(q2)} at ~140K")
    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
