#!/usr/bin/env python3
"""
flash_ceiling_headtohead_v2.py
==============================
Honest re-run of the flash ceiling head-to-head.

Why v2 exists: the original `flash_ceiling_headtohead.py` proves nothing. The
router applies request-level compression BEFORE routing, so a 105K-token
payload was silently collapsed to ~14K tokens and the model never saw the
large context (every Aug-22 row logged `prompt_tokens: 14232` at every target
size). v2 fixes both defects:

  1. Sends `X-Compression-Level: off` so the payload reaches the model intact.
  2. Reports *measured* `usage.prompt_tokens` from the response rather than an
     assumed 4-chars/token, and sizes filler from that measured calibration.

Method: seed with real Hermes session content, pad with UNIQUE filler (so
repetition-collapse cannot shrink it), POST to the live router forcing one
model at a time, and read back which model actually served the request plus
whether it produced real content.

  - used_model == forced  + content present  -> model handled the context
  - used_model != forced                    -> router escalated away (failure)
  - content empty                           -> empty-stream (the failure mode
                                               the BIGGIE_FLASH_MAX_CONTEXT_TOKENS
                                               ceiling exists to avoid)

Read-only w.r.t. config and the router service: nothing is reconfigured, no
ceiling is moved. Uses only the documented request header.

Usage:
  python3 flash_ceiling_headtohead_v2.py --runs 2
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flash_ceiling_headtohead import load_real_context  # noqa: E402

ROUTER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# Target the band the ceiling guards.
TARGET_TOKENS = [105_000, 125_000, 145_000]

# Fallback calibration; replaced at runtime by a measured probe (see
# calibrate()) because chars-per-token for numeric-heavy filler is ~2.3, not
# the 4.0 the original harness assumed.
CHARS_PER_TOKEN_FALLBACK = 2.4

MODELS = [
    "deepseek-v4-flash:cloud",    # the model the ceiling blocks above 100K
    "deepseek-v4.1-flash:cloud",  # the current default first rung
    "glm-5.3:cloud",              # incumbent that the ceiling escalates to
]


def unique_filler(n_chars: int) -> str:
    """Filler with every line distinct, so dedup/collapse cannot shrink it."""
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


def calibrate(seed, model: str = "glm-5.3:cloud") -> float:
    """Measure chars-per-token empirically so targets land in the real band.

    Sends a known-size payload with compression off and divides sent chars by
    the model-reported prompt_tokens. Without this, sizing error compounds and
    a "145K" test can silently land at 90K.
    """
    probe_chars = 200_000
    payload = list(seed) + [{"role": "user", "content": unique_filler(probe_chars)}]
    body = {
        "model": model,
        "messages": payload,
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        ROUTER_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Compression-Level": "off"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        parsed = json.loads(resp.read().decode())
    pt = (parsed.get("usage") or {}).get("prompt_tokens") or 0
    if pt <= 0:
        return CHARS_PER_TOKEN_FALLBACK
    total_chars = sum(len(m.get("content") or "") for m in payload)
    cpt = total_chars / pt
    print(f"calibration: {total_chars:,} chars / {pt:,} prompt_tokens = {cpt:.2f} chars/token")
    return cpt


def build(seed, target_tokens: int, chars_per_token: float):
    base_chars = int(target_tokens * chars_per_token)
    seed_chars = sum(len(m.get("content") or "") for m in seed)
    need = max(0, base_chars - seed_chars)
    payload = list(seed) + [{"role": "user", "content": unique_filler(need)}]
    return payload, seed_chars, need


def probe(model: str, payload, max_tokens: int = 256, timeout: int = 900) -> dict:
    body = {
        "model": model,
        "messages": payload,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        ROUTER_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            # Critical: stop the router pre-compressing the payload away.
            "X-Compression-Level": "off",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            parsed = json.loads(resp.read().decode())
    except Exception as e:
        return {
            "forced": model,
            "error": f"{type(e).__name__}: {e}",
            "latency_s": round(time.time() - t0, 1),
            "sent_chars": len(data),
        }
    ch = (parsed.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    usage = parsed.get("usage") or {}
    used = parsed.get("model") or ""
    return {
        "forced": model,
        "used_model": used,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": ch.get("finish_reason"),
        "content_len": len(content),
        "empty_content": not content.strip(),
        "escalated_away": used.split(":")[0] != model.split(":")[0],
        "latency_s": round(time.time() - t0, 1),
        "sent_chars": len(data),
        "content_head": content[:120].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=TARGET_TOKENS)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    sys_prompt, real_msgs = load_real_context()
    seed = [{"role": "system", "content": sys_prompt}] + list(real_msgs)
    print(f"seed: {len(sys_prompt)} char system prompt, {len(real_msgs)} real messages")

    chars_per_token = calibrate(seed)

    results = []
    for target in args.sizes:
        payload, seed_chars, filler = build(seed, target, chars_per_token)
        print(
            f"\n=== target {target:,} tokens | seed {seed_chars:,} + "
            f"unique filler {filler:,} chars ==="
        )
        for model in args.models:
            for run in range(1, args.runs + 1):
                r = probe(model, payload)
                r.update(target_tokens=target, run=run, seed_chars=seed_chars)
                results.append(r)
                pt = r.get("prompt_tokens")
                print(
                    f"  [r{run}] {model:<26} used={str(r.get('used_model')):<14} "
                    f"prompt_tokens={pt if pt is not None else '-':>8} "
                    f"finish={r.get('finish')!s:<8} "
                    f"content_len={r.get('content_len','-')!s:<6} "
                    f"esc={r.get('escalated_away')} {r.get('latency_s')}s "
                    f"{r.get('error','')}"
                )

    out = Path(__file__).resolve().parent / "flash_ceiling_results_v2.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results -> {out}")

    print("\n=== VERDICT PER MODEL ===")
    for model in args.models:
        rows = [r for r in results if r["forced"] == model]
        ok = [r for r in rows if r.get("used_model") and not r.get("escalated_away")
              and not r.get("empty_content")]
        pts = [r.get("prompt_tokens") for r in rows if r.get("prompt_tokens")]
        print(
            f"  {model:<26} served_itself={len(ok)}/{len(rows)}  "
            f"prompt_tokens={min(pts) if pts else '-'}..{max(pts) if pts else '-'}"
        )


if __name__ == "__main__":
    main()
