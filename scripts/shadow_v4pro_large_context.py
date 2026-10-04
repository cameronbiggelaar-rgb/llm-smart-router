#!/usr/bin/env python3
"""
shadow_v4pro_large_context.py  (v2 — corrected)
===============================================
Three-way forced-model head-to-head: does deepseek-v4-pro serve >=150K
contexts, or does it empty-stream like the models the glm rungs exist to catch?

WHY v2 EXISTS — v1's results were INVALID (three harness defects):
  1. max_tokens=256. These models emit *reasoning* tokens before content, so a
     256-token budget was spent entirely on reasoning and `content` came back
     empty with finish_reason=length. That would have been misread as "the
     model empty-streamed". v2 uses a real budget AND inspects `reasoning`
     separately, so "reasoning-only" is never scored as an empty stream.
  2. Seed too large. The real seed alone was ~200K tokens, so the 150K target
     overshot and got zero filler. v2 caps the seed so filler actually fills.
  3. No cap detection. A transient account-wide 429 (Ollama monthly credit cap)
     turned every call into a 503 and would have been scored as a model
     failure. v2 preflights and aborts loudly instead of scoring 503s.

Method (kept from the proven v2 ceiling harness):
  * X-Compression-Level: off  — stop the router pre-collapsing the payload
  * measured chars-per-token calibration (NOT the 4.0 assumption)
  * UNIQUE filler so repetition-collapse cannot shrink the payload

Read-only w.r.t. config and the live router; nothing reconfigured.

Usage:
  python3 shadow_v4pro_large_context.py --sizes 150000 200000 250000 --runs 2
"""
import argparse
import json
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

ROUTER_URL = "http://127.0.0.1:8080/v1/chat/completions"
STATE_DB = Path.home() / ".hermes" / "state.db"

SEED_SID = "20260911_065143_8e021a05"
SEED_CHAR_CAP = 40_000          # keep the seed small so filler does the sizing

MODELS = [
    "deepseek-v4.1-flash:cloud",   # incumbent workhorse ($0.15 in)
    "glm-5.2:cloud",               # current large-context insurance ($1.40 in)
    "deepseek-v4-pro",             # proposed replacement ($0.66 in)
]

CHARS_PER_TOKEN_FALLBACK = 3.2
# Must match what production actually gives a session_compression call. The
# endpoint injects 16,384 for compression; a smaller budget is spent entirely on
# the model's *reasoning* tokens, so content never starts and every call scores
# as "reasoning only" -- a harness artifact, not a model failure.
MAX_TOKENS = 16384
CAL_TIMEOUT = 900
PROBE_TIMEOUT = 900

# A bare data dump with NO instruction makes a reasoning model burn its whole
# output budget deciding what was asked (v4-pro did this twice: ~33K reasoning
# tokens, finish=length, zero content). That is a property of the no-task
# prompt, not of the model as a summariser -- production always sends a real
# task. So the run scores the model on the task it is actually hired for.
INSTRUCTION = (
    "Summarise the conversation above into a structured checkpoint summary. "
    "Preserve: key facts, decisions made, file paths, command outputs, and any "
    "unresolved questions. Be concise and factual."
)


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


def load_real_context(sid: str = SEED_SID):
    """Real system prompt + real messages, trimmed to SEED_CHAR_CAP chars.

    Trimmed from the OLDEST end so the most recent (most representative) turns
    survive. Without the cap the seed alone exceeds the smallest target and the
    test silently stops measuring the band it claims to measure.
    """
    conn = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    sp = conn.execute("SELECT system_prompt FROM sessions WHERE id=?", (sid,)).fetchone()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? AND active=1 "
        "ORDER BY timestamp ASC, id ASC",
        (sid,),
    ).fetchall()
    conn.close()
    sys_prompt = (sp[0] if sp and sp[0] else "") or "You are a helpful assistant."

    kept, used = [], len(sys_prompt)
    for role, content in reversed(rows):
        c = content or ""
        if not c:
            continue
        if used + len(c) > SEED_CHAR_CAP and kept:
            break
        kept.append({"role": role, "content": c})
        used += len(c)
    return sys_prompt, list(reversed(kept))


def post(model, payload, max_tokens=MAX_TOKENS, timeout=PROBE_TIMEOUT) -> dict:
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
        headers={"Content-Type": "application/json", "X-Compression-Level": "off"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            parsed = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:300]
        except Exception:
            pass
        return {
            "http_status": e.code,
            "error": f"HTTP {e.code}",
            "error_detail": detail,
            "latency_s": round(time.time() - t0, 1),
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "latency_s": round(time.time() - t0, 1),
        }

    ch = (parsed.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    usage = parsed.get("usage") or {}
    used = parsed.get("model") or ""
    return {
        "used_model": used,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": ch.get("finish_reason"),
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        # A reasoning-only reply is NOT an empty stream: the model did work and
        # the budget simply ran out before content started. Scoring it as empty
        # is the exact mistake v1 made.
        "served_content": bool(content.strip()),
        "reasoning_only": bool(not content.strip() and reasoning.strip()),
        "empty_stream": bool(not content.strip() and not reasoning.strip()),
        "escalated_away": bool(used) and used.split(":")[0] != model.split(":")[0],
        "latency_s": round(time.time() - t0, 1),
        "content_head": content[:120].replace("\n", " "),
    }


def check_holds(models) -> bool:
    """Abort if any forced model is currently circuit-open or rate-held.

    Run 1 was invalidated exactly this way: v4-pro's breaker had opened on the
    earlier credit-cap 429s, so all six v4-pro probes returned an instant
    breaker-503 that scored as a model failure. The preflight call only
    exercises MODELS[0], so it sails straight past a hold on another model.
    """
    db = Path(__file__).resolve().parent.parent / "data" / "router_state.db"
    if not db.exists():
        print(f"hold-check: {db} not found — skipping")
        return True
    now = time.time()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = dict(
            (m, (rl, co))
            for m, rl, co in conn.execute(
                "SELECT model, COALESCE(rate_limited_until,0), "
                "COALESCE(circuit_open_until,0) FROM model_state"
            )
        )
    except sqlite3.Error as e:
        print(f"hold-check: query failed ({e}) — skipping")
        return True
    finally:
        conn.close()

    blocked = []
    for m in models:
        rl, co = rows.get(m, (0, 0))
        if rl > now or co > now:
            left = max(rl, co) - now
            blocked.append(f"{m} ({'rate-held' if rl > now else 'circuit-open'} "
                           f"{left:.0f}s left)")
    if blocked:
        print("HOLD-CHECK FAILED — these forced models are not callable right now:")
        for b in blocked:
            print(f"  - {b}")
        print("Aborting: a breaker-503 would be scored as a model failure "
              "(this invalidated run 1).")
        return False
    print(f"hold-check ok ({len(models)} models callable)")
    return True


def preflight() -> bool:
    """One tiny call. Distinguishes 'account cap / service down' from 'model
    failed the big context', so a 503 is never scored as a model result."""
    r = post(MODELS[0], [{"role": "user", "content": "reply with the single word: ok"}],
             max_tokens=64, timeout=120)
    if r.get("error"):
        print(f"PREFLIGHT FAILED: {r.get('error')} {r.get('error_detail','')}")
        print("Aborting: account/service-level failure would be misread as a model result.")
        return False
    print(f"preflight ok (used={r.get('used_model')} "
          f"content_len={r.get('content_len')} {r.get('latency_s')}s)")
    return True


def calibrate(seed, model=MODELS[0]) -> float:
    probe_chars = 150_000
    payload = list(seed) + [{"role": "user", "content": unique_filler(probe_chars)}]
    r = post(model, payload, max_tokens=16)
    pt = r.get("prompt_tokens") or 0
    if pt <= 0:
        print(f"calibration: no usage ({r.get('error','')}) — using fallback")
        return CHARS_PER_TOKEN_FALLBACK
    total_chars = sum(len(m.get("content") or "") for m in payload)
    cpt = total_chars / pt
    print(f"calibration: {total_chars:,} chars / {pt:,} prompt_tokens = {cpt:.2f} chars/token")
    return cpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[150_000, 200_000, 250_000])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--no-instruction", action="store_true",
                    help="reproduce the bare-dump run (no task) for comparison")
    ap.add_argument("--out", default="shadow_v4pro_results.json",
                    help="results filename (relative to this script's dir)")
    args = ap.parse_args()

    if not check_holds(args.models):
        raise SystemExit(3)
    if not preflight():
        raise SystemExit(2)

    sys_prompt, real_msgs = load_real_context()
    seed = [{"role": "system", "content": sys_prompt}] + list(real_msgs)
    seed_chars = sum(len(m["content"]) for m in seed)
    print(f"seed: sid={SEED_SID} {len(real_msgs)} real messages, {seed_chars:,} chars")

    cpt = calibrate(seed)

    results = []
    for target in args.sizes:
        base_chars = int(target * cpt)
        need = max(0, base_chars - seed_chars)
        payload = list(seed) + [{"role": "user", "content": unique_filler(need)}]
        if not args.no_instruction:
            # Real task last, so the model is scored on its actual job.
            payload.append({"role": "user", "content": INSTRUCTION})
        sent_chars = sum(len(m["content"]) for m in payload)
        print(f"\n=== target {target:,} tok | seed {seed_chars:,} + filler {need:,} "
              f"= {sent_chars:,} chars (~{int(sent_chars/cpt):,} tok) ===")
        for model in args.models:
            for run in range(1, args.runs + 1):
                r = post(model, payload)
                r.update(forced=model, target_tokens=target, run=run,
                         est_tokens=int(sent_chars / cpt))
                results.append(r)
                print(f"  [r{run}] {model:<26} used={str(r.get('used_model')):<20} "
                      f"pt={str(r.get('prompt_tokens')):>8} "
                      f"fin={str(r.get('finish')):<8} "
                      f"content={str(r.get('content_len')):<6} "
                      f"reason={str(r.get('reasoning_len')):<6} "
                      f"empty={r.get('empty_stream')} esc={r.get('escalated_away')} "
                      f"{r.get('latency_s')}s {r.get('error','')}")

    out = Path(__file__).resolve().parent / args.out
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results -> {out}")

    print("\n=== VERDICT PER FORCED MODEL ===")
    for model in args.models:
        mine = [r for r in results if r.get("forced") == model]
        served = [r for r in mine if r.get("served_content")]
        reason_only = [r for r in mine if r.get("reasoning_only")]
        empty = [r for r in mine if r.get("empty_stream")]
        errs = [r for r in mine if r.get("error")]
        esc = [r for r in mine if r.get("escalated_away")]
        print(f"  {model:<26} served={len(served)}/{len(mine)} "
              f"reasoning_only={len(reason_only)} empty={len(empty)} "
              f"escalated={len(esc)} errors={len(errs)}")
        for r in errs:
            print(f"      err @{r.get('target_tokens')}: {r.get('error')} "
                  f"{r.get('error_detail','')[:120]}")


if __name__ == "__main__":
    main()
