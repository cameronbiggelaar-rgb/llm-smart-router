#!/usr/bin/env python3
"""
ceiling_final_test.py
=====================
Completes the evidence for both context ceilings in the router.

TEST 1 - the BIGGIE_FLASH_MAX_CONTEXT_TOKENS=100000 ceiling
  The code comment says flash is "prone to the empty-stream problem on very
  large contexts". That is a claim about CONTEXT SIZE. The 2x2 below separates
  context size from output budget, which earlier runs confounded:

      context  \  max_tokens      256        8192
      small    (20K)            ?  (run)    know: ok
      large    (140K)           ?  (run)    know: ok (3/3, pt=141,617)

  If large/256 fails while small/8192 succeeds, the failure tracks the OUTPUT
  BUDGET, not context size - which means the size-based ceiling is
  mis-justified.

TEST 2 - the MODEL_CONTEXT_CEILING glm-5.3 = 150000 ceiling
  Code comment claims glm-5.3 "degenerates into repeated-token garbage" above
  150K. Test at ~160K and score the output for repetition/degeneracy, so the
  claim is checked rather than assumed.

Read-only: request header only. No config, service, or ceiling changes.
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flash_ceiling_headtohead import load_real_context  # noqa: E402

ROUTER_URL = "http://127.0.0.1:8080/v1/chat/completions"
FLASH = "deepseek-v4.1-flash:cloud"
GLM53 = "glm-5.3:cloud"


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


def empty_events(since="8 min ago"):
    try:
        out = subprocess.run(
            ["sudo", "journalctl", "-u", "biggie-llm-endpoint", "--since", since,
             "--no-pager"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return -1
    return sum(1 for ln in out.splitlines() if "empty_content" in ln)


def degeneracy_score(text: str) -> dict:
    """Detect the 'repeated-token garbage' failure mode."""
    if not text.strip():
        return {"words": 0, "unique_ratio": 0.0, "max_line_repeat": 0,
                "max_ngram_repeat": 0, "degenerate": True}
    words = text.split()
    # longest run of identical words
    run = best = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    line_counts = Counter(lines)
    max_line_repeat = max(line_counts.values()) if line_counts else 0
    # repeated 5-grams
    grams = Counter(tuple(words[i:i+5]) for i in range(max(0, len(words)-4)))
    max_ngram_repeat = max(grams.values()) if grams else 0
    return {
        "words": len(words),
        "unique_ratio": round(len(set(words)) / len(words), 3),
        "max_word_run": best,
        "max_line_repeat": max_line_repeat,
        "max_ngram_repeat": max_ngram_repeat,
        "degenerate": best > 5 or max_ngram_repeat > 3,
    }


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
        "content_head": content[:150].replace("\n", " "),
        "degeneracy": degeneracy_score(content),
    }


def build(seed, seed_chars, target_tokens, cpt):
    need = max(0, int(target_tokens * cpt) - seed_chars)
    return list(seed) + [{"role": "user", "content": unique_filler(need)}]


def main():
    sys_prompt, real_msgs = load_real_context()
    seed = [{"role": "system", "content": sys_prompt}] + list(real_msgs)
    seed_chars = sum(len(m.get("content") or "") for m in seed)
    rows = []

    # ---------- TEST 1: the 2x2 ----------
    print(f"seed_chars={seed_chars:,}\n=== TEST 1: budget x context (forced v4.1-flash) ===")
    for label, target, cpt in (("small 20K", 20_000, 2.77), ("large 140K", 140_000, 2.77)):
        payload = build(seed, seed_chars, target, cpt)
        for mx in (256, 8192):
            for run in (1, 2):
                r = probe(FLASH, payload, mx)
                r.update(test="T1", context_label=label, target_tokens=target,
                         max_tokens=mx, forced=FLASH, run=run)
                rows.append(r)
                print(f"  {label:<11} max_tokens={mx:<5} [r{run}] "
                      f"used={str(r.get('used_model')):<14} pt={r.get('prompt_tokens') or '-':>7} "
                      f"fin={str(r.get('finish')):<8} content={r.get('content_len','-')!s:<6} "
                      f"reason={r.get('reasoning_len','-')!s:<7} "
                      f"esc={r.get('escalated_away')} empty_evts={r.get('router_empty_events')} "
                      f"{r.get('latency_s')}s", flush=True)

    # ---------- TEST 2: glm-5.3 above its 150K ceiling ----------
    print("\n=== TEST 2: glm-5.3 at ~160K (above its 150K MODEL_CONTEXT_CEILING) ===")
    cpt = 2.77
    for attempt in range(3):
        payload = build(seed, seed_chars, 160_000, cpt)
        r = probe(GLM53, payload, 8192)
        pt = r.get("prompt_tokens") or 0
        print(f"  [calib {attempt+1}] pt={pt:,} used={r.get('used_model')}", flush=True)
        if pt:
            cpt = (seed_chars + max(0, int(160_000 * cpt) - seed_chars)) / pt
        if pt >= 155_000:
            break

    for run in range(1, 4):
        r = probe(GLM53, payload, 8192)
        r.update(test="T2", context_label="160K glm-5.3", target_tokens=160_000,
                 max_tokens=8192, forced=GLM53, run=run)
        rows.append(r)
        d = r.get("degeneracy") or {}
        print(f"  [r{run}] used={str(r.get('used_model')):<14} pt={r.get('prompt_tokens') or '-':>7} "
              f"fin={str(r.get('finish')):<8} content={r.get('content_len','-')!s:<7} "
              f"words={d.get('words','-')} uniq={d.get('unique_ratio','-')} "
              f"wordrun={d.get('max_word_run','-')} ngram={d.get('max_ngram_repeat','-')} "
              f"DEGENERATE={d.get('degenerate')} {r.get('latency_s')}s", flush=True)

    out = Path(__file__).resolve().parent / "ceiling_final_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows -> {out}")

    print("\n=== TEST 1 VERDICT (forced v4.1-flash) ===")
    for label in ("small 20K", "large 140K"):
        for mx in (256, 8192):
            rs = [r for r in rows if r.get("test") == "T1"
                  and r.get("context_label") == label and r.get("max_tokens") == mx]
            ok = sum(1 for r in rs if not r.get("escalated_away") and not r.get("empty_content"))
            pts = [r.get("prompt_tokens") for r in rs if r.get("prompt_tokens")]
            print(f"  {label:<11} max_tokens={mx:<5}: served_itself={ok}/{len(rs)} "
                  f"prompt_tokens={min(pts) if pts else '-'}")

    print("\n=== TEST 2 VERDICT (glm-5.3 > 150K ceiling) ===")
    rs = [r for r in rows if r.get("test") == "T2"]
    deg = sum(1 for r in rs if (r.get("degeneracy") or {}).get("degenerate"))
    ok = sum(1 for r in rs if not r.get("empty_content"))
    pts = [r.get("prompt_tokens") for r in rs if r.get("prompt_tokens")]
    print(f"  runs={len(rs)} prompt_tokens={min(pts) if pts else '-'}..{max(pts) if pts else '-'} "
          f"produced_content={ok}/{len(rs)} degenerate={deg}/{len(rs)}")


if __name__ == "__main__":
    main()
