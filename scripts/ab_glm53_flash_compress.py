#!/usr/bin/env python3
"""Compression A/B: glm-5.3-flash vs the deepseek flash incumbents.

Question: is glm-5.3-flash a better (cheaper, no-worse) alternative to
deepseek-v4.x-flash for the session-compression work?

Method mirrors the established v4.1-flash A/B (scripts/ab_direct.py) so results
are comparable to references/ollama-unit-economics.md:

  * the REAL production summariser prompt is assembled by
    agent.context_compressor.ContextCompressor._build_summary_prompt itself;
  * identical input per session across legs => differences are the MODEL;
  * NO max_tokens cap (production sends none: context_compressor.py:645);
  * direct ollama.com call, because the router overrides the client model for
    session_compression, so a not-yet-wired model cannot be reached any other way.

Adds glm-5.3-flash and measures, per leg:
  sections emitted / numbered actions / file paths / chars,
  fact-coverage  = share of source numbers reproduced,
  hallucinated   = share of summary numbers absent from source,
  + USD and credit-units cost.

Credential is read from .env at runtime and never printed.

Usage:
  python3 scripts/ab_glm53_flash_compress.py [--sids ...] [--models ...]
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
import types
import urllib.request
from pathlib import Path

sys.path.insert(0, "/home/hermes/.hermes/hermes-agent")

BASE = "https://ollama.com/v1/chat/completions"
STATE = "/home/hermes/.hermes/state.db"
ENV = Path.home() / ".hermes" / ".env"
OUT = Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data"

# models.py prices, USD per 1M tokens (input, output) + credit ratio.
PRICE = {
    "glm-5.3-flash":       (0.15, 0.50, 1.0),
    "deepseek-v4.1-flash": (0.15, 0.60, 1.0),
    "deepseek-v4-flash":   (0.22, 0.66, 1.47),
    "glm-5.3":             (1.50, 4.50, 3.0),
}

SECTIONS = ["Active Task", "Goal", "Constraints & Preferences", "Completed Actions",
            "Active State", "Blocked", "Key Decisions", "Resolved Questions",
            "Relevant Files", "Critical Context"]

DEGEN = re.compile(r"Let me analyze|Let me analyse|Here is|I'll summarize|"
                   r"The conversation is about|Let me go through")


def api_key():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("OLLAMA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OLLAMA_API_KEY not found")


def real_prompt(content, budget=10000, focus=None, has_user_turn=True):
    """Assemble the production prompt via Hermes' own builder."""
    from agent.context_compressor import ContextCompressor

    shim = types.SimpleNamespace(
        _previous_summary=None,
        tail_mode="lean",
        _summary_template_sections=ContextCompressor._summary_template_sections,
    )
    return ContextCompressor._build_summary_prompt(
        shim, content, budget, focus, "", has_user_turn)


def load_session(sid):
    con = sqlite3.connect(f"file:{STATE}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT role, content FROM messages WHERE session_id=? AND active=1 "
        "AND content IS NOT NULL ORDER BY timestamp ASC, id ASC", (sid,)).fetchall()
    con.close()
    return [{"role": r, "content": c} for r, c in rows if c]


def truncate_turns(msgs, target_chars):
    """Greedy tail-walk: keep the most recent turns that fit."""
    out, used = [], 0
    for m in reversed(msgs):
        c = len(m["content"]) + len(m["role"]) + 4
        if used + c > target_chars and out:
            break
        out.append(m)
        used += c
        if used >= target_chars:
            break
    return list(reversed(out or msgs[-1:]))


def render_turns(msgs):
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in msgs)


def call(model, prompt, key, max_tokens=0, timeout=2400):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "stream": False}
    if max_tokens:                      # 0 => omit, matching production
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        ch = (out.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        u = out.get("usage") or {}
        return {
            "ok": True,
            "content": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning") or msg.get("reasoning_content") or "").strip(),
            "served_by": out.get("model"),
            "finish": ch.get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "latency": round(time.time() - t0, 1),
        }
    except Exception as e:
        txt = ""
        rd = getattr(e, "read", None)
        if callable(rd):
            try:
                txt = rd().decode()[:400]
            except Exception:
                pass
        return {"ok": False, "error": f"{type(e).__name__}: {e} {txt}",
                "latency": round(time.time() - t0, 1)}


NUM = re.compile(r"\b\d[\d,]*\.?\d*\b")


def nums(text):
    out = set()
    for m in NUM.findall(text or ""):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def score(summary, source):
    src, smy = nums(source), nums(summary)
    # ignore 1-2 digit integers: too common to be evidence of coverage
    src_facts = {n for n in src if n >= 100}
    smy_facts = {n for n in smy if n >= 100}
    return {
        "sections_present": sum(1 for s in SECTIONS if f"## {s}" in summary),
        "sections_total": len(SECTIONS),
        "chars": len(summary),
        "file_paths": len(set(re.findall(r"(?:/[\w.\-]+){3,}", summary))),
        "hex_ids": len(set(re.findall(r"\b[0-9a-f]{8,40}\b", summary))),
        "numbered_actions": len(re.findall(r"^\d+\.\s", summary, re.M)),
        "coverage_pct": round(100 * len(src_facts & smy_facts) / len(src_facts), 1) if src_facts else None,
        "hallucinated_pct": round(100 * len(smy_facts - src_facts) / len(smy_facts), 1) if smy_facts else None,
        "source_facts": len(src_facts),
        "prose_degen": bool(DEGEN.search(summary)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", nargs="+", default=[
        "20260911_104856_871268",   # same session as the prior v4.1-flash A/B
        "20260911_065143_8e021a05",
        "20260711_220431_67d5337f",
    ])
    ap.add_argument("--models", nargs="+", default=[
        "glm-5.3-flash", "deepseek-v4.1-flash", "deepseek-v4-flash"])
    ap.add_argument("--target_chars", type=int, default=500000)
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--tag", default="_glm53flash")
    args = ap.parse_args()

    key = api_key()
    results = []
    for sid in args.sids:
        msgs = load_session(sid)
        if not msgs:
            print(f"\n########## {sid}: NO MESSAGES — skipped ##########", flush=True)
            continue
        turns = truncate_turns(msgs, args.target_chars)
        rendered = render_turns(turns)
        prompt = real_prompt(rendered, budget=args.max_tokens or 10000)
        nchars = sum(len(m["content"]) for m in turns)
        print(f"\n########## {sid}: {len(turns)} turns, {nchars:,} chars "
              f"-> prompt {len(prompt):,} chars ##########", flush=True)
        rec = {"sid": sid, "turns": len(turns), "chars": nchars,
               "prompt_chars": len(prompt), "legs": {}}
        for model in args.models:
            print(f"  -> {model} ...", flush=True)
            res = call(model, prompt, key, args.max_tokens)
            pi, po, ratio = PRICE.get(model, (0.15, 0.60, 1.0))
            pt = res.get("prompt_tokens") or 0
            ct = res.get("completion_tokens") or 0
            res["cost_units"] = round(pt * ratio, 0)
            res["usd"] = round(pt / 1e6 * pi + ct / 1e6 * po, 5)
            body = res.get("content") or res.get("reasoning") or ""
            res["score"] = score(body, rendered)
            rec["legs"][model] = res
            s = res["score"]
            print(f"     fin={res.get('finish')} served={res.get('served_by')} "
                  f"pt={pt} ct={ct} units≈{res['cost_units']:.0f} ${res['usd']} "
                  f"secs={s['sections_present']}/{s['sections_total']} "
                  f"acts={s['numbered_actions']} paths={s['file_paths']} "
                  f"cov={s['coverage_pct']}% halluc={s['hallucinated_pct']}% "
                  f"degen={s['prose_degen']} {res.get('latency')}s", flush=True)
            if not res.get("ok"):
                print(f"     ERR: {res.get('error')}", flush=True)
        results.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ab_compress{args.tag}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {path}")

    print("\n===== SUMMARY (mean over sessions) =====")
    for model in args.models:
        legs = [r["legs"][model] for r in results
                if model in r["legs"] and r["legs"][model].get("ok")]
        if not legs:
            print(f"  {model:24s} no successful legs")
            continue
        def mean(k, sub=None):
            v = [l["score"][k] if sub is None else l[sub] for l in legs]
            v = [x for x in v if x is not None]
            return round(sum(v) / len(v), 2) if v else None
        print(f"  {model:24s} cov={mean('coverage_pct')}% halluc={mean('hallucinated_pct')}% "
              f"secs={mean('sections_present')} acts={mean('numbered_actions')} "
              f"paths={mean('file_paths')} chars={mean('chars')} "
              f"usd=${mean(None,'usd')} units={mean(None,'cost_units')} "
              f"lat={mean(None,'latency')}s n={len(legs)}")


if __name__ == "__main__":
    main()
