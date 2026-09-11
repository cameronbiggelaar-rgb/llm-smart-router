#!/usr/bin/env python3
"""Direct-provider compression A/B using the REAL Hermes summariser prompt.

Forces each model at ollama.com, bypassing the router's compression ladder
(which always overrides the client's model choice for session_compression).
Same input for every leg => differences reflect the MODEL, not the payload.

The prompt is assembled by agent.context_compressor.ContextCompressor
._build_summary_prompt itself, so the A/B sees exactly what production sends.

Credential is read from .env at runtime and never printed.
"""
import argparse, json, os, re, sqlite3, sys, time, urllib.request, types

sys.path.insert(0, "/home/hermes/.hermes/hermes-agent")

BASE = "https://ollama.com/v1/chat/completions"
STATE = "/home/hermes/.hermes/state.db"
ENV = "/home/hermes/.hermes/.env"
OUT = "/home/hermes/.hermes/skills/llm-smart-router/data/ab_direct"

RATIO = {
    "deepseek-v4.1-flash": 0.68,
    "deepseek-v4-flash": 1.0,
    "glm-5.3": 1.5,
    "glm-5.2": 1.5,
}


def api_key():
    for line in open(ENV):
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
        "SELECT role, content FROM messages WHERE session_id=? AND active=1 AND content IS NOT NULL "
        "ORDER BY timestamp ASC, id ASC", (sid,)).fetchall()
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


def call(model, prompt, key, max_tokens, timeout=1800):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False,
    }
    # Production sends NO max_tokens cap on the summary call (see
    # context_compressor.py:645). 0 => omit the field entirely.
    if max_tokens:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        ch = (out.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        u = out.get("usage") or {}
        return {
            "ok": bool(content or reasoning),
            "content": content,
            "reasoning": reasoning,
            "served_by": out.get("model"),
            "finish": ch.get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "latency": round(time.time() - t0, 1),
        }
    except Exception as e:
        body_txt = ""
        rd = getattr(e, "read", None)
        if callable(rd):
            try:
                body_txt = rd().decode()[:400]
            except Exception:
                pass
        return {"ok": False, "error": f"{type(e).__name__}: {e} {body_txt}",
                "latency": round(time.time() - t0, 1)}


SECTIONS = ["Active Task", "Goal", "Constraints & Preferences", "Completed Actions",
            "Active State", "Blocked", "Key Decisions", "Resolved Questions",
            "Relevant Files", "Critical Context"]


def score(text):
    """Structural fidelity: does it emit the required sections as real headings,
    and does it carry concrete artefacts rather than prose?"""
    secs = sum(1 for s in SECTIONS if f"## {s}" in text)
    return {
        "sections_present": secs,
        "sections_total": len(SECTIONS),
        "chars": len(text),
        "file_paths": len(set(re.findall(r"(?:/[\w.\-]+){3,}", text))),
        "numbers": len(set(re.findall(r"\b\d{3,}\b", text))),
        "hex_ids": len(set(re.findall(r"\b[0-9a-f]{8,40}\b", text))),
        "numbered_actions": len(re.findall(r"^\d+\.\s", text, re.M)),
        "prose_dump": bool(re.search(r"Let me analyze|Let me analyse|Here is|I'll summarize|The conversation is about", text)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", nargs="+", default=["20260911_104856_871268"])
    ap.add_argument("--models", nargs="+",
                    default=["deepseek-v4.1-flash", "deepseek-v4-flash", "glm-5.3"])
    ap.add_argument("--target_chars", type=int, default=310000)
    ap.add_argument("--max_tokens", type=int, default=10000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    key = api_key()
    results = []
    for sid in args.sids:
        msgs = load_session(sid)
        turns = truncate_turns(msgs, args.target_chars)
        rendered = render_turns(turns)
        prompt = real_prompt(rendered, budget=args.max_tokens)
        nchars = sum(len(m["content"]) for m in turns)
        print(f"\n########## {sid}: {len(turns)} turns, {nchars:,} chars, prompt {len(prompt):,} chars ##########",
              flush=True)
        rec = {"sid": sid, "turns": len(turns), "chars": nchars, "prompt_chars": len(prompt), "legs": {}}
        for model in args.models:
            print(f"  -> {model} ...", flush=True)
            res = call(model, prompt, key, args.max_tokens)
            pt = res.get("prompt_tokens") or 0
            res["cost_units"] = round(pt * RATIO.get(model.split(":")[0], 1.0), 0)
            body = res.get("content") or res.get("reasoning") or ""
            res["score"] = score(body)
            rec["legs"][model] = res
            s = res["score"]
            print(f"     fin={res.get('finish')} served={res.get('served_by')} pt={res.get('prompt_tokens')} "
                  f"ct={res.get('completion_tokens')} units≈{res['cost_units']:.0f} "
                  f"secs={s['sections_present']}/{s['sections_total']} acts={s['numbered_actions']} "
                  f"paths={s['file_paths']} nums={s['numbers']} prose={s['prose_dump']} {res.get('latency')}s",
                  flush=True)
            if not res.get("ok"):
                print(f"     ERR: {res.get('error')}", flush=True)
        results.append(rec)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    path = f"{OUT}{args.tag}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
