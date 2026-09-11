#!/usr/bin/env python3
"""A/B quality measurement: change-1 (compress earlier on flash, <100K)
vs current routing (glm-5.3 at native 130-160K).

For each real recent session, build two payloads:
  - FLASH path (change 1): the conversation truncated to ~90K, forced to deepseek-v4-flash
  - GLM path (current):  the conversation at native size, forced to glm-5.3
Run both through the live router, capture summaries + real token usage + cost,
and emit a side-by-side for quality scoring + the cost delta.

Cost: Ollama credit units = input_tokens * ratio. ratio flash=0.5, glm-5.3=1.5.
"""
import sqlite3, json, time, urllib.request, argparse, sys

ROUTER = "http://127.0.0.1:8080/v1/chat/completions"
STATE = "/home/hermes/.hermes/state.db"
RATIO = {"deepseek-v4-flash:cloud": 0.5, "glm-5.3:cloud": 1.5}
CHARS_PER_TOK = 4.0

def load_session(sid):
    con = sqlite3.connect(f"file:{STATE}?mode=ro", uri=True); cur = con.cursor()
    sp = cur.execute("SELECT system_prompt FROM sessions WHERE id=?", (sid,)).fetchone()
    msgs = cur.execute(
        "SELECT role, content FROM messages WHERE session_id=? AND active=1 AND content IS NOT NULL "
        "ORDER BY timestamp ASC, id ASC", (sid,)).fetchall()
    con.close()
    return (sp[0] or "You are a helpful assistant."), [{"role": r, "content": c} for r, c in msgs if c]

def truncate_to(msgs, target_chars, sys_prompt):
    """Keep full recent messages, drop oldest until under target_chars (system + kept).

    Walk the tail greedily; if the newest single message already overshoots, keep
    just the newest message (never return an empty body) but always cap at target.
    """
    used = len(sys_prompt)
    out = []
    for m in reversed(msgs):
        c = len(m["content"])
        if used + c > target_chars and out:
            break
        out.append(m); used += c
        if used >= target_chars:
            break
    if not out and msgs:
        out = [msgs[-1]]  # never empty
    return [{"role": "system", "content": sys_prompt}] + list(reversed(out))

def call(model, messages, timeout=300):
    body = {"model": model, "messages": messages, "max_tokens": 4000,
            "temperature": 0.0, "stream": False}
    req = urllib.request.Request(ROUTER, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        choice = (out.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        used = out.get("model")
        fin = choice.get("finish_reason")
        pt = out.get("usage", {}).get("prompt_tokens")
        ct = out.get("usage", {}).get("completion_tokens")
        return {"ok": True, "content": content, "used_model": used, "finish": fin,
                "prompt_tokens": pt, "completion_tokens": ct, "latency": round(time.time()-t0, 1)}
    except Exception as e:
        return {"ok": False, "error": str(e), "latency": round(time.time()-t0, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", nargs="+",
        default=["20260911_104856_871268", "20260911_065143_8e021a05", "20260909_095333_bd3306"])
    ap.add_argument("--flash_target", type=int, default=90000)   # ~flash cap 100K
    args = ap.parse_args()

    all_out = []
    for sid in args.sids:
        sp, msgs = load_session(sid)
        native_chars = len(sp) + sum(len(m["content"]) for m in msgs)
        flash_msgs = truncate_to(msgs, args.flash_target, sp)

        print(f"\n########## {sid} ##########")
        print(f"  native: {len(msgs):>3} msgs, {native_chars:>8} chars (~{native_chars//4:,} tok) -> GLM path")
        print(f"  flash : {len(flash_msgs)-1:>3} msgs, {args.flash_target:>8} char cap         -> FLASH path (change 1)")

        # Current path: glm-5.3 on native size
        glm = call("glm-5.3:cloud", [{"role":"system","content":sp}, *msgs])
        # Change-1 path: flash on truncated (<100K)
        fd = call("deepseek-v4-flash:cloud", flash_msgs)

        glm_cost = 0; fd_cost = 0
        if glm.get("ok") and glm.get("prompt_tokens"):
            glm_cost = round(glm["prompt_tokens"] * RATIO.get("glm-5.3:cloud", 1.5), 0)
        if fd.get("ok") and fd.get("prompt_tokens"):
            fd_cost = round(fd["prompt_tokens"] * RATIO.get("deepseek-v4-flash:cloud", 0.5), 0)

        rec = {"sid": sid, "native_chars": native_chars, "glm": glm, "flash": fd,
               "glm_cost_units": glm_cost, "flash_cost_units": fd_cost,
               "savings_units": glm_cost - fd_cost}
        all_out.append(rec)

        print(f"  GLM  : fin={glm.get('finish')} pt={glm.get('prompt_tokens')} "
              f"ct={glm.get('completion_tokens')} units≈{glm_cost} ok={glm.get('ok')}")
        print(f"  FLASH: fin={fd.get('finish')} pt={fd.get('prompt_tokens')} "
              f"ct={fd.get('completion_tokens')} units≈{fd_cost} ok={fd.get('ok')}")

    # Save full summaries for quality scoring
    out_path = "/home/hermes/.hermes/skills/llm-smart-router/data/compression_ab_result.json"
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved full summaries + usage to {out_path}")
    print("\n=== COST SUMMARY (Ollama credit units) ===")
    tot_glm = sum(r["glm_cost_units"] for r in all_out if r.get("glm_cost_units"))
    tot_fd  = sum(r["flash_cost_units"] for r in all_out if r.get("flash_cost_units"))
    print(f"  GLM path total : {tot_glm:>8} units")
    print(f"  FLASH path total: {tot_fd:>8} units")
    print(f"  SAVINGS         : {tot_glm-tot_fd:>8} units ({(100*(tot_glm-tot_fd)/tot_glm if tot_glm else 0):.0f}%)")

if __name__ == "__main__":
    main()
