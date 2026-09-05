#!/usr/bin/env python3
"""Probe the force_model override path to gpt-6-astra (explicit Lane 2).

gpt-6-astra is tier 14 and excluded from auto-routing entirely — reachable ONLY
via a force_model override (the body 'model' field). Sending model="gpt-6-astra"
is NOT the biggie-router sentinel, so the endpoint honours it as force_model.

This verifies the explicit force path end-to-end and that the log records it as
the force_model lane (distinct from rethink/rearchitect tier-13 and escalation).

Usage:
  python3 probe_force_astra.py                 # fire + verify
  python3 probe_force_astra.py --dry           # show payload, no network call
  python3 probe_force_astra.py --count         # report force_model lane count
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DB = str(Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db")

# Explicit force_model override -> gpt-6-astra, which auto-routing can NEVER pick.
MODEL = "gpt-6-astra"
PROMPT = "Reply with exactly: ASTRA_FORCED_OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print payload, no network call")
    ap.add_argument("--count", action="store_true", help="just report force_model lane count")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    if args.count:
        n = conn.execute(
            "SELECT COUNT(*) FROM router_logs WHERE routing_reason LIKE '%forced%'"
        ).fetchone()[0]
        print(f"force_model lane log rows: {n}")
        return

    payload = {
        "model": MODEL,
        "stream": False,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    body = json.dumps(payload).encode()

    if args.dry:
        print("DRY RUN — would POST to:", ENDPOINT)
        print(json.dumps(payload, indent=2))
        return

    print(f"POST {ENDPOINT}")
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            r = json.loads(resp.read().decode())
        print("HTTP", resp.status)
        print("ROUTED MODEL:", r.get("model") or r.get("model_used") or "?")
        content = (
            r.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(r.get("choices"), list)
            else r.get("output_text", "")
        )
        print("OUTPUT:", content)
    except urllib.error.HTTPError as e:
        print("HTTP ERROR", e.code, e.read().decode()[:500])
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print("REQUEST FAILED:", e)
        sys.exit(1)

    row = conn.execute(
        "SELECT routing_reason, final_model, escalated FROM router_logs "
        "ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    if row:
        print("--- router_logs.db latest row ---")
        print("  reason:", row["routing_reason"])
        print("  final_model:", row["final_model"])
        print("  escalated:", row["escalated"])
        reason = (row["routing_reason"] or "").lower()
        final = (row["final_model"] or "").lower()
        if "force" in reason and "astra" in final:
            print("  ✅ recorded as force_model lane -> gpt-6-astra")
        elif final == "gpt-6-astra" and "force" not in reason:
            print("  ⚠️  reached astra but NOT as force_model — inspect reason")
        else:
            print("  ⚠️  not routed to astra via force — inspect routing_reason")
    else:
        print("no rows in router_logs.db yet")


if __name__ == "__main__":
    main()
