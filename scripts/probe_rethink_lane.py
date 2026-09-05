#!/usr/bin/env python3
"""Probe the rethink/rearchitect routing lane (tier 13 -> gpt-5.6-sol).

Sends a chat-completion to the Biggie LLM endpoint whose prompt triggers the
planning/rethink sub-type in routing_table.yaml, then verifies the routed model
and the routing_lane recorded in router_logs.db.

This is the normal-routing path to a strong model (NOT force_model), so it
exercises the rethink lane end-to-end. gpt-6-astra stays force_model-only.

Usage:
  python3 probe_rethink_lane.py                 # fire a rethink probe + verify
  python3 probe_rethink_lane.py --dry           # show what would be sent, no call
  python3 probe_rethink_lane.py --count         # report rethink lane count in DB
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

# Endpoint the Hermes biggie-router provider points at (service listens on 8080).
ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DB = str(Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db")

# Prompt must classify as 'planning' (PLANNING_KEYWORDS) AND contain a rethink
# sub-type keyword so match_sub_type returns tier 13.
PROMPT = (
    "Please rethink the whole architecture of this system and plan the migration. "
    "Reply with exactly: RETHINK_OK"
)

# Model sentinel = the router itself, so it is NOT treated as a force_model
# override — capability routing fires normally.
MODEL = "biggie-router"


def _lane_lookup(conn):
    """Return the rethink_rearchitect / escalation / force / normal counts."""
    row = conn.execute(
        "SELECT routing_reason, final_model, escalated FROM router_logs "
        "ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print payload, no network call")
    ap.add_argument("--count", action="store_true", help="just report rethink-lane count")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    if args.count:
        n = conn.execute(
            "SELECT COUNT(*) FROM router_logs WHERE routing_reason LIKE '%rethink%'"
        ).fetchone()[0]
        print(f"rethink/rearchitect lane log rows: {n}")
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

    # Verify the lane recorded in the log DB.
    row = _lane_lookup(conn)
    if row:
        print("--- router_logs.db latest row ---")
        print("  reason:", row["routing_reason"])
        print("  final_model:", row["final_model"])
        print("  escalated:", row["escalated"])
        reason = (row["routing_reason"] or "").lower()
        final = (row["final_model"] or "").lower()
        escalated = bool(row["escalated"])
        # Lane heuristic matches check-routing-stats.py: tier 13 via normal
        # routing (escalation=0, reason not a force_model string) == the rethink
        # trigger, the ONLY normal-routing path to gpt-5.6-sol.
        if "force" in reason:
            print("  ⚠️  recorded as force_model lane (probe did NOT exercise rethink)")
        elif escalated:
            print("  ⚠️  recorded as escalation lane (probe did NOT exercise rethink)")
        elif final == "gpt-5.6-sol" and "13" in reason:
            print("  ✅ recorded as rethink/rearchitect lane -> gpt-5.6-sol")
        else:
            print("  ⚠️  lane not identified as rethink — inspect routing_reason")
    else:
        print("no rows in router_logs.db yet")


if __name__ == "__main__":
    main()
