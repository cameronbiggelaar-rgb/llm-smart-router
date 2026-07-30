"""Data collector for the LLM Smart Router.

Reads Hermes session data (state.db), extracts features including model fit
signals (complexity, corrections, subagent awareness), and writes observations
to the router SQLite database.

This is a read-only observer — it never modifies the Hermes sessions DB.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import (
    RouterLog,
    init_db,
    insert_router_log,
    get_collection_state,
    set_collection_state,
    estimate_cost,
)
from feature_extractor import extract_features

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_HERMES_HOME = Path.home() / ".hermes"
DEFAULT_SESSIONS_DB = "state.db"
DEFAULT_ROUTER_DB = str(Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db")


class RouterCollector:
    """Collects model call data from Hermes sessions and logs to router DB.

    Usage:
        collector = RouterCollector()
        count = collector.collect()
        print(f"Collected {count} new observations")
    """

    def __init__(
        self,
        hermes_home: str | Path = DEFAULT_HERMES_HOME,
        sessions_db: str = DEFAULT_SESSIONS_DB,
        router_db: str = DEFAULT_ROUTER_DB,
    ):
        self.hermes_home = Path(hermes_home)
        self.sessions_db_path = str(self.hermes_home / sessions_db)
        self.router_db_path = router_db

    def collect(self, since: str | None = None) -> int:
        """Run the data collection pipeline.

        Args:
            since: ISO timestamp to collect from. If None, uses the last
                   collection timestamp from the router DB (or beginning of time).

        Returns:
            Number of new observations collected.
        """
        # Open router DB (creates schema if needed)
        router_conn = init_db(self.router_db_path)

        # Determine collection window
        if since is None:
            since = get_collection_state(router_conn, "last_collected_at", "")

        # Open Hermes sessions DB (read-only)
        if not Path(self.sessions_db_path).exists():
            logger.warning("Hermes sessions DB not found at %s", self.sessions_db_path)
            return 0

        try:
            hermes_conn = sqlite3.connect(f"file:{self.sessions_db_path}?mode=ro", uri=True)
        except sqlite3.OperationalError as e:
            logger.error("Cannot open Hermes sessions DB: %s", e)
            return 0

        try:
            sessions = self._get_sessions(hermes_conn, since)
            count = 0

            # Build a lookup of all sessions for parent resolution
            all_sessions = self._get_all_sessions(hermes_conn)

            for session_row in sessions:
                session_id = session_row["id"]
                usage_rows = self._get_usage(hermes_conn, session_id)
                first_msg = self._get_first_message(hermes_conn, session_id)
                messages = self._get_messages(hermes_conn, session_id)

                # Resolve subagent chain
                parent_session = None
                delegation_depth = 0
                parent_id = session_row.get("parent_session_id", "") or ""
                if parent_id:
                    parent_session = all_sessions.get(parent_id)
                    # Walk up the chain to compute depth
                    current = parent_id
                    while current and current in all_sessions:
                        delegation_depth += 1
                        current = all_sessions[current].get("parent_session_id", "") or ""

                log = extract_features(
                    session_row,
                    usage_rows,
                    first_msg,
                    messages=messages,
                    parent_session=parent_session,
                    delegation_depth=delegation_depth,
                )
                if log is None:
                    continue

                # Enrich with cost estimate
                if log.estimated_cost_usd == 0.0 and log.input_tokens > 0:
                    log.estimated_cost_usd = estimate_cost(
                        log.model_used, log.input_tokens, log.output_tokens, router_conn
                    )

                insert_router_log(router_conn, log)
                count += 1

            # Update collection state
            now = datetime.now(timezone.utc).isoformat()
            set_collection_state(router_conn, "last_collected_at", now)
            set_collection_state(router_conn, "last_collect_count", str(count))
            set_collection_state(router_conn, "last_collect_time", str(int(time.time())))

            logger.info("Collected %d observations", count)
            return count

        finally:
            hermes_conn.close()
            router_conn.close()

    def _get_sessions(self, conn: sqlite3.Connection, since: str | None) -> List[Dict[str, Any]]:
        """Query sessions from the Hermes DB, optionally filtered by time."""
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        if since:
            try:
                dt = datetime.fromisoformat(since)
                since_ts = dt.timestamp()
                cur.execute(
                    "SELECT * FROM sessions WHERE started_at > ? ORDER BY started_at ASC",
                    (since_ts,),
                )
            except (ValueError, TypeError):
                cur.execute("SELECT * FROM sessions ORDER BY started_at ASC LIMIT 500")
        else:
            cur.execute("SELECT * FROM sessions ORDER BY started_at ASC LIMIT 500")

        return [dict(row) for row in cur.fetchall()]

    def _get_all_sessions(self, conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
        """Build a lookup of all sessions by ID for parent resolution."""
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, model, parent_session_id FROM sessions")
        return {row["id"]: dict(row) for row in cur.fetchall()}

    def _get_usage(self, conn: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
        """Query model usage records for a session."""
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM session_model_usage WHERE session_id = ?",
            (session_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def _get_first_message(self, conn: sqlite3.Connection, session_id: str) -> str:
        """Get the first user message text from a session."""
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id ASC LIMIT 1",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            return ""
        content = row["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            return json.dumps(content)
        return str(content) if content else ""

    def _get_messages(self, conn: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session (for correction detection)."""
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        rows = cur.fetchall()
        result = []
        for row in rows:
            content = row["content"]
            if isinstance(content, str):
                result.append({"role": row["role"], "content": content})
            elif isinstance(content, (dict, list)):
                result.append({"role": row["role"], "content": json.dumps(content)})
            else:
                result.append({"role": row["role"], "content": str(content) if content else ""})
        return result


def main():
    """CLI entry point for the collector."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Smart Router - Data Collector")
    parser.add_argument("--since", help="ISO timestamp to collect from")
    parser.add_argument("--hermes-home", default=str(DEFAULT_HERMES_HOME), help="Hermes home directory")
    parser.add_argument("--router-db", default=DEFAULT_ROUTER_DB, help="Router database path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--recreate", action="store_true", help="Recreate the router DB from scratch")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.recreate:
        db_path = Path(args.router_db)
        if db_path.exists():
            db_path.unlink()
            print(f"Deleted existing router DB at {db_path}")

    collector = RouterCollector(
        hermes_home=args.hermes_home,
        router_db=args.router_db,
    )
    count = collector.collect(since=args.since)
    print(f"Collected {count} observations")


if __name__ == "__main__":
    main()
