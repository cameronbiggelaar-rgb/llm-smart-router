"""Cost savings report for the LLM Smart Router.

Generates a summary of model usage, costs, and potential savings
from the router_logs database.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import (
    CostSummary,
    MODEL_COST_ORDER,
    get_model_cost,
    init_db,
)


def get_cost_summary(
    router_db: str,
    since: str | None = None,
    days: int = 7,
) -> CostSummary:
    """Generate a cost summary from router_logs data.

    Args:
        router_db: Path to the router SQLite database.
        since: ISO timestamp to filter from (overrides days).
        days: Number of days to look back (default 7).

    Returns:
        CostSummary dataclass with aggregated metrics.
    """
    conn = init_db(router_db)
    try:
        # Determine time window
        if since is None:
            from datetime import timedelta
            since_dt = datetime.now(timezone.utc) - timedelta(days=days)
            since = since_dt.isoformat()

        # Get all logs in the period
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM router_logs WHERE timestamp >= ? ORDER BY timestamp",
            (since,),
        )
        rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return CostSummary(period_start=since, period_end=datetime.now(timezone.utc).isoformat())

        # Aggregate
        total_cost = sum(r.get("estimated_cost_usd", 0) or 0 for r in rows)
        total_calls = len(rows)

        # Cost by model
        by_model: Dict[str, float] = {}
        for r in rows:
            model = r.get("model_used", "unknown")
            by_model[model] = by_model.get(model, 0) + (r.get("estimated_cost_usd", 0) or 0)

        # Cost by task type
        by_task: Dict[str, float] = {}
        for r in rows:
            task = r.get("task_type", "other")
            by_task[task] = by_task.get(task, 0) + (r.get("estimated_cost_usd", 0) or 0)

        # Compute "cost if expensive" — what it would cost if every call
        # used the most expensive model in the chain (gpt-5.5)
        cost_if_expensive = 0.0
        for r in rows:
            model = r.get("model_used", "unknown")
            mc = get_model_cost(model, conn)
            if mc is None:
                # Unknown model — use gpt-5.5 pricing as fallback
                gpt55 = get_model_cost("gpt-5.5", conn)
                if gpt55:
                    inp = r.get("input_tokens", 0) or 0
                    out = r.get("output_tokens", 0) or 0
                    cost_if_expensive += (inp / 1_000_000 * gpt55.input_cost_per_1m) + \
                                         (out / 1_000_000 * gpt55.output_cost_per_1m)
            else:
                # Use the most expensive model's pricing
                expensive_model = MODEL_COST_ORDER[-1] if MODEL_COST_ORDER else "gpt-5.5"
                expensive_mc = get_model_cost(expensive_model, conn)
                if expensive_mc and expensive_mc.input_cost_per_1m > mc.input_cost_per_1m:
                    inp = r.get("input_tokens", 0) or 0
                    out = r.get("output_tokens", 0) or 0
                    cost_if_expensive += (inp / 1_000_000 * expensive_mc.input_cost_per_1m) + \
                                         (out / 1_000_000 * expensive_mc.output_cost_per_1m)
                else:
                    cost_if_expensive += r.get("estimated_cost_usd", 0) or 0

        savings = cost_if_expensive - total_cost
        savings_pct = (savings / cost_if_expensive * 100) if cost_if_expensive > 0 else 0.0

        return CostSummary(
            total_cost_usd=round(total_cost, 4),
            cost_if_expensive_usd=round(cost_if_expensive, 4),
            savings_usd=round(savings, 4),
            savings_pct=round(savings_pct, 1),
            total_calls=total_calls,
            by_model=by_model,
            by_task=by_task,
            period_start=since,
            period_end=rows[-1].get("timestamp", datetime.now(timezone.utc).isoformat()),
        )
    finally:
        conn.close()


def generate_fit_report(router_db: str, since: str | None = None, days: int = 7) -> str:
    """Generate a model fit report showing how well models match task complexity.

    Analyzes complexity scores, correction rates, subagent usage, and
    model overkill/underpowered signals.
    """
    conn = init_db(router_db)
    try:
        if since is None:
            from datetime import timedelta
            since_dt = datetime.now(timezone.utc) - timedelta(days=days)
            since = since_dt.isoformat()

        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM router_logs WHERE timestamp >= ? ORDER BY timestamp",
            (since,),
        )
        rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return "No data in period."

        total = len(rows)
        subagent_count = sum(1 for r in rows if r.get("is_subagent"))
        corrected = sum(1 for r in rows if r.get("user_correction_count", 0) > 0)
        total_corrections = sum(r.get("user_correction_count", 0) or 0 for r in rows)
        switched = sum(1 for r in rows if r.get("model_switched"))
        avg_complexity = sum(r.get("complexity_score", 0) or 0 for r in rows) / total if total else 0

        # By model stats
        model_stats: Dict[str, Dict] = {}
        for r in rows:
            m = r.get("model_used", "unknown")
            if m not in model_stats:
                model_stats[m] = {"calls": 0, "corrections": 0, "total_complexity": 0.0, "subagent_calls": 0}
            model_stats[m]["calls"] += 1
            model_stats[m]["corrections"] += r.get("user_correction_count", 0) or 0
            model_stats[m]["total_complexity"] += r.get("complexity_score", 0) or 0
            if r.get("is_subagent"):
                model_stats[m]["subagent_calls"] += 1

        lines = []
        lines.append("=" * 60)
        lines.append("  LLM Smart Router — Model Fit Report")
        lines.append("=" * 60)
        lines.append(f"  Period: {since[:10]} to {rows[-1].get('timestamp', 'now')[:10]}")
        lines.append(f"  Total sessions: {total}")
        lines.append(f"  Subagent sessions: {subagent_count} ({subagent_count/total*100:.0f}%)")
        lines.append(f"  Sessions with corrections: {corrected} ({corrected/total*100:.1f}%)")
        lines.append(f"  Total corrections: {total_corrections}")
        lines.append(f"  Model switches: {switched}")
        lines.append(f"  Avg complexity score: {avg_complexity:.3f}")
        lines.append("")

        # Model fit table
        lines.append(f"  {'Model':<25} {'Calls':>6} {'Sub':>5} {'Corr':>5} {'AvgCpx':>7}")
        lines.append("  " + "-" * 50)
        for m, s in sorted(model_stats.items(), key=lambda x: -x[1]["calls"]):
            avg_cpx = s["total_complexity"] / s["calls"] if s["calls"] else 0
            lines.append(f"  {m:<25} {s['calls']:>6} {s['subagent_calls']:>5} {s['corrections']:>5} {avg_cpx:>7.3f}")
        lines.append("")

        # Task type fit
        lines.append("  Fit by Task Type:")
        lines.append(f"  {'Task':<20} {'Calls':>6} {'Corr':>5} {'AvgCpx':>7} {'Sub%':>6}")
        lines.append("  " + "-" * 46)
        for task in ["coding", "qa", "research", "planning", "debugging", "other"]:
            task_rows = [r for r in rows if r.get("task_type") == task]
            if not task_rows:
                continue
            t_corr = sum(r.get("user_correction_count", 0) or 0 for r in task_rows)
            t_cpx = sum(r.get("complexity_score", 0) or 0 for r in task_rows) / len(task_rows)
            t_sub = sum(1 for r in task_rows if r.get("is_subagent")) / len(task_rows) * 100
            lines.append(f"  {task:<20} {len(task_rows):>6} {t_corr:>5} {t_cpx:>7.3f} {t_sub:>5.0f}%")
        lines.append("")

        # Key insights
        lines.append("  Key Insights:")
        high_corr_models = [(m, s) for m, s in model_stats.items()
                           if s["corrections"] > 0 and s["calls"] > 0]
        if high_corr_models:
            worst = max(high_corr_models, key=lambda x: x[1]["corrections"] / x[1]["calls"])
            corr_rate = worst[1]["corrections"] / worst[1]["calls"]
            lines.append(f"  ⚠️  Highest correction rate: {worst[0]} ({corr_rate:.1%} of calls corrected)")

        if subagent_count > 0:
            sub_pct = subagent_count / total * 100
            lines.append(f"  ℹ️  {sub_pct:.0f}% of sessions are subagents — consider routing subagent")
            lines.append("     tasks to cheaper models by default, escalating only on failure.")

        if avg_complexity < 0.3:
            lines.append(f"  💡  Avg complexity is low ({avg_complexity:.2f}) — most tasks could use cheaper models.")
        elif avg_complexity > 0.7:
            lines.append(f"  💡  Avg complexity is high ({avg_complexity:.2f}) — expensive models are justified.")

        lines.append("=" * 60)
        return "\n".join(lines)

    finally:
        conn.close()


def generate_report(router_db: str, since: str | None = None, days: int = 7) -> str:
    """Generate a human-readable compute budget report.

    Since all models are flat-subscription (ChatGPT $20/mo, Ollama Cloud $100/mo,
    local free), the "cost" values are RELATIVE COMPUTE UNITS — a dimensionless
    measure of subscription budget consumption. This helps identify which models
    and task types consume the most capacity.

    Args:
        router_db: Path to the router SQLite database.
        since: ISO timestamp to filter from.
        days: Number of days to look back.

    Returns:
        Formatted report string.
    """
    summary = get_cost_summary(router_db, since=since, days=days)

    lines = []
    lines.append("=" * 60)
    lines.append("  LLM Smart Router — Compute Budget Report")
    lines.append("=" * 60)
    lines.append(f"  Period: {summary.period_start[:10]} to {summary.period_end[:10]}")
    lines.append(f"  Total calls: {summary.total_calls}")
    lines.append("")
    lines.append("  ┌─────────────────────────────────────────────────────────┐")
    lines.append(f"  │ Total compute units:  {summary.total_cost_usd:>10.2f}                    │")
    lines.append(f"  │ If all gpt-5.5:       {summary.cost_if_expensive_usd:>10.2f}                    │")
    lines.append(f"  │ Savings (units):      {summary.savings_usd:>10.2f}                    │")
    lines.append(f"  │ Savings %:            {summary.savings_pct:>7.1f}%                   │")
    lines.append("  └─────────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append("  Note: All models are flat-subscription (ChatGPT $20/mo,")
    lines.append("  Ollama Cloud $100/mo, local free). 'Compute units' are a")
    lines.append("  relative measure of subscription budget consumption.")
    lines.append("")

    # By model
    if summary.by_model:
        lines.append("  Compute by Model:")
        lines.append(f"  {'Model':<25} {'Units':>10} {'%':>7}")
        lines.append("  " + "-" * 44)
        for model, cost in sorted(summary.by_model.items(), key=lambda x: -x[1]):
            pct = (cost / summary.total_cost_usd * 100) if summary.total_cost_usd > 0 else 0
            lines.append(f"  {model:<25} {cost:>10.2f}  {pct:>5.1f}%")
        lines.append("")

    # By task type
    if summary.by_task:
        lines.append("  Compute by Task Type:")
        lines.append(f"  {'Task':<20} {'Units':>10} {'%':>7} {'Calls':>7}")
        lines.append("  " + "-" * 46)
        for task, cost in sorted(summary.by_task.items(), key=lambda x: -x[1]):
            pct = (cost / summary.total_cost_usd * 100) if summary.total_cost_usd > 0 else 0
            lines.append(f"  {task:<20} {cost:>10.2f}  {pct:>5.1f}%")
        lines.append("")

    lines.append("  Recommendation:")
    if summary.savings_usd > 0:
        lines.append(f"    ✅ Smart routing saved {summary.savings_usd:.2f} compute units in this period.")
    else:
        lines.append("    ⚠️  No savings yet — data collection is still in Phase 1.")
    lines.append("    Phase 2 (classifier) will enable automatic routing to cheaper models,")
    lines.append("    preserving expensive model capacity for complex work.")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """CLI entry point for the report generator."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Smart Router - Cost Report")
    parser.add_argument("--router-db", default=str(
        Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db"
    ), help="Router database path")
    parser.add_argument("--since", help="ISO timestamp to filter from")
    parser.add_argument("--days", type=int, default=7, help="Days to look back (default: 7)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--fit", action="store_true", help="Show model fit report instead of compute budget")
    args = parser.parse_args()

    if args.fit:
        print(generate_fit_report(args.router_db, since=args.since, days=args.days))
    elif args.json:
        summary = get_cost_summary(args.router_db, since=args.since, days=args.days)
        import json
        print(json.dumps({
            "total_cost_usd": summary.total_cost_usd,
            "cost_if_expensive_usd": summary.cost_if_expensive_usd,
            "savings_usd": summary.savings_usd,
            "savings_pct": summary.savings_pct,
            "total_calls": summary.total_calls,
            "by_model": summary.by_model,
            "by_task": summary.by_task,
            "period_start": summary.period_start,
            "period_end": summary.period_end,
        }, indent=2))
    else:
        print(generate_report(args.router_db, since=args.since, days=args.days))


if __name__ == "__main__":
    main()
