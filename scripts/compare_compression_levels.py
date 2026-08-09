#!/usr/bin/env python3
"""Compare Biggie compression levels for captured samples.

Usage:
  python3 scripts/compare_compression_levels.py data/compression_samples/sample.json
  python3 scripts/compare_compression_levels.py data/compression_samples/

The script is non-destructive: it reads captured raw messages and reports the
output size/savings for off/lite/standard/structural/aggressive. It does not
call any external LLMs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compression import compress_messages  # noqa: E402

LEVELS = ("off", "lite", "standard", "structural", "aggressive")


def iter_sample_paths(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(path.glob("*.json"))
    else:
        yield path


def load_sample(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def message_chars(messages: List[Dict[str, Any]]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def compare_sample(path: Path) -> Dict[str, Any]:
    sample = load_sample(path)
    messages = sample.get("messages", [])
    workload_type = sample.get("workload_type", "session_compression")
    context_tokens = int(sample.get("context_tokens", 0) or 0)
    original_chars = message_chars(messages)
    rows = []
    for level in LEVELS:
        compressed, stats = compress_messages(
            messages,
            level,
            workload_type=workload_type,
            context_tokens=context_tokens,
        )
        out_chars = message_chars(compressed)
        rows.append({
            "requested_level": level,
            "effective_level": stats.get("level", level),
            "output_chars": out_chars,
            "savings_pct": stats.get("savings_pct", 0.0),
            "compression_time_ms": stats.get("compression_time_ms", 0.0),
        })
    return {
        "sample": str(path),
        "request_id": sample.get("request_id", ""),
        "workload_type": workload_type,
        "context_tokens": context_tokens,
        "input_chars": original_chars,
        "captured_level": sample.get("compression_level", ""),
        "captured_stats": sample.get("compression_stats", {}),
        "levels": rows,
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"\nSample: {report['sample']}")
    print(f"request_id={report['request_id']} workload={report['workload_type']} context_tokens={report['context_tokens']} input_chars={report['input_chars']}")
    print("level        effective     output_chars  savings   time_ms")
    print("---------------------------------------------------------")
    for row in report["levels"]:
        print(
            f"{row['requested_level']:<12} {row['effective_level']:<12} "
            f"{row['output_chars']:>12}  {row['savings_pct']:>6.1f}%  "
            f"{row['compression_time_ms']:>7.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="Sample JSON file or directory")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = ap.parse_args()

    reports = [compare_sample(p) for p in iter_sample_paths(args.path)]
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        if not reports:
            print("No sample JSON files found.")
        for report in reports:
            print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
