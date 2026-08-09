"""Opt-in capture of large compression payloads for offline comparison.

This module intentionally does no privacy redaction. Samples are for local,
non-destructive analysis and should stay out of git via .gitignore.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "compression_samples"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _message_chars(messages: List[Dict[str, Any]]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def sampling_enabled(headers: Optional[Dict[str, str]] = None) -> bool:
    """Return True if env or request headers explicitly enable sampling."""
    headers = headers or {}
    header_value = ""
    for k, v in headers.items():
        if k.lower() == "x-compression-sample":
            header_value = str(v).strip().lower()
            break
    if header_value in {"1", "true", "yes", "on"}:
        return True
    if header_value in {"0", "false", "no", "off"}:
        return False
    return _env_bool("BIGGIE_COMPRESSION_SAMPLE", False)


def sample_config() -> Dict[str, Any]:
    return {
        "dir": Path(os.environ.get("BIGGIE_COMPRESSION_SAMPLE_DIR", str(DEFAULT_SAMPLE_DIR))),
        "max_samples": int(os.environ.get("BIGGIE_COMPRESSION_SAMPLE_MAX", "50")),
        "min_chars": int(os.environ.get("BIGGIE_COMPRESSION_SAMPLE_MIN_CHARS", "250000")),
        "min_context_tokens": int(os.environ.get("BIGGIE_COMPRESSION_SAMPLE_MIN_CONTEXT_TOKENS", "50000")),
    }


def should_capture_sample(
    messages: List[Dict[str, Any]],
    workload_type: str,
    context_tokens: int,
    headers: Optional[Dict[str, str]] = None,
) -> bool:
    """Gate sample capture to large session-compression payloads only."""
    if not sampling_enabled(headers):
        return False
    if workload_type != "session_compression":
        return False
    cfg = sample_config()
    return context_tokens >= cfg["min_context_tokens"] or _message_chars(messages) >= cfg["min_chars"]


def _existing_sample_count(sample_dir: Path) -> int:
    if not sample_dir.exists():
        return 0
    return len(list(sample_dir.glob("*.json")))


def capture_compression_sample(
    *,
    request_id: str,
    messages: List[Dict[str, Any]],
    workload_type: str,
    context_tokens: int,
    requested_model: str,
    selected_model: str,
    compression_level: str,
    compression_stats: Dict[str, Any],
    route_metadata: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """Write a raw sample JSON file if sampling is enabled and under cap.

    Returns the written path, or None when capture is disabled/skipped. This is
    deliberately best-effort; callers should never fail a user request because
    sample capture failed.
    """
    if not should_capture_sample(messages, workload_type, context_tokens, headers=headers):
        return None

    cfg = sample_config()
    sample_dir: Path = cfg["dir"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    if _existing_sample_count(sample_dir) >= cfg["max_samples"]:
        return None

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe_req = "".join(c if c.isalnum() or c in "-_" else "-" for c in (request_id or "no-request"))[:80]
    path = sample_dir / f"{ts}-{safe_req}.json"
    payload = {
        "schema": "biggie.compression_sample.v1",
        "captured_at": time.time(),
        "request_id": request_id,
        "workload_type": workload_type,
        "context_tokens": context_tokens,
        "input_chars": _message_chars(messages),
        "requested_model": requested_model,
        "selected_model": selected_model,
        "compression_level": compression_level,
        "compression_stats": compression_stats,
        "route_metadata": route_metadata or {},
        "messages": messages,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
