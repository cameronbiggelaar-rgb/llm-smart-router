"""Batch 5 — stamp the running build, so staleness is a fact and not a guess.

Why this exists
---------------
`biggie-llm-endpoint` served **21 hours of pre-patch code** while the file on
disk was current. `/health` answered HTTP 200 for the whole window and reported a
**hardcoded `"version": "1.0.0"`** — the health endpoint could not, even in
principle, reveal that the process was stale.

The only signal available was a log-format mismatch: the old
`Backend local request failed:` (empty `str(e)`) appeared 21 times *after* the
batch-4 patch landed, with 0 typed lines. That detector only works if the change
happened to alter a log string, and only if a human reads the journal.

Root cause: Python binds a module at process start; the unit is `Type=simple` /
`Restart=on-failure` / `CanReload=no`, so systemd only re-execs on a CRASH and
editing the file does nothing. This test file makes "what did this process
actually load, and has the file changed since?" a first-class question.

Design notes
------------
* Stamp is captured **once at import** (that is the honest record of what loaded).
* Staleness is detected by comparing each tracked file's current mtime to the
  mtime captured at import — a `stat()` per call, strictly more sensitive than a
  SHA comparison because it catches **uncommitted** edits, which is this repo's
  normal state.
* The stamp **fails open** to `"unknown"`: a stamping bug must never be able to
  take the endpoint down.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402

ENDPOINT_SRC = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()


#  the stamp 

def test_stamp_has_the_four_expected_fields():
    """A stamp that does not say WHICH build, WHEN, and from WHAT files is not a
    stamp — it is another green light."""
    stamp = ep._capture_build_stamp()
    for key in ("git_sha", "source_mtime", "started_at"):
        assert key in stamp, f"stamp missing {key!r}: {stamp}"


def test_stamp_tracks_both_the_endpoint_and_the_router_module():
    """A stale `router.py` is exactly how the batch-1/2/3 routing fixes would
    silently fail to apply, so it must be tracked alongside the endpoint itself.
    Tracking only the entry-point file would report 'fresh' while the routing
    policy served traffic from a version that predates the fix."""
    stamp = ep._capture_build_stamp()
    tracked = stamp["source_mtime"]
    assert isinstance(tracked, dict) and tracked, f"source_mtime not a map: {tracked!r}"
    names = {Path(p).name for p in tracked}
    assert "biggie_llm_endpoint.py" in names, names
    assert "router.py" in names, names


def test_stamp_source_mtime_values_are_numbers():
    stamp = ep._capture_build_stamp()
    for path, mtime in stamp["source_mtime"].items():
        assert isinstance(mtime, (int, float)), f"{path} mtime not numeric: {mtime!r}"


def test_stamp_reports_a_real_git_sha():
    """The sha is the identity of the build. 'unknown' is a legitimate fallback
    but in a real checkout on this host it must resolve to something."""
    stamp = ep._capture_build_stamp()
    assert isinstance(stamp["git_sha"], str) and stamp["git_sha"]
    assert stamp["git_sha"] != "unknown", (
        "git_sha should resolve in a real checkout; "
        f"got {stamp['git_sha']!r}"
    )


def test_stamp_fails_open_when_git_is_unavailable(monkeypatch):
    """A stamping bug must never be able to take down the endpoint. With git
    forced to fail the helper returns 'unknown' and does not raise."""
    real_run = subprocess.run

    def _boom(*a, **kw):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(ep.subprocess, "run", _boom)
    try:
        stamp = ep._capture_build_stamp()
    except Exception as exc:  # noqa: BLE001 - that is the point of the test
        pytest.fail(f"_capture_build_stamp must fail open, raised {exc!r}")
    finally:
        monkeypatch.setattr(ep.subprocess, "run", real_run)
    assert stamp["git_sha"] == "unknown", stamp


#  staleness detection 

def test_steady_state_reports_not_stale():
    """The common case must be clean. A staleness check that reports stale when
    nothing changed is worse than no check: it trains people to ignore it."""
    assert ep._newer_than_stamp(ep._capture_build_stamp()) == []


def test_an_edit_after_import_is_detected():
    """The actual regression this exists for: touch a tracked file to a FUTURE
    mtime, exactly as an edit would, and the helper must name that file."""
    stamp = ep._capture_build_stamp()
    target = SCRIPTS_DIR / "router.py"
    original = target.stat().st_mtime
    future = time.time() + 3600
    try:
        os.utime(target, (future, future))
        changed = ep._newer_than_stamp(stamp)
    finally:
        os.utime(target, (original, original))
    assert [Path(p).name for p in changed] == ["router.py"], (
        f"expected exactly router.py, got {changed}"
    )


def test_detection_is_clean_again_after_restore():
    """Guards the restore in the test above — a leaked future mtime would make
    every later test in the suite see a permanently stale endpoint."""
    assert ep._newer_than_stamp(ep._capture_build_stamp()) == []


#  /health surfaces it 

def _health_payload():
    import asyncio

    return asyncio.run(ep.health())


def test_health_exposes_a_build_block():
    payload = _health_payload()
    assert "build" in payload, sorted(payload)
    build = payload["build"]
    for key in ("git_sha", "source_mtime", "started_at", "stale"):
        assert key in build, f"build block missing {key!r}: {build}"


def test_health_build_block_is_not_stale_in_the_steady_state():
    build = _health_payload()["build"]
    assert build["stale"] is False, build
    assert build.get("changed_files") == [], build


def test_health_still_reports_the_legacy_version_field():
    """Kept for compatibility — it is no longer the only identity available, but
    removing it would break existing consumers."""
    assert _health_payload()["version"] == "1.0.0"


def test_health_reports_stale_when_a_tracked_file_changes():
    """End-to-end: the detection must reach the surface a human or the ops job
    actually looks at. A helper that works but is not wired up detects nothing."""
    build0 = _health_payload()["build"]
    assert build0["stale"] is False
    # Re-stamp as if this process had loaded long ago, then edit a file.
    target = SCRIPTS_DIR / "router.py"
    original = target.stat().st_mtime
    try:
        past = time.time() - 7200
        os.utime(target, (past, past))
        ep._BUILD_STAMP["source_mtime"][str(target)] = past
        future = time.time() + 3600
        os.utime(target, (future, future))
        build = _health_payload()["build"]
    finally:
        os.utime(target, (original, original))
        ep._BUILD_STAMP["source_mtime"][str(target)] = ep._source_mtime(SCRIPTS_DIR / "router.py")
    assert build["stale"] is True, build
    assert "router.py" in [Path(p).name for p in build["changed_files"]], build
