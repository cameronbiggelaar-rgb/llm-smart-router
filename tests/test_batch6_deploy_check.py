"""Batch 6 — the daily ops job must flag a stale deployment.

Why this exists
---------------
Batch 5 taught the endpoint to report what it actually loaded (`build.stale`).
That is only useful if something looks at it without a human remembering to.
`biggie-router-ops.timer` already runs daily and already knows how to report —
it is the natural place for a staleness check.

The endpoint served 21h of pre-patch code with a green `/health`, so the failure
mode this guards is: the fix is committed, tested, deployed-by-hope, and the
daily job happily reports unit economics while serving last week's routing.

Design rules pinned here
------------------------
* **Detection only.** It never restarts anything. An automatic restart on a
  detected mismatch could mask a crash-loop (restart → stale → restart), which is
  a far worse failure than a stale build.
* **Never fatal to `maintain`.** A stale build is a real finding but it must not
  abort the rollup/purge/report work — that would turn an information problem
  into a monitoring outage. It warns and continues.
* **Unreachable endpoint is UNKNOWN, not stale and not an exception.** An
  endpoint that is down is a different problem, reported by its own liveness
  check; conflating the two would make this check cry wolf during restarts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import router_ops  # noqa: E402


def _health(sha: str, stale: bool = False, changed=None) -> dict:
    return {
        "status": "ok",
        "build": {
            "git_sha": sha,
            "stale": stale,
            "changed_files": changed or [],
            "started_at": "2026-09-16T14:30:40+10:00",
            "source_mtime": {},
        },
    }


def _patch_health(monkeypatch, payload_or_exc, status_code=200):
    """Stub the module's health fetch.

    `_fetch_health` returns the PARSED payload (dict) or None — it does its own
    HTTP and swallows failures. So an exception here stands for "unreachable",
    which the real helper reports as None.
    """
    def _fake(*a, **kw):
        if isinstance(payload_or_exc, Exception):
            return None
        return payload_or_exc

    monkeypatch.setattr(router_ops, "_fetch_health", _fake, raising=False)


def _patch_head(monkeypatch, sha: str):
    monkeypatch.setattr(router_ops, "_repo_head_sha", lambda *a, **kw: sha, raising=False)


#  the comparison 

def test_deploy_check_is_a_registered_subcommand():
    """It has to be invocable, or it is a library nobody calls."""
    src = (SCRIPTS_DIR / "router_ops.py").read_text()
    assert 'add_parser("deploy-check"' in src, "deploy-check subcommand not registered"


def test_matching_sha_reports_ok(monkeypatch, capsys):
    _patch_health(monkeypatch, _health("abc1234"))
    _patch_head(monkeypatch, "abc1234")
    rc = router_ops.cmd_deploy_check(None, argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "OK" in out.upper(), out


def test_mismatched_sha_is_reported_as_stale(monkeypatch, capsys):
    """The whole point: the running build is not the current commit."""
    _patch_health(monkeypatch, _health("old1111"))
    _patch_head(monkeypatch, "new2222")
    rc = router_ops.cmd_deploy_check(None, argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert "STALE" in out.upper(), out
    assert "old1111" in out and "new2222" in out, out
    assert rc != 0, "a stale deployment must signal non-zero via the command rc"


def test_endpoint_stale_flag_alone_is_enough(monkeypatch, capsys):
    """Even if the shas happen to match (e.g. uncommitted edits after a commit),
    the endpoint's own mtime verdict must be surfaced."""
    _patch_health(monkeypatch, _health("same111", stale=True, changed=["/x/router.py"]))
    _patch_head(monkeypatch, "same111")
    rc = router_ops.cmd_deploy_check(None, argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert "STALE" in out.upper(), out
    assert "router.py" in out, out
    assert rc != 0


def test_unreachable_endpoint_is_unknown_and_not_an_exception(monkeypatch, capsys):
    """An endpoint that is down is a liveness problem, not a staleness one.
    Reporting it as stale would cry wolf on every restart."""
    _patch_health(monkeypatch, ConnectionError("connection refused"))
    _patch_head(monkeypatch, "abc1234")
    rc = router_ops.cmd_deploy_check(None, argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert "UNKNOWN" in out.upper(), out
    assert "STALE" not in out.upper(), out
    assert rc == 0, "cannot reach != needs a restart; must not fail the job"


def test_unknown_git_sha_does_not_produce_a_false_stale(monkeypatch, capsys):
    """'unknown' means 'could not determine', which is not the same as 'different'.
    Treating it as a mismatch would flag every non-checkout host forever."""
    _patch_health(monkeypatch, _health("unknown"))
    _patch_head(monkeypatch, "abc1234")
    rc = router_ops.cmd_deploy_check(None, argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert "STALE" not in out.upper(), out
    assert "UNKNOWN" in out.upper(), out
    assert rc == 0


def test_json_mode_emits_machine_readable_output(monkeypatch, capsys):
    _patch_health(monkeypatch, _health("old1111"))
    _patch_head(monkeypatch, "new2222")
    router_ops.cmd_deploy_check(None, argparse.Namespace(json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "STALE", payload
    assert payload["running_sha"] == "old1111", payload
    assert payload["repo_sha"] == "new2222", payload


#  maintain must not be broken by it 

def test_maintain_warns_but_does_not_abort_on_stale(monkeypatch, capsys):
    """Turning a staleness finding into a failed daily job would convert an
    information problem into a monitoring outage — the rollup/purge/report work
    must still run."""
    calls: list[str] = []

    monkeypatch.setattr(
        router_ops, "cmd_rollup", lambda conn, args: (calls.append("rollup"), 0)[1]
    )
    monkeypatch.setattr(
        router_ops, "cmd_purge", lambda conn, args: (calls.append("purge"), 0)[1]
    )
    monkeypatch.setattr(
        router_ops, "cmd_report", lambda conn, args: (calls.append("report"), 0)[1]
    )
    monkeypatch.setattr(
        router_ops, "cmd_propose", lambda conn, args: (calls.append("propose"), 0)[1]
    )
    # a stale deployment is detected...
    _patch_health(monkeypatch, _health("old1111"))
    _patch_head(monkeypatch, "new2222")

    rc = router_ops.cmd_maintain(
        None,
        argparse.Namespace(days=1, retention=30, batch=1000),
    )
    out = capsys.readouterr().out
    # ...every stage still ran...
    assert calls == ["rollup", "purge", "report", "propose"], calls
    # ...the job did not fail...
    assert rc == 0, rc
    # ...and it warned.
    assert "STALE" in out.upper(), out


def test_maintain_does_not_crash_when_the_endpoint_is_down(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        router_ops, "cmd_rollup", lambda conn, args: (calls.append("rollup"), 0)[1]
    )
    monkeypatch.setattr(
        router_ops, "cmd_purge", lambda conn, args: (calls.append("purge"), 0)[1]
    )
    monkeypatch.setattr(
        router_ops, "cmd_report", lambda conn, args: (calls.append("report"), 0)[1]
    )
    monkeypatch.setattr(
        router_ops, "cmd_propose", lambda conn, args: (calls.append("propose"), 0)[1]
    )
    _patch_health(monkeypatch, ConnectionError("refused"))
    _patch_head(monkeypatch, "abc1234")
    rc = router_ops.cmd_maintain(None, argparse.Namespace(days=1, retention=30, batch=1000))
    assert rc == 0
    assert calls == ["rollup", "purge", "report", "propose"], calls


#  the real helper must not raise 

def test_fetch_health_returns_none_or_dict_and_never_raises(monkeypatch):
    """The helper is called by a daily job; an exception here would be an outage
    in the monitoring that exists to catch outages."""
    import httpx

    def _boom(*a, **kw):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(router_ops.httpx, "get", _boom)
    assert router_ops._fetch_health() is None


def test_pre_stamp_running_endpoint_is_stale_not_unknown(monkeypatch, capsys):
    """Measured live 2026-09-16: the running endpoint returned no `build` block at
    all while the source on disk had the feature — i.e. it was serving pre-stamp
    code. Answering UNKNOWN there is honest but useless, because that is exactly
    the deploy-lag case this check exists to catch."""
    _patch_health(monkeypatch, {"status": "ok", "version": "1.0.0"})  # no build block
    _patch_head(monkeypatch, "abc1234")
    monkeypatch.setattr(router_ops, "_endpoint_source_has_stamp", lambda: True)
    verdict = router_ops.evaluate_deploy({"status": "ok"}, "abc1234")
    assert verdict["status"] == "STALE", verdict
    rc = router_ops.cmd_deploy_check(None, argparse.Namespace(json=False))
    assert "STALE" in capsys.readouterr().out.upper()
    assert rc != 0


def test_missing_build_block_is_unknown_when_the_source_lacks_the_feature(monkeypatch):
    """If the deployed source has no stamp feature either, a missing block is not
    evidence of anything — do not invent a failure."""
    monkeypatch.setattr(router_ops, "_endpoint_source_has_stamp", lambda: False)
    verdict = router_ops.evaluate_deploy({"status": "ok"}, "abc1234")
    assert verdict["status"] == "UNKNOWN", verdict


def test_repo_head_sha_never_raises():
    sha = router_ops._repo_head_sha()
    assert sha is None or isinstance(sha, str)
