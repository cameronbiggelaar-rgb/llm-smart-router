"""B29 — static contracts for the two endpoint wirings that break the ratchet.

Both defects are *wiring* defects, not logic defects: the router functions were
already correct, but the endpoint did not tell them what it was doing. Nothing
behavioural catches a dropped keyword argument passed through a helper — the
generic path just silently resumes and routes compression to the ChatGPT plan
again. So these are parsed from source, like the B13 contracts.

  1. The escalation call site must pass ``workload_type`` (and ``context_tokens``).
     Without ``workload_type`` the router cannot know a compaction failure is a
     summariser problem and it climbs the generic tool ladder instead.
  2. The handler must re-derive ``requires_tools`` from the *normalised* body,
     after ``_normalize_compression_request`` has stripped tools. If it keeps
     reading the original body, compression is re-classified as tool-requiring
     and Fix 2 is undone at the call site.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import router  # noqa: E402

ENDPOINT_SRC = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()


def _tree() -> ast.Module:
    return ast.parse(ENDPOINT_SRC)


def _calls_to(name: str):
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == name:
            yield node


# --------------------------------------------------------------------------
# 1. Escalation call site carries compression context.
# --------------------------------------------------------------------------


def test_escalate_on_failure_call_sites_pass_workload_type():
    """Defect class: compression escalating to the ChatGPT plan.

    ``escalate_on_failure`` only holds compression on the summariser ladder when
    it is told ``workload_type="session_compression"``. A call site that omits it
    silently restores the historical defect.
    """
    sites = list(_calls_to("escalate_on_failure"))
    assert sites, "no escalate_on_failure call sites found — did the wiring change?"

    offenders = [
        (node.lineno, sorted(kw.arg for kw in node.keywords if kw.arg))
        for node in sites
        if "workload_type" not in {kw.arg for kw in node.keywords}
    ]
    assert not offenders, (
        "escalate_on_failure is called without workload_type, so a session-"
        f"compression failure will promote to the ChatGPT plan: line(s) {offenders}"
    )


def test_escalate_on_failure_call_sites_pass_context_tokens():
    """Context size selects the ladder rung; omitting it silently downgrades."""
    sites = list(_calls_to("escalate_on_failure"))
    offenders = [
        (node.lineno, sorted(kw.arg for kw in node.keywords if kw.arg))
        for node in sites
        if "context_tokens" not in {kw.arg for kw in node.keywords}
    ]
    assert not offenders, (
        "escalate_on_failure is called without context_tokens, so ladder "
        f"selection ignores 100K+ compaction sizes: line(s) {offenders}"
    )


def test_escalate_on_failure_accepts_the_keywords_call_sites_pass():
    """Parity: every keyword at every call site must be a real parameter."""
    valid = set(inspect.signature(router.escalate_on_failure).parameters)
    offenders = []
    for node in _calls_to("escalate_on_failure"):
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in valid:
                offenders.append((node.lineno, kw.arg))
    assert not offenders, (
        f"escalate_on_failure call sites pass unknown keywords {offenders}; "
        f"valid parameters: {sorted(valid)}"
    )


# --------------------------------------------------------------------------
# 2. requires_tools is derived after normalisation.
# --------------------------------------------------------------------------


def test_requires_tools_is_reassigned_after_normalisation():
    """Ordering contract: normalise first, then derive the flag.

    If ``requires_tools = bool(body.get("tools"))`` is the last assignment to
    that name, the strip produced by ``_normalize_compression_request`` is
    ignored and compression is offered-tools again downstream.
    """
    src_lines = ENDPOINT_SRC.splitlines()

    normalize_at = None
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_normalize_compression_request":
            normalize_at = node.lineno
            break

    assert normalize_at is not None, (
        "_normalize_compression_request is never called — Fix 2 is not wired"
    )

    # Every assignment to requires_tools that occurs at or after the call.
    later_assignments = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", "") == "requires_tools" and node.lineno >= normalize_at:
                    later_assignments.append((node.lineno, ast.unparse(node.value)))

    assert later_assignments, (
        "requires_tools is never re-derived after _normalize_compression_request, "
        "so the tool-strip does not affect tool-requirement classification"
    )

    for lineno, value_src in later_assignments:
        assert "_requires_tools" in value_src, (
            f"line {lineno}: requires_tools re-derived as {value_src!r}, which "
            "re-reads the raw body instead of consulting the flag recorded by "
            "_normalize_compression_request. A pre-strip snapshot restores the "
            "tool-requiring misclassification."
        )


def test_normalizer_contract_strips_tools_for_compression_only():
    """Direct behavioural sanity on the helper the static checks depend on."""
    compression_body = {
        "messages": [{"role": "user", "content": "summarize"}],
        "tools": [{"type": "function", "function": {"name": "t"}}],
        "tool_choice": "auto",
    }
    out = ep._normalize_compression_request(dict(compression_body), "session_compression")
    assert "tools" not in out
    assert "tool_choice" not in out
    assert out["_requires_tools"] is False

    other_body = {
        "messages": [{"role": "user", "content": "summarize"}],
        "tools": [{"type": "function", "function": {"name": "t"}}],
    }
    out2 = ep._normalize_compression_request(dict(other_body), "general")
    assert out2["tools"], "non-compression workloads must keep their tool schemas"
