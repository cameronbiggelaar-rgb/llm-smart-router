"""B30 fix 1 — the openai-codex branch must not forward sampling params.

Codex's ``/responses`` endpoint rejects sampling knobs with
``400 Unsupported parameter: <name>``. The endpoint used to forward
``temperature``/``top_p``/``stop``/``frequency_penalty``/``presence_penalty``
into the Codex body, so any client that set ``temperature`` deterministically
broke every Codex request. Escalation then re-sent the same params to the same
rejecting endpoint (measured: 18 upstream 400s from ONE client request).

Fix 1 is the codex-specific strip; fix 2 (deterministic-4xx classification) is
covered in test_b30_deterministic_4xx_breaker.py.
"""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

# biggie_llm_endpoint lazily imports Hermes' own codex adapter for the codex
# branch, resolving it from `Path.home()`. tests/conftest.py deliberately
# redirects HOME to a temp dir (production-DB isolation guard), so that lazy
# insert points at nothing here. Derive the real location from this skill's own
# path instead — `.hermes/skills/llm-smart-router/scripts/...` -> `.hermes`.
HERMES_AGENT = ROOT.parents[1] / "hermes-agent"

for _p in (str(SCRIPTS_DIR), str(HERMES_AGENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import biggie_llm_endpoint as ep  # noqa: E402

ENDPOINT_SRC = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()

CODE_SAMPLING_PARAMS = (
    "temperature",
    "top_p",
    "stop",
    "frequency_penalty",
    "presence_penalty",
)


# --------------------------------------------------------------------------
# Static contract — the codex branch itself carries no forwarding loop.
# --------------------------------------------------------------------------


def _codex_branch_node(tree):
    """Locate the `if provider == "openai-codex":` body inside proxy_to_backend."""
    for fn in ast.walk(tree):
        if isinstance(fn, ast.AsyncFunctionDef) and fn.name == "proxy_to_backend":
            for node in ast.walk(fn):
                if not isinstance(node, ast.If):
                    continue
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "provider"
                    and any(
                        isinstance(c, ast.Constant) and c.value == "openai-codex"
                        for c in test.comparators
                    )
                ):
                    return node
    return None


def test_codex_branch_has_no_sampling_param_forwarding_loop():
    """RED before fix 1: the codex branch iterated params into the body."""
    tree = ast.parse(ENDPOINT_SRC)
    branch = _codex_branch_node(tree)
    assert branch is not None, "could not locate the openai-codex branch"

    offenders = []
    for node in ast.walk(branch):
        if isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
            names = {
                elt.value
                for elt in node.iter.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
            if names & set(CODE_SAMPLING_PARAMS):
                offenders.append((node.lineno, sorted(names)))

    assert not offenders, (
        "codex branch still forwards sampling params to /responses; Codex "
        f"rejects these with 400 Unsupported parameter: {offenders}"
    )


def test_ollama_branch_still_forwards_sampling_params():
    """Guard against over-stripping: Ollama genuinely accepts temperature."""
    tree = ast.parse(ENDPOINT_SRC)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
            names = {
                elt.value
                for elt in node.iter.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
            if "temperature" in names and "max_tokens" in names:
                found = True
    assert found, (
        "the Ollama/OpenAI-compatible forwarding loop disappeared; the codex "
        "strip must not change behaviour on backends that accept temperature"
    )


def test_codex_denylist_documents_the_restriction():
    assert hasattr(ep, "CODEX_UNSUPPORTED_PARAMS"), (
        "CODEX_UNSUPPORTED_PARAMS denylist missing — the Codex restriction is "
        "undocumented and can silently regress"
    )
    for p in CODE_SAMPLING_PARAMS + ("max_tokens", "max_output_tokens"):
        assert p in ep.CODEX_UNSUPPORTED_PARAMS, f"{p} must be in the Codex denylist"


# --------------------------------------------------------------------------
# Behavioural contract — assert the real body handed to the Codex endpoint.
# --------------------------------------------------------------------------


SSE_PAYLOAD = [
    "event: response.completed",
    'data: {"type":"response.completed","response":{"output":[{"type":"message",'
    '"content":[{"type":"output_text","text":"hi"}]}]}}',
]


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    async def aread(self):
        return b""

    async def aiter_lines(self):
        for line in self._payload:
            yield line

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "ok"}, "done_reason": "stop",
                "prompt_eval_count": 1, "eval_count": 1, "model": "m"}


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    """Captures the JSON body posted to a backend."""

    def __init__(self, payload):
        self.captured = None
        self._resp = _FakeResponse(payload)

    def stream(self, method, url, json=None, headers=None):
        self.captured = {"method": method, "url": url, "json": json}
        return _FakeStreamCtx(self._resp)

    async def post(self, url, json=None, headers=None):
        self.captured = {"method": "POST", "url": url, "json": json}
        return self._resp


def _codex_backend():
    return {
        "provider": "openai-codex",
        "backend_model": "gpt-5.5",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "test-token",
    }


def _drive(monkeypatch, backend, request_body):
    fake = _FakeClient(SSE_PAYLOAD)
    monkeypatch.setattr(ep, "_get_httpx_client", lambda: fake)
    asyncio.run(
        ep.proxy_to_backend(
            backend, [{"role": "user", "content": "say hi"}], request_body
        )
    )
    assert fake.captured is not None, "no request was issued to the backend"
    return fake.captured["json"]


def test_temperature_absent_from_real_codex_body(monkeypatch):
    """The exact production defect: temperature reached the Codex body."""
    body = _drive(
        monkeypatch,
        _codex_backend(),
        {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "say hi"}],
            "temperature": 0.7,
            "max_tokens": 32,
        },
    )
    leaked = [p for p in CODE_SAMPLING_PARAMS + ("max_tokens", "max_output_tokens")
              if p in body]
    assert not leaked, (
        f"Codex rejects {leaked} with 400 Unsupported parameter. "
        f"Body keys sent: {sorted(body)}"
    )


def test_codex_body_keeps_required_fields(monkeypatch):
    """The strip must not remove what Codex actually needs."""
    body = _drive(
        monkeypatch,
        _codex_backend(),
        {"model": "gpt-5.5", "messages": [{"role": "user", "content": "say hi"}],
         "temperature": 0.7},
    )
    assert body["model"] == "gpt-5.5"
    assert body["input"], "converted input items must survive the strip"
    assert body["stream"] is True
    assert body["store"] is False


def test_codex_url_targets_responses(monkeypatch):
    fake = _FakeClient(SSE_PAYLOAD)
    monkeypatch.setattr(ep, "_get_httpx_client", lambda: fake)
    asyncio.run(
        ep.proxy_to_backend(
            _codex_backend(),
            [{"role": "user", "content": "hi"}],
            {"model": "gpt-5.5"},
        )
    )
    assert fake.captured["url"].endswith("/responses")


def test_ollama_still_receives_temperature(monkeypatch):
    """Regression guard: the strip is Codex-specific, not global."""
    backend = {
        "provider": "local",
        "backend_model": "llama3.1:8b",
        "base_url": "http://127.0.0.1:11434",
        "api_key": "",
    }
    body = _drive(
        monkeypatch, backend,
        {"model": "llama3.1:8b", "temperature": 0.3, "max_tokens": 16},
    )
    assert body.get("temperature") == 0.3, (
        "Ollama accepts temperature and must still receive it"
    )

