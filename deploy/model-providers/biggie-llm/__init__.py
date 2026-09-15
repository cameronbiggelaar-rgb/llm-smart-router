"""biggie-llm route profile: declare prompt_cache_key support.

WHY THIS EXISTS
---------------
Hermes' chat-completions transport can send a session-scoped
``prompt_cache_key``, but only when the resolved provider profile sets
``supports_prompt_cache_key``. The generic ``custom`` profile leaves it False
(deliberately: many OpenAI-compatible endpoints reject unknown top-level
fields), so requests through this router carried NO session identity.

Without an identity the router must DERIVE one from the prompt text, and the
derived key was a system-prompt hash: every conversation on a profile collided
onto one key. Consequences measured on 2026-09-15:
  - the "per-session 40/hr" compression guard was effectively GLOBAL (one pot,
    drawn by three concurrent conversations);
  - ``session_compression_overflow`` labelled a session "runaway" when a
    *different* session had consumed the shared budget.

The router accepts ``prompt_cache_key`` (it is an OpenAI-compatible endpoint and
already reads the field on the Codex path), so declaring support here is
accurate, not aspirational.

WHY IT INHERITS INSTEAD OF CONSTRUCTING
---------------------------------------
A freshly-constructed ``ProviderProfile`` resets every field it does not name to
the class default, which silently changes wire behaviour beyond this fix. So the
generic ``custom`` profile is the base and only the intended fields are
overridden. ``aliases`` is overridden rather than inherited precisely because the
generic profile's aliases (``ollama``, ``local``, ``vllm``, ``llamacpp``) must
NOT be repointed at this router.

REGISTERED NAMES
----------------
``custom:biggie-llm`` with alias ``biggie-llm``. Both spellings resolve, because
``agent.provider`` may carry either the bare configured name (``biggie-llm``) or
the ``custom:``-prefixed id — and a bare name that misses the registry falls
through to ``None``, silently disabling this flag.

User plugins under ``$HERMES_HOME/plugins/model-providers/`` are registered AFTER
bundled profiles and replace a same-named entry. Discovery is once per process,
so a long-lived gateway needs a restart to pick this up.
"""
import dataclasses
import logging

from providers import register_provider, get_provider_profile

logger = logging.getLogger(__name__)

# Bundled plugins load FIRST (providers/__init__.py step 1, user plugins step 2),
# so the generic custom profile is already registered when this module runs.
_base = get_provider_profile("custom")
if _base is None:  # pragma: no cover - defensive; visible as a failed plugin load
    raise RuntimeError(
        "biggie-llm provider plugin: the generic 'custom' profile is not "
        "registered, so there is nothing to inherit from. Refusing to register "
        "a from-scratch profile that would silently reset unrelated wire "
        "behaviour."
    )

biggie_llm = dataclasses.replace(
    _base,
    name="custom:biggie-llm",
    # Do NOT inherit the generic aliases (ollama/local/vllm/llamacpp) — those
    # must keep resolving to the generic custom profile.
    aliases=("biggie-llm",),
    display_name="Biggie LLM Router",
    description="Local smart model router (biggie-llm-endpoint)",
    base_url="http://127.0.0.1:8080/v1",
    # The router forwards prompt_cache_key; it does not reject unknown fields.
    supports_prompt_cache_key=True,
)

register_provider(biggie_llm)
