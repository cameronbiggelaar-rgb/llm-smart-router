"""B35 acceptance: the DEPLOYED plugin must make the transport emit a
session-scoped prompt_cache_key for the name the transport ACTUALLY asks for.

The previous version of this test passed while B35 was dead because it queried
the name I chose ('custom:biggie-llm') instead of the name the transport uses
('custom'). This version asserts on the real lookup, and includes a NEGATIVE
control (flag off => no key) so a green cannot be vacuous.
"""
import sys, dataclasses
sys.path.insert(0, "/home/hermes/.hermes/hermes-agent")

from providers import get_provider_profile, _discover_providers, _REGISTRY, register_provider
_discover_providers()

fails = []
def check(cond, label, detail=""):
    print(("PASS " if cond else "FAIL ") + label + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(label)

# The name the transport passes: agent.provider for a custom_providers route.
p = get_provider_profile("custom")
print("agent.provider='custom' resolves to class:", type(p).__name__)
print("  flag:", getattr(p, "supports_prompt_cache_key", None))

check(p is not None, "profile resolves for 'custom'")
check(getattr(p, "supports_prompt_cache_key", False) is True,
      "supports_prompt_cache_key is True", f"got {getattr(p,'supports_prompt_cache_key',None)}")

# Behaviour inherited from CustomProfile must survive.
check(hasattr(p, "build_api_kwargs_extras"), "inherits build_api_kwargs_extras")
check(hasattr(p, "fetch_models"), "inherits fetch_models")
try:
    p.build_api_kwargs_extras(base_url="http://127.0.0.1:11434")
    check(True, "inherited build_api_kwargs_extras callable")
except Exception as e:
    check(False, "inherited build_api_kwargs_extras callable", f"{type(e).__name__}: {e}")

# Nothing else may have moved. Compare against a pristine bundled profile.
import importlib
bundled_path = "/home/hermes/.hermes/hermes-agent/plugins/model-providers/custom/__init__.py"
spec = importlib.util.spec_from_file_location("_pristine_custom", bundled_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
pristine = mod.CustomProfile(name="custom",
                             aliases=("ollama", "local", "vllm", "llamacpp", "llama.cpp", "llama-cpp"),
                             env_vars=(), base_url="")
before = {f.name: getattr(pristine, f.name) for f in dataclasses.fields(pristine)}
after  = {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}
diffs = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
print("field diffs vs pristine:", diffs)
check(set(diffs) <= {"supports_prompt_cache_key"},
      "ONLY supports_prompt_cache_key differs", str(list(diffs)))

# The real transport function, driven with the resolved flag.
from agent.transports.chat_completions import _add_prompt_cache_key
def emit(sid, flag):
    kw = {}
    _add_prompt_cache_key(
        kw, messages=[{"role": "system", "content": "sys"},
                      {"role": "user", "content": "hello world"}],
        tools=None, supports_prompt_cache_key=flag, session_id=sid, cache_scope_id=None)
    return kw.get("prompt_cache_key")

flag = bool(getattr(p, "supports_prompt_cache_key", False))
a = emit("20260915_100000_aaaaaa", flag)
b = emit("20260915_100000_bbbbbb", flag)
print("keys:", {"A": a, "B": b})

check(a is not None and b is not None, "a key is emitted for every request", f"{a} / {b}")
check(a != b, "two sessions produce DIFFERENT keys")
check(a is not None and emit("20260915_100000_aaaaaa", flag) == a,
      "key is STABLE across turns of one session", f"{a!r} (a None would make this vacuous)")
check(a is not None and len(a) <= 64, "key is provider-safe (<=64 chars)")

# NEGATIVE CONTROL: flag off must emit nothing (proves the test can fail).
neg = emit("20260915_100000_aaaaaa", False)
check(neg is None, "NEGATIVE CONTROL: flag off emits no key", f"got {neg!r}")

print()
if fails:
    print("FAILURES:", fails); sys.exit(1)
print("ALL B35 ACCEPTANCE CHECKS PASS")
