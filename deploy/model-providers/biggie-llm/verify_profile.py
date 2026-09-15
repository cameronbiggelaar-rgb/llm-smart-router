"""Behavioural-difference check for the biggie-llm provider plugin.

The earlier version flagged every differing field, which is noise: name/aliases/
display_name/description/base_url are MEANT to differ. What must NOT differ is
anything that changes wire behaviour. Assert that set explicitly.
"""
import sys, dataclasses
sys.path.insert(0, "/home/hermes/.hermes/hermes-agent")

from providers import get_provider_profile, _REGISTRY, _discover_providers
_discover_providers()

mine = get_provider_profile("custom:biggie-llm")
generic = _REGISTRY.get("custom")

assert mine is not None, "plugin did not register"
assert mine is not generic, "lookup fell through to the generic profile"

f_mine = {f.name: getattr(mine, f.name) for f in dataclasses.fields(mine)}
f_gen = {f.name: getattr(generic, f.name) for f in dataclasses.fields(generic)}

# Fields that legitimately differ (identity / routing target / the fix itself).
INTENDED = {
    "name", "aliases", "display_name", "description", "base_url",
    "supports_prompt_cache_key",
}
# The single behavioural change this plugin is FOR.
EXPECTED_FLIP = "supports_prompt_cache_key"

diffs = sorted(k for k in set(f_mine) | set(f_gen) if f_mine.get(k) != f_gen.get(k))
unexpected = [k for k in diffs if k not in INTENDED]

print("all differing fields:")
for k in diffs:
    tag = "intended" if k in INTENDED else "UNINTENDED"
    print(f"   [{tag}] {k}: {f_gen.get(k)!r} -> {f_mine.get(k)!r}")

print(f"\nbehavioural differences beyond the fix: {unexpected or 'NONE'}")

# Explicitly assert the fields that would silently break the wire.
for field, why in [
    ("api_mode", "would change the transport used"),
    ("supports_vision", "would disable image handling"),
    ("supports_vision_tool_messages", "would change tool-message encoding"),
    ("supports_model_listing", "would change /models behaviour"),
    ("auth_type", "would change authentication"),
]:
    a, b = f_mine.get(field), f_gen.get(field)
    status = "OK" if a == b else "MISMATCH"
    print(f"   {status:8} {field}: {a!r} (generic {b!r}) — {why}")

print()
assert not unexpected, f"UNINTENDED BEHAVIOURAL DIFFS: {unexpected}"
assert f_mine[EXPECTED_FLIP] is True
assert f_gen[EXPECTED_FLIP] is False, "generic profile should remain False"
print("PASS: only the intended fields differ; no behavioural field changed.")
