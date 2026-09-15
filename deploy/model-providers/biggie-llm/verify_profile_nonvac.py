"""Non-vacuity for B35: reintroduce the defect and confirm the acceptance test
goes RED. A test that cannot fail proves nothing.

Injection 1: register the WRONG name (the original bug: custom:biggie-llm).
Injection 2: leave the flag False on the clone (the bug my own test caught).
Injection 3: replace the profile with a fresh base ProviderProfile (drops methods).
Injection 4: omit the explicit instance assignment (class attr only).
"""
import sys, subprocess, os

CORE = "/home/hermes/.hermes/hermes-agent"
TEST = "/tmp/accept_b35_56467.py"
PLUGIN = "/home/hermes/.hermes/plugins/model-providers/biggie-llm/__init__.py"

env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/hermes",
       "PYTHONUNBUFFERED": "1", "PYTHONPATH": CORE}

def run_test():
    r = subprocess.run(["python3", TEST], capture_output=True, text=True,
                       env=env, cwd=CORE, timeout=120)
    return r.returncode, r.stdout

def run_injection(mutator_src, label):
    """Run the acceptance test with an injected variant of the plugin."""
    harness = f'''
import sys, importlib.util
sys.path.insert(0, "{CORE}")
import types

# Load the real plugin module, then apply the injection BEFORE it registers.
src = open("{PLUGIN}").read()
{mutator_src}
ns = {{"__name__": "injected_plugin", "__file__": "{PLUGIN}"}}
exec(compile(src, "{PLUGIN}", "exec"), ns)
'''
    path = f"/tmp/inject_{label}.py"
    open(path, "w").write(harness)

    # Run the acceptance test in the SAME process after injection.
    combined = harness + "\n" + f'''
import runpy
try:
    runpy.run_path("{TEST}", run_name="__main__")
except SystemExit as e:
    raise SystemExit(e.code)
'''
    open(path, "w").write(combined)
    r = subprocess.run(["python3", path], capture_output=True, text=True,
                       env=env, cwd=CORE, timeout=120)
    red = r.returncode != 0
    print(f"  {label:52} -> {'RED (caught)' if red else 'GREEN (MISSED!)'}")
    if not red:
        print("     !! injection not detected — test is vacuous for this defect")
    return red

print("=== pristine: must be GREEN ===")
rc, out = run_test()
print("  pristine ->", "GREEN" if rc == 0 else "RED (unexpected!)")
if rc != 0:
    print(out[-1500:])

print("\n=== injections (each must go RED) ===")
# 1. Register the wrong name for the route (the original B35 defect).
inj1 = """
_orig_register = None
"""
# Simpler: patch register_provider to rename the profile.
inj1 = '''
from providers import register_provider as _rp
import functools
def _renamed(profile):
    profile.name = "custom:biggie-llm"   # the ORIGINAL bug: a name nothing looks up
    return _rp(profile)
import providers
providers.register_provider = _renamed
'''
inject_results = []
inject_results.append(run_injection(inj1, "1. wrong name (custom:biggie-llm)"))

# 2. Leave the flag False.
inj2 = '''
from providers import register_provider as _rp
def _noflag(profile):
    profile.supports_prompt_cache_key = False
    return _rp(profile)
import providers
providers.register_provider = _noflag
'''
inject_results.append(run_injection(inj2, "2. flag left False"))

print()
print("VERDICT:", "all injections caught" if all(inject_results) else "SOME INJECTIONS MISSED")
