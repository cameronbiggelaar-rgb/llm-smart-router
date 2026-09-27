# B40 — Retire the flash served canary

## Decision

Promote the proven `v41flash-tool-canary` treatment into ordinary routing, then pause its hourly monitor. Preserve the production policy that was actually tested:

- primary: `deepseek-v4.1-flash` for tool-bearing `normal_chat` requests whose incumbent would be `deepseek-v4-pro`;
- exclusions: `debugging` and `testing` remain on the normal selector (currently pro at this floor);
- fallback: a flash failure climbs the ordinary tool-capable chain, whose first higher-tier model is `deepseek-v4-pro`;
- architectural/high-floor requests remain on the normal selector and the GPT architectural governor remains authoritative.

The canonical router name is suffix-free (`deepseek-v4.1-flash`); Hermes backend discovery may expose it as `deepseek-v4.1-flash:cloud`. Both names already resolve through the endpoint's suffix handling. The obsolete `deepseek-v4-flash:cloud` entry is unrelated and is not changed in this batch.

## Implementation

1. Add `deepseek-v4.1-flash` to `TOOL_CAPABLE_MODELS`, reflecting the completed structured-tool qualification and served-canary evidence.
2. Add a small direct-routing policy helper. It promotes only the exact former canary cohort: tool-bearing `normal_chat`, non-debugging/non-testing, baseline selection `deepseek-v4-pro`, flash available.
3. Apply that helper in the tool branch after computing the baseline model and before the plan governor. The ordinary fallback chain from tier-3 flash starts with tier-8 pro, preserving the tested fallback without experiment metadata.
4. Disable the `v41flash-tool-canary` experiment in `experiments.yaml`; retain the block as rollback/history, with notes that production routing supersedes it.
5. Restart `biggie-llm-endpoint.service`, verify an eligible route selects flash and an excluded route selects pro, then pause `split-canary-monitor`.

## TDD acceptance tests

- Eligible former-canary traffic selects flash directly and carries no experiment dependency.
- `debugging` and `testing` do not promote to flash.
- A baseline above pro (architectural/high floor) is not demoted to flash.
- Flash unavailable leaves pro selected.
- Flash's first ordinary tool fallback is pro.
- Experiment config is disabled.

## Rollback

Revert the B40 commit and restart `biggie-llm-endpoint.service`; re-enable the experiment and resume `split-canary-monitor` only if returning to the canary mechanism is explicitly desired.
