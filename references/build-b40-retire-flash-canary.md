# B40 — Flash canary retirement build record

## Change

Promoted the proven `v41flash-tool-canary` treatment into normal router policy:

- `deepseek-v4.1-flash` is now a verified tool-capable model.
- Tool-bearing `normal_chat` traffic that would otherwise select
  `deepseek-v4-pro` is promoted to flash.
- `debugging` and `testing` remain on pro.
- Higher-floor selections are not demoted.
- If flash is unavailable, the baseline pro selection remains in place.
- If flash fails after selection, the ordinary tool fallback chain starts with
  `deepseek-v4-pro`.
- The `v41flash-tool-canary` experiment is disabled and retained only as rollout
  history.

The canonical deployed model name is `deepseek-v4.1-flash`. The legacy
`deepseek-v4-flash` entry remains separately configured by prior decision; this
change does not reinterpret or remove that model.

## Tests

TDD regression module: `tests/test_flash_canary_retirement.py`.

Targeted router/governor/endpoint suite:

- **169 passed**

Full suite (`tests/`, excluding the production DB isolation test):

- **977 passed**
- **10 failed**
- **4 deselected**

The same ten failures are the known data-drift set from the pre-change baseline:
five split-canary monitor tests, four optimiser evidence tests whose dated
fixtures have fallen outside the live 14-day window, and one historical
streaming-row-count assertion. No new failure class was introduced.

## Operational sequence

1. Commit the router/config/test changes.
2. Restart `biggie-llm-endpoint.service` externally to this process.
3. Verify endpoint health and one eligible/excluded live routing decision.
4. Pause `split-canary-monitor` only after direct routing is confirmed.
