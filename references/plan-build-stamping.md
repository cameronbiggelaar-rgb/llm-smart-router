# Plan — build stamping so staleness is detectable, not guessed

Status: IMPLEMENTED 2026-09-16 (B5 `84568c6`, B6 `7c9bf55`)
Thread: 56467

## Problem (measured, not hypothetical)

`biggie-llm-endpoint` served **21h of pre-patch code** while the file on disk was
current. `/health` returned HTTP 200 the whole time and reports a **hardcoded
`"version": "1.0.0"`** — so the health endpoint *cannot* reveal staleness.

The only signal was a log-format mismatch: `Backend local request failed:` with
nothing after the colon (the batch-4 fix makes it name the exception class), which
appeared 21 times *after* the patch landed with 0 typed lines. That is a terrible
detector: it only works if the patch happened to change a log string, and only if
a human is reading the journal.

Root cause of the underlying incident: Python binds a module at process start.
`Type=simple` / `Restart=on-failure` / `CanReload=no` means systemd only re-execs
on a **crash**, so editing the deployment file does nothing to the running process.

Cost of the gap: a fix can be "done" (committed, tested, 653 passing) and still
not be serving traffic, with a green health check saying everything is fine.

## Design

Stamp the running process at import with what it actually loaded, then make
"on-disk code is newer than what I loaded" a first-class, cheap, queryable fact.

### What is captured (once, at import)

| field | source | why |
|---|---|---|
| `git_sha` | `git rev-parse --short HEAD` in repo root | identify *which* build is running |
| `source_mtime` | mtime of each tracked source file | detect edits after load |
| `started_at` | import time (ISO) | correlate with journal / `ActiveEnterTimestamp` |

### The staleness check itself

At `/health`, `stat()` the tracked source files and compare their mtime to the
captured value. If any is **newer than the stamp**, the process is running
pre-edit code → `stale: true`, plus the names of the changed files.

Why mtime and not a re-`git rev-parse`:
- **Cheaper** — a `stat()` per request vs a subprocess.
- **Strictly more sensitive** — catches uncommitted edits, which a SHA comparison
  misses. Working-with-uncommitted-changes is the normal state of this repo.

The `git_sha` is still reported (captured at import) so you can tell *which* build
is live; the mtime comparison answers *"is it the current one"*.

### Failure mode to design out

`git` may be absent, slow, or the repo may not be a repo. The stamp **must not**
raise or block startup — it fails open to `"unknown"`. A stamping bug must never
be able to take down the endpoint.

### Tracked source files

- `biggie_llm_endpoint.py` (the process itself)
- `router.py` (the routing policy it imports — a stale `router.py` is exactly how
  the batch-1/2/3 fixes would silently not apply)

## Deliverables

### (A) Endpoint exposes its build

- `_capture_build_stamp()` — import-time, fail-open.
- `_newer_than_stamp()` — pure(ish) helper, returns changed filenames.
- `/health` gains a `build` block: `git_sha`, `source_mtime`, `started_at`,
  `stale`, `changed_files`.
- Keep `version` for compatibility; it is no longer the only identity available.

### (B) The ops job flags a mismatch

- New `router_ops.py deploy-check` subcommand: read the endpoint's `/health`,
  read the repo's current `git rev-parse HEAD`, compare, and report
  `OK` / `STALE` with the changed files.
- Wired into `maintain` as a **non-fatal warning** so the existing
  `biggie-router-ops.timer` catches staleness on its own schedule without a
  staleness condition failing (and thus masking) the rest of the daily job.

## Test plan (TDD, batch per deliverable)

### Batch 1 — stamp + `/health`
RED tests (`tests/test_batch5_build_stamp.py`):
1. `_capture_build_stamp()` returns the four keys, `source_mtime` non-empty.
2. Tracked paths include **both** the endpoint and `router.py`.
3. `git_sha` is a non-empty string; **fail-open** — with git forced to fail the
   helper returns `"unknown"` and does **not** raise.
4. `_newer_than_stamp()` is empty when nothing changed (the steady state — a test
   that passes only when stale would be backwards).
5. Behavioural: touch a tracked file to a **future** mtime after import ⇒ the
   helper names exactly that file. Restore mtime after.
6. `/health` response carries the `build` block with `stale` False in the steady
   state, and `version` still present.

### Batch 2 — ops job
RED tests (`tests/test_batch6_deploy_check.py`):
1. `deploy-check` is a registered subcommand.
2. Matching SHA ⇒ exit 0, reports OK.
3. Mismatched SHA ⇒ reports STALE and names it (non-zero or explicit marker).
4. `/health` unreachable ⇒ reports UNKNOWN, does **not** crash the job.
5. `maintain` invokes the check but a stale result does **not** abort `maintain`.

Non-vacuity probe per batch: revert just the guard/implementation ⇒ exactly the
new tests go red, everything else stays green.

## Out of scope

- Auto-restarting on staleness. Detection only — an automatic restart on a
  detected mismatch could mask a real crash-loop and is a separate decision.
- Changing `Type=`/adding `ExecReload`. Would help, but the detection must exist
  first; and `Type=notify` needs a working `sd_notify`, which is its own change.

## Rollout

Code + tests committed per convention (unpushed). **Activation needs
`sudo systemctl restart biggie-llm-endpoint`** — the fix is itself subject to the
staleness it detects, so verify with the new signature (`build.stale` false and a
`git_sha` matching HEAD) rather than a 200.


---

## Completion (2026-09-16)

Shipped as two batches, both committed (unpushed, per convention).

### B5 `84568c6` — endpoint stamps its build
- `_capture_build_stamp()` / `_newer_than_stamp()` / `_source_mtime()` in
  `biggie_llm_endpoint.py`; stamp taken once at import.
- `/health` gains a `build` block (`git_sha`, `source_mtime`, `started_at`,
  `stale`, `changed_files`); `version` kept for compatibility.
- Tracks **both** `biggie_llm_endpoint.py` and `router.py`.
- Fails open to `"unknown"`.
- Tests: `tests/test_batch5_build_stamp.py` (12). Suite 653 → **665**.

### B6 `7c9bf55` — the ops job flags it
- `router_ops.py`: `_fetch_health()`, `_repo_head_sha()`, `evaluate_deploy()`,
  `cmd_deploy_check` (+ `deploy-check` subcommand, `--json`).
- `maintain` calls it as a **non-fatal warning**.
- Tests: `tests/test_batch6_deploy_check.py` (13). Suite 665 → **678**.

### Deviation from plan (worth recording)

The plan assumed a missing `build` block means UNKNOWN. Measured against the live
endpoint, that was wrong in the most important case: the running process returned
**no build block at all** (it predated B5), which is *proof* of a stale
deployment — the exact scenario the check exists for — and reporting UNKNOWN
there would be honest but useless. Added `_endpoint_source_has_stamp()`: if the
deployed source has the feature but the running process does not report it, the
verdict is STALE. Two tests pin both sides.

Second deviation: no `httpx`/`subprocess` imports or endpoint URL constant existed
in `router_ops.py`; both were added.

### Verification
- Full suite **678 passed**.
- Live CLI works: `python3 scripts/router_ops.py deploy-check` → **STALE**
  against the currently-running (pre-B5) endpoint, `rc=1`; matching sha → OK.
- `maintain --days 1` ran end-to-end on the real DB, warned about staleness, and
  still completed rollup + report + propose.

### Status: NOT YET LIVE
This work is subject to the staleness it detects — the running endpoint needs
`sudo systemctl restart biggie-llm-endpoint` before `deploy-check` can report OK.
Until then it correctly reports its own deployment as STALE.
