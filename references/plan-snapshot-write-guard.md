# Plan — Snapshot write guard (`--save-snapshot`)

## Problem

`scripts/check-routing-stats.py` persists a cumulative snapshot
(`data/routing_stats_snapshot.json`) at the end of `main()` so the nightly
digest can report genuine day-over-day deltas. Today that write is
**unconditional**: any invocation — the 9:30pm cron *and* any ad-hoc run
(verification, `--today`, `--brief`, `--json`) — overwrites the baseline.

Two failure modes follow:

1. **Baseline perturbation / shift.** A manual run overwrites the cron's
   baseline with near-current values, so the next cron diffs against a
   minutes-old baseline instead of yesterday's. Worse, a `--today` manual run
   writes a *today-only* cumulative (total_calls = today's count), which the
   next cron then diffs against — a wildly wrong "day-over-day" delta.
2. **Concurrent-write corruption.** `_save_snapshot` writes a fixed
   `SNAPSHOT_FILE + ".tmp"` then `Path.replace`s it. Two processes writing at
   once interleave into the shared tmp and can move a half-written file into
   place; the next `_load_snapshot` sees invalid JSON and reports "first run —
   no prior baseline" (the exact symptom seen 2026-10-23).

There is also a latent parent-dir fragility: `_save_snapshot` raises
`FileNotFoundError` if `data/` is absent (masked today only because an earlier
probe in `main()` creates it).

## Decision

Make snapshot persistence **opt-in** via an explicit `--save-snapshot` flag,
defaulting to *off*:

- **Manual / ad-hoc runs** (no flag) read the snapshot for deltas but never
  write it — they cannot perturb, shift, or race-corrupt the baseline.
- **The nightly cron** passes `--save-snapshot` and owns the write.

Also harden `_save_snapshot`:
- `Path(SNAPSHOT_FILE).parent.mkdir(parents=True, exist_ok=True)` before write.
- Write to a **per-process unique** tmp (`<file>.tmp.<pid>`) so concurrent
  runs cannot interleave, then `os.replace` atomically into place.

Opt-in (not opt-out `--no-save-snapshot`) because the failure mode is "someone
forgot to *not* clobber"; defaulting to read-only makes forgetting safe.

## Change surface

- `scripts/check-routing-stats.py`
  - `main()`: add `save_snapshot = "--save-snapshot" in sys.argv`; gate the
    `_save_snapshot(snapshot)` call; add a one-line note in the full report's
    day-over-day section when the run does not persist.
  - `_save_snapshot()`: mkdir parents + unique tmp + `os.replace`.
- Cron job `router-lanes-nightly` (`0d503d276f05`): append `--save-snapshot`
  to the command in its prompt.

## Tests (TDD — RED first)

In `tests/test_report_calculations.py`:

- Extend `_run` with an `extra_args` parameter.
- Update `test_snapshot_second_run_reports_deltas` to pass `--save-snapshot`
  on the runs that must persist (the flag is now required to write).
- Add `test_manual_run_without_flag_does_not_overwrite_snapshot`:
  run 1 with `--save-snapshot` writes baseline 10; run 2 *without* the flag
  (15 calls) leaves the file at 10 and still reports the +5 delta against it.

## Rollout

- Commit touches only `check-routing-stats.py`, the test file, and this plan.
- Cron update is a separate config change (no code deploy). Next cron
  (2026-10-24 21:30) will load the existing 2026-10-23 baseline and report the
  first real day-over-day deltas since the guard landed.
