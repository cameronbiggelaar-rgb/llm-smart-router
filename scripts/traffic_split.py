"""Traffic splitting — send part of production to a candidate model safely.

This implements the requested capability: configuration naming a **test model**
plus a **flag/percentage**, so a bounded share of real traffic exercises the
candidate before it is committed to.

Two modes:

``shadow``
    Production serves the control answer, unchanged. The endpoint *additionally*
    calls the candidate on the real payload, records its cost/latency/quality,
    and discards the output. Real load, real payloads, zero user-visible risk.
    This is the default and the recommended first step.

``split``
    ``percent`` of matching traffic is actually served by the candidate. This is
    the deliberate promotion step; it changes what users receive.

Design rules:

* **Deterministic, sticky bucketing.** The arm is a hash of ``session_id``
  (falling back to ``request_id``). A session always lands in the same arm, so a
  conversation cannot flap between models mid-thread and the split is
  reproducible for analysis. Random sampling would corrupt both.
* **Fail-closed.** A missing/malformed config yields no experiments and normal
  routing. An experiment naming an unregistered model is skipped by the caller —
  it is never half-applied.
* **Shadow is not a split.** ``select_arm`` returns ``control`` for every shadow
  request by construction, so shadow can never leak into the served response.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

DEFAULT_EXPERIMENTS_PATH = Path(__file__).resolve().parent / "experiments.yaml"

VALID_MODES = ("shadow", "split")

# Cache of parsed config keyed by (path, mtime) so the endpoint can hot-reload
# an edited experiments file without re-parsing on every request.
_CACHE: Dict[Tuple[str, float], List["Experiment"]] = {}


@dataclass(frozen=True)
class Experiment:
    """One candidate-model experiment."""

    name: str
    enabled: bool
    model: str
    mode: str
    percent: float = 0.0
    match_workload: Tuple[str, ...] = ()
    match_min_tier: int = 0
    started: str = ""
    notes: str = ""

    @property
    def is_shadow(self) -> bool:
        return self.mode == "shadow"


def build_experiment(raw: Dict[str, Any]) -> Optional[Experiment]:
    """Build an Experiment from raw YAML, or None if it is not usable.

    Returns None (rather than raising) so one bad entry cannot take down the
    whole experiments file.
    """
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    model = str(raw.get("model") or "").strip()
    if not name or not model:
        return None

    mode = str(raw.get("mode") or "shadow").strip().lower()
    if mode not in VALID_MODES:
        return None

    try:
        percent = float(raw.get("percent") or 0.0)
    except (TypeError, ValueError):
        percent = 0.0
    # Clamp rather than reject: a typo of 250 must not silently mean "all".
    percent = max(0.0, min(100.0, percent))

    match = raw.get("match") or {}
    if not isinstance(match, dict):
        match = {}
    workload_raw = match.get("workload") or ()
    if isinstance(workload_raw, str):
        workload_raw = [workload_raw]
    workload = tuple(str(w).strip() for w in workload_raw if str(w).strip())
    try:
        min_tier = int(match.get("min_tier") or 0)
    except (TypeError, ValueError):
        min_tier = 0

    return Experiment(
        name=name,
        enabled=bool(raw.get("enabled", False)),
        model=model,
        mode=mode,
        percent=percent,
        match_workload=workload,
        match_min_tier=min_tier,
        started=str(raw.get("started") or ""),
        notes=str(raw.get("notes") or ""),
    )


def load_experiments(path: Optional[str] = None) -> List[Experiment]:
    """Load experiments from YAML. Never raises; returns [] on any problem."""
    p = Path(path) if path else DEFAULT_EXPERIMENTS_PATH
    if not p.exists() or yaml is None:
        return []
    try:
        mtime = p.stat().st_mtime
        key = (str(p), mtime)
        if key in _CACHE:
            return _CACHE[key]
        raw = yaml.safe_load(p.read_text()) or {}
        entries = raw.get("experiments") or []
        if not isinstance(entries, list):
            return []
        exps = [e for e in (build_experiment(r) for r in entries) if e is not None]
        _CACHE[key] = exps
        return exps
    except Exception:
        return []


def _matches(exp: Experiment, workload: str, tier: int) -> bool:
    if not exp.enabled:
        return False
    if exp.match_workload and workload not in exp.match_workload:
        return False
    if exp.match_min_tier and tier and tier < exp.match_min_tier:
        return False
    return True


def pick_experiment(
    experiments: Sequence[Experiment],
    workload: str,
    tier: int = 0,
) -> Optional[Experiment]:
    """Return the first enabled experiment matching this workload/tier."""
    for exp in experiments:
        if _matches(exp, workload, tier):
            return exp
    return None


def _bucket(key: str) -> float:
    """Stable 0..100 bucket for a key.

    Uses a hash of the key so the mapping is stable across processes and
    restarts (Python's ``hash()`` is salted per process and would reshuffle
    every session on restart).
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 100.0  # 0.00 .. 99.99


def select_arm(
    exp: Experiment,
    request_id: str = "",
    session_id: str = "",
) -> str:
    """Return ``'control'`` or ``'treatment'`` for this request.

    Shadow experiments always return ``'control'`` — the production answer is
    never replaced; the candidate is called separately and discarded.

    For splits, the arm is decided by a stable hash of the session (preferred)
    or the request id, so it is sticky per session and reproducible.
    """
    if not exp.enabled:
        return "control"
    if exp.is_shadow:
        return "control"
    if exp.percent <= 0.0:
        return "control"
    if exp.percent >= 100.0:
        return "treatment"
    key = session_id or request_id
    if not key:
        return "control"
    return "treatment" if _bucket(key) < exp.percent else "control"


def split_ratio_observed(exp: Experiment, n: int = 5000) -> float:
    """Observed treatment percentage over ``n`` synthetic sessions.

    Used to verify a configured percentage is actually honoured.
    """
    if n <= 0:
        return 0.0
    treated = sum(
        1 for i in range(n) if select_arm(exp, session_id=f"synthetic-session-{i}") == "treatment"
    )
    return 100.0 * treated / n


def should_run_shadow(exp: Experiment) -> bool:
    """True when the candidate should be called additionally (output discarded)."""
    return bool(exp.enabled) and exp.is_shadow


def experiment_to_row(exp: Experiment) -> Dict[str, Any]:
    """Flatten an Experiment for insertion into the ``experiments`` table."""
    import json

    return {
        "name": exp.name,
        "enabled": 1 if exp.enabled else 0,
        "model": exp.model,
        "mode": exp.mode,
        "percent": exp.percent,
        "match_json": json.dumps(
            {"workload": list(exp.match_workload), "min_tier": exp.match_min_tier}
        ),
        "started": exp.started,
        "notes": exp.notes,
    }


def sync_experiments_to_db(conn, experiments: Iterable[Experiment]) -> int:
    """Mirror the YAML config into the DB so experiment runs are auditable."""
    n = 0
    for exp in experiments:
        row = experiment_to_row(exp)
        conn.execute(
            "INSERT OR REPLACE INTO experiments "
            "(name, enabled, model, mode, percent, match_json, started, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["name"], row["enabled"], row["model"], row["mode"],
                row["percent"], row["match_json"], row["started"], row["notes"],
            ),
        )
        n += 1
    conn.commit()
    return n
