# LLM Smart Router — Architecture

> **Goal:** A data-driven system that routes each task to the cheapest model that can successfully complete it, optimising for cost without sacrificing quality.

## Design Principle

**No gateway patching.** Hermes core is a narrow waist — we don't modify it. Instead, the router reads existing Hermes session data (sessions DB + session_model_usage) and builds a feature set from what's already tracked. This is a **read-only observer** pattern.

## Data Flow

```
┌─────────────────────┐     ┌──────────────────────┐
│  Hermes Sessions DB │────▶│  Router Data Pipeline │
│  (sessions.db)      │     │                       │
│                     │     │  1. Query sessions    │
│  ┌───────────────┐  │     │  2. Extract features  │
│  │ sessions      │  │     │  3. Classify task     │
│  │ messages      │  │     │  4. Log to router DB │
│  │ model_usage   │  │     │  5. (Phase 2) Predict│
│  └───────────────┘  │     └──────────────────────┘
└─────────────────────┘              │
                                     ▼
                          ┌──────────────────────┐
                          │  Router SQLite DB    │
                          │  (router_logs.db)    │
                          │                      │
                          │  router_logs         │
                          │  model_costs         │
                          │  cost_savings        │
                          └──────────────────────┘
```

## Module Contracts

### `models.py` — Data definitions

```python
@dataclass
class RouterLog:
    """One row in router_logs table — a single model call observation."""
    session_id: str
    model_used: str
    provider: str
    task_type: str          # 'coding' | 'qa' | 'research' | 'planning' | 'debugging' | 'other'
    prompt_length: int      # chars
    context_length: int     # chars (session history)
    tool_call_count: int
    contains_code_blocks: bool
    has_keywords: bool
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    estimated_cost_usd: float
    success: bool
    retry_count: int
    escalated: bool
    user_corrected: bool
    error_type: str | None
    timestamp: str          # ISO 8601

@dataclass
class ModelCost:
    """Known model cost per 1M tokens (USD). Source: provider pricing."""
    model: str
    provider: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    effective_date: str     # ISO date when this pricing was verified
```

### `feature_extractor.py` — Feature extraction

```python
def extract_features(session_row: dict, usage_rows: list[dict]) -> RouterLog:
    """Build a RouterLog from a Hermes session + its model_usage records."""

def classify_task(session_row: dict) -> str:
    """Heuristic task classifier based on session metadata + message content."""

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost using known pricing table."""
```

### `collector.py` — Data collection pipeline

```python
class RouterCollector:
    """Reads Hermes sessions DB, extracts features, writes to router DB."""

    def __init__(self, hermes_home: str = "~/.hermes"):
        self.hermes_db = Path(hermes_home) / "sessions.db"
        self.router_db = Path(hermes_home) / "skills" / "llm-smart-router" / "data" / "router_logs.db"

    def collect(self, since: str | None = None) -> int:
        """Query sessions since timestamp, extract features, insert rows. Returns count."""

    def _get_sessions(self, since: str | None) -> list[dict]: ...
    def _get_usage(self, session_id: str) -> list[dict]: ...
    def _get_messages(self, session_id: str, limit: int = 5) -> list[dict]: ...
```

### `report.py` — Cost savings report

```python
def generate_report(router_db: str) -> str:
    """Generate a cost-savings summary from router_logs data."""

def cost_summary(router_db: str) -> dict:
    """Return dict with: total_cost, cost_if_expensive, savings, by_model, by_task."""
```

### `train.py` — Phase 2: Classifier training

```python
def train_classifier(router_db: str, output_path: str) -> dict:
    """Train a Random Forest classifier on collected data. Returns metrics."""

def predict(features: dict, classifier_path: str) -> str:
    """Predict cheapest successful model for given features."""
```

## Data Model (SQLite)

### `router_logs` table

```sql
CREATE TABLE router_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    model_used TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT 'other',
    prompt_length INTEGER NOT NULL DEFAULT 0,
    context_length INTEGER NOT NULL DEFAULT 0,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    contains_code_blocks INTEGER NOT NULL DEFAULT 0,
    has_keywords INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_seconds REAL NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1,
    retry_count INTEGER NOT NULL DEFAULT 0,
    escalated INTEGER NOT NULL DEFAULT 0,
    user_corrected INTEGER NOT NULL DEFAULT 0,
    error_type TEXT
);
CREATE INDEX idx_router_logs_timestamp ON router_logs(timestamp);
CREATE INDEX idx_router_logs_model ON router_logs(model_used);
CREATE INDEX idx_router_logs_task ON router_logs(task_type);
```

### `model_costs` table

```sql
CREATE TABLE model_costs (
    model TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    input_cost_per_1m REAL NOT NULL,
    output_cost_per_1m REAL NOT NULL,
    effective_date TEXT NOT NULL
);
```

## Test Strategy

| Layer | What it tests | How |
|---|---|---|
| **Unit** | `classify_task()` with known inputs | Assert correct task type for coding/qa/research/planning prompts |
| **Unit** | `extract_features()` with mock session data | Assert correct RouterLog fields |
| **Unit** | `estimate_cost()` with known model prices | Assert correct cost calculation |
| **Integration** | `RouterCollector` against a temp SQLite DB | Create temp sessions DB with known data, run collector, assert router DB has correct rows |
| **Integration** | `generate_report()` against populated router DB | Assert report contains expected sections |
| **E2E** | Full collect → report pipeline | Run against real sessions DB (read-only), assert no crashes |

## Error Handling

- **Hermes DB not found:** Log warning, return 0 rows collected (graceful degradation)
- **Missing fields in session data:** Use defaults (0, '', 'other') — never crash on partial data
- **SQLite write failure:** Log error, abort current batch (don't corrupt existing data)
- **Concurrent access:** Router DB uses WAL mode for safe concurrent reads/writes

## Cost Model (Phase 1 pricing)

Pricing model:
- **All models are flat-subscription or free** — no per-call cost
- **gpt-5.5 (ChatGPT OAuth):** $20/mo flat, rate-limited
- **Ollama Cloud:** $100/mo flat
- **Local:** Free

The "cost" values are **relative compute units** — a dimensionless measure of how much subscription budget each model call consumes. This lets the router prefer cheaper models for simple tasks, preserving expensive model capacity for complex work.

Relative compute ratios (deepseek-v4-flash = 1.0x baseline):

| Model | Provider | Ratio | Input units/1M | Output units/1M |
|---|---|---|---|---|
| llama3.1:8b | local | 0.0x | 0.00 | 0.00 |
| qwen3:14b | local | 0.0x | 0.00 | 0.00 |
| deepseek-v4-flash | ollama-cloud | 1.0x | 0.50 | 1.50 |
| minimax-m2.7:cloud | ollama-cloud | 2.0x | 1.00 | 3.00 |
| glm-5 | ollama-cloud | 2.0x | 1.00 | 3.00 |
| glm-5.1 | ollama-cloud | 2.5x | 1.25 | 3.75 |
| glm-5.2 | ollama-cloud | 3.0x | 1.50 | 4.50 |
| deepseek-v4-pro | ollama-cloud | 4.0x | 2.00 | 6.00 |
| deepseek-v3.1:671b | ollama-cloud | 10.0x | 5.00 | 15.00 |
| gpt-5.5 | openai-codex | 30.0x | 15.00 | 60.00 |
