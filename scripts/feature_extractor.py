"""Feature extraction for the LLM Smart Router.

Extracts features from Hermes session data and classifies tasks
into types for the router pipeline. Includes model fit signals:
complexity scoring, correction detection, and subagent awareness.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from models import RouterLog, MODEL_CAPABILITY_TIERS


# ── Task classification keywords ─────────────────────────────────────────────

CODING_KEYWORDS = [
    "def ", "class ", "function", "import ", "refactor", "debug", "bug",
    "error", "traceback", "exception", "syntax", "compile", "test_",
    "pytest", "unittest", "async def", "lambda", "return ", "yield ",
    "try:", "except", "finally:", "with ", "while ",
    "git ", "commit", "push", "pull", "merge", "branch",
    "docker", "dockerfile", "makefile", "npm ", "pip ",
    "api ", "endpoint", "route", "middleware", "database",
    "sql", "query", "select ", "insert ", "update ", "delete ",
    "json", "yaml", "toml", "config", "schema",
    "sort a list", "parse json", "write a", "implement",
]

PLANNING_KEYWORDS = [
    "plan the", "architecture design", "design a", "strategy for",
    "approach to", "roadmap", "timeline", "milestone", "proposal", "spec",
    "decision", "trade-off", "evaluate options",
    "how should we", "what if we", "consider", "recommend",
]

RESEARCH_KEYWORDS = [
    "what is", "how does", "explain", "compare", "research", "find",
    "search for", "look up", "investigate", "analyze",
    "tell me about", "what are", "define", "describe",
    "latest", "news", "update on", "status of",
]

DEBUGGING_KEYWORDS = [
    "why is", "why does", "why isn't", "why aren't",
    "not working", "broken", "failing", "crash", "hang",
    "unexpected", "wrong", "incorrect", "mismatch",
    "fix this", "help debug", "root cause",
    "debug this",
]

# ── Complexity scoring signals ────────────────────────────────────────────────

# Phrases that indicate multi-step / complex instructions
COMPLEXITY_INSTRUCTION_PATTERNS = [
    r"first\b.*\bthen\b", r"step\s+\d+", r"in this order",
    r"make sure", r"ensure that", r"don't forget", r"remember to",
    r"after that", r"once you've", r"before you",
    r"if.*then.*else", r"depending on", r"based on",
]

# Format/output constraints
FORMAT_CONSTRAINTS = [
    "in json", "as json", "as a table", "in a table", "formatted as",
    "in markdown", "as markdown", "in yaml", "as yaml",
    "csv format", "as csv", "in csv",
    "return only", "output only", "respond with only",
    "bullet points", "numbered list",
    "with columns", "with headers",
]

# Niche / domain-specific references that smaller models may not handle well
NICHE_REFERENCES = [
    "kubernetes", "k8s", "terraform", "ansible", "helm",
    "react native", "flutter", "swiftui", "jetpack compose",
    "pytorch", "tensorflow", "jax", "cuda", "cublas",
    "llvm", "clang", "wasm", "webassembly",
    "postgres", "postgresql", "mongodb", "redis", "kafka",
    "graphql", "grpc", "protobuf", "websocket",
    "oauth", "oidc", "saml", "jwt", "mtls",
    "arm64", "aarch64", "risc-v", "avx512",
    "hermes agent", "hermes gateway", "hermes skill",
    "openai codex", "responses api",
]

# Correction signal patterns — user messages that indicate the model got it wrong
CORRECTION_PATTERNS = [
    r"^no[.,!]?\s", r"^that'?s not", r"^that'?s wrong",
    r"^not what", r"^i didn't", r"^i meant",
    r"^actually[.,!]?\s", r"^instead[.,!]?\s",
    r"^you didn't", r"^you misunderstood",
    r"^re-?read", r"^try again",
    r"^that'?s incorrect", r"^wrong[.,!]?\s",
    r"^my ask was", r"^what i asked",
    r"^let me re-?phrase", r"^to clarify",
    r"^that doesn't", r"^that isn't",
    r"^i said", r"^as i said",
    r"^please re-?read", r"^read my",
    r"^you ignored", r"^you missed",
    r"^that'?s not what i", r"^not what i",
    r"^that'?s still", r"^still wrong",
]


def classify_task(
    prompt: str = "",
    tool_call_count: int = 0,
    session_model: str = "",
) -> str:
    """Classify a task into one of: coding, qa, research, planning, debugging, other.

    Uses heuristic keyword matching on the prompt text, with tool call
    count and session model as secondary signals.

    Args:
        prompt: The user's message text (first message in session).
        tool_call_count: Number of tool calls in the session.
        session_model: The model used for this session.

    Returns:
        One of: 'coding', 'qa', 'research', 'planning', 'debugging', 'other'.
    """
    text = (prompt or "").lower()

    # Debugging keywords checked first (most specific)
    if any(kw in text for kw in DEBUGGING_KEYWORDS):
        return "debugging"

    # Planning keywords checked before coding (design/architecture phrases)
    if any(kw in text for kw in PLANNING_KEYWORDS):
        return "planning"

    # Coding keywords
    if any(kw in text for kw in CODING_KEYWORDS):
        return "coding"

    # Research keywords
    if any(kw in text for kw in RESEARCH_KEYWORDS):
        return "research"

    # High tool call count suggests complex/coding work
    if tool_call_count > 3:
        return "coding"

    # Default
    return "qa"


def contains_code_blocks(text: str) -> bool:
    """Check if text contains markdown code blocks or inline code."""
    if not text:
        return False
    return bool(re.search(r"```|`[^`]+`", text))


def has_router_keywords(text: str) -> bool:
    """Check if text contains keywords that suggest complex routing needed."""
    if not text:
        return False
    keywords = ["refactor", "debug", "plan", "design", "architecture", "optimize"]
    return any(kw in text.lower() for kw in keywords)


# ── Model fit scoring ─────────────────────────────────────────────────────────

def score_complexity(prompt: str, tool_call_count: int, message_count: int) -> float:
    """Score task complexity from 0.0 (trivial) to 1.0 (very complex).

    Factors:
    - Prompt length (longer = more complex)
    - Multi-step instructions detected
    - Tool call count (more tools = more complex)
    - Session length (longer sessions = more complex)
    - Code blocks present
    """
    if not prompt:
        return 0.0

    text = prompt.lower()
    score = 0.0

    # Prompt length factor (up to 0.3)
    length_score = min(len(prompt) / 2000, 0.3)
    score += length_score

    # Multi-step instruction patterns (up to 0.2)
    for pattern in COMPLEXITY_INSTRUCTION_PATTERNS:
        if re.search(pattern, text):
            score += 0.1
            break

    # Tool call count factor (up to 0.2)
    if tool_call_count > 10:
        score += 0.2
    elif tool_call_count > 5:
        score += 0.15
    elif tool_call_count > 2:
        score += 0.1

    # Session length factor (up to 0.15)
    if message_count > 50:
        score += 0.15
    elif message_count > 20:
        score += 0.1
    elif message_count > 10:
        score += 0.05

    # Code blocks (up to 0.15)
    if contains_code_blocks(prompt):
        score += 0.15

    return min(score, 1.0)


def count_instructions(prompt: str) -> int:
    """Count explicit instructions/requirements in a prompt."""
    if not prompt:
        return 0
    count = 0
    text = prompt.lower()

    # Count numbered/bullet items
    count += len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+", prompt))

    # Count imperative sentences (starts with verb-like word)
    count += len(re.findall(r"(?:^|[.!?\n])\s*(?:[A-Z][a-z]+)\s+(?:the|a|an|this|that|your|my|me|it|us|them|him|her)\s+", prompt))

    # Count "need", "must", "should", "require" patterns
    count += len(re.findall(r"\b(?:need|must|should|require|ensure|make sure|don't forget)\b", text))

    # Count sentences with action verbs at start (looser match)
    count += len(re.findall(r"(?:^|[.!?\n])\s*(?:Do|Make|Create|Write|Build|Find|Get|Run|Add|Remove|Update|Change|Fix|Check|Test|Use|Set|Configure|Install|Deploy|Start|Stop|Enable|Disable)\s", prompt))

    return count


def has_format_constraint(prompt: str) -> bool:
    """Check if prompt specifies output format constraints."""
    if not prompt:
        return False
    text = prompt.lower()
    return any(kw in text for kw in FORMAT_CONSTRAINTS)


def has_niche_references(prompt: str) -> bool:
    """Check if prompt references niche/domain-specific concepts."""
    if not prompt:
        return False
    text = prompt.lower()
    return any(kw in text for kw in NICHE_REFERENCES)


def is_correction_message(message_text: str) -> bool:
    """Detect if a user message is a correction of the previous assistant response."""
    if not message_text:
        return False
    text = message_text.strip()
    return any(re.match(pattern, text, re.IGNORECASE) for pattern in CORRECTION_PATTERNS)


def count_corrections(messages: List[Dict[str, Any]]) -> int:
    """Count the number of user corrections in a session's message history.

    A correction is a user message that follows an assistant message and
    matches correction patterns.
    """
    if not messages or len(messages) < 3:
        return 0

    count = 0
    for i in range(1, len(messages) - 1):
        prev = messages[i - 1]
        curr = messages[i]
        next_msg = messages[i + 1]

        # Pattern: user → assistant → user (correction)
        if (prev.get("role") == "user"
                and curr.get("role") == "assistant"
                and next_msg.get("role") == "user"):
            content = next_msg.get("content", "")
            if isinstance(content, str) and is_correction_message(content):
                count += 1

    return count


def compute_model_fit(
    model_used: str,
    complexity_score: float,
    user_correction_count: int,
    success: bool,
) -> Dict[str, Any]:
    """Compute model fit metrics for a session.

    Returns:
        Dict with:
        - model_tier: capability tier of the model used
        - min_tier_needed: estimated minimum tier needed for this complexity
        - overkill: whether a much more capable model was used than needed
        - underpowered: whether the model was likely too weak
    """
    tier = MODEL_CAPABILITY_TIERS.get(model_used, 5)

    # Estimate minimum tier needed from complexity
    if complexity_score < 0.2:
        min_tier = 1  # local 8b is fine
    elif complexity_score < 0.4:
        min_tier = 3  # flash is fine
    elif complexity_score < 0.6:
        min_tier = 5  # glm-5.1 level
    elif complexity_score < 0.8:
        min_tier = 7  # deepseek-v4-pro level
    else:
        min_tier = 9  # need top tier

    # Adjust for corrections (signal we underpowered)
    if user_correction_count > 0 and success:
        min_tier = min(min_tier + user_correction_count, 10)

    overkill = (tier - min_tier) >= 4
    underpowered = tier < min_tier

    return {
        "model_tier": tier,
        "min_tier_needed": min_tier,
        "overkill": overkill,
        "underpowered": underpowered,
    }


# ── Main extraction function ─────────────────────────────────────────────────

def extract_features(
    session_row: Dict[str, Any],
    usage_rows: List[Dict[str, Any]],
    first_message_text: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
    parent_session: Optional[Dict[str, Any]] = None,
    delegation_depth: int = 0,
) -> Optional[RouterLog]:
    """Build a RouterLog from a Hermes session row and its model_usage records.

    Args:
        session_row: A row from the sessions table.
        usage_rows: Rows from session_model_usage for this session.
        first_message_text: The text of the first user message.
        messages: All messages in the session (for correction detection).
        parent_session: The parent session row (if this is a subagent).
        delegation_depth: How deep in the delegation chain.

    Returns:
        A RouterLog if data is sufficient, None if the session has no usage data.
    """
    if not usage_rows:
        return None

    # Aggregate usage across all model calls in this session
    total_input = sum(u.get("input_tokens", 0) or 0 for u in usage_rows)
    total_output = sum(u.get("output_tokens", 0) or 0 for u in usage_rows)
    total_cost = sum(u.get("estimated_cost_usd", 0) or 0 for u in usage_rows)

    # Use the primary model from the session
    model_used = session_row.get("model", "") or (usage_rows[0].get("model", "") if usage_rows else "")
    provider = session_row.get("billing_provider", "") or (usage_rows[0].get("billing_provider", "") if usage_rows else "")

    # Session-level metrics
    tool_call_count = session_row.get("tool_call_count", 0) or 0
    message_count = session_row.get("message_count", 0) or 0

    # Compute latency from session timing
    started_at = session_row.get("started_at")
    ended_at = session_row.get("ended_at")
    latency = 0.0
    if started_at and ended_at:
        latency = float(ended_at) - float(started_at)

    # Task classification
    task_type = classify_task(
        prompt=first_message_text,
        tool_call_count=tool_call_count,
        session_model=model_used,
    )

    # Feature extraction
    prompt_length = len(first_message_text) if first_message_text else 0
    code_blocks = contains_code_blocks(first_message_text)
    keywords = has_router_keywords(first_message_text)

    # Context length estimate (rough: message_count * avg message size)
    context_length = message_count * 500

    # Determine success (session completed without error)
    end_reason = session_row.get("end_reason", "")
    success = end_reason not in ("error", "cancelled", "interrupted") if end_reason else True

    # ── Model fit signals ──
    complexity = score_complexity(first_message_text, tool_call_count, message_count)
    instr_count = count_instructions(first_message_text)
    fmt_constraint = has_format_constraint(first_message_text)
    niche_refs = has_niche_references(first_message_text)

    # Subagent detection
    parent_id = session_row.get("parent_session_id", "") or ""
    is_sub = bool(parent_id)
    parent_model = parent_session.get("model", "") if parent_session else ""

    # Correction detection
    correction_count = count_corrections(messages or [])

    # Model switch detection — check if the model changed mid-session
    model_switched = False
    if usage_rows:
        models_used = set(u.get("model", "") for u in usage_rows if u.get("model"))
        model_switched = len(models_used) > 1

    # Timestamp
    timestamp = ""
    if started_at:
        from datetime import datetime, timezone
        timestamp = datetime.fromtimestamp(float(started_at), tz=timezone.utc).isoformat()

    return RouterLog(
        session_id=session_row.get("id", ""),
        model_used=model_used,
        provider=provider,
        task_type=task_type,
        prompt_length=prompt_length,
        context_length=context_length,
        tool_call_count=tool_call_count,
        contains_code_blocks=code_blocks,
        has_keywords=keywords,
        input_tokens=total_input,
        output_tokens=total_output,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        success=success,
        retry_count=0,
        escalated=False,
        user_corrected=correction_count > 0,
        error_type=None if success else end_reason,
        timestamp=timestamp,
        # New fit signals
        complexity_score=round(complexity, 4),
        instruction_count=instr_count,
        has_format_constraint=fmt_constraint,
        has_niche_references=niche_refs,
        is_subagent=is_sub,
        parent_model=parent_model,
        delegation_depth=delegation_depth,
        user_correction_count=correction_count,
        model_switched=model_switched,
        session_message_count=message_count,
    )
