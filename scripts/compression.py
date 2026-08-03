"""Prompt compression module for Biggie LLM Endpoint.

Implements RTK-style command-aware filtering and Caveman-style
filler removal, adapted for Hermes agent sessions.

Compression levels:
  - off:      No compression
  - lite:     Whitespace cleanup, ANSI removal (~15% savings)
  - standard: Lite + filler removal, phrase condensing (~30% savings)
  - aggressive: Standard + tool output filtering, dedup (~50% savings)

All levels preserve semantic meaning — only remove noise.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

# ── Configuration ─────────────────────────────────────────────────────────────

# Default compression level (can be overridden via env var or request header)
DEFAULT_LEVEL = "standard"

# ── ANSI / Control sequence removal ──────────────────────────────────────────

_ANSI_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_PROGRESS_BAR_PATTERN = re.compile(
    r"[─━═▬▭▮▯█▓▒░■□▪▫▸▹►▻◄◅◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯⏣⏤⏥⏦⏧⏨⏩⏪⏫⏬⏭⏮⏯⏰⏱⏲⏳⏴⏵⏶⏷⏸⏹⏺]"
    r"|\d+%|[\|\/\\\-]\s*$",
    re.MULTILINE,
)

# ── Filler words / verbose phrases ──────────────────────────────────────────

_FILLER_WORDS = re.compile(
    r"\b(?:"
    r"please|kindly|basically|actually|essentially|honestly|literally|"
    r"simply|just\s(?:want|need|like|wanted|needed)|"
    r"in\s+order\s+to|as\s+a\s+result\s+of|due\s+to\s+the\s+fact\s+that|"
    r"at\s+this\s+point\s+in\s+time|in\s+the\s+event\s+that|"
    r"in\s+the\s+process\s+of|for\s+the\s+purpose\s+of|"
    r"with\s+regard\s+to|on\s+a\s+regular\s+basis|"
    r"would\s+you\s+mind|if\s+you\s+could\s+possibly|"
    r"i\s+(?:think|believe|feel|would\s+say)\s+that"
    r")\b",
    re.IGNORECASE,
)

# ── Command output patterns (RTK-style) ─────────────────────────────────────

_COMMAND_PATTERNS = {
    "git_status": re.compile(r"^\s*(?:M\s|A\s|\?\?|D\s|R\s|C\s|UU|AA|MM)\s+", re.MULTILINE),
    "git_diff_header": re.compile(r"^diff --git a/.* b/.*$", re.MULTILINE),
    "git_diff_hunk": re.compile(r"^@@ -\d+,\d+ \+\d+,\d+ @@", re.MULTILINE),
    "npm_install": re.compile(r"^npm\s+(?:install|i|add)\s", re.MULTILINE),
    "npm_audit": re.compile(r"^npm\s+audit", re.MULTILINE),
    "test_runner": re.compile(r"^(?:PASS|FAIL|PASSED|FAILED|✓|✗|√|×|ok\s+\d+|not\s+ok\s+\d+)", re.MULTILINE),
    "test_suite": re.compile(r"^(?:Test\s+Suites|Tests:|Suites:|Tests\s+run)", re.MULTILINE),
    "build_output": re.compile(r"^(?:\[.*?\]\s*)?(?:Building|Compiling|Bundling|Transforming|Processing)", re.MULTILINE),
    "eslint": re.compile(r"^\d+:\d+\s+(?:error|warning)\s+", re.MULTILINE),
    "docker_log": re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", re.MULTILINE),
    "shell_prompt": re.compile(r"^\$\s+", re.MULTILINE),
    "traceback": re.compile(r"^(?:Traceback|File\s+\".*?\",\s+line\s+\d+)", re.MULTILINE),
    "coverage": re.compile(r"^(?:Statements|Branches|Functions|Lines)\s*:", re.MULTILINE),
}

# Lines to always keep (errors, warnings, summaries)
_KEEP_PATTERNS = [
    re.compile(r"(?:error|Error|ERROR|fail|Fail|FAIL)", re.MULTILINE),
    re.compile(r"(?:warning|Warning|WARNING)", re.MULTILINE),
    re.compile(r"(?:summary|Summary|SUMMARY)", re.MULTILINE),
    re.compile(r"(?:changed|insertion|deletion)", re.MULTILINE),
    re.compile(r"(?:^\d+\s+(?:passing|failing|pending|skipped))", re.MULTILINE),
    re.compile(r"(?:exit\s+code|Exit\s+code)", re.MULTILINE),
]

# ── Compression functions ────────────────────────────────────────────────────


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (color codes, cursor movements, etc.)."""
    return _ANSI_PATTERN.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse multiple blank lines and trailing spaces."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def remove_filler(text: str) -> str:
    """Remove filler words and condense verbose phrases."""
    text = _FILLER_WORDS.sub("", text)
    # Clean up double spaces left by removal
    text = re.sub(r"  +", " ", text)
    return text


def remove_progress_bars(text: str) -> str:
    """Remove progress bar characters and percentage indicators."""
    return _PROGRESS_BAR_PATTERN.sub("", text)


def deduplicate_lines(text: str) -> str:
    """Remove consecutive duplicate lines (common in build/test output)."""
    lines = text.split("\n")
    result = []
    prev = ""
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev:
            continue
        result.append(line)
        prev = stripped
    return "\n".join(result)


def filter_tool_output(text: str) -> str:
    """Filter verbose tool output — keep errors, warnings, summaries.

    For each block of tool output, keeps:
    - Lines matching error/warning patterns
    - Summary lines
    - First and last few lines of each block
    """
    lines = text.split("\n")
    result = []
    in_tool_block = False
    block_lines = []
    block_has_error = False

    for line in lines:
        stripped = line.strip()

        # Detect start of tool output block
        if any(p.match(stripped) for p in _KEEP_PATTERNS):
            block_has_error = True

        # Check if this is a command output line
        is_command = any(p.search(stripped) for p in _COMMAND_PATTERNS.values())

        if is_command or (stripped and not stripped.startswith("#") and not stripped.startswith("//")):
            if not in_tool_block:
                in_tool_block = True
                block_lines = [line]
                block_has_error = bool(
                    any(p.search(stripped) for p in _KEEP_PATTERNS)
                )
            else:
                block_lines.append(line)
        else:
            if in_tool_block:
                # Flush the block
                if block_has_error or len(block_lines) <= 5:
                    result.extend(block_lines)
                else:
                    # Keep first 2 and last 2 lines
                    result.extend(block_lines[:2])
                    result.append(f"  ... ({len(block_lines) - 4} lines filtered by RTK)")
                    result.extend(block_lines[-2:])
                in_tool_block = False
                block_lines = []
                block_has_error = False
            result.append(line)

    # Flush remaining block
    if in_tool_block:
        if block_has_error or len(block_lines) <= 5:
            result.extend(block_lines)
        else:
            result.extend(block_lines[:2])
            result.append(f"  ... ({len(block_lines) - 4} lines filtered by RTK)")
            result.extend(block_lines[-2:])

    return "\n".join(result)


# ── Main compression pipeline ────────────────────────────────────────────────


def compress_messages(
    messages: List[Dict],
    level: str = DEFAULT_LEVEL,
) -> Tuple[List[Dict], Dict]:
    """Compress a list of chat messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        level: Compression level ('off', 'lite', 'standard', 'aggressive')

    Returns:
        Tuple of (compressed_messages, stats_dict)
    """
    if level == "off":
        return messages, {
            "level": "off",
            "input_chars": sum(len(m.get("content", "")) for m in messages if isinstance(m.get("content"), str)),
            "output_chars": 0,
            "savings_pct": 0.0,
            "compression_time_ms": 0.0,
        }

    t0 = time.time()
    input_chars = 0
    output_chars = 0
    compressed = []

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            compressed.append(msg)
            continue

        input_chars += len(content)
        text = content

        # Lite: always applied
        text = strip_ansi(text)
        text = collapse_whitespace(text)
        text = remove_progress_bars(text)

        # Standard: add filler removal
        if level in ("standard", "aggressive"):
            text = remove_filler(text)

        # Aggressive: add tool output filtering and dedup
        if level == "aggressive":
            text = filter_tool_output(text)
            text = deduplicate_lines(text)

        new_msg = dict(msg)
        new_msg["content"] = text
        compressed.append(new_msg)
        output_chars += len(text)

    elapsed = (time.time() - t0) * 1000
    savings = 0.0
    if input_chars > 0:
        savings = (1 - output_chars / input_chars) * 100

    stats = {
        "level": level,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "savings_pct": round(savings, 1),
        "compression_time_ms": round(elapsed, 2),
    }

    return compressed, stats
