#!/usr/bin/env bash
# Limp-home status checker for cron jobs and shell scripts.
#
# Usage:
#   check-limp-home.sh              # Always exits 0 (non-LLM jobs run regardless)
#   check-limp-home.sh --needs-llm  # Exits 1 if limp-home is active (LLM jobs pause)
#   check-limp-home.sh --message    # Prints human-readable status, always exits 0
#   check-limp-home.sh --needs-llm --message  # Prints status, exits 1 if paused
#
# Non-LLM jobs (data processing, file ops, Python scripts) should NOT use
# --needs-llm — they run regardless of limp-home status.
# LLM-dependent jobs (batch audits, subagent chains, AI visibility checks)
# should use --needs-llm to pause gracefully when cloud is exhausted.
set -euo pipefail

SIGNAL_FILE="$HOME/.hermes/data/limp_home_status.json"
NEEDS_LLM=false
MESSAGE=false

for arg in "$@"; do
    case "$arg" in
        --needs-llm) NEEDS_LLM=true ;;
        --message) MESSAGE=true ;;
    esac
done

# Non-LLM jobs always run — limp-home doesn't affect them
if [ "$NEEDS_LLM" = false ]; then
    if [ "$MESSAGE" = true ]; then
        echo "🟢 Running (no LLM needed — limp-home irrelevant)"
    fi
    exit 0
fi

# LLM-dependent jobs check limp-home status
if [ ! -f "$SIGNAL_FILE" ]; then
    [ "$MESSAGE" = true ] && echo "🟢 Normal operation — cloud available"
    exit 0
fi

ACTIVE=$(python3 -c "import json; d=json.load(open('$SIGNAL_FILE')); print('true' if d.get('active') else 'false')" 2>/dev/null || echo "false")

if [ "$ACTIVE" = "true" ]; then
    if [ "$MESSAGE" = true ]; then
        DURATION=$(python3 -c "import json; d=json.load(open('$SIGNAL_FILE')); print(int(d.get('duration_seconds', 0)))" 2>/dev/null || echo "0")
        echo "⚠️  Limp-home active — cloud models exhausted (${DURATION}s) — pausing LLM-dependent work"
    fi
    exit 1
else
    [ "$MESSAGE" = true ] && echo "🟢 Normal operation — cloud available"
    exit 0
fi
