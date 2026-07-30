#!/usr/bin/env bash
# Limp-home status checker for cron jobs and shell scripts.
# Exits 0 if normal, 1 if limp-home is active.
# With --message, prints a human-readable status.
set -euo pipefail

SIGNAL_FILE="$HOME/.hermes/data/limp_home_status.json"

if [ ! -f "$SIGNAL_FILE" ]; then
    # No signal file = not in limp-home
    [ "${1:-}" = "--message" ] && echo "🟢 Normal operation"
    exit 0
fi

ACTIVE=$(python3 -c "import json; d=json.load(open('$SIGNAL_FILE')); print('true' if d.get('active') else 'false')" 2>/dev/null || echo "false")

if [ "$ACTIVE" = "true" ]; then
    if [ "${1:-}" = "--message" ]; then
        DURATION=$(python3 -c "import json; d=json.load(open('$SIGNAL_FILE')); print(int(d.get('duration_seconds', 0)))" 2>/dev/null || echo "0")
        echo "⚠️  Limp-home active — cloud models exhausted, running on local only (${DURATION}s)"
    fi
    exit 1
else
    [ "${1:-}" = "--message" ] && echo "🟢 Normal operation"
    exit 0
fi
