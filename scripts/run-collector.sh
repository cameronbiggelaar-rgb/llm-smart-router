#!/usr/bin/env bash
# LLM Smart Router — Data Collection Cron Wrapper
# Runs the collector silently, only reports on errors.
set -euo pipefail

cd "$HOME/.hermes/skills/llm-smart-router/scripts"

# Run collector, capture output
OUTPUT=$(python3 collector.py 2>&1) || {
    echo "❌ LLM Router collector failed: $OUTPUT"
    exit 1
}

echo "✅ Router data collected: $OUTPUT"
