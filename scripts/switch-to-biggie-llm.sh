#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Switch: Hermes → Biggie LLM Endpoint
# ═══════════════════════════════════════════════════════════════════════════════
# Creates a backup of config.yaml, then adds the custom_providers section
# and switches the default model to biggie-router.
#
# Rollback: bash rollback-biggie-llm.sh --apply
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

HERMES_CONFIG="$HOME/.hermes/config.yaml"
BACKUP_DIR="$HOME/.hermes/config-backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_PATH="$BACKUP_DIR/config.yaml.backup.$TIMESTAMP"

echo "══════════════════════════════════════════════════════════════════"
echo " Biggie LLM Endpoint — Switch Over"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Create backup ─────────────────────────────────────────────────────

echo "─── Step 1: Backup current config ───"
mkdir -p "$BACKUP_DIR"
cp "$HERMES_CONFIG" "$BACKUP_PATH"
echo "   ✅ Backup created: $BACKUP_PATH"
echo ""

# ── Step 2: Verify the endpoint is running ────────────────────────────────────

echo "─── Step 2: Verify Biggie LLM Endpoint is running ───"
if curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "   ✅ Endpoint is running on 127.0.0.1:8080"
else
    echo "   ❌ Endpoint is NOT running!"
    echo "   Start it with: sudo systemctl start biggie-llm-endpoint"
    echo "   Then re-run this script."
    exit 1
fi
echo ""

# ── Step 3: Add custom_providers and switch model ─────────────────────────────

echo "─── Step 3: Update config.yaml ───"

# Check if custom_providers already exists
if grep -q "custom_providers:" "$HERMES_CONFIG" 2>/dev/null; then
    echo "   ⚠️  custom_providers already exists in config — merging"
fi

# Check if biggie-llm is already configured
if grep -q "biggie-llm" "$HERMES_CONFIG" 2>/dev/null; then
    echo "   ⚠️  biggie-llm already configured — skipping provider addition"
else
    # Add custom_providers section before the model section
    # Using a temp file approach for safety
    python3 -c "
import yaml

with open('$HERMES_CONFIG') as f:
    config = yaml.safe_load(f) or {}

# Add custom_providers
if 'custom_providers' not in config:
    config['custom_providers'] = {}

config['custom_providers']['biggie-llm'] = {
    'base_url': 'http://127.0.0.1:8080/v1',
    'api_key': '',
}

# Switch model to biggie-router
config['model'] = {
    'default': 'biggie-router',
    'provider': 'biggie-llm',
}

with open('$HERMES_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print('   ✅ custom_providers added and model switched to biggie-router')
"
fi
echo ""

# ── Step 4: Verify the change ────────────────────────────────────────────────

echo "─── Step 4: Verify ───"
if grep -q "biggie-llm" "$HERMES_CONFIG" 2>/dev/null; then
    echo "   ✅ biggie-llm provider configured"
else
    echo "   ❌ biggie-llm not found in config — something went wrong"
    echo "   Rollback: bash $HOME/.hermes/skills/llm-smart-router/scripts/rollback-biggie-llm.sh --apply"
    exit 1
fi

if grep -q "biggie-router" "$HERMES_CONFIG" 2>/dev/null; then
    echo "   ✅ Default model set to biggie-router"
else
    echo "   ⚠️  Default model may not be biggie-router — check config"
fi

echo ""

# ── Summary ───────────────────────────────────────────────────────────────────

echo "══════════════════════════════════════════════════════════════════"
echo " Switch complete!"
echo ""
echo " Hermes is now pointing to the Biggie LLM Endpoint."
echo " The endpoint will route each request to the cheapest capable model."
echo ""
echo " To roll back at any time:"
echo "   bash $HOME/.hermes/skills/llm-smart-router/scripts/rollback-biggie-llm.sh --apply"
echo ""
echo " Backup saved at:"
echo "   $BACKUP_PATH"
echo "══════════════════════════════════════════════════════════════════"
