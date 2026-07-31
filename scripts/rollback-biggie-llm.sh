#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Rollback: Biggie LLM Endpoint → Original Hermes config
# ═══════════════════════════════════════════════════════════════════════════════
# Run this to undo the switch. It:
#   1. Restores the original config.yaml from backup
#   2. Stops and disables the biggie-llm-endpoint service
#   3. Verifies Hermes can reach the original provider
#
# Usage:
#   bash rollback-biggie-llm.sh          # dry-run — shows what would happen
#   bash rollback-biggie-llm.sh --apply  # actually rolls back
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

HERMES_CONFIG="$HOME/.hermes/config.yaml"
BACKUP_DIR="$HOME/.hermes/config-backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DRY_RUN=true

# Parse args
if [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN=false
fi

echo "══════════════════════════════════════════════════════════════════"
echo " Biggie LLM Endpoint — Rollback"
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo " Mode:        $($DRY_RUN && echo 'DRY-RUN (no changes)' || echo 'APPLY')"
echo " Config:      $HERMES_CONFIG"
echo " Backup dir:  $BACKUP_DIR"
echo " Timestamp:   $TIMESTAMP"
echo ""

# ── Step 1: Find the most recent backup ──────────────────────────────────────

LATEST_BACKUP=""
if [[ -d "$BACKUP_DIR" ]]; then
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/config.yaml.backup.* 2>/dev/null | head -1)
fi

if [[ -z "$LATEST_BACKUP" ]]; then
    echo "⚠️  No backup found in $BACKUP_DIR"
    echo "   Checking if config.yaml has the biggie-llm custom_providers section..."
    if grep -q "biggie-llm" "$HERMES_CONFIG" 2>/dev/null; then
        echo "   ✅ biggie-llm section found — config was switched."
        echo ""
        echo "   ❌ Cannot roll back automatically — no backup to restore from."
        echo "   Manual fix: remove the 'custom_providers' section and restore"
        echo "   model.default to 'gpt-5.5' and model.provider to 'openai-codex'."
        exit 1
    else
        echo "   ✅ No biggie-llm section found — config is already rolled back or was never switched."
        exit 0
    fi
fi

echo " Found backup: $(basename "$LATEST_BACKUP")"
echo " Created:      $(stat -c '%y' "$LATEST_BACKUP" 2>/dev/null || stat -f '%Sm' "$LATEST_BACKUP" 2>/dev/null)"
echo ""

# ── Step 2: Restore config.yaml ──────────────────────────────────────────────

echo "─── Step 1: Restore config.yaml ───"
if $DRY_RUN; then
    echo "   Would: cp '$LATEST_BACKUP' '$HERMES_CONFIG'"
else
    cp "$LATEST_BACKUP" "$HERMES_CONFIG"
    echo "   ✅ Restored config.yaml from backup"
fi

# ── Step 3: Stop and disable the service ──────────────────────────────────────

echo ""
echo "─── Step 2: Stop biggie-llm-endpoint service ───"

if systemctl is-active --quiet biggie-llm-endpoint 2>/dev/null; then
    if $DRY_RUN; then
        echo "   Would: sudo systemctl stop biggie-llm-endpoint"
        echo "   Would: sudo systemctl disable biggie-llm-endpoint"
    else
        sudo systemctl stop biggie-llm-endpoint
        sudo systemctl disable biggie-llm-endpoint
        echo "   ✅ Service stopped and disabled"
    fi
else
    echo "   ⏭️  Service not running — skipping"
fi

# ── Step 4: Verify ────────────────────────────────────────────────────────────

echo ""
echo "─── Step 3: Verify ───"

if $DRY_RUN; then
    echo "   Would check:"
    echo "     - Config no longer references biggie-llm"
    echo "     - Service is stopped"
    echo "     - Hermes can reach the original provider"
else
    # Check config
    if grep -q "biggie-llm" "$HERMES_CONFIG" 2>/dev/null; then
        echo "   ⚠️  Config still references biggie-llm — manual cleanup needed"
    else
        echo "   ✅ Config restored — no biggie-llm references"
    fi

    # Check service
    if systemctl is-active --quiet biggie-llm-endpoint 2>/dev/null; then
        echo "   ⚠️  Service still running — may need manual stop"
    else
        echo "   ✅ Service stopped"
    fi

    # Check the original model is set
    if grep -q "default: gpt-5.5" "$HERMES_CONFIG" 2>/dev/null; then
        echo "   ✅ Default model restored to gpt-5.5"
    else
        echo "   ⚠️  Default model may not be gpt-5.5 — check config"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════════════"
if $DRY_RUN; then
    echo " DRY-RUN complete. No changes made."
    echo ""
    echo " To actually roll back, run:"
    echo "   bash $0 --apply"
else
    echo " Rollback complete. Hermes is back to the original config."
    echo ""
    echo " To re-enable the Biggie LLM Endpoint later:"
    echo "   sudo systemctl enable --now biggie-llm-endpoint"
    echo "   # Then re-add custom_providers to config.yaml"
fi
echo "══════════════════════════════════════════════════════════════════"
