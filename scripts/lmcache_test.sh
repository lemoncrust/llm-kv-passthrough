#!/usr/bin/env bash
# Real-manager validation with small temporary tiers and a process memory cap.
set -euo pipefail
exec bash scripts/test-safe.sh tests/integration -m integration "$@"
