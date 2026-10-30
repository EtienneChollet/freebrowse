#!/usr/bin/env bash
# Serve the built application with source scans read-only and isolated AI sessions.
set -euo pipefail

freebrowse_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${DATA_DIR:?Set DATA_DIR to the directory containing the source scan}"
export NIIVUE_BUILD_DIR="${NIIVUE_BUILD_DIR:-$freebrowse_root/frontend/dist}"
export AI_DIR="${AI_DIR:-$freebrowse_root/.runtime/ai-sessions}"
export MODELS_DIR="${MODELS_DIR:-$freebrowse_root/models}"
export ENABLE_AI=true ENABLE_AI_HISTORY=true ENABLE_QA=false
export SERVERLESS_MODE=false READ_ONLY_DATA=true
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$freebrowse_root/.runtime/matplotlib}"
freebrowse_python="${FREEBROWSE_PYTHON:-$freebrowse_root/backend/.venv/bin/python}"

mkdir -p "$AI_DIR" "$MPLCONFIGDIR"
test -f "$NIIVUE_BUILD_DIR/index.html"
test -x "$freebrowse_python"
exec "$freebrowse_python" -m uvicorn server:app \
  --app-dir "$freebrowse_root/backend/src" \
  --host "${FREEBROWSE_HOST:-127.0.0.1}" --port "${FREEBROWSE_PORT:-8102}" --workers 1
