#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-audio_turn_detector}"
CONDA_BIN="${CONDA_BIN:-conda}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_DIR}/.conda/${ENV_NAME}}"

source "${PROJECT_DIR}/scripts/env.sh"
load_env_file "${PROJECT_DIR}/.env"

FUNASR_HOST="${FUNASR_HOST:-127.0.0.1}"
FUNASR_PORT="${FUNASR_PORT:-10095}"
FUNASR_DEVICE="${FUNASR_DEVICE:-cpu}"
FUNASR_NGPU="${FUNASR_NGPU:-0}"
FUNASR_NCPU="${FUNASR_NCPU:-4}"

cd "${REPO_ROOT}/asr_module/funasr"

exec "${CONDA_BIN}" run --no-capture-output -p "${ENV_PREFIX}" \
  python funasr_2pass_server.py \
    --host "${FUNASR_HOST}" \
    --port "${FUNASR_PORT}" \
    --device "${FUNASR_DEVICE}" \
    --ngpu "${FUNASR_NGPU}" \
    --ncpu "${FUNASR_NCPU}" \
    "$@"
