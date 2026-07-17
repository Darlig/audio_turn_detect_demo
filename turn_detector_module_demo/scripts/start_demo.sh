#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-audio_turn_detector}"
CONDA_BIN="${CONDA_BIN:-conda}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_DIR}/.conda/${ENV_NAME}}"

source "${PROJECT_DIR}/scripts/env.sh"
load_env_file "${PROJECT_DIR}/.env"
configure_ca_bundle

cd "${PROJECT_DIR}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
export no_proxy="${no_proxy:-127.0.0.1,localhost,::1}"
EFFECTIVE_LIVEKIT_URL="${LIVEKIT_URL:-ws://127.0.0.1:8890}"
if [[ "${EFFECTIVE_LIVEKIT_URL}" == ws://127.0.0.1* || "${EFFECTIVE_LIVEKIT_URL}" == ws://localhost* ]]; then
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
fi
exec "${CONDA_BIN}" run --no-capture-output -p "${ENV_PREFIX}" turn-detector-module-demo "$@"
