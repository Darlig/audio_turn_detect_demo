#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-audio_turn_detector}"
CONDA_BIN="${CONDA_BIN:-conda}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_DIR}/.conda/${ENV_NAME}}"

source "${PROJECT_DIR}/scripts/env.sh"
load_env_file "${PROJECT_DIR}/.env"

export LIVEKIT_URL="${LIVEKIT_URL:-ws://127.0.0.1:8890}"
export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
export LIVEKIT_AGENT_NAME="${LIVEKIT_AGENT_NAME:-cascade-voice-agent}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/asr_module:${REPO_ROOT}/llm_module:${REPO_ROOT}/tts_module:${REPO_ROOT}/agents/livekit-agents:${PYTHONPATH:-}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
export no_proxy="${no_proxy:-127.0.0.1,localhost,::1}"

if [[ "${LIVEKIT_URL}" == ws://127.0.0.1* || "${LIVEKIT_URL}" == ws://localhost* ]]; then
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
fi

cd "${PROJECT_DIR}"

if [[ $# -eq 0 ]]; then
  set -- dev
fi

exec "${CONDA_BIN}" run --no-capture-output -p "${ENV_PREFIX}" voice-agent "$@"
