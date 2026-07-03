#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/env.sh"
load_env_file "${PROJECT_DIR}/.env"

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
DEMO_HOST="${DEMO_HOST:-0.0.0.0}"
DEMO_PORT="${DEMO_PORT:-8090}"
DEMO_SSL_CERT_FILE="${DEMO_SSL_CERT_FILE:-}"
DEMO_SSL_KEY_FILE="${DEMO_SSL_KEY_FILE:-}"

if [[ -z "${DEMO_SSL_CERT_FILE}" && -z "${DEMO_SSL_KEY_FILE}" ]]; then
  default_cert="${PROJECT_DIR}/certs/dev-cert.pem"
  default_key="${PROJECT_DIR}/certs/dev-key.pem"
  if [[ -f "${default_cert}" && -f "${default_key}" ]]; then
    DEMO_SSL_CERT_FILE="${default_cert}"
    DEMO_SSL_KEY_FILE="${default_key}"
  fi
fi

if [[ -n "${DEMO_SSL_CERT_FILE}" || -n "${DEMO_SSL_KEY_FILE}" ]]; then
  if [[ -z "${DEMO_SSL_CERT_FILE}" || -z "${DEMO_SSL_KEY_FILE}" ]]; then
    echo "Both DEMO_SSL_CERT_FILE and DEMO_SSL_KEY_FILE are required for HTTPS." >&2
    exit 1
  fi
  DEMO_SCHEME="https"
else
  DEMO_SCHEME="http"
fi

mkdir -p "${LOG_DIR}"

pids=()

cleanup() {
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_bg() {
  local name="$1"
  shift
  echo "starting ${name}; log=${LOG_DIR}/${name}.log"
  "$@" >"${LOG_DIR}/${name}.log" 2>&1 &
  pids+=("$!")
}

start_bg livekit "${PROJECT_DIR}/scripts/start_livekit_server.sh"
sleep "${LIVEKIT_START_DELAY:-1}"

start_bg funasr "${PROJECT_DIR}/scripts/start_funasr_server.sh"
sleep "${FUNASR_START_DELAY:-3}"

web_args=(--host "${DEMO_HOST}" --port "${DEMO_PORT}")
if [[ "${DEMO_SCHEME}" == "https" ]]; then
  web_args+=(--ssl-cert-file "${DEMO_SSL_CERT_FILE}" --ssl-key-file "${DEMO_SSL_KEY_FILE}")
fi
start_bg web "${PROJECT_DIR}/scripts/start_demo.sh" "${web_args[@]}"
sleep "${WEB_START_DELAY:-1}"

start_bg voice-agent "${PROJECT_DIR}/scripts/start_voice_agent.sh" dev

echo "stack started"
echo "web: ${DEMO_SCHEME}://${DEMO_HOST}:${DEMO_PORT}"
echo "logs: ${LOG_DIR}"

wait -n "${pids[@]}"
