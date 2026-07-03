#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/env.sh"
load_env_file "${PROJECT_DIR}/.env"

DEFAULT_BIN="${PROJECT_DIR}/../../turn_detect_demo/bin/livekit-server"
LIVEKIT_SERVER_BIN="${LIVEKIT_SERVER_BIN:-${DEFAULT_BIN}}"
LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
LIVEKIT_BIND="${LIVEKIT_BIND:-0.0.0.0}"
LIVEKIT_PORT="${LIVEKIT_PORT:-8890}"
LIVEKIT_UDP_PORT="${LIVEKIT_UDP_PORT:-8892}"
LIVEKIT_RTC_TCP_PORT="${LIVEKIT_RTC_TCP_PORT:-8891}"
LIVEKIT_NODE_IP="${LIVEKIT_NODE_IP:-${NODE_IP:-}}"
LIVEKIT_USE_EXTERNAL_IP="${LIVEKIT_USE_EXTERNAL_IP:-false}"

if [[ ! -x "${LIVEKIT_SERVER_BIN}" ]]; then
  echo "livekit-server not found or not executable: ${LIVEKIT_SERVER_BIN}" >&2
  echo "Set LIVEKIT_SERVER_BIN=/path/to/livekit-server." >&2
  exit 1
fi

if [[ -z "${LIVEKIT_NODE_IP}" ]]; then
  LIVEKIT_NODE_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}')"
fi

args=(
  --dev
  --bind "${LIVEKIT_BIND}"
  --port "${LIVEKIT_PORT}"
  --udp-port "${LIVEKIT_UDP_PORT}"
  --rtc.tcp_port "${LIVEKIT_RTC_TCP_PORT}"
  --keys "${LIVEKIT_API_KEY}: ${LIVEKIT_API_SECRET}"
)

if [[ -n "${LIVEKIT_NODE_IP}" ]]; then
  args+=(--node-ip "${LIVEKIT_NODE_IP}" --rtc.node_ip.ipv4 "${LIVEKIT_NODE_IP}")
  echo "LiveKit advertising node IP: ${LIVEKIT_NODE_IP}"
else
  echo "LiveKit node IP auto-detection failed; set LIVEKIT_NODE_IP=LAN_IP for LAN clients." >&2
fi

if [[ "${LIVEKIT_USE_EXTERNAL_IP}" == "true" ]]; then
  args+=(--rtc.use_external_ip)
fi

exec "${LIVEKIT_SERVER_BIN}" "${args[@]}"
