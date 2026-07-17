#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/env.sh"
load_env_file "${PROJECT_DIR}/.env"

CERT_FILE="${DEMO_SSL_CERT_FILE:-${PROJECT_DIR}/certs/dev-cert.pem}"
KEY_FILE="${DEMO_SSL_KEY_FILE:-${PROJECT_DIR}/certs/dev-key.pem}"
HOST_IP="${1:-${DEMO_HTTPS_IP:-${LIVEKIT_NODE_IP:-}}}"

if [[ -z "${HOST_IP}" ]]; then
  HOST_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}' || true)"
fi

if [[ -z "${HOST_IP}" ]]; then
  echo "Could not auto-detect LAN IP. Usage: $0 LAN_IP" >&2
  exit 1
fi

mkdir -p "$(dirname "${CERT_FILE}")" "$(dirname "${KEY_FILE}")"

openssl req \
  -x509 \
  -newkey rsa:2048 \
  -sha256 \
  -days 365 \
  -nodes \
  -keyout "${KEY_FILE}" \
  -out "${CERT_FILE}" \
  -subj "/CN=${HOST_IP}" \
  -addext "subjectAltName=IP:${HOST_IP},IP:127.0.0.1,DNS:localhost"

chmod 600 "${KEY_FILE}"

cat <<EOF
Created:
  ${CERT_FILE}
  ${KEY_FILE}

Start HTTPS demo with:
  bash scripts/start_demo.sh --host 0.0.0.0 --port 8090 \\
    --ssl-cert-file ${CERT_FILE} \\
    --ssl-key-file ${KEY_FILE}
EOF
