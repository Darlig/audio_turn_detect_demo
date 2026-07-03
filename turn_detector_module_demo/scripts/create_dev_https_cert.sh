#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="${PROJECT_DIR}/certs"
HOST_IP="${1:-${DEMO_HTTPS_IP:-}}"

if [[ -z "${HOST_IP}" ]]; then
  HOST_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}')"
fi

if [[ -z "${HOST_IP}" ]]; then
  echo "Could not auto-detect LAN IP. Usage: $0 192.168.0.179" >&2
  exit 1
fi

mkdir -p "${CERT_DIR}"

openssl req \
  -x509 \
  -newkey rsa:2048 \
  -sha256 \
  -days 365 \
  -nodes \
  -keyout "${CERT_DIR}/dev-key.pem" \
  -out "${CERT_DIR}/dev-cert.pem" \
  -subj "/CN=${HOST_IP}" \
  -addext "subjectAltName=IP:${HOST_IP},IP:127.0.0.1,DNS:localhost"

chmod 600 "${CERT_DIR}/dev-key.pem"

cat <<EOF
Created:
  ${CERT_DIR}/dev-cert.pem
  ${CERT_DIR}/dev-key.pem

Start HTTPS demo with:
  bash scripts/start_demo.sh --host 0.0.0.0 --port 8090 \\
    --ssl-cert-file ${CERT_DIR}/dev-cert.pem \\
    --ssl-key-file ${CERT_DIR}/dev-key.pem
EOF
