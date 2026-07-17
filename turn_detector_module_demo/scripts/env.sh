#!/usr/bin/env bash

load_env_file() {
  local env_file="$1"
  [[ -f "${env_file}" ]] || return 0

  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue

    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi

    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -z "${!key+x}" ]] || continue
    export "${key}=${value}"
  done <"${env_file}"
}

configure_ca_bundle() {
  if [[ -n "${SSL_CERT_FILE:-}" && ! -r "${SSL_CERT_FILE}" ]]; then
    echo "CA bundle is not readable: ${SSL_CERT_FILE}" >&2
    return 1
  fi
  if [[ -n "${REQUESTS_CA_BUNDLE:-}" && ! -r "${REQUESTS_CA_BUNDLE}" ]]; then
    echo "Requests CA bundle is not readable: ${REQUESTS_CA_BUNDLE}" >&2
    return 1
  fi
  if [[ -n "${SSL_CERT_FILE:-}" && -z "${REQUESTS_CA_BUNDLE:-}" ]]; then
    export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-${SSL_CERT_FILE}}"
  fi
}
