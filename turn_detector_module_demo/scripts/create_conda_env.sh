#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-audio_turn_detector}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_BIN="${CONDA_BIN:-conda}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_DIR}/.conda/${ENV_NAME}}"

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "conda env already exists: ${ENV_PREFIX}"
else
  "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi

"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -m pip install -U pip
"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -m pip install -e "${PROJECT_DIR}[dev,voice]"

echo "ready: conda activate ${ENV_PREFIX}"
