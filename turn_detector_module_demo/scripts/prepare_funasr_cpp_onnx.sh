#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

source "${PROJECT_DIR}/scripts/env.sh"
load_env_file "${PROJECT_DIR}/.env"

FUNASR_CPP_ROOT="${FUNASR_CPP_ROOT:-${REPO_ROOT}/asr_module/funasr_cpp_onnx}"
FUNASR_CPP_REPO_URL="${FUNASR_CPP_REPO_URL:-https://github.com/modelscope/FunASR.git}"
FUNASR_CPP_REPO_REF="${FUNASR_CPP_REPO_REF:-main}"
FUNASR_CPP_UPSTREAM_DIR="${FUNASR_CPP_UPSTREAM_DIR:-${FUNASR_CPP_ROOT}/upstream/FunASR}"
FUNASR_CPP_DEPS_DIR="${FUNASR_CPP_DEPS_DIR:-${FUNASR_CPP_ROOT}/deps}"
FUNASR_CPP_BUILD_DIR="${FUNASR_CPP_BUILD_DIR:-${FUNASR_CPP_ROOT}/build/websocket}"
FUNASR_CPP_ONNXRUNTIME_URL="${FUNASR_CPP_ONNXRUNTIME_URL:-https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-linux-x64-1.14.0.tgz}"
FUNASR_CPP_FFMPEG_URL="${FUNASR_CPP_FFMPEG_URL:-https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-linux64-gpl-shared.tar.xz}"
FUNASR_CPP_ONNXRUNTIME_DIR="${FUNASR_CPP_ONNXRUNTIME_DIR:-${FUNASR_CPP_DEPS_DIR}/onnxruntime}"
FUNASR_CPP_FFMPEG_DIR="${FUNASR_CPP_FFMPEG_DIR:-${FUNASR_CPP_DEPS_DIR}/ffmpeg}"
FUNASR_CPP_ONNXRUNTIME_ARCHIVE="${FUNASR_CPP_ONNXRUNTIME_ARCHIVE:-${FUNASR_CPP_DEPS_DIR}/onnxruntime.tgz}"
FUNASR_CPP_FFMPEG_ARCHIVE="${FUNASR_CPP_FFMPEG_ARCHIVE:-${FUNASR_CPP_DEPS_DIR}/ffmpeg.tar.xz}"
FUNASR_CPP_BUILD_JOBS="${FUNASR_CPP_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"

mkdir -p "${FUNASR_CPP_ROOT}" "${FUNASR_CPP_DEPS_DIR}" "${FUNASR_CPP_BUILD_DIR}"

fetch_archive() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output}"
  else
    wget "${url}" -O "${output}"
  fi
}

if [[ ! -d "${FUNASR_CPP_UPSTREAM_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${FUNASR_CPP_REPO_REF}" "${FUNASR_CPP_REPO_URL}" "${FUNASR_CPP_UPSTREAM_DIR}"
else
  git -C "${FUNASR_CPP_UPSTREAM_DIR}" fetch --depth 1 origin "${FUNASR_CPP_REPO_REF}"
  git -C "${FUNASR_CPP_UPSTREAM_DIR}" checkout FETCH_HEAD
fi

fetch_archive "${FUNASR_CPP_ONNXRUNTIME_URL}" "${FUNASR_CPP_ONNXRUNTIME_ARCHIVE}"
fetch_archive "${FUNASR_CPP_FFMPEG_URL}" "${FUNASR_CPP_FFMPEG_ARCHIVE}"

if [[ ! -d "${FUNASR_CPP_ONNXRUNTIME_DIR}" ]]; then
  mkdir -p "${FUNASR_CPP_ONNXRUNTIME_DIR}"
  tar -xzf "${FUNASR_CPP_ONNXRUNTIME_ARCHIVE}" \
    -C "${FUNASR_CPP_ONNXRUNTIME_DIR}" --strip-components=1
fi

if [[ ! -d "${FUNASR_CPP_FFMPEG_DIR}" ]]; then
  mkdir -p "${FUNASR_CPP_FFMPEG_DIR}"
  tar -xf "${FUNASR_CPP_FFMPEG_ARCHIVE}" \
    -C "${FUNASR_CPP_FFMPEG_DIR}" --strip-components=1
fi

cmake \
  -S "${FUNASR_CPP_UPSTREAM_DIR}/runtime/websocket" \
  -B "${FUNASR_CPP_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_DIR="${FUNASR_CPP_ONNXRUNTIME_DIR}" \
  -DFFMPEG_DIR="${FUNASR_CPP_FFMPEG_DIR}"

cmake --build "${FUNASR_CPP_BUILD_DIR}" --parallel "${FUNASR_CPP_BUILD_JOBS}"

echo "FunASR C++/ONNX websocket runtime is ready."
echo "binary: ${FUNASR_CPP_BUILD_DIR}/bin/funasr-wss-server-2pass"
