#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

source "${PROJECT_DIR}/scripts/env.sh"
load_env_file "${PROJECT_DIR}/.env"

ENV_NAME="${ENV_NAME:-audio_turn_detector}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_DIR}/.conda/${ENV_NAME}}"
FUNASR_CPP_ROOT="${FUNASR_CPP_ROOT:-${REPO_ROOT}/asr_module/funasr_cpp_onnx}"
FUNASR_CPP_DEPS_DIR="${FUNASR_CPP_DEPS_DIR:-${FUNASR_CPP_ROOT}/deps}"
FUNASR_CPP_BUILD_DIR="${FUNASR_CPP_BUILD_DIR:-${FUNASR_CPP_ROOT}/build/websocket}"
FUNASR_CPP_BIN="${FUNASR_CPP_BIN:-${FUNASR_CPP_BUILD_DIR}/bin/funasr-wss-server-2pass}"
FUNASR_CPP_ONNXRUNTIME_LIB_DIR="${FUNASR_CPP_ONNXRUNTIME_LIB_DIR:-${FUNASR_CPP_DEPS_DIR}/onnxruntime/lib}"
FUNASR_CPP_FFMPEG_LIB_DIR="${FUNASR_CPP_FFMPEG_LIB_DIR:-${FUNASR_CPP_DEPS_DIR}/ffmpeg/lib}"
FUNASR_CPP_HOST="${FUNASR_CPP_HOST:-127.0.0.1}"
FUNASR_CPP_PORT="${FUNASR_CPP_PORT:-10095}"
FUNASR_CPP_DOWNLOAD_MODEL_DIR="${FUNASR_CPP_DOWNLOAD_MODEL_DIR:-${FUNASR_CPP_ROOT}/models}"
FUNASR_CPP_MODEL_DIR="${FUNASR_CPP_MODEL_DIR:-damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx}"
FUNASR_CPP_ONLINE_MODEL_DIR="${FUNASR_CPP_ONLINE_MODEL_DIR:-damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx}"
FUNASR_CPP_VAD_DIR="${FUNASR_CPP_VAD_DIR:-damo/speech_fsmn_vad_zh-cn-16k-common-onnx}"
FUNASR_CPP_PUNC_DIR="${FUNASR_CPP_PUNC_DIR:-damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx}"
FUNASR_CPP_ITN_DIR="${FUNASR_CPP_ITN_DIR:-thuduj12/fst_itn_zh}"
FUNASR_CPP_LM_DIR="${FUNASR_CPP_LM_DIR:-}"
FUNASR_CPP_HOTWORD_FILE="${FUNASR_CPP_HOTWORD_FILE:-${FUNASR_CPP_ROOT}/hotwords.txt}"

case "${FUNASR_CPP_DOWNLOAD_MODEL_DIR}" in
  /*) ;;
  *) FUNASR_CPP_DOWNLOAD_MODEL_DIR="${REPO_ROOT}/${FUNASR_CPP_DOWNLOAD_MODEL_DIR}" ;;
esac

case "${FUNASR_CPP_HOTWORD_FILE}" in
  /*) ;;
  *) FUNASR_CPP_HOTWORD_FILE="${REPO_ROOT}/${FUNASR_CPP_HOTWORD_FILE}" ;;
esac

resolve_model_dir() {
  local model_dir="$1"
  if [[ "${model_dir}" != /* && -d "${FUNASR_CPP_DOWNLOAD_MODEL_DIR}/${model_dir}" ]]; then
    model_dir="${FUNASR_CPP_DOWNLOAD_MODEL_DIR}/${model_dir}"
  fi
  printf '%s' "${model_dir}"
}

FUNASR_CPP_MODEL_DIR="$(resolve_model_dir "${FUNASR_CPP_MODEL_DIR}")"
FUNASR_CPP_ONLINE_MODEL_DIR="$(resolve_model_dir "${FUNASR_CPP_ONLINE_MODEL_DIR}")"
FUNASR_CPP_VAD_DIR="$(resolve_model_dir "${FUNASR_CPP_VAD_DIR}")"
FUNASR_CPP_PUNC_DIR="$(resolve_model_dir "${FUNASR_CPP_PUNC_DIR}")"
FUNASR_CPP_ITN_DIR="$(resolve_model_dir "${FUNASR_CPP_ITN_DIR}")"
if [[ -n "${FUNASR_CPP_LM_DIR}" ]]; then
  FUNASR_CPP_LM_DIR="$(resolve_model_dir "${FUNASR_CPP_LM_DIR}")"
fi

cpu_count="$(nproc 2>/dev/null || echo 4)"
FUNASR_CPP_DECODER_THREAD_NUM="${FUNASR_CPP_DECODER_THREAD_NUM:-${cpu_count}}"
FUNASR_CPP_MODEL_THREAD_NUM="${FUNASR_CPP_MODEL_THREAD_NUM:-1}"
FUNASR_CPP_IO_THREAD_NUM="${FUNASR_CPP_IO_THREAD_NUM:-$(( (FUNASR_CPP_DECODER_THREAD_NUM + 15) / 16 ))}"

if [[ ! -x "${FUNASR_CPP_BIN}" ]]; then
  echo "FunASR C++/ONNX binary not found: ${FUNASR_CPP_BIN}" >&2
  echo "Run: bash ${PROJECT_DIR}/scripts/prepare_funasr_cpp_onnx.sh" >&2
  exit 1
fi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  export PATH="${ENV_PREFIX}/bin:${PATH}"
fi

funasr_library_dirs=(
  "${FUNASR_CPP_BUILD_DIR}/src"
  "${FUNASR_CPP_BUILD_DIR}/yaml-cpp"
  "${FUNASR_CPP_BUILD_DIR}/openfst/src/lib"
  "${FUNASR_CPP_BUILD_DIR}/openfst/src/script"
  "${FUNASR_CPP_BUILD_DIR}/glog"
  "${FUNASR_CPP_BUILD_DIR}/gflags"
  "${FUNASR_CPP_BUILD_DIR}/_deps/portaudio-build"
  "${FUNASR_CPP_ONNXRUNTIME_LIB_DIR}"
  "${FUNASR_CPP_FFMPEG_LIB_DIR}"
)
export LD_LIBRARY_PATH="$(IFS=:; echo "${funasr_library_dirs[*]}"):${LD_LIBRARY_PATH:-}"

mkdir -p "${FUNASR_CPP_DOWNLOAD_MODEL_DIR}"
if [[ ! -f "${FUNASR_CPP_HOTWORD_FILE}" ]]; then
  : > "${FUNASR_CPP_HOTWORD_FILE}"
fi

host_args=()
help_text="$("${FUNASR_CPP_BIN}" --help 2>&1 || true)"
if [[ "${help_text}" == *"--host"* ]]; then
  host_args+=(--host "${FUNASR_CPP_HOST}")
elif [[ "${help_text}" == *"--listen-ip"* ]]; then
  host_args+=(--listen-ip "${FUNASR_CPP_HOST}")
elif [[ "${FUNASR_CPP_HOST}" != "127.0.0.1" && "${FUNASR_CPP_HOST}" != "0.0.0.0" ]]; then
  echo "warning: this funasr-wss-server-2pass build does not expose a host bind flag in --help" >&2
fi

cmd=(
  "${FUNASR_CPP_BIN}"
  --download-model-dir "${FUNASR_CPP_DOWNLOAD_MODEL_DIR}"
  --model-dir "${FUNASR_CPP_MODEL_DIR}"
  --online-model-dir "${FUNASR_CPP_ONLINE_MODEL_DIR}"
  --vad-dir "${FUNASR_CPP_VAD_DIR}"
  --punc-dir "${FUNASR_CPP_PUNC_DIR}"
  --itn-dir "${FUNASR_CPP_ITN_DIR}"
  --lm-dir "${FUNASR_CPP_LM_DIR}"
  --decoder-thread-num "${FUNASR_CPP_DECODER_THREAD_NUM}"
  --model-thread-num "${FUNASR_CPP_MODEL_THREAD_NUM}"
  --io-thread-num "${FUNASR_CPP_IO_THREAD_NUM}"
  --port "${FUNASR_CPP_PORT}"
  --certfile 0
  --hotword "${FUNASR_CPP_HOTWORD_FILE}"
)
cmd+=("${host_args[@]}")

exec "${cmd[@]}"
