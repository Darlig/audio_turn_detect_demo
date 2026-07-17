from __future__ import annotations

import logging
import os

from livekit.agents import stt

from .funasr_cpp_onnx_stt import FunASRCppOnnxSTT
from .funasr_stt import FunASRSTT


DEFAULT_STT_PROVIDER = "funasr_cpp_onnx"
logger = logging.getLogger("cascade-voice-agent")


def create_stt(provider: str | None = None) -> stt.STT:
    selected = (provider or os.getenv("AGENT_STT_PROVIDER") or DEFAULT_STT_PROVIDER).strip().lower()
    normalized = selected.replace("-", "_")
    if normalized in {"funasr_cpp_onnx", "cpp_onnx", "funasr_cpp"}:
        logger.info("using STT provider: funasr_cpp_onnx")
        return FunASRCppOnnxSTT()
    if normalized in {"funasr_python", "python_funasr", "funasr"}:
        logger.info("using STT provider: funasr_python")
        return FunASRSTT()
    raise ValueError(
        "unsupported AGENT_STT_PROVIDER="
        f"{selected!r}; expected funasr_cpp_onnx or funasr_python"
    )
