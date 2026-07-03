from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class ASRResult:
    text: str
    mode: str
    is_final: bool
    raw: dict[str, Any]

    @property
    def display_text(self) -> str:
        return str(self.raw.get("partial_text") or self.text).strip()


def parse_server_message(message: str | bytes) -> ASRResult | None:
    if isinstance(message, bytes):
        message = message.decode("utf-8", errors="ignore")
    if not message:
        return None

    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return None
    text = str(data.get("text") or data.get("preds") or "").strip()
    mode = str(data.get("mode") or data.get("type") or "")
    is_final = bool(
        data.get("is_final")
        or data.get("final")
        or mode in {"2pass-offline", "offline", "final"}
    )
    if not text:
        return None
    return ASRResult(text=text, mode=mode, is_final=is_final, raw=data)


def start_message(
    *,
    mode: str = "2pass",
    chunk_size: tuple[int, int, int] = (8, 8, 4),
    chunk_interval: int = 10,
    wav_name: str = "microphone",
    hotwords: str = "",
    semantic_turn_detection: bool = False,
) -> str:
    return json.dumps(
        {
            "mode": mode,
            "chunk_size": list(chunk_size),
            "chunk_interval": chunk_interval,
            "wav_name": wav_name,
            "is_speaking": True,
            "hotwords": hotwords,
            "semantic_turn_detection": semantic_turn_detection,
        },
        ensure_ascii=False,
    )


def stop_message() -> str:
    return json.dumps({"is_speaking": False}, ensure_ascii=False)
