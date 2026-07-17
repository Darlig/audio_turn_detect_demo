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
    if not text:
        return None

    mode = str(data.get("mode") or data.get("type") or "")
    mode_lower = mode.lower()
    is_online = mode_lower in {"2pass-online", "online", "partial"}
    is_final = mode_lower in {"2pass-offline", "offline", "final"} or (
        not is_online and bool(data.get("is_final") or data.get("final"))
    )

    return ASRResult(text=text, mode=mode, is_final=is_final, raw=data)


def start_message(
    *,
    mode: str = "2pass",
    chunk_size: tuple[int, int, int] = (5, 10, 5),
    sample_rate: int = 16000,
    wav_name: str = "livekit-agent",
    hotwords: str = "",
    use_itn: bool = True,
) -> str:
    return json.dumps(
        {
            "mode": mode,
            "wav_name": wav_name,
            "wav_format": "pcm",
            "is_speaking": True,
            "chunk_size": list(chunk_size),
            "audio_fs": sample_rate,
            "hotwords": hotwords,
            "itn": use_itn,
        },
        ensure_ascii=False,
    )


def stop_message() -> str:
    return json.dumps({"is_speaking": False}, ensure_ascii=False)
