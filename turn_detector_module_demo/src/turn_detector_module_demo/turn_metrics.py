from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable


@dataclass
class TurnTiming:
    seq: int
    speech_started_at: float | None = None
    speech_ended_at: float | None = None
    audio_tail_source: str | None = None
    provider_speech_ended_at: float | None = None
    provider_audio_tail_source: str | None = None
    endpoint_sent_at: float | None = None
    asr_final_at: float | None = None
    asr_transcript: str | None = None
    eou_confirmed_at: float | None = None
    llm_claimed: bool = False
    llm_started_at: float | None = None
    llm_first_token_at: float | None = None
    llm_finished_at: float | None = None
    tts_claimed: bool = False
    tts_started_at: float | None = None
    tts_first_audio_at: float | None = None
    playback_started_at: float | None = None


MetricsCallback = Callable[[dict[str, Any]], None]

_lock = Lock()
_turns: dict[int, TurnTiming] = {}
_pending_vad_tails: list[float] = []
_metrics_callback: MetricsCallback | None = None


def set_metrics_callback(callback: MetricsCallback | None) -> None:
    global _metrics_callback
    with _lock:
        _metrics_callback = callback


def reset_turn_metrics() -> None:
    with _lock:
        _turns.clear()
        _pending_vad_tails.clear()


def record_vad_speech_tail(*, speech_ended_at: float) -> None:
    with _lock:
        _pending_vad_tails.append(speech_ended_at)
        del _pending_vad_tails[:-8]


def record_asr_turn(
    seq: int,
    *,
    speech_started_at: float | None,
    speech_ended_at: float | None,
    endpoint_sent_at: float | None,
    asr_final_at: float,
    audio_tail_source: str | None = None,
    transcript: str | None = None,
) -> TurnTiming:
    with _lock:
        timing = _turns.setdefault(seq, TurnTiming(seq=seq))
        timing.speech_started_at = speech_started_at
        timing.provider_speech_ended_at = speech_ended_at
        timing.provider_audio_tail_source = audio_tail_source
        vad_tail = _consume_latest_vad_tail(asr_final_at)
        timing.speech_ended_at = vad_tail if vad_tail is not None else speech_ended_at
        timing.audio_tail_source = "livekit_vad" if vad_tail is not None else audio_tail_source
        timing.endpoint_sent_at = endpoint_sent_at
        timing.asr_final_at = asr_final_at
        timing.asr_transcript = transcript
        return timing


def record_eou_confirmed(
    *,
    eou_confirmed_at: float,
    transcript: str | None = None,
) -> TurnTiming | None:
    with _lock:
        timing = _latest_turn(
            lambda item: item.asr_final_at is not None and item.eou_confirmed_at is None
        )
        if timing is None:
            timing = _latest_turn(lambda item: item.asr_final_at is not None)
        if timing is None:
            return None
        timing.eou_confirmed_at = eou_confirmed_at
        if transcript:
            timing.asr_transcript = transcript
        return timing


def claim_llm_turn_sequence(fallback_seq: int) -> int:
    with _lock:
        timing = _latest_turn(
            lambda item: item.asr_final_at is not None and not item.llm_claimed
        )
        if timing is None:
            return _unused_fallback_seq(fallback_seq)
        timing.llm_claimed = True
        return timing.seq


def record_llm_turn(
    seq: int,
    *,
    llm_started_at: float,
    llm_finished_at: float,
) -> TurnTiming:
    with _lock:
        timing = _turns.setdefault(seq, TurnTiming(seq=seq))
        timing.llm_started_at = llm_started_at
        timing.llm_finished_at = llm_finished_at
        return timing


def record_llm_first_token(
    seq: int,
    *,
    llm_started_at: float,
    llm_first_token_at: float,
) -> TurnTiming:
    with _lock:
        timing = _turns.setdefault(seq, TurnTiming(seq=seq))
        timing.llm_started_at = llm_started_at
        timing.llm_first_token_at = llm_first_token_at
        return timing


def claim_tts_turn_sequence(fallback_seq: int) -> int:
    with _lock:
        timing = _latest_turn(
            lambda item: (
                item.asr_final_at is not None
                and item.llm_claimed
                and not item.tts_claimed
            )
        )
        if timing is None:
            return _unused_fallback_seq(fallback_seq)
        timing.tts_claimed = True
        return timing.seq


def record_tts_first_audio(
    seq: int,
    *,
    tts_started_at: float,
    tts_first_audio_at: float,
) -> TurnTiming:
    with _lock:
        timing = _turns.setdefault(seq, TurnTiming(seq=seq))
        timing.tts_started_at = tts_started_at
        timing.tts_first_audio_at = tts_first_audio_at
        return timing


def record_playback_started(*, playback_started_at: float) -> dict[str, Any] | None:
    with _lock:
        timing = _latest_turn(
            lambda item: item.tts_first_audio_at is not None
            and item.playback_started_at is None
            and item.asr_final_at is not None
        )
        if timing is None:
            return None
        timing.playback_started_at = playback_started_at
        seq = timing.seq
    return publish_turn_latency_metrics(seq)


def publish_turn_latency_metrics(seq: int) -> dict[str, Any] | None:
    with _lock:
        timing = _turns.get(seq)
        callback = _metrics_callback
        if (
            timing is None
            or timing.asr_final_at is None
            or timing.speech_ended_at is None
            or timing.playback_started_at is None
        ):
            return None
        payload = turn_latency_payload(timing)

    if callback is not None:
        callback(payload)
    return payload


def turn_latency_payload(timing: TurnTiming) -> dict[str, Any]:
    audio_tail_to_asr_final_ms = elapsed_ms(timing.speech_ended_at, timing.asr_final_at)
    asr_final_to_eou_confirmed_ms = elapsed_ms(timing.asr_final_at, timing.eou_confirmed_at)
    llm_input_to_first_token_ms = elapsed_ms(timing.llm_started_at, timing.llm_first_token_at)
    if llm_input_to_first_token_ms is None:
        llm_input_to_first_token_ms = elapsed_ms(timing.llm_started_at, timing.llm_finished_at)
    tts_text_to_first_audio_ms = elapsed_ms(timing.tts_started_at, timing.tts_first_audio_at)
    eou_to_llm_input_ms = signed_elapsed_ms(timing.eou_confirmed_at, timing.llm_started_at)
    llm_first_token_to_tts_text_ms = signed_elapsed_ms(
        timing.llm_first_token_at, timing.tts_started_at
    )
    tts_first_audio_to_playback_ms = elapsed_ms(
        timing.tts_first_audio_at, timing.playback_started_at
    )
    audio_tail_to_tts_first_audio_ms = elapsed_ms(
        timing.speech_ended_at, timing.tts_first_audio_at
    )
    speech_tail_to_playback_started_ms = elapsed_ms(
        timing.speech_ended_at, timing.playback_started_at
    )
    llm_started_before_eou = (
        timing.llm_started_at is not None
        and timing.eou_confirmed_at is not None
        and timing.llm_started_at < timing.eou_confirmed_at
    )
    return {
        "type": "latency_metrics",
        "seq": timing.seq,
        "metrics": {
            "audio_tail_to_asr_final_ms": audio_tail_to_asr_final_ms,
            "asr_final_to_eou_confirmed_ms": asr_final_to_eou_confirmed_ms,
            "llm_input_to_first_token_ms": llm_input_to_first_token_ms,
            "tts_text_to_first_audio_ms": tts_text_to_first_audio_ms,
            "eou_to_llm_input_ms": eou_to_llm_input_ms,
            "llm_first_token_to_tts_text_ms": llm_first_token_to_tts_text_ms,
            "tts_first_audio_to_playback_ms": tts_first_audio_to_playback_ms,
            "audio_tail_to_tts_first_audio_ms": audio_tail_to_tts_first_audio_ms,
            "speech_tail_to_playback_started_ms": speech_tail_to_playback_started_ms,
            # Compatibility aliases for older logs.
            "asr": audio_tail_to_asr_final_ms,
            "eou_wait": asr_final_to_eou_confirmed_ms,
            "llm": llm_input_to_first_token_ms,
            "tts": tts_text_to_first_audio_ms,
            "total": speech_tail_to_playback_started_ms,
        },
        "debug": {
            "audioTailSource": timing.audio_tail_source,
            "providerAudioTailSource": timing.provider_audio_tail_source,
            "preemptiveGenerationUsed": llm_started_before_eou,
            "llmStartedBeforeEou": llm_started_before_eou,
            "hasEouConfirmed": timing.eou_confirmed_at is not None,
        },
        "anchors": {
            "speechStartedAt": timing.speech_started_at,
            "audioTailAt": timing.speech_ended_at,
            "providerAudioTailAt": timing.provider_speech_ended_at,
            "endpointSentAt": timing.endpoint_sent_at,
            "asrFinalAt": timing.asr_final_at,
            "eouConfirmedAt": timing.eou_confirmed_at,
            "llmStartedAt": timing.llm_started_at,
            "llmFirstTokenAt": timing.llm_first_token_at,
            "ttsStartedAt": timing.tts_started_at,
            "ttsFirstAudioAt": timing.tts_first_audio_at,
            "playbackStartedAt": timing.playback_started_at,
        },
    }


def elapsed_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, (end - start) * 1000.0)


def signed_elapsed_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _latest_turn(predicate: Callable[[TurnTiming], bool]) -> TurnTiming | None:
    for seq in sorted(_turns, reverse=True):
        timing = _turns[seq]
        if predicate(timing):
            return timing
    return None


def _unused_fallback_seq(fallback_seq: int) -> int:
    seq = -abs(fallback_seq or 1)
    while seq in _turns:
        seq -= 1
    return seq


def _consume_latest_vad_tail(asr_final_at: float) -> float | None:
    candidates = [tail for tail in _pending_vad_tails if tail <= asr_final_at]
    if not candidates:
        return None
    selected = candidates[-1]
    _pending_vad_tails[:] = [tail for tail in _pending_vad_tails if tail > selected]
    return selected


def format_metric(value: float | None) -> str:
    if value is None:
        return "na"
    return f"{value:.1f}"
