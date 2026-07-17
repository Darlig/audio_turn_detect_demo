from __future__ import annotations

import pytest

from turn_detector_module_demo.turn_metrics import (
    TurnTiming,
    publish_turn_latency_metrics,
    record_asr_turn,
    record_playback_started,
    record_tts_first_audio,
    record_vad_speech_tail,
    reset_turn_metrics,
    set_metrics_callback,
    turn_latency_payload,
)


@pytest.fixture(autouse=True)
def _clean_metrics_state():
    reset_turn_metrics()
    set_metrics_callback(None)
    yield
    reset_turn_metrics()
    set_metrics_callback(None)


def test_vad_tail_replaces_provider_tail_and_preserves_debug_anchor() -> None:
    record_vad_speech_tail(speech_ended_at=10.0)

    timing = record_asr_turn(
        1,
        speech_started_at=8.0,
        speech_ended_at=10.88,
        endpoint_sent_at=10.89,
        asr_final_at=10.9,
        audio_tail_source="funasr_timestamp",
    )

    assert timing.speech_ended_at == 10.0
    assert timing.audio_tail_source == "livekit_vad"
    assert timing.provider_speech_ended_at == 10.88
    assert timing.provider_audio_tail_source == "funasr_timestamp"


def test_complete_latency_chain_sums_to_server_total_with_preemption() -> None:
    timing = TurnTiming(
        seq=2,
        speech_ended_at=10.0,
        asr_final_at=10.9,
        eou_confirmed_at=11.0,
        llm_started_at=10.95,
        llm_first_token_at=10.96,
        tts_started_at=10.97,
        tts_first_audio_at=11.17,
        playback_started_at=11.2,
    )

    metrics = turn_latency_payload(timing)["metrics"]
    stages = (
        "audio_tail_to_asr_final_ms",
        "asr_final_to_eou_confirmed_ms",
        "eou_to_llm_input_ms",
        "llm_input_to_first_token_ms",
        "llm_first_token_to_tts_text_ms",
        "tts_text_to_first_audio_ms",
        "tts_first_audio_to_playback_ms",
    )

    assert sum(metrics[name] for name in stages) == pytest.approx(metrics["total"])
    assert metrics["eou_to_llm_input_ms"] == pytest.approx(-50.0)
    assert metrics["total"] == pytest.approx(1200.0)


def test_metrics_publish_only_after_server_playback_starts() -> None:
    received: list[dict[str, object]] = []
    set_metrics_callback(received.append)
    record_asr_turn(
        3,
        speech_started_at=1.0,
        speech_ended_at=2.0,
        endpoint_sent_at=2.1,
        asr_final_at=2.2,
    )
    record_tts_first_audio(3, tts_started_at=2.3, tts_first_audio_at=2.5)

    assert publish_turn_latency_metrics(3) is None
    payload = record_playback_started(playback_started_at=2.55)

    assert payload is not None
    assert payload["metrics"]["total"] == pytest.approx(550.0)
    assert received == [payload]
