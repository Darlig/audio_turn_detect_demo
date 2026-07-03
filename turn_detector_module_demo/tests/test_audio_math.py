from __future__ import annotations

import numpy as np

from turn_detector_module_demo.audio_math import decision_from_score, pcm16_rms01


def test_decision_uses_threshold() -> None:
    assert decision_from_score(0.36, 0.355) is True
    assert decision_from_score(0.35, 0.355) is False
    assert decision_from_score(0.9, None) is False


def test_pcm16_rms01_handles_silence_and_signal() -> None:
    silence = np.zeros(160, dtype=np.int16).tobytes()
    signal = np.full(160, 16384, dtype=np.int16).tobytes()

    assert pcm16_rms01(silence) == 0.0
    assert 0.49 < pcm16_rms01(signal) < 0.51
