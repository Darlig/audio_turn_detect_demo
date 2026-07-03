from __future__ import annotations

import math

import numpy as np


def pcm16_rms01(pcm16le: bytes | memoryview) -> float:
    data = bytes(pcm16le)
    if not data:
        return 0.0
    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    rms = math.sqrt(float(np.mean(samples.astype(np.float32) ** 2)))
    return min(1.0, rms / 32768.0)


def decision_from_score(score: float, threshold: float | None) -> bool:
    return bool(threshold is not None and score >= threshold)
