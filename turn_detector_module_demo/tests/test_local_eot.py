from __future__ import annotations

import numpy as np


def test_local_eot_predict_runs() -> None:
    from livekit.local_inference import EOT, EOT_MAX_SAMPLES

    model = EOT()
    score = model.predict(np.zeros(EOT_MAX_SAMPLES, dtype=np.int16))

    assert 0.0 <= score <= 1.0
