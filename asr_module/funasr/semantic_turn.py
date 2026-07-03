from __future__ import annotations

import asyncio
import multiprocessing as mp

from livekit.agents import llm
from livekit.agents.ipc.inference_proc_executor import InferenceProcExecutor
from livekit.plugins.turn_detector.base import EOUModelBase


class StandaloneMultilingualEOUModel(EOUModelBase):
    def __init__(self, inference_executor: InferenceProcExecutor) -> None:
        super().__init__(
            model_type="multilingual",
            inference_executor=inference_executor,
            load_languages=True,
        )

    def _inference_method(self) -> str:
        return "lk_end_of_utterance_multilingual"


class SemanticTurnDetector:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._model: StandaloneMultilingualEOUModel | None = None
        self._executor: InferenceProcExecutor | None = None
        self.unavailable_reason: str | None = None

    async def predict(self, text: str) -> float | None:
        if not self.enabled or not text.strip() or self.unavailable_reason:
            return None
        await self._ensure_model()
        if self._model is None:
            return None

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="user", content=text)
        return float(await self._model.predict_end_of_turn(chat_ctx))

    async def warmup(self) -> None:
        if self.enabled:
            await self._ensure_model()

    async def aclose(self) -> None:
        if self._executor:
            await self._executor.aclose()
            self._executor = None
            self._model = None

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import livekit.plugins.turn_detector.multilingual  # noqa: F401
        from livekit.agents.inference_runner import _InferenceRunner

        runner = _InferenceRunner.registered_runners.get("lk_end_of_utterance_multilingual")
        if runner is None:
            self.unavailable_reason = "multilingual EOU runner is not registered"
            return

        loop = asyncio.get_running_loop()
        self._executor = InferenceProcExecutor(
            runners={"lk_end_of_utterance_multilingual": runner},
            initialize_timeout=5 * 60,
            close_timeout=5,
            memory_warn_mb=2000,
            memory_limit_mb=0,
            ping_interval=5,
            ping_timeout=60,
            high_ping_threshold=2.5,
            mp_ctx=mp.get_context("spawn"),
            loop=loop,
            http_proxy=None,
        )
        await self._executor.start()
        await self._executor.initialize()
        self._model = StandaloneMultilingualEOUModel(self._executor)


class PartialTranscript:
    def __init__(self) -> None:
        self.text = ""

    def reset(self) -> None:
        self.text = ""

    def update(self, partial: str) -> str:
        partial = partial.strip()
        if not partial:
            return self.text

        if partial.startswith(self.text):
            self.text = partial
        elif not self.text.endswith(partial):
            self.text += partial

        return self.text
