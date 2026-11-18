"""Small scripted counterpart used before suite policies exist."""

from __future__ import annotations

from collections.abc import Mapping

from lambda_duplex.adapters.base import MediaFrame


class ScriptedCounterpart:
    """Speak fixed lines on fixed ticks and remember what it heard."""

    def __init__(self, script: Mapping[int, str], finish_after: int | None = None) -> None:
        self._script = dict(script)
        self._finish_after = finish_after
        self._last_tick = -1
        self.heard: list[tuple[int, MediaFrame]] = []

    async def start(self, seed: int) -> None:
        self._last_tick = -1
        self.heard.clear()

    async def hear(self, frame: MediaFrame, tick: int) -> None:
        self.heard.append((tick, frame))

    async def render(self, tick: int) -> MediaFrame:
        self._last_tick = tick
        return MediaFrame(tick=tick, text=self._script.get(tick))

    def is_finished(self) -> bool:
        return self._finish_after is not None and self._last_tick >= self._finish_after

    async def stop(self) -> None:
        return None
