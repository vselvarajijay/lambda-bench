"""Media clock used by every conversation."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TICK_MS = 160


@dataclass
class MediaClock:
    """Integer-tick clock that never depends on wall time in lockstep mode."""

    tick_ms: int = DEFAULT_TICK_MS
    tick: int = 0

    def __post_init__(self) -> None:
        if self.tick_ms <= 0:
            raise ValueError("tick_ms must be positive")
        if self.tick < 0:
            raise ValueError("tick must not be negative")

    @property
    def now_ms(self) -> int:
        return self.tick * self.tick_ms

    def advance(self) -> int:
        self.tick += 1
        return self.tick

    def samples_per_tick(self, sample_rate: int) -> int:
        samples = sample_rate * self.tick_ms
        if samples % 1000:
            raise ValueError(f"{self.tick_ms} ms is not a whole number of samples")
        return samples // 1000
