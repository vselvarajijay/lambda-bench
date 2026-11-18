"""Small helpers for 16-bit mono PCM."""

from __future__ import annotations

BYTES_PER_SAMPLE = 2
INTERNAL_SAMPLE_RATE = 24_000


def silence(samples: int) -> bytes:
    """Return silent PCM for a sample count."""
    if samples < 0:
        raise ValueError("samples must not be negative")
    return bytes(samples * BYTES_PER_SAMPLE)


def sample_count(pcm: bytes) -> int:
    """Return the number of whole samples in PCM data."""
    if len(pcm) % BYTES_PER_SAMPLE:
        raise ValueError("PCM data must contain whole 16-bit samples")
    return len(pcm) // BYTES_PER_SAMPLE


class PlayoutBuffer:
    """Release buffered audio at exactly one tick per call."""

    def __init__(self, samples_per_tick: int, sample_rate: int = INTERNAL_SAMPLE_RATE) -> None:
        if samples_per_tick <= 0 or sample_rate <= 0:
            raise ValueError("samples_per_tick and sample_rate must be positive")
        self.samples_per_tick = samples_per_tick
        self.sample_rate = sample_rate
        self._buffer = bytearray()

    @property
    def pending_samples(self) -> int:
        return len(self._buffer) // BYTES_PER_SAMPLE

    def push(self, pcm: bytes) -> None:
        sample_count(pcm)
        self._buffer.extend(pcm)

    def pop_tick(self) -> bytes:
        size = self.samples_per_tick * BYTES_PER_SAMPLE
        chunk = bytes(self._buffer[:size])
        del self._buffer[:size]
        return chunk + bytes(size - len(chunk))

    def truncate(self) -> int:
        """Drop unplayed audio and return the dropped duration in milliseconds."""
        dropped = self.pending_samples
        self._buffer.clear()
        return round(dropped * 1000 / self.sample_rate)
