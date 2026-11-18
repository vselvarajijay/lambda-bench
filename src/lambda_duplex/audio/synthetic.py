"""Deterministic speech-shaped audio for tests and offline runs."""

from __future__ import annotations

import hashlib

import numpy as np

from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE

WORD_MS = 240
GAP_MS = 60


class SyntheticSpeech:
    """Render text to a repeatable voiced signal. It is not intelligible speech."""

    name = "synthetic"

    def __init__(self, seed: int = 0, sample_rate: int = INTERNAL_SAMPLE_RATE) -> None:
        self.seed = seed
        self.sample_rate = sample_rate

    def _word(self, word: str, index: int) -> np.ndarray:
        digest = hashlib.sha256(f"{self.seed}:{index}:{word}".encode()).digest()
        pitch = 95 + digest[0] % 110
        samples = self.sample_rate * WORD_MS // 1000
        time = np.arange(samples) / self.sample_rate
        signal = np.zeros(samples)
        for harmonic in range(1, 9):
            phase = digest[harmonic] / 255 * 2 * np.pi
            signal += np.sin(2 * np.pi * pitch * harmonic * time + phase) / harmonic
        envelope = np.sin(np.pi * np.arange(samples) / max(samples - 1, 1)) ** 0.5
        result: np.ndarray = signal * envelope
        return result / max(float(np.max(np.abs(result))), 1e-9) * 0.35

    def render(self, text: str) -> bytes:
        """Return 16-bit mono PCM with duration set by the word count."""
        words = text.split() or ["_"]
        gap = np.zeros(self.sample_rate * GAP_MS // 1000)
        parts: list[np.ndarray] = []
        for index, word in enumerate(words):
            parts.extend((self._word(word, index), gap))
        signal = np.concatenate(parts[:-1])
        return np.round(signal * 32767).astype("<i2").tobytes()
