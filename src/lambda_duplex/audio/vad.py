"""Speech activity detection with the bundled Silero VAD ONNX model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

VAD_SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64
WINDOW_MS = WINDOW_SAMPLES * 1000 / VAD_SAMPLE_RATE
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


@dataclass(frozen=True)
class VadConfig:
    """Thresholds for turning speech probabilities into segments."""

    threshold: float = 0.5
    negative_threshold: float = 0.35
    min_speech_ms: int = 120
    min_silence_ms: int = 200
    pad_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Segment:
    """Speech interval in milliseconds."""

    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class SileroVad:
    """Compute frame-level speech probabilities for 16 kHz PCM."""

    def __init__(self) -> None:
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        model = resources.files("lambda_duplex.audio").joinpath("assets/silero_vad.onnx")
        self._session = ort.InferenceSession(
            model.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def probabilities(self, pcm: bytes) -> np.ndarray:
        """Return one speech probability per complete or padded 32 ms window."""
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        padding = -audio.size % WINDOW_SAMPLES
        audio = np.pad(audio, (0, padding))
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        rate = np.array(VAD_SAMPLE_RATE, dtype=np.int64)
        results = np.zeros(audio.size // WINDOW_SAMPLES, dtype=np.float32)
        for index in range(results.size):
            window = audio[index * WINDOW_SAMPLES : (index + 1) * WINDOW_SAMPLES][None, :]
            model_input = np.concatenate((context, window), axis=1)
            output, state = self._session.run(
                None, {"input": model_input, "state": state, "sr": rate}
            )
            context = model_input[:, -CONTEXT_SAMPLES:]
            results[index] = float(output[0, 0])
        return results


def segments_from_probabilities(
    probabilities: np.ndarray, duration_ms: float, config: VadConfig | None = None
) -> list[Segment]:
    """Apply hysteresis, minimum durations, merging, and padding."""
    settings = config or VadConfig()
    raw: list[list[float]] = []
    start: float | None = None
    last_speech = 0.0
    for index, probability in enumerate(probabilities):
        time_ms = index * WINDOW_MS
        if probability >= settings.threshold:
            if start is None:
                start = time_ms
            last_speech = time_ms + WINDOW_MS
        elif start is not None and (
            probability < settings.negative_threshold
            and time_ms + WINDOW_MS - last_speech >= settings.min_silence_ms
        ):
            raw.append([start, last_speech])
            start = None
    if start is not None:
        raw.append([start, last_speech])
    segments: list[Segment] = []
    for begin, end in raw:
        if end - begin < settings.min_speech_ms:
            continue
        padded_start = max(0.0, begin - settings.pad_ms)
        padded_end = min(duration_ms, end + settings.pad_ms)
        if segments and padded_start <= segments[-1].end_ms:
            segments[-1] = Segment(segments[-1].start_ms, padded_end)
        else:
            segments.append(Segment(padded_start, padded_end))
    return segments
