"""Transcribe model speech and store the result with a run."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE
from lambda_duplex.trajectory.store import StoredRun, TrajectoryError, load_run

WHISPER_RATE = 16_000
PADDING_MS = 200
DEFAULT_WHISPER_MODEL = "small.en"
NO_SPEECH_THRESHOLD = 0.6


@dataclass(frozen=True)
class TranscriptWord:
    word: str
    start_ms: float
    end_ms: float
    probability: float | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    words: tuple[TranscriptWord, ...]


class SpeechTranscriber(Protocol):
    """Convert 16 kHz PCM to text."""

    name: str

    def transcribe(self, pcm: bytes) -> Transcript: ...


class FasterWhisperTranscriber:
    """CPU-friendly faster-whisper transcription with stable decoding settings."""

    def __init__(
        self,
        model: str = DEFAULT_WHISPER_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found,import-untyped,unused-ignore]

        version = importlib.metadata.version("faster-whisper")
        self.name = f"faster-whisper:{version}:{Path(model).name}:{compute_type}"
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, pcm: bytes) -> Transcript:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=True,
        )
        words: list[TranscriptWord] = []
        parts: list[str] = []
        for segment in segments:
            if float(segment.no_speech_prob) >= NO_SPEECH_THRESHOLD:
                continue
            parts.append(segment.text.strip())
            words.extend(
                TranscriptWord(
                    word.word.strip(),
                    float(word.start) * 1000,
                    float(word.end) * 1000,
                    float(word.probability),
                )
                for word in segment.words or ()
            )
        return Transcript(" ".join(part for part in parts if part), tuple(words))


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return word edit distance divided by reference word count."""
    expected = normalized_words(reference)
    actual = normalized_words(hypothesis)
    if not expected:
        raise ValueError("reference must contain words")
    previous = list(range(len(actual) + 1))
    for row, word in enumerate(expected, 1):
        current = [row]
        for column, candidate in enumerate(actual, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (word != candidate),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def injected_rows(run: StoredRun) -> list[dict[str, Any]]:
    """Return adapter text from stored model events."""
    return [
        {
            "time_ms": event.time_ms,
            "text": event.data["text"],
            "word_count": len(normalized_words(event.data["text"])),
            "words": [],
            "source": "injected",
        }
        for event in run.trajectory.events
        if event.source == "model" and event.type == "spoke"
    ]


def transcribe_run(
    path: Path, transcriber: SpeechTranscriber | None = None
) -> list[dict[str, Any]]:
    """Write ``transcript.jsonl`` from model audio, or from injected text."""
    run = load_run(path)
    if transcriber is None:
        rows = injected_rows(run)
        _write_rows(path / "transcript.jsonl", rows)
        return rows
    if run.manifest.get("audio") is None:
        raise TrajectoryError(f"{path} has no audio")
    from lambda_duplex.audio.timeline import build_timeline
    from lambda_duplex.audio.tracks import read_track, resample_pcm

    timeline_path = path / "timeline.json"
    if timeline_path.is_file():
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    else:
        timeline = build_timeline(path)
    pcm, rate = read_track(path / "audio" / "model.flac")
    samples_per_ms = rate / 1000
    rows = []
    for index, segment in enumerate(timeline["model"]["segments"]):
        start_ms = max(0.0, segment["start_ms"] - PADDING_MS)
        end_ms = min(timeline["duration_ms"], segment["end_ms"] + PADDING_MS)
        start = round(start_ms * samples_per_ms) * 2
        end = round(end_ms * samples_per_ms) * 2
        clip = resample_pcm(pcm[start:end], rate, WHISPER_RATE)
        result = transcriber.transcribe(clip)
        words = [
            {
                **asdict(word),
                "start_ms": word.start_ms + start_ms,
                "end_ms": word.end_ms + start_ms,
            }
            for word in result.words
        ]
        rows.append(
            {
                "segment_index": index,
                "start_ms": segment["start_ms"],
                "end_ms": segment["end_ms"],
                "text": result.text,
                "word_count": len(normalized_words(result.text)),
                "words": words,
                "source": transcriber.name,
            }
        )
    _write_rows(path / "transcript.jsonl", rows)
    return rows


class LiveTranscriber:
    """Run blocking transcription without keeping the process alive after a timeout."""

    def __init__(self, transcriber: SpeechTranscriber) -> None:
        self._transcriber = transcriber
        self.name = transcriber.name

    async def transcribe(self, pcm: bytes) -> str:
        from lambda_duplex.audio.tracks import resample_pcm

        speech = resample_pcm(pcm, INTERNAL_SAMPLE_RATE, WHISPER_RATE)
        loop = asyncio.get_running_loop()
        result: asyncio.Future[str] = loop.create_future()

        def finish(text: str | None, error: BaseException | None) -> None:
            if result.done():
                return
            if error is not None:
                result.set_exception(error)
            else:
                result.set_result(text or "")

        def publish(text: str | None, error: BaseException | None) -> None:
            try:
                loop.call_soon_threadsafe(finish, text, error)
            except RuntimeError:
                return

        def work() -> None:
            try:
                text = self._transcriber.transcribe(speech).text
            except BaseException as error:
                publish(None, error)
            else:
                publish(text, None)

        threading.Thread(target=work, daemon=True).start()
        return await result


class ScriptedTranscriber:
    """Deterministic transcriber for tests and replay."""

    def __init__(
        self,
        texts: Sequence[str] | Callable[[bytes], str],
        name: str = "scripted",
        failure: str | None = None,
    ) -> None:
        self.name = name
        self._texts = texts
        self._failure = failure
        self._index = 0

    async def transcribe(self, pcm: bytes) -> str:
        if callable(self._texts):
            return self._texts(pcm)
        if self._index >= len(self._texts):
            raise RuntimeError(self._failure or "no recorded transcript remains")
        text = self._texts[self._index]
        self._index += 1
        return text
