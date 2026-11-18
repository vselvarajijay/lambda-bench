from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from lambda_duplex import cli
from lambda_duplex.adapters.base import Capabilities, MediaFrame
from lambda_duplex.adapters.mock import MockAdapter
from lambda_duplex.audio.synthetic import SyntheticSpeech
from lambda_duplex.audio.tracks import resample_pcm
from lambda_duplex.audio.transcription import (
    FasterWhisperTranscriber,
    LiveTranscriber,
    ScriptedTranscriber,
    Transcript,
    TranscriptWord,
    normalized_words,
    transcribe_run,
    word_error_rate,
)
from lambda_duplex.counterpart.turn_taking import TurnController, TurnPersona
from lambda_duplex.orchestrator import Orchestrator, OrchestratorConfig
from lambda_duplex.orchestrator.scripted import ScriptedCounterpart
from lambda_duplex.suites.base import (
    ConversationView,
    DeliveredSpan,
    ModelUtterance,
    PolicyOutput,
    SuiteEvent,
)
from lambda_duplex.suites.loader import load_suite, valid_scenarios
from lambda_duplex.suites.runtime import PolicyCounterpart, ScenarioRun, run_scenario
from lambda_duplex.trajectory.replay import verify_run
from lambda_duplex.trajectory.store import RunSpec, TrajectoryError, find_run_dirs, write_run

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "suites/demo"
JFK_TEXT = (
    "And so my fellow Americans ask not what your country can do for you "
    "ask what you can do for your country"
)


class FakeOffline:
    name = "fake-asr"

    def __init__(self) -> None:
        self.clips: list[bytes] = []

    def transcribe(self, pcm: bytes) -> Transcript:
        self.clips.append(pcm)
        word = TranscriptWord("hello", 10, 20, 0.9)
        return Transcript("hello there", (word,))


def transcribed_run(text: str = "What is the code word?") -> tuple[ScenarioRun, MockAdapter]:
    loaded = load_suite(str(DEMO))
    scenario = valid_scenarios(loaded)[0]
    speech = SyntheticSpeech(0)
    adapter = MockAdapter({1: "this adapter text is not trusted"}, echo=False, speech=speech)
    run = asyncio.run(
        run_scenario(
            loaded.suite,
            scenario,
            adapter,
            max_ticks=60,
            speech=speech,
            transcriber=ScriptedTranscriber([text], "fake-asr"),
        )
    )
    return run, adapter


def store(path: Path, run: ScenarioRun, adapter: MockAdapter, mode: str = "whisper") -> Path:
    suite = load_suite(str(DEMO)).suite
    spec = RunSpec("demo", "mock", "audio", 0, 60, 160, "clean", audio_backend="synthetic")
    transcript = {"mode": mode, "model": "fake-asr"} if mode == "whisper" else {"mode": mode}
    return write_run(
        path / "run",
        suite,
        spec,
        adapter.describe(),
        run.trajectory,
        run.result,
        run.decision,
        tracks=run.tracks,
        condition=run.condition_details,
        transcript=transcript,
    )


def test_word_error_rate_counts_edits() -> None:
    assert normalized_words("Hello, THERE!") == ["hello", "there"]
    assert word_error_rate("one two three four", "one 2 three") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        word_error_rate("", "x")


def test_counterpart_uses_transcript_not_adapter_text() -> None:
    run, _ = transcribed_run()

    events = run.outcome.events
    transcript = next(event for event in events if event.type == "model_transcript")
    assert transcript.data["text"] == "What is the code word?"
    assert transcript.data["source"] == "fake-asr"
    assert not any(event.type == "heard" for event in events)
    assert run.decision.passed is True


def test_wrong_transcript_changes_the_outcome() -> None:
    run, _ = transcribed_run("Nice weather today.")

    assert run.decision.passed is False


def test_transcript_covers_only_heard_speech_and_waits_for_gap() -> None:
    run, _ = transcribed_run()

    transcript = next(event for event in run.outcome.events if event.type == "model_transcript")

    assert transcript.tick == transcript.data["end_tick"] + 5


def test_transcribed_run_replays_from_recorded_text(tmp_path: Path) -> None:
    run, adapter = transcribed_run()
    path = store(tmp_path, run, adapter)

    report = asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))

    assert report.ok, report.problems


def test_changed_recorded_transcript_fails_replay(tmp_path: Path) -> None:
    run, adapter = transcribed_run()
    path = store(tmp_path, run, adapter)
    rows = (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    changed = []
    for line in rows:
        row = json.loads(line)
        if row["type"] == "model_transcript":
            row["data"]["text"] = "Nice weather today."
        changed.append(json.dumps(row))
    (path / "events.jsonl").write_text("\n".join(changed) + "\n", encoding="utf-8")

    report = asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))

    assert report.replay_ok is False


def test_text_mode_rejects_transcriber() -> None:
    loaded = load_suite(str(DEMO))
    scenario = valid_scenarios(loaded)[0]

    with pytest.raises(ValueError, match="audio mode"):
        asyncio.run(
            run_scenario(
                loaded.suite,
                scenario,
                MockAdapter(echo=False),
                transcriber=ScriptedTranscriber(["x"]),
            )
        )


def audio_cli_run(out: Path) -> Path:
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]
    assert cli.main([*arguments, "--audio-backend", "synthetic", "--out", str(out)]) == 0
    return find_run_dirs(out)[0]


def test_injected_transcript_is_labeled(tmp_path: Path) -> None:
    path = audio_cli_run(tmp_path / "runs")

    rows = transcribe_run(path)

    assert rows[0]["text"] == "What is the code word?"
    assert rows[0]["source"] == "injected"
    assert json.loads((path / "manifest.json").read_text())["transcript"] == {"mode": "injected"}


def test_offline_transcript_uses_vad_segments_and_absolute_word_times(tmp_path: Path) -> None:
    path = audio_cli_run(tmp_path / "runs")
    fake = FakeOffline()

    rows = transcribe_run(path, fake)

    assert fake.clips
    assert rows[0]["source"] == "fake-asr"
    assert rows[0]["word_count"] == 2
    assert rows[0]["words"][0]["start_ms"] >= 10
    assert (path / "transcript.jsonl").is_file()


def test_transcript_word_counts_reach_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = audio_cli_run(tmp_path / "runs")
    transcribe_run(path, FakeOffline())
    capsys.readouterr()

    assert (path / "timeline.json").is_file()
    assert cli.main(["metrics", str(path), "--group", "timing"]) == 0
    assert "response_rate" in capsys.readouterr().out


def test_text_run_cannot_use_whisper(capsys: pytest.CaptureFixture[str]) -> None:
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--transcript", "whisper"]

    assert cli.main(arguments) == 2
    assert "audio mode" in capsys.readouterr().err


def test_offline_transcription_requires_audio(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    assert cli.main(["run", "--suite", str(DEMO), "--model", "mock", "--out", str(out)]) == 0

    with pytest.raises(TrajectoryError, match="no audio"):
        transcribe_run(find_run_dirs(out)[0], FakeOffline())


@pytest.fixture(scope="module")
def whisper() -> FasterWhisperTranscriber:
    pytest.importorskip("faster_whisper")
    return FasterWhisperTranscriber()


def jfk_clip() -> tuple[bytes, float]:
    source = Path(__file__).parent.parent / "fixtures/audio/jfk.flac"
    audio, rate = sf.read(source, dtype="int16", always_2d=False)
    mono = np.asarray(audio.mean(axis=1) if audio.ndim > 1 else audio, dtype="<i2")
    return resample_pcm(mono.tobytes(), rate, 16_000), mono.size / rate


@pytest.mark.local
def test_whisper_accuracy_and_speed(whisper: FasterWhisperTranscriber) -> None:
    pcm, seconds = jfk_clip()

    started = time.perf_counter()
    result = whisper.transcribe(pcm)
    elapsed = time.perf_counter() - started

    assert word_error_rate(JFK_TEXT, result.text) <= 0.10
    assert elapsed / seconds < 1.0
    assert result.words


@pytest.mark.local
def test_whisper_does_not_invent_text_from_silence(whisper: FasterWhisperTranscriber) -> None:
    assert whisper.transcribe(bytes(16_000 * 2 * 2)).text == ""


TICK_SAMPLES = 3840


def speech_tick(level: int = 8000) -> bytes:
    high = level.to_bytes(2, "little", signed=True) * 480
    low = (level // 8).to_bytes(2, "little", signed=True) * 480
    return (high + low) * 4


class SilentPolicy:
    def __init__(self) -> None:
        self.utterances: list[str] = []

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        self.utterances.append(utterance.text)
        return PolicyOutput()

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        return None

    def on_delivered(self, span: DeliveredSpan) -> tuple[SuiteEvent, ...]:
        return ()

    def is_finished(self, view: ConversationView) -> bool:
        return False


class RecordingTranscriber:
    name = "recording"

    def __init__(self) -> None:
        self.clips: list[bytes] = []

    async def transcribe(self, pcm: bytes) -> str:
        self.clips.append(pcm)
        return f"utterance {len(self.clips)}"


def feed(counterpart: PolicyCounterpart, frames: list[bytes], first_tick: int = 0) -> None:
    async def run() -> None:
        for offset, pcm in enumerate(frames):
            tick = first_tick + offset
            await counterpart.hear(MediaFrame(tick, pcm, 24_000), tick + 1)

    asyncio.run(run())


def live_counterpart(
    transcriber: RecordingTranscriber, turns: TurnController | None = None
) -> tuple[PolicyCounterpart, SilentPolicy]:
    policy = SilentPolicy()
    speaker = PolicyCounterpart(
        policy, 160, SyntheticSpeech(1), turns=turns, transcriber=transcriber
    )
    return speaker, policy


def test_open_speech_is_transcribed_when_conversation_ends() -> None:
    transcriber = RecordingTranscriber()
    speaker, policy = live_counterpart(transcriber)
    feed(speaker, [speech_tick()] * 5)

    asyncio.run(speaker.finish(MediaFrame(5, speech_tick(), 24_000), 6))

    assert len(transcriber.clips) == 1
    assert policy.utterances == []
    assert [item[0] for item in speaker.drain_events()] == ["model_transcript"]


def test_tick_limit_flushes_pending_model_speech() -> None:
    loaded = load_suite(str(DEMO))
    scenario = valid_scenarios(loaded)[0]
    speech = SyntheticSpeech(0)
    adapter = MockAdapter({1: "What is the code word?"}, echo=False, speech=speech)
    transcriber = ScriptedTranscriber(["What is the code word?"], "fake-asr")

    run = asyncio.run(
        run_scenario(
            loaded.suite,
            scenario,
            adapter,
            max_ticks=9,
            speech=speech,
            transcriber=transcriber,
        )
    )

    assert any(event.type == "model_transcript" for event in run.outcome.events)


def test_clip_includes_quiet_context_around_detected_speech() -> None:
    transcriber = RecordingTranscriber()
    speaker, _ = live_counterpart(transcriber)
    quiet = speech_tick(200)
    frames = [bytes(TICK_SAMPLES * 2), quiet, quiet, *[speech_tick()] * 3, quiet, quiet]

    feed(speaker, [*frames, *[bytes(TICK_SAMPLES * 2)] * 4])

    assert len(transcriber.clips[0]) == (2 + 3 + 2) * TICK_SAMPLES * 2
    assert transcriber.clips[0].startswith(quiet)


def test_long_utterance_is_flushed_without_a_pause() -> None:
    transcriber = RecordingTranscriber()
    speaker, policy = live_counterpart(transcriber)

    feed(speaker, [speech_tick()] * 100)

    assert policy.utterances == ["utterance 1"]


def test_interrupt_decision_gets_partial_transcript() -> None:
    class Decider:
        async def should_interrupt(self, model_text: str, run_ms: int) -> bool:
            return False

    settings = TurnPersona(talk_limit_ms=320, interruptiveness=0.0)
    turns = TurnController(settings, 1, 160, Decider())
    transcriber = RecordingTranscriber()
    speaker, _ = live_counterpart(transcriber, turns)

    async def run() -> None:
        for tick in range(14):
            await speaker.hear(MediaFrame(tick, speech_tick(), 24_000), tick + 1)
            await speaker.render(tick + 1)

    asyncio.run(run())

    names = [item[0] for item in speaker.drain_events()]
    assert "model_transcript_partial" in names
    assert turns.latest_model_text.startswith("utterance")


def test_realtime_cannot_use_blocking_transcription() -> None:
    class RealtimeMock(MockAdapter):
        capabilities = Capabilities(clock_modes=frozenset({"realtime"}))

    config = OrchestratorConfig(1, clock_mode="realtime", trust_model_text=False)

    with pytest.raises(ValueError, match="realtime"):
        Orchestrator(RealtimeMock(), ScriptedCounterpart({}), config)


def test_failed_transcription_run_replays(tmp_path: Path) -> None:
    class FailingTranscriber:
        name = "fake-asr"

        async def transcribe(self, pcm: bytes) -> str:
            raise RuntimeError("out of memory")

    loaded = load_suite(str(DEMO))
    scenario = valid_scenarios(loaded)[0]
    speech = SyntheticSpeech(0)
    adapter = MockAdapter({1: "What is the code word?"}, echo=False, speech=speech)
    run = asyncio.run(
        run_scenario(
            loaded.suite,
            scenario,
            adapter,
            max_ticks=60,
            speech=speech,
            transcriber=FailingTranscriber(),
        )
    )
    assert run.outcome.status == "counterpart_error"
    path = store(tmp_path, run, adapter)

    report = asyncio.run(verify_run(path, loaded.suite))

    assert report.ok, report.problems


def test_live_transcriber_timeout_does_not_need_worker_to_finish() -> None:
    release = threading.Event()

    class Hanging:
        name = "hanging"

        def transcribe(self, pcm: bytes) -> Transcript:
            release.wait(5)
            return Transcript("late", ())

    async def run() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(LiveTranscriber(Hanging()).transcribe(bytes(9600)), 0.01)

    asyncio.run(run())
    release.set()


def test_stale_transcript_is_rejected_by_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = audio_cli_run(tmp_path / "runs")
    transcribe_run(path, FakeOffline())
    rows = (path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0]) | {"start_ms": 1.0}
    (path / "transcript.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert cli.main(["metrics", str(path)]) == 2
    assert "does not match" in capsys.readouterr().err


def test_metrics_report_transcript_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = audio_cli_run(tmp_path / "runs")
    transcribe_run(path, FakeOffline())
    capsys.readouterr()

    assert cli.main(["metrics", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["word_count_source"] == "fake-asr"


def test_invalid_whisper_model_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("faster_whisper")
    path = audio_cli_run(tmp_path / "runs")

    assert cli.main(["transcribe", str(path), "--whisper-model", "not-a-model"]) == 2
    assert "cannot transcribe" in capsys.readouterr().err


def test_injected_transcribe_works_without_audio_packages(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    assert cli.main(["run", "--suite", str(DEMO), "--model", "mock", "--out", str(out)]) == 0
    program = f"""
import sys
for name in ("numpy", "soundfile", "soxr", "onnxruntime"):
    sys.modules[name] = None
from lambda_duplex import cli
assert cli.main(["transcribe", {str(out)!r}, "--backend", "injected"]) == 0
"""

    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
