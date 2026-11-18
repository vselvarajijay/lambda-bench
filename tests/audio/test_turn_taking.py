from __future__ import annotations

import asyncio
import json
import wave
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from lambda_duplex import cli
from lambda_duplex.adapters.base import MediaFrame
from lambda_duplex.adapters.mock import MockAdapter
from lambda_duplex.audio.synthetic import SyntheticSpeech
from lambda_duplex.audio.timeline import build_timeline
from lambda_duplex.counterpart.turn_taking import (
    INTERRUPT_TEXT,
    LLMInterruptDecider,
    SpeechGate,
    TurnAction,
    TurnController,
    TurnPersona,
)
from lambda_duplex.llm import FakeProvider, LLMGateway
from lambda_duplex.suites.base import (
    ConversationView,
    DeliveredSpan,
    ModelUtterance,
    PolicyOutput,
    Scenario,
    SpeechSegment,
    SuiteEvent,
)
from lambda_duplex.suites.loader import load_suite, valid_scenarios
from lambda_duplex.suites.runtime import PolicyCounterpart, ScenarioRun, run_scenario
from lambda_duplex.trajectory.replay import verify_run
from lambda_duplex.trajectory.store import (
    RunSpec,
    TrajectoryError,
    find_run_dirs,
    load_run,
    write_run,
)

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "suites/demo"
TICK_BYTES = 3840 * 2
LOUD = (b"\x00\x20" * 480 + b"\x40\x00" * 480) * 4
QUIET = bytes(TICK_BYTES)


def persona(**changes: object) -> TurnPersona:
    values: dict[str, object] = {
        "talk_limit_ms": 1000_000,
        "interruptiveness": 0.0,
        "silence_tolerance_ms": 1000_000,
        "backchannel_min_ms": 1000_000,
        "backchannel_max_ms": 2000_000,
    }
    values.update(changes)
    return TurnPersona(**values)  # type: ignore[arg-type]


def collect(
    controller: TurnController,
    ticks: int,
    model_from: int | None = None,
    playing: bool = False,
    pending: bool = False,
) -> list[tuple[int, TurnAction]]:
    async def run() -> list[tuple[int, TurnAction]]:
        found: list[tuple[int, TurnAction]] = []
        for tick in range(ticks):
            speaking = model_from is not None and tick >= model_from
            controller.observe_model(tick, speaking)
            found.extend(
                (tick, action) for action in await controller.decide(tick, playing, pending)
            )
        return found

    return asyncio.run(run())


def test_counterpart_yields_when_overlap_reaches_limit() -> None:
    actions = collect(TurnController(persona(), 1, 160), 20, model_from=3, playing=True)

    assert (actions[0][0], actions[0][1].kind) == (9, "yield")


def test_interrupt_waits_for_talk_limit_and_decision_slot() -> None:
    controller = TurnController(persona(talk_limit_ms=1600, interruptiveness=1.0), 1, 160)

    actions = collect(controller, 30, model_from=0)

    assert (actions[0][0], actions[0][1].kind) == (13, "interrupt")
    assert controller.may_start_speech()


def test_zero_interruptiveness_never_interrupts() -> None:
    controller = TurnController(persona(talk_limit_ms=160), 1, 160)

    assert collect(controller, 100, model_from=0) == []
    assert not controller.may_start_speech()


def test_backchannel_is_forced_by_maximum_gap() -> None:
    settings = persona(backchannel_min_ms=800, backchannel_max_ms=800, backchannels_per_second=0.0)

    actions = collect(TurnController(settings, 1, 160), 11, model_from=0)

    assert [(tick, action.kind) for tick, action in actions] == [
        (4, "backchannel"),
        (9, "backchannel"),
    ]


def test_silence_prompt_happens_once_until_activity_returns() -> None:
    controller = TurnController(persona(silence_tolerance_ms=800), 1, 160)

    actions = collect(controller, 20)

    assert [(tick, action.kind) for tick, action in actions] == [(5, "prompt")]


def test_inserts_depend_on_who_is_speaking() -> None:
    settings = persona(speech_inserts_per_minute=60_000)

    assert collect(TurnController(settings, 1, 160), 1, playing=True)[0][1].kind == "vocal_tic"
    assert collect(TurnController(settings, 1, 160), 1)[0][1].kind == "non_directed"


def test_decisions_are_repeatable_for_a_seed() -> None:
    settings = TurnPersona(talk_limit_ms=160, interruptiveness=0.5, speech_inserts_per_minute=30)

    assert collect(TurnController(settings, 8, 160), 500, model_from=10) == collect(
        TurnController(settings, 8, 160), 500, model_from=10
    )


def test_scenario_persona_values_map_to_settings() -> None:
    settings = TurnPersona.from_scenario({"patience": 0.5, "interruptiveness": 0.15}, True)

    assert settings.talk_limit_ms == 9000
    assert settings.interruptiveness == 0.15
    assert settings.speech_inserts_per_minute > 0
    with pytest.raises(ValueError, match="patience"):
        TurnPersona.from_scenario({"patience": 2}, False)
    with pytest.raises(ValueError, match="interruptiveness"):
        TurnPersona.from_scenario({"interruptiveness": 2}, False)


def test_llm_decider_can_interrupt_and_replay() -> None:
    gateway = LLMGateway("record", FakeProvider(lambda request, call: '{"interrupt": true}'))
    decider = LLMInterruptDecider(gateway, "fake")
    controller = TurnController(persona(talk_limit_ms=160), 1, 160, decider)

    assert collect(controller, 2, model_from=0)[0][1].kind == "interrupt"

    replay = LLMGateway("replay", entries=gateway.entries)
    replayed = TurnController(
        persona(talk_limit_ms=160), 1, 160, LLMInterruptDecider(replay, "fake")
    )
    assert collect(replayed, 2, model_from=0)[0][1].kind == "interrupt"


def test_invalid_llm_turn_response_is_rejected() -> None:
    gateway = LLMGateway("record", FakeProvider(lambda request, call: '{"interrupt": "yes"}'))

    with pytest.raises(ValueError, match="boolean"):
        asyncio.run(LLMInterruptDecider(gateway, "fake").should_interrupt("hello", 1000))


class Policy:
    def __init__(self, outputs: Mapping[int, PolicyOutput] | None = None) -> None:
        self.outputs = dict(outputs or {})
        self.spans: list[DeliveredSpan] = []

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        return PolicyOutput()

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        return self.outputs.get(view.tick)

    def on_delivered(self, span: DeliveredSpan) -> tuple[SuiteEvent, ...]:
        self.spans.append(span)
        return ()

    def is_finished(self, view: ConversationView) -> bool:
        return False


def play(counterpart: PolicyCounterpart, ticks: int, model_from: int) -> list[MediaFrame]:
    async def run() -> list[MediaFrame]:
        frames = []
        for tick in range(ticks):
            if tick:
                pcm = LOUD if tick - 1 >= model_from else QUIET
                await counterpart.hear(MediaFrame(tick - 1, pcm, 24_000), tick)
            frames.append(await counterpart.render(tick))
        return frames

    return asyncio.run(run())


def counterpart(policy: Policy, settings: TurnPersona) -> PolicyCounterpart:
    speech = SyntheticSpeech(1)
    return PolicyCounterpart(policy, 160, speech, turns=TurnController(settings, 1, 160))


def test_yield_cuts_current_speech_without_credit() -> None:
    long_speech = SpeechSegment("one two three four five six seven eight", ("fact",))
    policy = Policy({0: PolicyOutput((long_speech,))})
    speaker = counterpart(policy, persona())

    play(speaker, 12, model_from=2)

    span = policy.spans[0]
    assert span.interrupted is True
    assert span.tags == ()
    assert span.end_tick == 8


def test_queued_speech_waits_for_model_to_stop() -> None:
    policy = Policy({2: PolicyOutput((SpeechSegment("hello"),))})
    speaker = counterpart(policy, persona())

    frames = play(speaker, 8, model_from=0)

    assert all(frame.text is None for frame in frames)


def test_interrupt_starts_speech_over_model() -> None:
    settings = persona(talk_limit_ms=160, interruptiveness=1.0)
    speaker = counterpart(Policy(), settings)

    frames = play(speaker, 15, model_from=0)

    assert frames[13].text == "Sorry, can I jump in?"


def test_overlay_is_mixed_without_replacing_speech_and_recorded() -> None:
    settings = persona(backchannel_min_ms=160, backchannel_max_ms=160)
    speaker = counterpart(Policy(), settings)

    frames = play(speaker, 8, model_from=0)
    events = speaker.drain_events()

    assert any(any(frame.pcm) for frame in frames)
    recorded = [data for name, _, data in events if name == "counterpart_event"]
    assert recorded[0]["kind"] == "backchannel"
    assert recorded[0]["end_ms"] > recorded[0]["start_ms"]


def scenario_with_persona(**values: object) -> tuple[object, Scenario]:
    loaded = load_suite(str(DEMO))
    scenario = valid_scenarios(loaded)[0]
    data = {**scenario.data, "persona": values}
    return loaded.suite, replace(scenario, data=data)


def free_run(adapter: MockAdapter, ticks: int = 120, **values: object) -> ScenarioRun:
    suite, scenario = scenario_with_persona(**values)
    speech = adapter._speech
    assert speech is not None
    return asyncio.run(
        run_scenario(
            suite,  # type: ignore[arg-type]
            scenario,
            adapter,
            seed=3,
            max_ticks=ticks,
            condition="ablate_turns",
            speech=speech,
        )
    )


def long_text() -> str:
    return " ".join(f"word{index}" for index in range(80))


def actions(run: ScenarioRun) -> list[str]:
    return [
        str(event.data["action"]) for event in run.outcome.events if event.type == "turn_action"
    ]


def test_scripted_long_turn_triggers_listener_actions() -> None:
    adapter = MockAdapter({1: long_text()}, echo=False, speech=SyntheticSpeech(1))

    run = free_run(adapter, patience=0.0, interruptiveness=1.0, yield_when_interrupting_ms=320)

    found = actions(run)
    assert "backchannel" in found
    assert "interrupt" in found
    assert "yield" in found


def store(path: Path, run: ScenarioRun, adapter: MockAdapter) -> Path:
    suite = load_suite(str(DEMO)).suite
    spec = RunSpec("demo", "mock", "audio", 3, 120, 160, "ablate_turns", audio_backend="synthetic")
    return write_run(
        path / "run",
        suite,
        spec,
        adapter.describe(),
        run.trajectory,
        run.result,
        run.decision,
        tracks=run.tracks,
        effects=run.effects,
        condition=run.condition_details,
    )


def test_timeline_and_metrics_receive_counterpart_events(tmp_path: Path) -> None:
    adapter = MockAdapter({1: long_text()}, echo=False, speech=SyntheticSpeech(1))
    run = free_run(adapter, patience=0.0, interruptiveness=1.0, yield_when_interrupting_ms=320)
    path = store(tmp_path, run, adapter)

    timeline = build_timeline(path)

    assert timeline["counterpart"]["events"][0]["kind"] == "backchannel"
    assert any(item["interrupted"] for item in timeline["counterpart"]["segments"])
    assert load_run(path).decision is not None


def test_cli_free_turn_run_replays(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "runs"
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]
    extra = ["--audio-backend", "synthetic", "--condition", "ablate_turns", "--seed", "3"]

    assert cli.main([*arguments, *extra, "--ticks", "80", "--out", str(out)]) == 0

    path = find_run_dirs(out)[0]
    report = asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))
    assert report.ok, report.problems
    assert json.loads((path / "manifest.json").read_text(encoding="utf-8"))["run"]["seed"] == 3


def test_strict_condition_has_no_turn_actions() -> None:
    loaded = load_suite(str(DEMO))
    scenario = valid_scenarios(loaded)[0]
    speech = SyntheticSpeech(1)
    adapter = MockAdapter({1: long_text()}, echo=False, speech=speech)

    run = asyncio.run(run_scenario(loaded.suite, scenario, adapter, max_ticks=80, speech=speech))

    assert actions(run) == []


def actions_for(
    controller: TurnController,
    speaking: list[bool],
    playing: list[bool] | None = None,
    pending: list[bool] | None = None,
) -> list[tuple[int, str]]:
    async def run() -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        for tick, value in enumerate(speaking):
            controller.observe_model(tick, value)
            active = playing[tick] if playing else False
            queued = pending[tick] if pending else False
            for action in await controller.decide(tick, active, queued):
                found.append((tick, action.kind))
        return found

    return asyncio.run(run())


def paused_speech(ticks: int) -> list[bool]:
    return [tick % 7 != 6 for tick in range(ticks)]


def test_short_model_pause_keeps_turn_state() -> None:
    settings = persona(talk_limit_ms=3000, interruptiveness=1.0)
    controller = TurnController(settings, 1, 160)

    found = actions_for(controller, paused_speech(40))

    assert (25, "interrupt") in found
    controller = TurnController(persona(), 1, 160)
    assert actions_for(controller, paused_speech(10), [True] * 10)[0] == (7, "yield")


def test_short_model_pause_does_not_open_the_floor() -> None:
    controller = TurnController(persona(), 1, 160)

    for tick, speaking in enumerate([True] * 5 + [False] * 3):
        controller.observe_model(tick, speaking)
    assert not controller.may_start_speech()

    controller.observe_model(8, False)
    assert controller.may_start_speech()


def tone_tick(level: int) -> bytes:
    return level.to_bytes(2, "little", signed=True) * 3840


def test_steady_noise_stops_counting_as_speech() -> None:
    gate = SpeechGate()

    results = [gate.update(tone_tick(800)) for _ in range(12)]

    assert results[-1] is False


def speech_fixture_ticks() -> list[bytes]:
    with wave.open(str(ROOT / "tests/fixtures/speech.wav"), "rb") as source:
        pcm = source.readframes(source.getnframes())
    return [pcm[i : i + TICK_BYTES] for i in range(0, len(pcm) - TICK_BYTES, TICK_BYTES)]


def test_real_and_synthetic_speech_pass_gate() -> None:
    fixture = SpeechGate()
    active = [tick for tick in speech_fixture_ticks()[3:13]]
    assert sum(fixture.update(tick) for tick in active) >= 8

    synthetic = SpeechGate()
    pcm = SyntheticSpeech(1).render(" ".join(["word"] * 20))
    ticks = [pcm[i : i + TICK_BYTES] for i in range(0, len(pcm) - TICK_BYTES, TICK_BYTES)]
    assert sum(synthetic.update(tick) for tick in ticks) >= len(ticks) * 0.8


def test_closing_line_can_start_while_model_talks() -> None:
    ending = PolicyOutput((SpeechSegment("goodbye"),), end_conversation=True)
    speaker = counterpart(Policy({3: ending}), persona())

    frames = play(speaker, 5, model_from=0)

    assert frames[3].text == "goodbye"


def test_interrupt_permission_ends_with_interrupting_speech() -> None:
    controller = TurnController(persona(talk_limit_ms=160, interruptiveness=1.0), 1, 160)
    actions_for(controller, [True] * 15)
    assert controller.interrupting

    controller.speech_delivered()

    assert not controller.may_start_speech()


def test_yield_blocks_reinterrupt_until_model_stops() -> None:
    settings = persona(talk_limit_ms=160, interruptiveness=1.0, yield_when_interrupted_ms=320)
    controller = TurnController(settings, 1, 160)
    speaking = [True] * 40 + [False] * 5 + [True] * 20
    playing = [tick < 2 for tick in range(65)]

    found = actions_for(controller, speaking, playing)

    assert found[0] == (1, "yield")
    assert next(tick for tick, kind in found if kind == "interrupt") >= 45


def test_prompt_is_not_repeated_after_its_own_playback() -> None:
    controller = TurnController(persona(silence_tolerance_ms=800), 1, 160)
    playing = [5 <= tick <= 8 for tick in range(40)]

    found = actions_for(controller, [False] * 40, playing)

    assert [kind for _, kind in found].count("prompt") == 1


def test_inserts_during_model_speech_use_both_types() -> None:
    kinds = set()
    for seed in range(12):
        controller = TurnController(persona(speech_inserts_per_minute=60_000), seed, 160)
        kinds.add(actions_for(controller, [True])[0][1])

    assert kinds == {"vocal_tic", "non_directed"}


def test_short_model_sound_does_not_force_yield_after_it_stops() -> None:
    controller = TurnController(persona(), 1, 160)
    speaking = [False] * 8 + [True] * 4 + [False] * 8

    assert actions_for(controller, speaking, [True] * 20) == []


def test_listener_actions_need_current_model_speech() -> None:
    settings = persona(
        talk_limit_ms=160,
        interruptiveness=1.0,
        backchannel_min_ms=160,
        backchannel_max_ms=160,
    )
    controller = TurnController(settings, 1, 160)
    speaking = [True] * 12 + [False] * 3

    found = actions_for(controller, speaking)

    assert all(tick < 12 for tick, kind in found if kind in {"interrupt", "backchannel"})


@pytest.mark.parametrize("value", [None, [1], "x"])
def test_invalid_persona_container_is_rejected(value: object) -> None:
    adapter = MockAdapter(echo=False, speech=SyntheticSpeech(1))
    suite, scenario = scenario_with_persona()
    bad = replace(scenario, data={**scenario.data, "persona": value})

    with pytest.raises(ValueError, match="persona"):
        asyncio.run(
            run_scenario(
                suite,  # type: ignore[arg-type]
                bad,
                adapter,
                max_ticks=1,
                condition="ablate_turns",
                speech=SyntheticSpeech(1),
            )
        )


def test_non_directed_speech_waits_for_model_turn_after_counterpart() -> None:
    controller = TurnController(persona(speech_inserts_per_minute=60_000), 1, 160)
    controller.speech_delivered()

    assert actions_for(controller, [False, False]) == []
    assert actions_for(controller, [True, False, False, False, False, False, False])


def test_injected_interrupt_is_not_credit_or_eligible() -> None:
    policy = Policy()
    speaker = counterpart(policy, persona(talk_limit_ms=160, interruptiveness=1.0))

    play(speaker, 40, model_from=0)
    delivered = [data for name, _, data in speaker.drain_events() if name == "segment_delivered"]

    assert policy.spans[0].text == INTERRUPT_TEXT
    assert policy.spans[0].tags == ()
    assert delivered[0]["injected"] is True


def test_timeline_marks_cut_and_injected_segments_ineligible(tmp_path: Path) -> None:
    adapter = MockAdapter({1: long_text()}, echo=False, speech=SyntheticSpeech(1))
    run = free_run(adapter, patience=0.0, interruptiveness=1.0, yield_when_interrupting_ms=320)
    path = store(tmp_path, run, adapter)

    segments = build_timeline(path)["counterpart"]["segments"]

    assert any(item["interrupted"] and item["eligible"] is False for item in segments)


def test_overlay_is_recorded_when_it_starts_and_clamped_to_run(tmp_path: Path) -> None:
    adapter = MockAdapter({1: long_text()}, echo=False, speech=SyntheticSpeech(1))
    suite, scenario = scenario_with_persona(patience=1.0, interruptiveness=0.0)
    speech = adapter._speech
    assert speech is not None
    run = asyncio.run(
        run_scenario(
            suite,  # type: ignore[arg-type]
            scenario,
            adapter,
            seed=3,
            max_ticks=80,
            condition="ablate_turns",
            speech=speech,
        )
    )
    backchannels = [event for event in run.outcome.events if event.type == "counterpart_event"]
    assert backchannels

    path = store(tmp_path, run, adapter)
    timeline = build_timeline(path)

    assert all(
        item["end_ms"] <= timeline["duration_ms"] for item in timeline["counterpart"]["events"]
    )


def test_turn_randomness_varies_by_scenario() -> None:
    settings = TurnPersona(speech_inserts_per_minute=60)
    first = collect(TurnController(settings, 11, 160), 400, model_from=0)
    second = collect(TurnController(settings, 12, 160), 400, model_from=0)

    assert first != second


@pytest.mark.parametrize(
    "values",
    [
        {"patience": None},
        {"patience": True},
        {"patience": "0.5"},
        {"patience": float("nan")},
        {"interruptiveness": -0.1},
        {"silence_tolerance_ms": -1},
        {"silence_tolerance_ms": 12.9},
        {"silence_tolerance_ms": float("inf")},
        {"yield_when_interrupted_ms": -5},
    ],
)
def test_invalid_persona_values_are_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TurnPersona.from_scenario(values, False)


def test_invalid_persona_dataclass_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        TurnPersona(talk_limit_ms=-1)
    with pytest.raises(ValueError):
        TurnPersona(backchannels_per_second=float("inf"))


def test_old_free_condition_run_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]
    extra = ["--audio-backend", "synthetic", "--condition", "ablate_turns", "--out", str(out)]
    assert cli.main([*arguments, *extra]) == 0
    path = find_run_dirs(out)[0]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["condition"]["applied"]["turn_taking"] = "scripted"
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TrajectoryError, match="predates"):
        asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))
