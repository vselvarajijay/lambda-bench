from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from lambda_duplex import cli
from lambda_duplex.suites.base import Scenario
from lambda_duplex.suites.loader import load_suite
from suites.interviewer.disclosure import DisclosureEngine, Unlock, UtteranceSignals
from suites.interviewer.simulate import simulate

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "suites/interviewer"
EXAMPLE = SUITE / "dev/ad_feedback_0007.yaml"
SCRIPT = ROOT / "tests/fixtures/good_interview.yaml"


def raw_example() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    return data


def scenario(raw: dict[str, Any] | None = None) -> Scenario:
    return load_suite(str(SUITE)).suite.load_scenario(raw or raw_example())


BASE = scenario()


def signals(**values: Any) -> UtteranceSignals:
    return UtteranceSignals.from_mapping(values, BASE)


def mark_disclosed(engine: DisclosureEngine, fact: str, now: int = 0) -> None:
    engine.unlocked[fact] = Unlock(fact, now, "test", (), "setup")
    engine.on_delivered(fact, now)


def fact_engine(
    unlock: dict[str, Any], prior: tuple[str, ...] = (), fact_index: int = 0
) -> DisclosureEngine:
    data = copy.deepcopy(raw_example())
    data["hidden_story"]["facts"][fact_index]["unlock"] = unlock
    engine = DisclosureEngine(scenario(data))
    for fact in prior:
        mark_disclosed(engine, fact)
    return engine


def unlocked(engine: DisclosureEngine, signal: UtteranceSignals, now: int = 10_000) -> bool:
    result = engine.on_utterance(signal, now, "u")
    return "f1" in {item.fact_id for item in result.unlocks}


@pytest.mark.parametrize(
    ("unlock", "prior", "signal", "expected"),
    [
        ({"asked_about": "takeaway"}, (), {"asked_about": ["takeaway"]}, True),
        ({"asked_about": "takeaway"}, (), {"asked_about": ["emotion"]}, False),
        ({"open_ended": True}, (), {"open_ended": True}, True),
        ({"open_ended": True}, (), {}, False),
        ({"not_leading": True}, (), {"not_leading": True}, True),
        ({"not_leading": True}, (), {}, False),
        ({"disclosed": "f2"}, ("f2",), {}, True),
        ({"disclosed": "f2"}, (), {}, False),
        ({"followed_up_on": "f2"}, ("f2",), {"followed_up_on": ["f2"]}, True),
        ({"followed_up_on": "f2"}, (), {"followed_up_on": ["f2"]}, False),
        ({"followed_up_on": "f2"}, ("f2",), {}, False),
        (
            {"acknowledged": {"disclosed": "f2"}},
            ("f2",),
            {"acknowledged": [{"disclosed": "f2"}]},
            True,
        ),
        ({"acknowledged": {"disclosed": "f2"}}, (), {"acknowledged": [{"disclosed": "f2"}]}, False),
        ({"not": {"asked_about": "emotion"}}, (), {"asked_about": ["emotion"]}, False),
        ({"not": {"asked_about": "emotion"}}, (), {}, True),
        (
            {"any": [{"asked_about": "emotion"}, {"open_ended": True}]},
            (),
            {"open_ended": True},
            True,
        ),
        (
            {"all": [{"asked_about": "emotion"}, {"open_ended": True}]},
            (),
            {"open_ended": True},
            False,
        ),
    ],
)
def test_predicate_table(
    unlock: dict[str, Any], prior: tuple[str, ...], signal: dict[str, Any], expected: bool
) -> None:
    assert unlocked(fact_engine(unlock, prior), signals(**signal)) is expected


def test_classified_conditions_must_hold_on_same_utterance() -> None:
    engine = fact_engine({"all": [{"asked_about": "takeaway"}, {"open_ended": True}]})

    assert not unlocked(engine, signals(asked_about=["takeaway"]), 1)
    assert not unlocked(engine, signals(open_ended=True), 2)
    assert unlocked(engine, signals(asked_about=["takeaway"], open_ended=True), 3)


def test_moment_acknowledgement_requires_completed_moment() -> None:
    engine = fact_engine({"acknowledged": {"moment": "m1"}}, ("f2",))
    signal = signals(acknowledged=[{"moment": "m1"}])

    assert not unlocked(engine, signal, 1)
    engine.on_moment("m1", 2)
    assert unlocked(engine, signal, 3)


def test_moment_must_be_known_and_triggered() -> None:
    engine = DisclosureEngine(scenario())

    with pytest.raises(ValueError, match="unknown moment"):
        engine.on_moment("missing", 1)
    with pytest.raises(ValueError, match="not been triggered"):
        engine.on_moment("m1", 1)


def test_elapsed_time_boundary() -> None:
    engine = fact_engine({"elapsed_since": {"event": {"disclosed": "f2"}, "min_s": 5}}, ("f2",))

    assert not unlocked(engine, signals(), 4_999)
    assert unlocked(engine, signals(), 5_000)


def test_interruption_at_or_after_event_blocks_condition() -> None:
    expression = {"no_interruption_since": {"event": {"disclosed": "f2"}}}
    clean = fact_engine(expression, ("f2",))
    interrupted = fact_engine(expression, ("f2",))
    interrupted.on_interruption(0)

    assert unlocked(clean, signals())
    assert not unlocked(interrupted, signals())


@pytest.mark.parametrize(
    "expression",
    [
        {"not": {"no_interruption_since": {"event": {"disclosed": "f2"}}}},
        {"not": {"elapsed_since": {"event": {"disclosed": "f2"}, "min_s": 1}}},
    ],
)
def test_negated_event_timing_requires_the_event(expression: dict[str, Any]) -> None:
    unlock = {"all": [{"open_ended": True}, expression]}

    assert not unlocked(fact_engine(unlock), signals(open_ended=True))


def test_fact_counts_as_disclosed_only_after_delivery() -> None:
    engine = DisclosureEngine(scenario())
    first = engine.on_utterance(signals(asked_about=["emotion"]), 1, "u1")

    assert [item.fact_id for item in first.unlocks] == ["f5"]
    follow_up = signals(followed_up_on=["f5"], open_ended=True)
    assert engine.on_utterance(follow_up, 2, "u2").unlocks == ()
    engine.on_delivered("f5", 3)
    assert [item.fact_id for item in engine.on_utterance(follow_up, 4, "u3").unlocks] == ["f2"]


def test_undelivered_unknown_fact_is_rejected() -> None:
    with pytest.raises(ValueError, match="not unlocked"):
        DisclosureEngine(scenario()).on_delivered("f3", 1)


def test_time_cannot_move_backward() -> None:
    engine = DisclosureEngine(scenario())
    engine.on_tick(5)

    with pytest.raises(ValueError, match="backward"):
        engine.on_tick(4)


def deep_ready() -> DisclosureEngine:
    engine = DisclosureEngine(scenario())
    for now, signal, fact in (
        (1_000, signals(asked_about=["emotion"]), "f5"),
        (2_000, signals(followed_up_on=["f5"], open_ended=True), "f2"),
    ):
        result = engine.on_utterance(signal, now, f"u{now}")
        assert [item.fact_id for item in result.unlocks] == [fact]
        engine.on_delivered(fact, now + 500)
    return engine


def arm(engine: DisclosureEngine, now: int = 3_000) -> int:
    result = engine.on_utterance(signals(followed_up_on=["f2"]), now, "deep")
    assert [item.fact_id for item in result.trailoffs] == ["f3"]
    assert result.trailoffs[0].utterance_id == "deep"
    return result.trailoffs[0].request_id


def test_deep_fact_unlocks_after_trailoff_silence() -> None:
    engine = deep_ready()
    request = arm(engine)
    assert engine.on_trailoff_played(request, 4_000)

    assert engine.on_tick(5_999) == ()
    unlock = engine.on_tick(6_000)
    assert [item.fact_id for item in unlock] == ["f3"]
    assert unlock[0].source == "silence"
    assert unlock[0].utterance_id == "deep"


def test_silence_requires_source_fact_to_be_disclosed() -> None:
    expression = {"silence_after_trailoff": {"fact": "f2", "min_s": 2}}
    engine = fact_engine(expression)

    assert engine.on_utterance(signals(), 1_000, "u").trailoffs == ()


def test_onset_cancels_armed_and_running_timers() -> None:
    armed = deep_ready()
    request = arm(armed)
    armed.on_substantive_onset(3_200)
    assert armed.on_trailoff_played(request, 4_000) is False
    assert armed.on_tick(9_000) == ()

    running = deep_ready()
    running.on_trailoff_played(arm(running), 4_000)
    running.on_substantive_onset(5_000)
    assert running.on_tick(9_000) == ()


def test_new_utterance_supersedes_old_trailoff_request() -> None:
    engine = deep_ready()
    old = arm(engine, 3_000)
    new = arm(engine, 3_200)

    assert old != new
    assert engine.on_trailoff_played(old, 3_300) is False
    assert engine.on_tick(9_000) == ()


def test_matured_silence_is_kept_when_onset_arrives_at_deadline() -> None:
    engine = deep_ready()
    engine.on_trailoff_played(arm(engine), 4_000)

    assert [item.fact_id for item in engine.on_substantive_onset(6_000)] == ["f3"]


def test_failed_silence_is_not_reused_later() -> None:
    unlock = {
        "any": [
            {
                "all": [
                    {"disclosed": "f2"},
                    {"followed_up_on": "f2"},
                    {"silence_after_trailoff": {"fact": "f2", "min_s": 2}},
                    {"not": {"disclosed": "f6"}},
                ]
            },
            {
                "all": [
                    {"disclosed": "f2"},
                    {"followed_up_on": "f2"},
                    {"silence_after_trailoff": {"fact": "f2", "min_s": 2}},
                    {"asked_about": "action"},
                ]
            },
        ]
    }
    engine = fact_engine(unlock, ("f2",), 2)
    request = engine.on_utterance(signals(followed_up_on=["f2"]), 1_000, "u").trailoffs[0]
    engine.on_trailoff_played(request.request_id, 1_500)
    mark_disclosed(engine, "f6", 2_000)
    assert engine.on_tick(3_500) == ()

    result = engine.on_utterance(signals(followed_up_on=["f2"], asked_about=["action"]), 4_000, "x")

    assert "f3" not in {item.fact_id for item in result.unlocks}


def test_multiple_silence_paths_use_earliest_valid_timer() -> None:
    paths = [
        {
            "all": [
                {"disclosed": "f2"},
                {"followed_up_on": "f2"},
                {"silence_after_trailoff": {"fact": "f2", "min_s": seconds}},
            ]
        }
        for seconds in (10, 2)
    ]
    engine = fact_engine({"any": paths}, ("f2",), 2)
    result = engine.on_utterance(signals(followed_up_on=["f2"]), 1_000, "u")
    assert len(result.trailoffs) == 1
    engine.on_trailoff_played(result.trailoffs[0].request_id, 1_500)

    unlocks = engine.on_tick(3_500)

    assert [item.fact_id for item in unlocks] == ["f3"]
    assert engine.on_tick(20_000) == ()


def test_scripted_pause_does_not_count_as_silence() -> None:
    engine = deep_ready()
    engine.on_trailoff_played(arm(engine), 4_000)

    engine.on_pause_started(5_000, 3_000)

    assert engine.on_tick(8_999) == ()
    assert [item.fact_id for item in engine.on_tick(9_000)] == ["f3"]


def test_simulator_uses_scripted_pause_length() -> None:
    steps = [
        *script_steps()[:4],
        {"moment": "m1", "after_s": 0},
        {"wait_s": 1.0, "after_s": 0},
    ]

    rows = simulate(scenario(), steps)

    assert not any(row["event"] == "disclosed" and row["fact_id"] == "f3" for row in rows)


def test_speech_during_silence_holds_timer_until_it_is_classified() -> None:
    engine = deep_ready()
    engine.on_trailoff_played(arm(engine), 4_160)
    engine.on_model_speech_start(5_160)

    assert engine.on_tick(6_240) == ()
    assert engine.on_model_speech_end(5_160, 7_160, substantive=True) == ()
    assert engine.on_tick(9_000) == ()


def test_backchannel_releases_silence_hold() -> None:
    engine = deep_ready()
    engine.on_trailoff_played(arm(engine), 4_000)
    engine.on_model_speech_start(5_000)
    engine.on_tick(5_400)

    assert engine.on_model_speech_end(5_000, 5_400, substantive=False) == ()
    assert [item.fact_id for item in engine.on_tick(6_000)] == ["f3"]


def test_speech_after_deadline_keeps_completed_silence() -> None:
    engine = deep_ready()
    engine.on_trailoff_played(arm(engine), 4_000)
    engine.on_model_speech_start(6_000)

    unlocks = engine.on_model_speech_end(6_000, 8_000, substantive=True)

    assert [(item.fact_id, item.time_ms) for item in unlocks] == [("f3", 6_000)]


def test_utterance_settles_silence_at_its_onset() -> None:
    engine = deep_ready()
    engine.on_trailoff_played(arm(engine), 4_000)

    early = engine.on_utterance(signals(), 7_000, "late", onset_ms=5_000)

    assert early.unlocks == ()
    assert engine.on_tick(9_000) == ()


def test_silence_before_trailoff_does_not_count() -> None:
    engine = deep_ready()
    arm(engine)

    assert engine.on_tick(20_000) == ()


def test_generic_prompt_cannot_reach_deep_fact() -> None:
    engine = deep_ready()

    result = engine.on_utterance(signals(open_ended=True, not_leading=True), 3_000, "generic")

    assert result.unlocks == ()
    assert result.trailoffs == ()
    assert engine.on_tick(30_000) == ()


def test_acknowledging_scripted_moment_is_another_deep_path() -> None:
    engine = deep_ready()
    engine.on_moment("m1", 3_000)

    result = engine.on_utterance(
        signals(followed_up_on=["f2"], acknowledged=[{"moment": "m1"}]), 4_000, "u"
    )

    assert [item.fact_id for item in result.unlocks] == ["f3"]


@pytest.mark.parametrize(
    "values",
    [
        [],
        {"unknown": True},
        {"open_ended": "yes"},
        {"acknowledged": ["m1"]},
        {"asked_about": "emotion"},
        {"asked_about": ["missing"]},
        {"followed_up_on": ["missing"]},
        {"acknowledged": [{"moment": "missing"}]},
    ],
)
def test_invalid_signals_are_rejected(values: Any) -> None:
    with pytest.raises(ValueError):
        UtteranceSignals.from_mapping(values, BASE)


def script_steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = yaml.safe_load(SCRIPT.read_text(encoding="utf-8"))["steps"]
    return steps


def test_good_script_reaches_every_fact_with_reasons() -> None:
    rows = simulate(scenario(), script_steps())

    assert [row["fact_id"] for row in rows if row["event"] == "disclosed"] == [
        "f1",
        "f5",
        "f2",
        "f3",
        "f6",
        "f4",
    ]
    deep = next(row for row in rows if row["event"] == "unlocked" and row["fact_id"] == "f3")
    assert deep["source"] == "silence"
    assert deep["utterance_id"] == "u3"
    assert any("silence_after_trailoff" in reason for reason in deep["reasons"])


def test_simulator_ticks_silence_during_after_delay() -> None:
    steps = [*script_steps()[:4], {"onset": True, "after_s": 10}]

    rows = simulate(scenario(), steps)

    assert any(row["event"] == "disclosed" and row["fact_id"] == "f3" for row in rows)


@pytest.mark.parametrize(
    "steps",
    [
        "bad",
        ["bad"],
        [{"wait_s": 1, "onset": True}],
        [{"wait_s": float("nan")}],
        [{"wait_s": float("inf")}],
        [{"wait_s": 1, "after_s": -1}],
        [{"wait_s": 1, "after": 1}],
        [{"onset": False}],
        [{"utterance": ["asked_about"]}],
        [{"moment": "missing"}],
        [{"wait_s": 1.0e308}],
        [{"wait_s": 1, "after_s": 10**400}],
    ],
)
def test_invalid_script_steps_are_rejected(steps: Any) -> None:
    with pytest.raises(ValueError):
        simulate(scenario(), steps)


def cli_arguments(script: Path) -> list[str]:
    return [
        "interviewer",
        "simulate",
        "dev/ad_feedback_0007.yaml",
        "--script",
        str(script),
        "--suite",
        str(SUITE),
    ]


def test_cli_simulation_is_repeatable(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(cli_arguments(SCRIPT)) == 0
    first = capsys.readouterr().out
    assert cli.main(cli_arguments(SCRIPT)) == 0

    assert capsys.readouterr().out == first
    assert json.loads(first)[0]["fact_id"] == "f1"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("steps:\n  - {wait_s: 1, onset: true}\n", "exactly one action"),
        ("wrong: []\n", "only steps"),
    ],
)
def test_cli_rejects_invalid_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, message: str
) -> None:
    script = tmp_path / "bad.yaml"
    script.write_text(content, encoding="utf-8")

    assert cli.main(cli_arguments(script)) == 2
    assert message in capsys.readouterr().err
