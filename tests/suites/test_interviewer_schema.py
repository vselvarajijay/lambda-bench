from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from lambda_duplex import cli
from lambda_duplex.suites.loader import (
    load_scenario_files,
    load_suite,
    validate_scenario_data,
)

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "suites/interviewer"
EXAMPLE = SUITE / "dev/ad_feedback_0007.yaml"
Mutation = Callable[[dict[str, Any]], None]


def example() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    return data


def facts(data: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = data["hidden_story"]["facts"]
    return result


def issues(data: dict[str, Any]) -> list[str]:
    result = validate_scenario_data(load_suite(str(SUITE)), data, "dev")
    return [f"{issue.path}: {issue.message}" for issue in result.issues]


def leak_in_brief(data: dict[str, Any]) -> None:
    data["brief"]["full"] += " " + facts(data)[2]["content"]


def leak_in_compact(data: dict[str, Any]) -> None:
    data["brief"]["compact"] += " " + facts(data)[0]["content"]


def paraphrase_leak(data: dict[str, Any]) -> None:
    facts(data)[0]["paraphrases"].append("family cooking during a neighborhood blackout")


def word_run_leak(data: dict[str, Any]) -> None:
    data["brief"]["full"] += " Their father's home oxygen equipment came up."


def contradiction_leak(data: dict[str, Any]) -> None:
    data["brief"]["compact"] += " I didn\u2019t really pay attention to the tagline"


def script_leak(data: dict[str, Any]) -> None:
    data["brief"]["full"] += " I don't know, a little close to home."


def unknown_fact(data: dict[str, Any]) -> None:
    facts(data)[2]["unlock"]["all"][0] = {"disclosed": "missing"}


def unknown_topic(data: dict[str, Any]) -> None:
    facts(data)[0]["unlock"] = {"asked_about": "missing"}


def fact_topic_unknown(data: dict[str, Any]) -> None:
    facts(data)[0]["topic"] = "missing"


def unknown_predicate(data: dict[str, Any]) -> None:
    facts(data)[0]["unlock"] = {"magic_phrase": "please"}


def bad_shape(data: dict[str, Any]) -> None:
    facts(data)[0]["unlock"] = {"asked_about": "takeaway", "open_ended": True}


def bad_argument(data: dict[str, Any]) -> None:
    facts(data)[2]["unlock"]["all"][2] = {"silence_after_trailoff": {"fact": "f2", "min_s": 0}}


def nan_wait(data: dict[str, Any]) -> None:
    facts(data)[2]["unlock"]["all"][2] = {
        "silence_after_trailoff": {"fact": "f2", "min_s": float("nan")}
    }


def long_wait(data: dict[str, Any]) -> None:
    facts(data)[2]["unlock"]["all"][2] = {"silence_after_trailoff": {"fact": "f2", "min_s": 999}}


def cycle(data: dict[str, Any]) -> None:
    facts(data)[1]["unlock"] = {"all": [{"disclosed": "f3"}, {"followed_up_on": "f3"}]}


def self_dependency(data: dict[str, Any]) -> None:
    facts(data)[3]["unlock"] = {"all": [{"disclosed": "f4"}, {"followed_up_on": "f4"}]}


def contradiction_clause(data: dict[str, Any]) -> None:
    facts(data)[0]["unlock"] = {
        "all": [{"asked_about": "takeaway"}, {"not": {"asked_about": "takeaway"}}]
    }


def ordering_contradiction(data: dict[str, Any]) -> None:
    facts(data)[2]["unlock"]["all"].append({"not": {"disclosed": "f5"}})


def surface_dependency(data: dict[str, Any]) -> None:
    facts(data)[0]["unlock"] = {"disclosed": "f2"}


def probe_without_follow_up(data: dict[str, Any]) -> None:
    facts(data)[1]["unlock"] = {"all": [{"asked_about": "emotion"}, {"open_ended": True}]}


def deep_without_probe(data: dict[str, Any]) -> None:
    facts(data)[2]["unlock"] = {"asked_about": "emotion"}


def deep_bypass(data: dict[str, Any]) -> None:
    facts(data)[2]["unlock"] = {"any": [{"disclosed": "f2"}, {"asked_about": "emotion"}]}


def negative_only(data: dict[str, Any]) -> None:
    facts(data)[0]["unlock"] = {"not": {"asked_about": "emotion"}}


def duplicate_fact(data: dict[str, Any]) -> None:
    facts(data)[1]["id"] = "f1"


def uncovered_topic(data: dict[str, Any]) -> None:
    data["brief"]["required_topics"].append({"id": "price", "label": "Price", "description": "x"})


def long_compact(data: dict[str, Any]) -> None:
    data["brief"]["compact"] = "word " * 121


def blank_full(data: dict[str, Any]) -> None:
    data["brief"]["full"] = "   "


def newline_id(data: dict[str, Any]) -> None:
    facts(data)[5]["id"] = "f6\n"


def unknown_top_level(data: dict[str, Any]) -> None:
    data["interviewer_notes"] = "hidden hint"


def nan_patience(data: dict[str, Any]) -> None:
    data["persona"]["patience"] = float("nan")


def bad_pause(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["script"] = "It felt [pause 8.0] odd."


def missing_pause(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["script"] = "It felt odd."


def soft_pause(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["severity"] = "soft"


def unknown_trigger_fact(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["trigger"] = {"on_disclosed": "missing"}


def unknown_trigger(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["trigger"] = {"at_random": True}


def list_trigger(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["trigger"] = {"on_disclosed": ["f2"]}


def moment_cycle(data: dict[str, Any]) -> None:
    facts(data)[1]["unlock"]["all"].append({"acknowledged": {"moment": "m1"}})


def self_moment(data: dict[str, Any]) -> None:
    data["scripted_moments"][2]["trigger"] = {"after_moment": "m3"}


def unknown_contradiction(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["script"] = {"use": "missing"}


def extra_script_key(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["script"] = {"use": "c1", "junk": 1}


def missing_memory_distance(data: dict[str, Any]) -> None:
    del data["scripted_moments"][1]["memory_distance"]


def wrong_near_window(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["trigger"]["max_elapsed_s"] = 120


def wrong_far_window(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["memory_distance"] = "far"


def unordered_window(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["trigger"]["min_elapsed_s"] = 86


def wrong_success(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["success"] = {"classified": "x"}


def false_pause_success(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["success"] = {"no_substantive_onset_during_pause": False}


def unknown_label(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["success"]["classified"] = "misspelled"


def bad_turns(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["success"]["within_turns"] = -1


def bad_acknowledgement(data: dict[str, Any]) -> None:
    facts(data)[3]["unlock"]["all"].append({"acknowledged": {"moment": "missing"}})


def bad_event_reference(data: dict[str, Any]) -> None:
    facts(data)[3]["unlock"]["all"].append({"elapsed_since": {"event": "m1", "min_s": 2}})


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (leak_in_brief, "leaks hidden"),
        (leak_in_compact, "leaks hidden"),
        (paraphrase_leak, "leaks hidden"),
        (word_run_leak, "leaks hidden"),
        (contradiction_leak, "leaks hidden"),
        (script_leak, "leaks hidden"),
        (unknown_fact, "unknown fact missing"),
        (unknown_topic, "unknown topic missing"),
        (fact_topic_unknown, "unknown topic missing"),
        (unknown_predicate, "unknown unlock predicate"),
        (bad_shape, "exactly one operator"),
        (bad_argument, "positive min_s"),
        (nan_wait, "positive min_s"),
        (long_wait, "shorter than the time budget"),
        (cycle, "unreachable"),
        (self_dependency, "unreachable"),
        (contradiction_clause, "contradictory"),
        (ordering_contradiction, "contradictory"),
        (surface_dependency, "surface facts"),
        (probe_without_follow_up, "specific follow-up"),
        (deep_without_probe, "deep facts"),
        (deep_bypass, "deep facts"),
        (negative_only, "ungated"),
        (duplicate_fact, "duplicate id f1"),
        (uncovered_topic, "has no hidden fact"),
        (long_compact, "at most 120"),
        (blank_full, "does not match"),
        (newline_id, "does not match"),
        (unknown_top_level, "Unevaluated"),
        (nan_patience, "finite"),
        (bad_pause, "2 to 4"),
        (missing_pause, "[pause N]"),
        (soft_pause, "must be hard"),
        (unknown_trigger_fact, "unknown fact missing"),
        (unknown_trigger, "unknown trigger"),
        (list_trigger, "unknown trigger"),
        (moment_cycle, "unreachable"),
        (self_moment, "unreachable"),
        (unknown_contradiction, "known contradiction"),
        (extra_script_key, "known contradiction"),
        (missing_memory_distance, "required for contradictions"),
        (wrong_near_window, "under 90"),
        (wrong_far_window, "after 180"),
        (unordered_window, "ordered"),
        (wrong_success, "onset rule"),
        (false_pause_success, "onset rule"),
        (unknown_label, "label"),
        (bad_turns, "within_turns"),
        (bad_acknowledgement, "unknown moment missing"),
        (bad_event_reference, "requires event"),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_invalid_scenarios_report_useful_errors(mutation: Mutation, expected: str) -> None:
    data = copy.deepcopy(example())

    mutation(data)

    assert any(expected in item for item in issues(data)), issues(data)


def surface_through_moment(data: dict[str, Any]) -> None:
    facts(data)[0]["unlock"] = {"acknowledged": {"moment": "m1"}}


def impossible_timing(data: dict[str, Any]) -> None:
    event = {"disclosed": "f5"}
    facts(data)[1]["unlock"]["all"].extend(
        [
            {"elapsed_since": {"event": event, "min_s": 10}},
            {"not": {"elapsed_since": {"event": event, "min_s": 5}}},
        ]
    )


def impossible_branch(data: dict[str, Any]) -> None:
    facts(data)[0]["unlock"] = {
        "any": [
            {"asked_about": "takeaway"},
            {"all": [{"open_ended": True}, {"not": {"open_ended": True}}]},
        ]
    }


def list_label(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["success"]["classified"] = ["x"]


def list_contradiction(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["script"] = {"use": ["c1"]}


def huge_budget(data: dict[str, Any]) -> None:
    data["brief"]["time_budget_s"] = 10**400


def huge_wait(data: dict[str, Any]) -> None:
    facts(data)[1]["unlock"]["all"].append(
        {"elapsed_since": {"event": {"disclosed": "f5"}, "min_s": 10**400}}
    )


def contradiction_after_moment(data: dict[str, Any]) -> None:
    data["scripted_moments"][1]["trigger"]["after_event"] = {"moment": "m1"}


def integer_onset(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["success"] = {"no_substantive_onset_during_pause": 1}


def extra_pause_marker(data: dict[str, Any]) -> None:
    data["scripted_moments"][0]["script"] = "It [pause 3.0] felt [Pause 9] odd."


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (surface_through_moment, "surface facts"),
        (impossible_timing, "contradictory"),
        (impossible_branch, "impossible branch"),
        (list_label, "label"),
        (list_contradiction, "known contradiction"),
        (huge_budget, "finite"),
        (huge_wait, "positive min_s"),
        (contradiction_after_moment, "follow a fact"),
        (integer_onset, "onset rule"),
        (extra_pause_marker, "[pause N]"),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_more_invalid_scenarios_are_reported(mutation: Mutation, expected: str) -> None:
    data = copy.deepcopy(example())

    mutation(data)

    assert any(expected in item for item in issues(data)), issues(data)


def test_valid_any_expression_can_rule_out_one_branch() -> None:
    data = example()
    facts(data)[1]["unlock"] = {
        "all": [
            {"followed_up_on": "f5"},
            {"any": [{"disclosed": "f5"}, {"disclosed": "f6"}]},
            {"not": {"disclosed": "f6"}},
        ]
    }

    assert issues(data) == []


def test_short_script_fragments_are_not_leaks() -> None:
    data = example()
    data["scripted_moments"][0]["script"] = "The ad... [pause 3.0] I mean, it really got to me."

    assert issues(data) == []


def test_example_is_valid() -> None:
    assert issues(example()) == []


def test_missing_required_section_is_rejected() -> None:
    data = example()
    del data["hidden_story"]

    assert any("hidden_story" in item for item in issues(data))


def test_hidden_data_is_not_given_to_model() -> None:
    loaded = load_suite(str(SUITE))
    scenario = loaded.suite.load_scenario(example())

    assert "hidden_story" not in scenario.data
    assert "scripted_moments" not in scenario.data
    assert "persona" not in scenario.data
    assert "hidden_story" in scenario.hidden
    assert "persona" in scenario.hidden


def test_cli_validates_interviewer_suite(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["suites", "validate", str(SUITE), "--split", "dev"]) == 0
    assert "validated 1 scenarios, 0 issues" in capsys.readouterr().out


def test_common_brief_words_do_not_trigger_leaks() -> None:
    data = example()
    data["brief"]["full"] += " Ask about software, hardware, and whether it was hard to say."

    assert issues(data) == []


def test_persona_runtime_fields_are_allowed() -> None:
    data = example()
    data["persona"]["yield_when_interrupted_ms"] = 1000
    data["persona"]["yield_when_interrupting_ms"] = 5000
    data["persona"]["turn_decider_model"] = "fake"

    assert issues(data) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("id: [", "invalid YAML"),
        ("a: 1\na: 2\n", "duplicate key"),
        ("a: &x {b: 1}\nc: *x\n", "aliases"),
        ("id: 2024-13-45\n", "invalid YAML"),
    ],
)
def test_unsafe_yaml_is_reported(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "suite.py").write_text(
        (SUITE.parent / "demo/suite.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/a.yaml").write_text(text, encoding="utf-8")

    files = load_scenario_files(load_suite(str(tmp_path)))

    assert message in files[0].issues[0].message
