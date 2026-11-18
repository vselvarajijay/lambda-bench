"""Golden tests for slice 2.4 overlap metrics.

Expected values are hand-derived from docs/metrics.md ("Shared definitions", "Overlap behavior") and
PLAN.md 7.1-7.2. Times are compared within 1 ms and ratios within 1e-6.

Case layout (see tests/golden/overlap_*.json): {name, description, input, expected}. `expected` maps
a metric name to either a bare value or {"value": v, "count": n, "sufficient": b}. Only the listed
metrics are checked, so a case asserts exactly the behavior it was written for. Cases tagged
AMBIGUOUS in their description assert one reading of a definition the docs leave open.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lambda_duplex import cli
from lambda_duplex.metrics.conversation import (
    conversation_metrics,
    metrics_to_dict,
    overlap_metrics,
    timeline_from_dict,
    timing_metrics,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
TIME_TOLERANCE_MS = 1.0
RATIO_TOLERANCE = 1e-6
CASE_FILES = [
    "overlap_interruptions.json",
    "overlap_yield_pause.json",
    "overlap_backchannels.json",
    "overlap_selectivity.json",
    "overlap_events_balance.json",
]
DOCUMENTED_OVERLAP_METRICS = {
    "interruption_benign_rate",
    "interruption_benign_per_minute",
    "interruption_pause_cut_in_rate",
    "interruption_pause_cut_in_per_minute",
    "interruption_mid_sentence_rate",
    "interruption_mid_sentence_per_minute",
    "yield_latency_mean_ms",
    "yield_rate",
    "resume_rate",
    "pause_tolerance",
    "backchannel_per_minute",
    "backchannel_well_placed_rate",
    "selectivity_backchannel",
    "selectivity_vocal_tic",
    "selectivity_non_directed",
}


def _read(name: str) -> Any:
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def _load(name: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = _read(name)["cases"]
    return cases


def _cli_fixture() -> dict[str, Any]:
    data = _read("interruptions.json")
    timeline_input = {k: v for k, v in data.items() if k not in ("expected", "name", "description")}
    return {"name": data["name"], "input": timeline_input, "expected": data["expected"]}


VALID_CASES = [_cli_fixture(), *(case for name in CASE_FILES for case in _load(name))]
INVALID_CASES = _load("overlap_invalid.json")


def _ids(cases: list[dict[str, Any]]) -> list[str]:
    return [case["name"] for case in cases]


def _check_value(name: str, got: Any, want: Any) -> None:
    if want is None or isinstance(want, bool):
        assert got is want, f"{name}: expected {want!r}, got {got!r}"
        return
    assert isinstance(got, (int, float)) and not isinstance(got, bool), f"{name}: got {got!r}"
    tolerance = TIME_TOLERANCE_MS if name.endswith("_ms") else RATIO_TOLERANCE
    assert got == pytest.approx(want, abs=tolerance), f"{name}: expected {want}, got {got}"


def assert_matches(actual: dict[str, dict[str, Any]], expected: dict[str, Any]) -> None:
    for name, want in expected.items():
        assert name in actual, f"metric {name} missing from output"
        if isinstance(want, dict):
            for field, wanted in want.items():
                if field == "value":
                    _check_value(name, actual[name]["value"], wanted)
                else:
                    assert actual[name][field] == wanted, f"{name}.{field}"
        else:
            _check_value(name, actual[name]["value"], want)


def _compute(timeline_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return metrics_to_dict(conversation_metrics(timeline_from_dict(timeline_input)))


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("case", VALID_CASES, ids=_ids(VALID_CASES))
def test_overlap_golden_case(case: dict[str, Any]) -> None:
    assert_matches(_compute(case["input"]), case["expected"])


@pytest.mark.parametrize("case", VALID_CASES, ids=_ids(VALID_CASES))
def test_overlap_metrics_alone_match_overlap_expectations(case: dict[str, Any]) -> None:
    actual = metrics_to_dict(overlap_metrics(timeline_from_dict(case["input"])))
    overlap_only = {k: v for k, v in case["expected"].items() if k in DOCUMENTED_OVERLAP_METRICS}
    assert_matches(actual, overlap_only)


@pytest.mark.parametrize("case", VALID_CASES, ids=_ids(VALID_CASES))
def test_cli_matches_golden_case(
    case: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["metrics", str(_write(tmp_path, case["input"]))]) == 0
    assert_matches(json.loads(capsys.readouterr().out), case["expected"])


def test_cli_metrics_on_interruptions_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["metrics", str(GOLDEN / "interruptions.json")]) == 0
    assert_matches(json.loads(capsys.readouterr().out), _cli_fixture()["expected"])


def test_cli_metrics_without_counterpart_events_key(capsys: pytest.CaptureFixture[str]) -> None:
    """basic_turns.json predates counterpart events. Model speech never overlaps a turn: zeros."""
    assert cli.main(["metrics", str(GOLDEN / "basic_turns.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert DOCUMENTED_OVERLAP_METRICS <= set(result)
    assert_matches(
        result,
        {
            "interruption_benign_rate": 0.0,
            "interruption_pause_cut_in_rate": 0.0,
            "interruption_mid_sentence_rate": 0.0,
            "interruption_mid_sentence_per_minute": 0.0,
            "backchannel_per_minute": 0.0,
            "backchannel_well_placed_rate": None,
            "yield_rate": None,
            "yield_latency_mean_ms": None,
            "resume_rate": None,
            "pause_tolerance": None,
            "selectivity_backchannel": None,
            "selectivity_vocal_tic": None,
            "selectivity_non_directed": None,
            "response_rate": 0.6666666667,
            "talk_time_ratio": 0.2241379310,
            "dead_air_ratio": 0.42,
        },
    )


def test_cli_output_is_sorted_and_complete(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["metrics", str(GOLDEN / "interruptions.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert list(result) == sorted(result)
    assert DOCUMENTED_OVERLAP_METRICS <= set(result)
    for entry in result.values():
        assert set(entry) == {"value", "count", "sufficient"}


def test_conversation_metrics_is_timing_plus_overlap() -> None:
    timeline = timeline_from_dict(_cli_fixture()["input"])
    combined = metrics_to_dict(conversation_metrics(timeline))
    assert combined == {
        **metrics_to_dict(timing_metrics(timeline)),
        **metrics_to_dict(overlap_metrics(timeline)),
    }


def test_every_documented_overlap_metric_is_asserted_by_some_case() -> None:
    asserted = {name for case in VALID_CASES for name in case["expected"]}
    assert DOCUMENTED_OVERLAP_METRICS <= asserted


def test_golden_case_names_are_unique() -> None:
    names = _ids(VALID_CASES) + _ids(INVALID_CASES)
    assert len(names) == len(set(names))


def test_golden_expectations_name_real_metrics() -> None:
    for case in VALID_CASES:
        assert set(case["expected"]) <= set(_compute(case["input"])), case["name"]


@pytest.mark.parametrize("case", INVALID_CASES, ids=_ids(INVALID_CASES))
def test_invalid_event_data_is_rejected(case: dict[str, Any]) -> None:
    assert case["expected_error"] == "ValueError"
    timeline = timeline_from_dict(case["input"])
    with pytest.raises(ValueError):
        overlap_metrics(timeline)
    with pytest.raises(ValueError):
        conversation_metrics(timeline)


@pytest.mark.parametrize("case", INVALID_CASES, ids=_ids(INVALID_CASES))
def test_cli_rejects_invalid_event_data(
    case: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["metrics", str(_write(tmp_path, case["input"]))]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot compute metrics" in captured.err


@pytest.mark.parametrize(
    "events",
    [
        [{"start_ms": 1000, "end_ms": 2000}],
        [{"kind": "vocal_tic", "start_ms": 1000}],
        [{"kind": "vocal_tic", "start_ms": "soon", "end_ms": 2000}],
    ],
    ids=["missing_kind", "missing_end", "non_numeric_start"],
)
def test_cli_rejects_malformed_event_fields(
    events: list[dict[str, Any]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"duration_ms": 5000, "counterpart": [], "model": [], "counterpart_events": events}
    assert cli.main(["metrics", str(_write(tmp_path, payload))]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot compute metrics" in captured.err


def test_events_do_not_change_interruption_classification() -> None:
    """A counterpart event is not a turn, so it adds no eligible turns or interruptions."""
    base = {
        "duration_ms": 10000,
        "counterpart": [{"start_ms": 0, "end_ms": 6000}],
        "model": [{"start_ms": 1000, "end_ms": 2200}],
    }
    with_event = {
        **base,
        "counterpart_events": [{"kind": "vocal_tic", "start_ms": 1500, "end_ms": 1800}],
    }
    plain = _compute(base)
    eventful = _compute(with_event)
    for name in (
        "interruption_mid_sentence_rate",
        "interruption_benign_rate",
        "interruption_pause_cut_in_rate",
        "response_rate",
    ):
        assert eventful[name] == plain[name], name
    assert eventful["interruption_mid_sentence_rate"]["count"] == 1
