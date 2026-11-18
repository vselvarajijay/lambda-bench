"""Golden tests for slice 2.3 timing and balance metrics.

Expected values are hand-derived from docs/metrics.md and PLAN.md 7.1-7.3. Each golden case keeps
its input and expected values side by side. Times are compared within 1 ms and ratios within 1e-6.

Expected layout:
  latency:  {count, mean, p50, p90}      -> response_latency_{mean,p50,p90}_ms, count and sufficient
  eligible: {count, response_rate, early_start_rate}
  talk_time_ratio, [talk_time_in_target], dead_air_ratio, goal_reached, time_to_goal_ms,
  model_turns_to_goal                    -> value only (the docs do not define their counts)
`sufficient` is checked as count >= 10 for the latency and rate metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lambda_duplex import cli
from lambda_duplex.metrics.conversation import (
    metrics_to_dict,
    timeline_from_dict,
    timing_metrics,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
TIME_TOLERANCE_MS = 1.0
RATIO_TOLERANCE = 1e-6
SUPPORT_THRESHOLD = 10
VALUE_ONLY = (
    "talk_time_ratio",
    "talk_time_in_target",
    "dead_air_ratio",
    "goal_reached",
    "time_to_goal_ms",
    "model_turns_to_goal",
)
CASE_FILES = [
    "timing_boundaries.json",
    "timing_backchannels.json",
    "timing_balance.json",
    "timing_efficiency.json",
]


def _load(name: str) -> list[dict[str, Any]]:
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))["cases"]


def _basic_case() -> dict[str, Any]:
    data = json.loads((GOLDEN / "basic_turns.json").read_text(encoding="utf-8"))
    expected = data["expected"]
    timeline_input = {k: v for k, v in data.items() if k not in ("expected", "name", "description")}
    return {"name": data["name"], "input": timeline_input, "expected": expected}


VALID_CASES = [_basic_case(), *(case for name in CASE_FILES for case in _load(name))]
INVALID_CASES = _load("timing_invalid.json")


def _ids(cases: list[dict[str, Any]]) -> list[str]:
    return [case["name"] for case in cases]


def _flatten(expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Expand the compact golden layout into per-metric expectations."""
    flat: dict[str, dict[str, Any]] = {}
    latency = expected["latency"]
    for key in ("mean", "p50", "p90"):
        flat[f"response_latency_{key}_ms"] = {
            "value": latency[key],
            "count": latency["count"],
            "sufficient": latency["count"] >= SUPPORT_THRESHOLD,
        }
    eligible = expected["eligible"]
    for key in ("response_rate", "early_start_rate"):
        flat[key] = {
            "value": eligible[key],
            "count": eligible["count"],
            "sufficient": eligible["count"] >= SUPPORT_THRESHOLD,
        }
    for key in VALUE_ONLY:
        if key in expected:
            flat[key] = {"value": expected[key]}
    return flat


def _check_value(name: str, got: Any, want: Any) -> None:
    if want is None or isinstance(want, bool):
        assert got is want, f"{name}: expected {want!r}, got {got!r}"
        return
    assert isinstance(got, (int, float)) and not isinstance(got, bool), f"{name}: got {got!r}"
    tolerance = TIME_TOLERANCE_MS if name.endswith("_ms") else RATIO_TOLERANCE
    assert got == pytest.approx(want, abs=tolerance), f"{name}: expected {want}, got {got}"


def assert_matches(actual: dict[str, dict[str, Any]], expected: dict[str, Any]) -> None:
    flat = _flatten(expected)
    assert set(actual) == set(flat)
    for name, want in flat.items():
        for field, wanted in want.items():
            if field == "value":
                _check_value(name, actual[name]["value"], wanted)
            else:
                assert actual[name][field] == wanted, f"{name}.{field}"


def _compute(timeline_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return metrics_to_dict(timing_metrics(timeline_from_dict(timeline_input)))


@pytest.mark.parametrize("case", VALID_CASES, ids=_ids(VALID_CASES))
def test_timing_golden_case(case: dict[str, Any]) -> None:
    assert_matches(_compute(case["input"]), case["expected"])


def test_target_metric_present_only_with_a_target() -> None:
    for case in VALID_CASES:
        has_target = case["input"].get("talk_target") is not None
        assert ("talk_time_in_target" in case["expected"]) == has_target, case["name"]
        assert ("talk_time_in_target" in _compute(case["input"])) == has_target, case["name"]


def test_golden_case_names_are_unique() -> None:
    names = _ids(VALID_CASES) + _ids(INVALID_CASES)
    assert len(names) == len(set(names))


@pytest.mark.parametrize("case", INVALID_CASES, ids=_ids(INVALID_CASES))
def test_invalid_timelines_are_rejected(case: dict[str, Any]) -> None:
    assert case["expected_error"] == "ValueError"
    timeline = timeline_from_dict(case["input"])
    with pytest.raises(ValueError):
        timing_metrics(timeline)


def test_metrics_to_dict_shape_is_sorted_and_complete() -> None:
    result = _compute(_basic_case()["input"])
    assert list(result) == sorted(result)
    for entry in result.values():
        assert set(entry) == {"value", "count", "sufficient"}


def test_cli_metrics_on_basic_turns(capsys: pytest.CaptureFixture[str]) -> None:
    path = GOLDEN / "basic_turns.json"
    assert cli.main(["metrics", str(path)]) == 0
    assert_matches(json.loads(capsys.readouterr().out), _basic_case()["expected"])


def test_cli_metrics_on_empty_conversation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"duration_ms": 0, "counterpart": [], "model": []}))
    assert cli.main(["metrics", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["response_rate"] == {"value": None, "count": 0, "sufficient": False}
    assert result["talk_time_ratio"]["value"] is None
    assert result["dead_air_ratio"]["value"] is None
    assert "talk_time_in_target" not in result


@pytest.mark.parametrize(
    "payload",
    [
        {"duration_ms": 5000, "counterpart": [{"start_ms": 0, "end_ms": 6000}], "model": []},
        {"counterpart": [], "model": []},
        {"duration_ms": 5000, "counterpart": [{"start_ms": 0}], "model": []},
    ],
    ids=["invalid_timeline", "missing_duration", "missing_turn_end"],
)
def test_cli_metrics_rejects_bad_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], payload: dict[str, Any]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    assert cli.main(["metrics", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot compute metrics" in captured.err
