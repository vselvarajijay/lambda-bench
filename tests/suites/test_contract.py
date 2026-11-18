from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from lambda_duplex.suites.base import Scenario
from lambda_duplex.suites.loader import (
    LoadedSuite,
    SuiteError,
    list_suites,
    load_scenario_files,
    load_suite,
    valid_scenarios,
    validate_scenario_data,
)
from lambda_duplex.suites.runtime import ScenarioRun, mock_for, run_scenario

ROOT = Path(__file__).resolve().parents[2]
SUITES = [
    ROOT / "suites/demo",
    ROOT / "tests/fixtures/suites/support_stub",
    ROOT / "tests/fixtures/suites/tutoring_stub",
]


def execute(loaded: LoadedSuite, scenario: Scenario, behavior: str) -> ScenarioRun:
    adapter = mock_for(loaded.suite, scenario, behavior)
    return asyncio.run(run_scenario(loaded.suite, scenario, adapter, max_ticks=20))


@pytest.mark.parametrize("path", SUITES, ids=lambda path: path.name)
def test_suite_contract(path: Path) -> None:
    loaded = load_suite(str(path))
    scenario = valid_scenarios(loaded)[0]

    good = execute(loaded, scenario, "good")
    bad = execute(loaded, scenario, "bad")

    assert good.outcome.status == "completed"
    assert good.decision.passed is True
    assert bad.outcome.status == "completed"
    assert bad.decision.passed is False


def test_disclosure_is_recorded_when_delivered() -> None:
    loaded = load_suite("demo", ROOT)
    scenario = valid_scenarios(loaded)[0]

    run = execute(loaded, scenario, "good")
    event_types = [(event.source, event.type) for event in run.outcome.events]

    spoken = event_types.index(("counterpart", "spoke"), event_types.index(("model", "spoke")))
    assert event_types.index(("suite", "code_word_disclosed")) == spoken + 1


def test_listing_hides_private_folders(tmp_path: Path) -> None:
    for name in ("visible", "_template"):
        folder = tmp_path / "suites" / name
        folder.mkdir(parents=True)
        (folder / "suite.py").write_text("", encoding="utf-8")

    assert list_suites(tmp_path) == ["visible"]


def test_unknown_suite_is_reported(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="unknown suite"):
        load_suite("missing", tmp_path)


def test_suite_without_factory_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "suite.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(SuiteError, match="create_suite"):
        load_suite(str(tmp_path))


@pytest.mark.parametrize(
    ("changes", "split", "expected"),
    [
        ({"code_word": None}, "dev", "code_word"),
        ({"id": "Bad ID"}, "dev", "id"),
        ({"split": "test"}, "dev", "split"),
        ({"brief": "The answer is lantern."}, "dev", "brief"),
    ],
)
def test_invalid_scenarios_are_rejected(
    changes: dict[str, object], split: str, expected: str
) -> None:
    loaded = load_suite("demo", ROOT)
    raw: dict[str, object] = {
        "id": "valid",
        "version": 1,
        "split": "dev",
        "brief": "Find the word.",
        "code_word": "lantern",
    }
    raw.update(changes)
    raw = {key: value for key, value in raw.items() if value is not None}

    result = validate_scenario_data(loaded, raw, "dev")

    assert any(issue.path == expected or expected in issue.message for issue in result.issues)


def test_duplicate_ids_and_bad_yaml_are_reported(tmp_path: Path) -> None:
    source = ROOT / "suites/demo/suite.py"
    (tmp_path / "suite.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    dev = tmp_path / "dev"
    dev.mkdir()
    scenario = {
        "id": "same",
        "version": 1,
        "split": "dev",
        "brief": "Find it.",
        "code_word": "lantern",
    }
    (dev / "a.yaml").write_text(yaml.safe_dump(scenario), encoding="utf-8")
    (dev / "b.yaml").write_text(yaml.safe_dump(scenario), encoding="utf-8")
    (dev / "c.yaml").write_text("id: [", encoding="utf-8")

    files = load_scenario_files(load_suite(str(tmp_path)))

    assert any("duplicate" in issue.message for item in files for issue in item.issues)
    assert any("invalid YAML" in issue.message for item in files for issue in item.issues)
