from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from lambda_duplex import cli
from lambda_duplex.suites.loader import load_suite

ROOT = Path(__file__).resolve().parents[2]


def test_new_suite_runs_without_edits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "hello"

    assert cli.main(["new-suite", "hello", "--out", str(path)]) == 0
    assert load_suite(str(path)).suite.manifest.name == "hello"
    assert cli.main(["suites", "validate", str(path)]) == 0
    assert cli.main(["run", "--suite", str(path), "--model", "mock"]) == 0
    assert "passed=true" in capsys.readouterr().out
    assert cli.main(["run", "--suite", str(path), "--model", "mock:bad", "--ticks", "8"]) == 0
    assert "passed=false" in capsys.readouterr().out
    assert "_template" not in (path / "DATASHEET.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["Hello", "1suite", "../escape", "has-dash", ""])
def test_invalid_names_are_rejected(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["new-suite", name, "--out", str(tmp_path / "new")]) == 2
    assert "suite names" in capsys.readouterr().err
    assert not (tmp_path / "new").exists()


def test_non_empty_destination_is_left_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keep = tmp_path / "keep.txt"
    keep.write_text("mine", encoding="utf-8")

    assert cli.main(["new-suite", "hello", "--out", str(tmp_path)]) == 2
    assert "not empty" in capsys.readouterr().err
    assert keep.read_text(encoding="utf-8") == "mine"
    assert not (tmp_path / "suite.py").exists()


def test_template_is_hidden_from_suite_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["suites", "list"]) == 0
    assert "_template" not in capsys.readouterr().out


def test_wheel_contains_template(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT)],
        check=True,
        capture_output=True,
    )

    with zipfile.ZipFile(next(tmp_path.glob("*.whl"))) as wheel:
        names = wheel.namelist()

    assert "lambda_duplex/_suite_template/suite.py" in names
    assert "lambda_duplex/_suite_template/dev/example_001.yaml" in names
