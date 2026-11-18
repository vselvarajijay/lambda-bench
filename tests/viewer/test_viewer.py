from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lambda_duplex import cli
from lambda_duplex.trajectory.store import find_run_dirs
from lambda_duplex.viewer.build import ViewerError, build_viewer

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "suites/demo"


def audio_run(path: Path) -> Path:
    out = path / "runs"
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]
    extra = ["--audio-backend", "synthetic", "--out", str(out)]
    assert cli.main([*arguments, *extra]) == 0
    return find_run_dirs(out)[0]


def run_data(folder: Path) -> dict[str, Any]:
    text = (folder / "run-data.js").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text.removeprefix("window.RUN_DATA = ").removesuffix(";\n"))
    return data


def test_viewer_build_is_self_contained(tmp_path: Path) -> None:
    page = build_viewer(audio_run(tmp_path), tmp_path / "view")
    folder = page.parent

    data = run_data(folder)

    assert page.name == "index.html"
    assert (folder / "app.js").is_file()
    assert (folder / "audio/mix.flac").is_file()
    assert data["events"][0]["label"] == "code word disclosed"
    assert data["transcript"][0]["speaker"] == "Counterpart"
    assert "http://" not in page.read_text(encoding="utf-8")
    assert "https://" not in (folder / "app.js").read_text(encoding="utf-8")
    assert "lantern" in json.dumps(data)


def test_viewer_refuses_non_empty_output(tmp_path: Path) -> None:
    output = tmp_path / "view"
    output.mkdir()
    (output / "keep").write_text("x", encoding="utf-8")

    with pytest.raises(ViewerError, match="not empty"):
        build_viewer(audio_run(tmp_path), output)


def test_cli_rejects_text_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "runs"
    assert cli.main(["run", "--suite", str(DEMO), "--model", "mock", "--out", str(out)]) == 0

    assert cli.main(["view", str(out), "--out", str(tmp_path / "view")]) == 2
    assert "no audio" in capsys.readouterr().err


def test_wheel_contains_viewer(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT)],
        check=True,
        capture_output=True,
    )

    with zipfile.ZipFile(next(tmp_path.glob("*.whl"))) as wheel:
        assert "lambda_duplex/_viewer/app.js" in wheel.namelist()


@pytest.mark.browser
def test_browser_loads_markers_and_seeks(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    page_path = build_viewer(audio_run(tmp_path), tmp_path / "view")
    data = run_data(page_path.parent)
    expected_ms = data["events"][0]["time_ms"]
    errors: list[str] = []

    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 800})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(page_path.as_uri())
        page.wait_for_function("document.getElementById('audio').readyState >= 1")
        marker = page.locator(".marker").first

        assert page.locator(".segment.counterpart").count() == len(data["counterpart"])
        assert page.locator(".segment.model").count() == len(data["model"])
        assert page.locator(".marker").count() == len(data["events"])
        position = marker.evaluate("item => parseFloat(item.style.left)")
        assert position == pytest.approx(expected_ms / data["duration_ms"] * 100, abs=0.01)

        marker.click(force=True)
        page.wait_for_function(
            "target => Math.abs(document.getElementById('audio').currentTime - target) < 0.05",
            arg=expected_ms / 1000,
        )
        row = page.locator("#transcript button").first
        target = float(row.get_attribute("data-time-ms") or 0) / 1000
        row.click()
        page.wait_for_function(
            "target => Math.abs(document.getElementById('audio').currentTime - target) < 0.05",
            arg=target,
        )
        page.set_viewport_size({"width": 390, "height": 800})
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        browser.close()

    assert errors == []
    assert overflow <= 0
