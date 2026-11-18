"""Create a new suite from the packaged template."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
TEMPLATE_NAME = "_template"
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".json"}


class ScaffoldError(RuntimeError):
    """A suite could not be created."""


def template_dir() -> Path:
    """Locate the template in an installed package or source checkout."""
    package_root = Path(__file__).resolve().parents[1]
    for candidate in (
        package_root / "_suite_template",
        package_root.parents[1] / "suites" / TEMPLATE_NAME,
    ):
        if (candidate / "suite.py").is_file():
            return candidate
    raise ScaffoldError("suite template is not available")


def create_suite(name: str, destination: Path) -> Path:
    """Copy the template to an empty destination and apply the suite name."""
    if not NAME_PATTERN.fullmatch(name):
        raise ScaffoldError("suite names must match [a-z][a-z0-9_]*")
    if destination.exists() and any(destination.iterdir()):
        raise ScaffoldError(f"{destination} is not empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        template_dir(),
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for path in destination.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace(TEMPLATE_NAME, name), encoding="utf-8")
    return destination
