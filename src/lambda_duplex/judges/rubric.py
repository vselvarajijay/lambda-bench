"""Rubric files: versioned, schema-checked lists of judge items."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from lambda_duplex.suites.loader import ScenarioLoader

ID_PATTERN = "^[a-z0-9][a-z0-9_-]*(?![\\s\\S])"
BUILTIN_RUBRICS = Path(__file__).resolve().parent / "rubrics"
RUBRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "version", "instructions", "items"],
    "properties": {
        "id": {"type": "string", "pattern": ID_PATTERN},
        "version": {"type": "string", "minLength": 1},
        "instructions": {"type": "string", "minLength": 1},
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "prompt", "scale"],
                "properties": {
                    "id": {"type": "string", "pattern": ID_PATTERN},
                    "prompt": {"type": "string", "minLength": 1},
                    "scale": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["min", "max"],
                        "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}},
                    },
                    "anchors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["score", "text"],
                            "properties": {
                                "score": {"type": "integer"},
                                "text": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                    "allow_na": {"type": "boolean"},
                    "min_evidence": {"type": "integer", "minimum": 1},
                },
            },
        },
    },
}


class RubricError(ValueError):
    """A rubric is missing or invalid."""


@dataclass(frozen=True)
class RubricItem:
    """One scored question in a rubric."""

    id: str
    prompt: str
    min_score: int
    max_score: int
    anchors: tuple[tuple[int, str], ...] = ()
    allow_na: bool = False
    min_evidence: int = 1

    def public(self) -> dict[str, Any]:
        """Return the item as shown to the judge."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "scale": {"min": self.min_score, "max": self.max_score},
            "anchors": [{"score": score, "text": text} for score, text in self.anchors],
            "allow_na": self.allow_na,
            "min_evidence": self.min_evidence,
        }


@dataclass(frozen=True)
class JudgeRubric:
    """A complete rubric and the hash of its definition."""

    id: str
    version: str
    instructions: str
    items: tuple[RubricItem, ...]
    sha256: str


def _parse_item(entry: Mapping[str, Any]) -> RubricItem:
    low, high = entry["scale"]["min"], entry["scale"]["max"]
    if low >= high:
        raise RubricError(f"item {entry['id']}: scale min must be below max")
    anchors = tuple((anchor["score"], anchor["text"]) for anchor in entry.get("anchors", ()))
    scores = [score for score, _ in anchors]
    if len(set(scores)) != len(scores):
        raise RubricError(f"item {entry['id']}: anchor scores must be unique")
    if any(not low <= score <= high for score in scores):
        raise RubricError(f"item {entry['id']}: anchor score is outside the scale")
    return RubricItem(
        entry["id"],
        entry["prompt"],
        low,
        high,
        anchors,
        entry.get("allow_na", False),
        entry.get("min_evidence", 1),
    )


def parse_rubric(raw: Any) -> JudgeRubric:
    """Validate a decoded rubric."""
    try:
        jsonschema.validate(raw, RUBRIC_SCHEMA)
    except jsonschema.ValidationError as error:
        location = "/".join(str(part) for part in error.absolute_path) or "$"
        raise RubricError(f"invalid rubric at {location}: {error.message}") from error
    items = tuple(_parse_item(entry) for entry in raw["items"])
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise RubricError("rubric item ids must be unique")
    body = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(body.encode()).hexdigest()
    return JudgeRubric(raw["id"], raw["version"], raw["instructions"], items, digest)


def load_rubric(path: Path) -> JudgeRubric:
    """Load one rubric file."""
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=ScenarioLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RubricError(f"cannot read rubric {path}: {error}") from error
    return parse_rubric(raw)


def is_rubric_path(name: str) -> bool:
    """Return whether a rubric argument names a file rather than an id."""
    return re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None


def resolve_rubric(name: str, suite_root: Path | None) -> JudgeRubric:
    """Find a rubric by path, then in the suite, then among built-in rubrics."""
    if is_rubric_path(name):
        return load_rubric(Path(name))
    folders = [BUILTIN_RUBRICS] if suite_root is None else [suite_root / "rubrics", BUILTIN_RUBRICS]
    for folder in folders:
        path = folder / f"{name}.yaml"
        if path.is_file():
            rubric = load_rubric(path)
            if rubric.id != name:
                raise RubricError(f"{path} declares id {rubric.id}, expected {name}")
            return rubric
    raise RubricError(f"unknown rubric: {name}")
