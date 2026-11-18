"""Suite discovery and scenario validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import jsonschema
import yaml

from lambda_duplex.adapters.base import Capabilities, ModelContext
from lambda_duplex.suites.base import Issue, Scenario, Split, Suite

ENTRY_POINT_GROUP = "lambda_duplex.suites"
CORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "version", "split"],
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*(?![\\s\\S])"},
        "version": {"type": "integer", "minimum": 1},
        "split": {"enum": ["dev", "test"]},
    },
}


class ScenarioLoader(yaml.SafeLoader):
    """YAML loader that rejects ambiguous scenario structure."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            event = self.peek_event()  # type: ignore[no-untyped-call]
            raise yaml.constructor.ConstructorError(
                None, None, "YAML aliases are not allowed", event.start_mark
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in keys:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            keys.add(key)
        return super().construct_mapping(node, deep)


def _construct_scenario_mapping(loader: ScenarioLoader, node: Any) -> dict[Any, Any]:
    return loader.construct_mapping(node, deep=True)


ScenarioLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_scenario_mapping
)


class SuiteError(RuntimeError):
    """A suite could not be found or loaded."""


@dataclass(frozen=True)
class LoadedSuite:
    """A suite and the directory holding its scenarios."""

    suite: Suite
    root: Path | None


@dataclass(frozen=True)
class ScenarioFile:
    """Validation result for one scenario file."""

    path: Path
    scenario: Scenario | None
    issues: tuple[Issue, ...]


def _suite_roots(start: Path) -> list[Path]:
    return [folder / "suites" for folder in (start, *start.parents) if (folder / "suites").is_dir()]


def _load_from_path(root: Path) -> LoadedSuite:
    module_file = root / "suite.py"
    if not module_file.is_file():
        raise SuiteError(f"{root} does not contain suite.py")
    digest = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(
        f"_lambda_suite_{digest}", module_file, submodule_search_locations=[str(root.resolve())]
    )
    if spec is None or spec.loader is None:
        raise SuiteError(f"could not import {module_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    factory = getattr(module, "create_suite", None)
    if factory is None:
        raise SuiteError(f"{module_file} does not define create_suite()")
    suite = factory()
    if not isinstance(suite, Suite):
        raise SuiteError(f"{module_file} did not return a valid suite")
    return LoadedSuite(suite, root.resolve())


def load_suite(reference: str, start: Path | None = None) -> LoadedSuite:
    """Load a suite by path, repository folder, or installed entry point."""
    base = (start or Path.cwd()).resolve()
    candidate = Path(reference).expanduser()
    if candidate.is_dir():
        return _load_from_path(candidate)
    if "/" not in reference and not reference.startswith("."):
        for root in _suite_roots(base):
            if (root / reference / "suite.py").is_file():
                return _load_from_path(root / reference)
        for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            if entry.name == reference:
                suite = entry.load()()
                if not isinstance(suite, Suite):
                    raise SuiteError(f"entry point {reference} did not return a valid suite")
                return LoadedSuite(suite, Path(inspect.getfile(type(suite))).resolve().parent)
    raise SuiteError(f"unknown suite: {reference}")


def list_suites(start: Path | None = None) -> list[str]:
    """List suites available from repository folders and entry points."""
    names = {entry.name for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}
    for root in _suite_roots((start or Path.cwd()).resolve()):
        names.update(
            path.parent.name
            for path in root.glob("*/suite.py")
            if not path.parent.name.startswith("_")
        )
    return sorted(names)


def _schema_issues(suite: Suite, raw: Mapping[str, Any]) -> list[Issue]:
    suite_schema = dict(suite.scenario_schema())
    suite_schema.setdefault("$id", f"urn:lambda-duplex:suite:{suite.manifest.name}")
    schema = {"allOf": [CORE_SCHEMA, suite_schema]}
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as error:
        return [Issue("$schema", f"invalid suite schema: {error.message}")]
    validator = jsonschema.Draft202012Validator(schema)
    return [
        Issue("/".join(str(part) for part in error.absolute_path) or "$", error.message)
        for error in sorted(validator.iter_errors(raw), key=lambda item: list(item.path).__str__())
    ]


def public_scenario(scenario: Scenario) -> Scenario:
    """Return the scenario view allowed to reach model-context code."""
    return replace(scenario, hidden={})


def _context_fields(context: ModelContext) -> dict[str, str]:
    return {
        "full": context.full,
        "compact": context.compact,
        "tools": json.dumps([asdict(tool) for tool in context.tools], default=str),
    }


def _leak_issues(suite: Suite, scenario: Scenario) -> list[Issue]:
    issues: set[tuple[str, str]] = set()
    secrets = [secret.casefold() for secret in scenario.secrets()]
    for tools in (False, True):
        capabilities = Capabilities(clock_modes=frozenset({"lockstep"}), tools=tools)
        context = suite.model_context(public_scenario(scenario), capabilities)
        for name, text in _context_fields(context).items():
            if any(secret in text.casefold() for secret in secrets):
                issues.add((f"model_context.{name}", "model context leaks hidden scenario text"))
    return [Issue(path, message) for path, message in sorted(issues)]


def validate_scenario_data(loaded: LoadedSuite, raw: Any, split: Split) -> ScenarioFile:
    """Validate raw scenario data against core and suite rules."""
    path = Path("<memory>")
    if not isinstance(raw, Mapping):
        return ScenarioFile(path, None, (Issue("$", "scenario must be a mapping"),))
    issues = _schema_issues(loaded.suite, raw)
    if raw.get("split") != split:
        issues.append(Issue("split", f"scenario is stored in {split} but declares another split"))
    if issues:
        return ScenarioFile(path, None, tuple(issues))
    scenario = loaded.suite.load_scenario(raw)
    found = [*loaded.suite.validate_scenario(scenario), *_leak_issues(loaded.suite, scenario)]
    return ScenarioFile(path, scenario, tuple(found))


def load_yaml_file(path: Path) -> Any:
    """Read YAML with the strict scenario loader."""
    return yaml.load(path.read_text(encoding="utf-8"), Loader=ScenarioLoader)


def load_scenario_files(loaded: LoadedSuite, split: Split = "dev") -> list[ScenarioFile]:
    """Load and validate all scenarios in a split, including duplicate id checks."""
    if loaded.root is None:
        raise SuiteError(f"{loaded.suite.manifest.name} has no scenario directory")
    files: list[ScenarioFile] = []
    seen: set[str] = set()
    folder = loaded.root / split
    paths = sorted([*folder.glob("*.yaml"), *folder.glob("*.yml")])
    if not paths:
        raise SuiteError(f"{loaded.suite.manifest.name} has no scenarios in split {split}")
    for path in paths:
        try:
            raw = load_yaml_file(path)
        except (yaml.YAMLError, RecursionError, TypeError, ValueError) as error:
            files.append(ScenarioFile(path, None, (Issue("$", f"invalid YAML: {error}"),)))
            continue
        result = validate_scenario_data(loaded, raw, split)
        issues = list(result.issues)
        if result.scenario is not None:
            if result.scenario.id in seen:
                issues.append(Issue("id", f"duplicate scenario id {result.scenario.id}"))
            seen.add(result.scenario.id)
        files.append(ScenarioFile(path, result.scenario, tuple(issues)))
    return files


def valid_scenarios(loaded: LoadedSuite, split: Split = "dev") -> list[Scenario]:
    """Return scenarios for a split, raising if any scenario is invalid."""
    files = load_scenario_files(loaded, split)
    problems = [
        f"{item.path}: {issue.path}: {issue.message}" for item in files for issue in item.issues
    ]
    if problems:
        raise SuiteError("; ".join(problems))
    return [cast(Scenario, item.scenario) for item in files]
