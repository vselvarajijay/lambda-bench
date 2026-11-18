"""Unlock grammar and static analysis for interview facts."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_DEPTH = 16
MAX_CLAUSES = 256
Atom = tuple[str, str]
Node = tuple[str, str]
PREDICATES = {
    "disclosed",
    "followed_up_on",
    "asked_about",
    "open_ended",
    "not_leading",
    "silence_after_trailoff",
    "elapsed_since",
    "no_interruption_since",
    "acknowledged",
}


@dataclass(frozen=True)
class Clause:
    """One way to satisfy an expression: all positive atoms and no negative atoms."""

    positive: frozenset[Atom]
    negative: frozenset[Atom]


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= sys.float_info.max
    return isinstance(value, float) and math.isfinite(value)


def valid_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def event_ref(value: Any) -> Node | None:
    """Return the fact or moment named by an event reference."""
    if not isinstance(value, Mapping) or len(value) != 1:
        return None
    kind, target = next(iter(value.items()))
    if kind not in {"disclosed", "moment"} or not valid_identifier(target):
        return None
    return ("fact" if kind == "disclosed" else "moment", target)


def _argument_error(name: str, value: Any) -> str | None:
    if name in {"disclosed", "followed_up_on", "asked_about"}:
        return None if valid_identifier(value) else "must name an identifier"
    if name in {"open_ended", "not_leading"}:
        return None if value is True else "must be true"
    if not isinstance(value, Mapping):
        return "must be a mapping"
    if name == "silence_after_trailoff":
        valid = (
            set(value) == {"fact", "min_s"}
            and valid_identifier(value["fact"])
            and is_finite_number(value["min_s"])
            and value["min_s"] > 0
        )
        return None if valid else "requires fact and positive min_s"
    if name == "elapsed_since":
        valid = (
            set(value) == {"event", "min_s"}
            and event_ref(value["event"]) is not None
            and is_finite_number(value["min_s"])
            and value["min_s"] > 0
        )
        return None if valid else "requires event and positive min_s"
    if name == "no_interruption_since":
        valid = set(value) == {"event"} and event_ref(value["event"]) is not None
        return None if valid else "requires event"
    return None if event_ref(value) is not None else "requires an event reference"


def expression_errors(expression: Any, path: str, depth: int = 0) -> list[tuple[str, str]]:
    """Return grammar problems as path and message pairs."""
    if depth > MAX_DEPTH:
        return [(path, "unlock expression is too deep")]
    if not isinstance(expression, Mapping) or len(expression) != 1:
        return [(path, "unlock expression must have exactly one operator")]
    name, value = next(iter(expression.items()))
    location = f"{path}/{name}"
    if name in {"all", "any"}:
        if not isinstance(value, list) or not value:
            return [(location, "must be a non-empty list")]
        errors: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            errors.extend(expression_errors(item, f"{location}/{index}", depth + 1))
        return errors
    if name == "not":
        return expression_errors(value, location, depth + 1)
    if name not in PREDICATES:
        return [(location, "unknown unlock predicate")]
    problem = _argument_error(str(name), value)
    return [] if problem is None else [(location, problem)]


def _atom(name: str, value: Any) -> Atom:
    if isinstance(value, Mapping) and "min_s" in value:
        value = {**value, "min_s": float(value["min_s"])}
    return name, json.dumps(value, sort_keys=True)


def _timing_key(atom: Atom) -> tuple[tuple[str, str], float] | None:
    name, encoded = atom
    if name not in {"elapsed_since", "silence_after_trailoff"}:
        return None
    value = json.loads(encoded)
    target = value["event"] if name == "elapsed_since" else value["fact"]
    return (name, json.dumps(target, sort_keys=True)), float(value["min_s"])


def self_contradictory(clause: Clause) -> bool:
    """Return whether a clause cannot hold regardless of conversation order."""
    if clause.positive & clause.negative:
        return True
    required: dict[tuple[str, str], float] = {}
    for atom in clause.positive:
        timing = _timing_key(atom)
        if timing is not None:
            key, minimum = timing
            required[key] = max(required.get(key, minimum), minimum)
    for atom in clause.negative:
        timing = _timing_key(atom)
        if timing is not None and required.get(timing[0], -1.0) >= timing[1]:
            return True
    return False


def any_branches(expression: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return every authored branch of an ``any`` expression."""
    name, value = next(iter(expression.items()))
    if name == "not":
        return any_branches(value)
    if name not in {"all", "any"}:
        return []
    branches = list(value) if name == "any" else []
    for item in value:
        branches.extend(any_branches(item))
    return branches


def _combine(left: list[Clause], right: list[Clause]) -> list[Clause]:
    combined = [
        Clause(a.positive | b.positive, a.negative | b.negative) for a in left for b in right
    ]
    if len(combined) > MAX_CLAUSES:
        raise ValueError("unlock expression is too complex")
    return combined


def clauses(expression: Mapping[str, Any], negated: bool = False) -> list[Clause]:
    """Expand an expression into alternative conjunctions."""
    name, value = next(iter(expression.items()))
    if name == "not":
        return clauses(value, not negated)
    if name in {"all", "any"}:
        parts = [clauses(item, negated) for item in value]
        conjunction = (name == "all") != negated
        if conjunction:
            result = parts[0]
            for part in parts[1:]:
                result = _combine(result, part)
            return result
        merged = [item for part in parts for item in part]
        if len(merged) > MAX_CLAUSES:
            raise ValueError("unlock expression is too complex")
        return merged
    atom = _atom(name, value)
    if negated:
        return [Clause(frozenset(), frozenset({atom}))]
    return [Clause(frozenset({atom}), frozenset())]


def atom_dependency(atom: Atom) -> Node | None:
    """Return the fact or moment that must exist before an atom can hold."""
    name, encoded = atom
    value = json.loads(encoded)
    if name in {"disclosed", "followed_up_on"}:
        return "fact", value
    if name == "silence_after_trailoff":
        return "fact", value["fact"]
    if name in {"elapsed_since", "no_interruption_since"}:
        return event_ref(value["event"])
    if name == "acknowledged":
        return event_ref(value)
    return None


def clause_dependencies(clause: Clause) -> set[Node]:
    return {node for atom in clause.positive if (node := atom_dependency(atom)) is not None}


def analyze_reachability(
    facts: Mapping[str, list[Clause]], moments: Mapping[str, set[Node]]
) -> tuple[set[Node], dict[str, list[Clause]]]:
    """Return reachable nodes and the satisfiable clauses for each fact."""
    nodes = {("fact", fact) for fact in facts} | {("moment", moment) for moment in moments}
    required: dict[Node, set[Node]] = {node: set(nodes) for node in nodes}

    def live(clause: Clause) -> bool:
        if self_contradictory(clause):
            return False
        dependencies = clause_dependencies(clause)
        prerequisites = set(dependencies)
        for node in dependencies:
            prerequisites |= required.get(node, set())
        forbidden = {
            ("fact", json.loads(value)) for name, value in clause.negative if name == "disclosed"
        }
        return not (forbidden & prerequisites)

    changed = True
    while changed:
        changed = False
        for fact, options in facts.items():
            node = ("fact", fact)
            options_required = []
            for clause in options:
                if not live(clause):
                    continue
                dependencies = clause_dependencies(clause)
                closure = set(dependencies)
                for item in dependencies:
                    closure |= required.get(item, set())
                options_required.append(closure)
            updated = set.intersection(*options_required) if options_required else set(nodes)
            if updated != required[node]:
                required[node] = updated
                changed = True
        for moment, dependencies in moments.items():
            node = ("moment", moment)
            closure = set(dependencies)
            for item in dependencies:
                closure |= required.get(item, set())
            if closure != required[node]:
                required[node] = closure
                changed = True
    satisfiable = {fact: [item for item in items if live(item)] for fact, items in facts.items()}
    reached: set[Node] = set()
    changed = True
    while changed:
        changed = False
        for fact, options in satisfiable.items():
            node = ("fact", fact)
            if node not in reached and any(clause_dependencies(c) <= reached for c in options):
                reached.add(node)
                changed = True
        for moment, dependencies in moments.items():
            node = ("moment", moment)
            if node not in reached and dependencies <= reached:
                reached.add(node)
                changed = True
    return reached, satisfiable
