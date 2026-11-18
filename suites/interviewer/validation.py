"""Static validation for interviewer scenarios."""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from lambda_duplex.suites.base import Issue, Scenario

from .unlock import (
    Clause,
    Node,
    analyze_reachability,
    any_branches,
    atom_dependency,
    clause_dependencies,
    clauses,
    event_ref,
    expression_errors,
    is_finite_number,
    self_contradictory,
)

MAX_COMPACT_WORDS = 120
LEAK_WORDS = 4
NEAR_LIMIT_S = 90
FAR_LIMIT_S = 180
PAUSE_PATTERN = re.compile(r"\[pause (\d+(?:\.\d+)?)\]")
PAUSE_LIKE_PATTERN = re.compile(r"\[pause\b[^\]]*\]", re.IGNORECASE)
LABELS = {
    "tangent": {"steered_back_politely"},
    "contradiction": {"probed_contradiction"},
    "emotional_disclosure": {"acknowledged_before_moving_on"},
}


def _duplicates(values: Sequence[str]) -> set[str]:
    return {value for value in values if values.count(value) > 1}


def _check_ids(path: str, values: Sequence[str], issues: list[Issue]) -> None:
    for value in sorted(_duplicates(values)):
        issues.append(Issue(path, f"duplicate id {value}"))


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", normalized)


def _shares_text(text: str, secret: str) -> bool:
    words = _words(text)
    secret_words = _words(secret)
    if not secret_words:
        return False
    size = min(LEAK_WORDS, len(secret_words))
    available = {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}
    return any(
        tuple(secret_words[i : i + size]) in available for i in range(len(secret_words) - size + 1)
    )


def _secret_lines(story: Mapping[str, Any], moments: Sequence[Mapping[str, Any]]) -> Iterable[str]:
    for fact in story["facts"]:
        yield fact["content"]
        yield from fact.get("paraphrases", ())
    for item in story.get("contradictions", ()):
        yield item["first"]
        yield item["second"]
    for moment in moments:
        script = moment.get("script")
        if isinstance(script, str):
            for part in PAUSE_LIKE_PATTERN.split(script):
                if len(_words(part)) >= LEAK_WORDS:
                    yield part


def _number_issues(value: Any, path: str) -> list[Issue]:
    if isinstance(value, float) and not math.isfinite(value):
        return [Issue(path, "must be a finite number")]
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > sys.float_info.max:
        return [Issue(path, "must be a finite number")]
    if isinstance(value, Mapping):
        return [issue for k, item in value.items() for issue in _number_issues(item, f"{path}/{k}")]
    if isinstance(value, list):
        return [
            issue for i, item in enumerate(value) for issue in _number_issues(item, f"{path}/{i}")
        ]
    return []


def _reference_issue(
    node: Node | None, facts: set[str], moments: set[str], path: str
) -> list[Issue]:
    if node is None:
        return [Issue(path, "must reference a fact or moment")]
    kind, target = node
    if target in (facts if kind == "fact" else moments):
        return []
    return [Issue(path, f"unknown {kind} {target}")]


def _atom_issues(
    atom: tuple[str, str],
    path: str,
    topics: set[str],
    facts: set[str],
    moments: set[str],
    budget: float,
) -> list[Issue]:
    name, encoded = atom
    value = json.loads(encoded)
    if name == "asked_about" and value not in topics:
        return [Issue(path, f"unknown topic {value}")]
    issues: list[Issue] = []
    node = atom_dependency(atom)
    if node is not None:
        issues.extend(_reference_issue(node, facts, moments, path))
    if isinstance(value, Mapping) and "min_s" in value and value["min_s"] >= budget:
        issues.append(Issue(path, "wait must be shorter than the time budget"))
    return issues


def _with_moment_dependencies(
    start: set[Node], moment_dependencies: Mapping[str, set[Node]]
) -> set[Node]:
    result: set[Node] = set()
    pending = list(start)
    while pending:
        node = pending.pop()
        if node in result:
            continue
        result.add(node)
        if node[0] == "moment":
            pending.extend(moment_dependencies.get(node[1], ()))
    return result


def _clause_tier_issues(
    fact: Mapping[str, Any],
    options: Sequence[Clause],
    tiers: Mapping[str, str],
    path: str,
    moment_dependencies: Mapping[str, set[Node]],
) -> list[Issue]:
    issues: list[Issue] = []
    tier = fact["tier"]
    for clause in options:
        all_dependencies = _with_moment_dependencies(
            clause_dependencies(clause), moment_dependencies
        )
        fact_dependencies = {target for kind, target in all_dependencies if kind == "fact"}
        names = {name for name, _ in clause.positive}
        if tier == "surface" and fact_dependencies:
            issues.append(Issue(f"{path}/tier", "surface facts cannot depend on other facts"))
        if tier == "probe" and "followed_up_on" not in names:
            issues.append(Issue(f"{path}/tier", "probe facts must require a specific follow-up"))
        if tier == "deep" and not any(
            tiers.get(item) in {"probe", "deep"} for item in fact_dependencies
        ):
            issues.append(Issue(f"{path}/tier", "deep facts must build on a probe or deep fact"))
    return list({(issue.path, issue.message): issue for issue in issues}.values())


def _trigger_dependencies(
    moment: Mapping[str, Any], path: str, budget: float
) -> tuple[set[Node], list[Issue]]:
    trigger = moment["trigger"]
    keys = set(trigger)
    if keys == {"on_disclosed"} and isinstance(trigger["on_disclosed"], str):
        return {("fact", trigger["on_disclosed"])}, []
    if keys == {"after_moment"} and isinstance(trigger["after_moment"], str):
        return {("moment", trigger["after_moment"])}, []
    if keys == {"after_event", "min_elapsed_s", "max_elapsed_s"}:
        node = event_ref(trigger["after_event"])
        low = trigger["min_elapsed_s"]
        high = trigger["max_elapsed_s"]
        issues: list[Issue] = []
        if node is None:
            issues.append(Issue(f"{path}/after_event", "must reference a fact or moment"))
        elif moment["type"] == "contradiction" and node[0] != "fact":
            issues.append(Issue(f"{path}/after_event", "contradictions must follow a fact"))
        if not is_finite_number(low) or not is_finite_number(high) or not 0 < low <= high:
            issues.append(Issue(path, "elapsed window must be positive and ordered"))
        elif high >= budget:
            issues.append(Issue(path, "elapsed window must fit inside the time budget"))
        return ({node} if node is not None else set()), issues
    return set(), [Issue(path, "unknown trigger")]


def _distance_issues(moment: Mapping[str, Any], path: str) -> list[Issue]:
    distance = moment.get("memory_distance")
    trigger = moment["trigger"]
    if distance is None:
        return [Issue(f"{path}/memory_distance", "is required for contradictions")]
    if "after_event" not in trigger:
        return [Issue(f"{path}/memory_distance", "requires an elapsed-time trigger")]
    low, high = trigger.get("min_elapsed_s"), trigger.get("max_elapsed_s")
    if not is_finite_number(low) or not is_finite_number(high):
        return []
    if distance == "near" and high >= NEAR_LIMIT_S:
        return [Issue(f"{path}/memory_distance", "near moments must finish under 90 seconds")]
    if distance == "far" and low <= FAR_LIMIT_S:
        return [Issue(f"{path}/memory_distance", "far moments must begin after 180 seconds")]
    return []


def _shape_issues(moment: Mapping[str, Any], path: str, contradictions: set[str]) -> list[Issue]:
    kind = moment["type"]
    success = moment["success"]
    script = moment.get("script")
    issues: list[Issue] = []
    if kind == "mid_thought_pause":
        if moment["severity"] != "hard":
            issues.append(Issue(f"{path}/severity", "scripted pauses must be hard"))
        onset_rule = success.get("no_substantive_onset_during_pause")
        if set(success) != {"no_substantive_onset_during_pause"} or onset_rule is not True:
            issues.append(Issue(f"{path}/success", "pause moments use the onset rule"))
        pauses = PAUSE_PATTERN.findall(script) if isinstance(script, str) else []
        markers = PAUSE_LIKE_PATTERN.findall(script) if isinstance(script, str) else []
        if len(pauses) != 1 or len(markers) != 1 or not 2 <= float(pauses[0]) <= 4:
            issues.append(Issue(f"{path}/script", "needs one [pause N] from 2 to 4 seconds"))
        return issues
    turns = success.get("within_turns")
    valid_turns = isinstance(turns, int) and not isinstance(turns, bool) and turns > 0
    if set(success) != {"classified", "within_turns"} or not valid_turns:
        issues.append(Issue(f"{path}/success", "needs classified and positive within_turns"))
    elif not isinstance(success["classified"], str) or success["classified"] not in LABELS[kind]:
        issues.append(Issue(f"{path}/success", f"unknown {kind} label"))
    if kind == "tangent" and not isinstance(script, str):
        issues.append(Issue(f"{path}/script", "tangent moments need a script"))
    if kind == "contradiction":
        target = script.get("use") if isinstance(script, Mapping) else None
        if (
            not isinstance(script, Mapping)
            or set(script) != {"use"}
            or not isinstance(target, str)
            or target not in contradictions
        ):
            issues.append(Issue(f"{path}/script", "must use a known contradiction"))
        issues.extend(_distance_issues(moment, path))
    if kind == "emotional_disclosure" and script is not None and not isinstance(script, str):
        issues.append(Issue(f"{path}/script", "must be text when present"))
    return issues


def validate_interview(scenario: Scenario) -> list[Issue]:
    """Return semantic problems in an interviewer scenario."""
    brief = scenario.data["brief"]
    story = scenario.hidden["hidden_story"]
    moments = scenario.hidden["scripted_moments"]
    facts = story["facts"]
    budget = brief["time_budget_s"]
    topics = [topic["id"] for topic in brief["required_topics"]]
    fact_ids = [fact["id"] for fact in facts]
    moment_ids = [moment["id"] for moment in moments]
    contradiction_ids = [item["id"] for item in story.get("contradictions", ())]
    fact_set, moment_set, topic_set = set(fact_ids), set(moment_ids), set(topics)
    tiers = {fact["id"]: fact["tier"] for fact in facts}
    issues = _number_issues({**scenario.data, **scenario.hidden}, "scenario")
    _check_ids("brief/required_topics", topics, issues)
    _check_ids("hidden_story/facts", fact_ids, issues)
    _check_ids("scripted_moments", moment_ids, issues)
    _check_ids("hidden_story/contradictions", contradiction_ids, issues)
    if len(brief["compact"].split()) > MAX_COMPACT_WORDS:
        issues.append(Issue("brief/compact", f"must be at most {MAX_COMPACT_WORDS} words"))
    for topic in topics:
        if not any(fact["topic"] == topic for fact in facts):
            issues.append(Issue("brief/required_topics", f"topic {topic} has no hidden fact"))
    for field in ("full", "compact"):
        if any(_shares_text(brief[field], line) for line in _secret_lines(story, moments)):
            issues.append(Issue(f"brief/{field}", "leaks hidden scenario text"))
    expressions: dict[str, list[Clause]] = {}
    paths: dict[str, str] = {}
    for index, fact in enumerate(facts):
        path = f"hidden_story/facts/{index}"
        paths[fact["id"]] = path
        if fact["topic"] not in topic_set:
            issues.append(Issue(f"{path}/topic", f"unknown topic {fact['topic']}"))
        errors = expression_errors(fact["unlock"], f"{path}/unlock")
        issues.extend(Issue(location, message) for location, message in errors)
        if errors:
            continue
        try:
            expressions[fact["id"]] = clauses(fact["unlock"])
        except ValueError as error:
            issues.append(Issue(f"{path}/unlock", str(error)))
            continue
        atoms = {atom for item in expressions[fact["id"]] for atom in item.positive | item.negative}
        for atom in sorted(atoms):
            issues.extend(
                _atom_issues(atom, f"{path}/unlock", topic_set, fact_set, moment_set, budget)
            )
    moment_dependencies: dict[str, set[Node]] = {}
    for index, moment in enumerate(moments):
        path = f"scripted_moments/{index}"
        dependencies, trigger_issues = _trigger_dependencies(moment, f"{path}/trigger", budget)
        issues.extend(trigger_issues)
        for node in dependencies:
            issues.extend(_reference_issue(node, fact_set, moment_set, f"{path}/trigger"))
        moment_dependencies[moment["id"]] = dependencies
        issues.extend(_shape_issues(moment, path, set(contradiction_ids)))
    reached, satisfiable = analyze_reachability(expressions, moment_dependencies)
    for fact in facts:
        fact_id = fact["id"]
        if fact_id not in expressions:
            continue
        path = paths[fact_id]
        if ("fact", fact_id) not in reached:
            issues.append(
                Issue(f"{path}/unlock", f"fact {fact_id} is unreachable or contradictory")
            )
        if any(not clause.positive for clause in satisfiable[fact_id]):
            issues.append(Issue(f"{path}/unlock", "contains an ungated path"))
        for branch in any_branches(fact["unlock"]):
            if all(self_contradictory(clause) for clause in clauses(branch)):
                issues.append(Issue(f"{path}/unlock", "contains an impossible branch"))
                break
        options = [clause for clause in satisfiable[fact_id] if clause.positive]
        issues.extend(_clause_tier_issues(fact, options, tiers, path, moment_dependencies))
    for index, moment in enumerate(moments):
        if ("moment", moment["id"]) not in reached:
            issues.append(Issue(f"scripted_moments/{index}/trigger", "moment is unreachable"))
    return issues
