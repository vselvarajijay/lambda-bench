"""Interviewer benchmark suite."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambda_duplex.adapters.base import Capabilities, ModelContext
from lambda_duplex.suites.base import (
    Check,
    CounterpartPolicy,
    Issue,
    PassDecision,
    Rubric,
    RunContext,
    RunResult,
    Scenario,
    SuiteManifest,
)

from .simulate import simulate
from .validation import validate_interview

HIDDEN_KEYS = ("hidden_story", "scripted_moments", "persona", "pass")


class InterviewerSuite:
    manifest = SuiteManifest(
        name="interviewer",
        version="0.1",
        default_condition="realistic",
        default_personas=("guarded_pauser",),
        talk_ratio_target=(0.10, 0.35),
    )

    def scenario_schema(self) -> Mapping[str, Any]:
        schema: dict[str, Any] = json.loads(
            (Path(__file__).parent / "schema.json").read_text(encoding="utf-8")
        )
        return schema

    def load_scenario(self, raw: Mapping[str, Any]) -> Scenario:
        return Scenario.from_raw(raw, HIDDEN_KEYS)

    def validate_scenario(self, scenario: Scenario) -> Sequence[Issue]:
        return validate_interview(scenario)

    def model_context(self, scenario: Scenario, capabilities: Capabilities) -> ModelContext:
        brief = scenario.data["brief"]
        return ModelContext(full=brief["full"], compact=brief["compact"])

    def simulate_script(
        self, scenario: Scenario, script: Mapping[str, Any]
    ) -> Sequence[Mapping[str, Any]]:
        if not isinstance(script, Mapping) or set(script) != {"steps"}:
            raise ValueError("script must contain only steps")
        return simulate(scenario, script["steps"])

    def create_policy(self, scenario: Scenario, context: RunContext) -> CounterpartPolicy:
        raise NotImplementedError("the interviewee policy is added in slice 3.3")

    def checks(self) -> Sequence[Check]:
        return ()

    def rubrics(self) -> Sequence[Rubric]:
        return ()

    def is_pass(self, result: RunResult) -> PassDecision:
        return PassDecision(False, ("interviewer scoring is added in slice 3.5",))

    def composite(self, result: RunResult) -> float | None:
        return None


def create_suite() -> InterviewerSuite:
    return InterviewerSuite()
