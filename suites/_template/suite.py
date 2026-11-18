"""Starter suite. Replace this behavior with your use case."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lambda_duplex.adapters.base import Capabilities, ModelContext
from lambda_duplex.suites.base import (
    Check,
    CheckKind,
    CheckResult,
    ConversationView,
    CounterpartPolicy,
    DeliveredSpan,
    Issue,
    ModelUtterance,
    PassDecision,
    PolicyOutput,
    Rubric,
    RunContext,
    RunResult,
    Scenario,
    SpeechSegment,
    SuiteEvent,
    SuiteManifest,
    Trajectory,
)

SUITE_NAME = "_template"


@dataclass
class StarterPolicy:
    answer: str
    revealed: bool = field(default=False)

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        if "answer" not in utterance.text.lower() or self.revealed:
            return PolicyOutput((SpeechSegment("Could you ask that another way?"),))
        self.revealed = True
        return PolicyOutput((SpeechSegment(f"It is {self.answer}.", ("answer",)),))

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        if view.tick == 0:
            return PolicyOutput((SpeechSegment("Hello."),))
        return None

    def on_delivered(self, span: DeliveredSpan) -> Sequence[SuiteEvent]:
        if "answer" in span.tags:
            return (SuiteEvent("goal_reached", {"tick": span.end_tick}),)
        return ()

    def is_finished(self, view: ConversationView) -> bool:
        return any("answer" in span.tags for span in view.delivered)


class GoalCheck:
    id = "goal_reached"
    kind: CheckKind = "deterministic"

    def compute(self, trajectory: Trajectory) -> CheckResult:
        found = bool(trajectory.suite_events("goal_reached"))
        return CheckResult(self.id, self.kind, found, found)


class StarterSuite:
    manifest = SuiteManifest(
        name=SUITE_NAME,
        version="1",
        default_condition="clean",
        default_personas=("neutral",),
        talk_ratio_target=(0.2, 0.8),
        goal_event="goal_reached",
    )

    def scenario_schema(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "required": ["brief", "answer"],
            "properties": {
                "brief": {"type": "string", "minLength": 1},
                "answer": {"type": "string", "minLength": 1},
            },
        }

    def load_scenario(self, raw: Mapping[str, Any]) -> Scenario:
        return Scenario.from_raw(raw, ("answer",))

    def validate_scenario(self, scenario: Scenario) -> Sequence[Issue]:
        if scenario.hidden["answer"].lower() in scenario.data["brief"].lower():
            return (Issue("brief", "brief leaks the hidden answer"),)
        return ()

    def model_context(self, scenario: Scenario, capabilities: Capabilities) -> ModelContext:
        brief = scenario.data["brief"]
        return ModelContext(full=brief, compact=brief)

    def create_policy(self, scenario: Scenario, context: RunContext) -> CounterpartPolicy:
        return StarterPolicy(scenario.hidden["answer"])

    def checks(self) -> Sequence[Check]:
        return (GoalCheck(),)

    def rubrics(self) -> Sequence[Rubric]:
        return ()

    def is_pass(self, result: RunResult) -> PassDecision:
        if result.check("goal_reached").passed:
            return PassDecision(True)
        return PassDecision(False, ("answer was not disclosed",))

    def composite(self, result: RunResult) -> float | None:
        return None

    def mock_scripts(self, scenario: Scenario) -> Mapping[str, Mapping[int, str]]:
        return {"good": {1: "What is the answer?"}, "bad": {1: "Nice weather today."}}


def create_suite() -> StarterSuite:
    return StarterSuite()
