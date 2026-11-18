"""Trivial suite: learn a code word held by the counterpart."""

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


@dataclass
class DemoPolicy:
    code_word: str
    revealed: bool = field(default=False)

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        if "code word" not in utterance.text.lower() or self.revealed:
            return PolicyOutput((SpeechSegment("Could you ask that another way?"),))
        self.revealed = True
        return PolicyOutput((SpeechSegment(f"It is {self.code_word}.", ("code_word",)),))

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        if view.tick == 0:
            return PolicyOutput((SpeechSegment("Hello."),))
        return None

    def on_delivered(self, span: DeliveredSpan) -> Sequence[SuiteEvent]:
        if "code_word" in span.tags:
            return (SuiteEvent("code_word_disclosed", {"tick": span.end_tick}),)
        return ()

    def is_finished(self, view: ConversationView) -> bool:
        return any("code_word" in span.tags for span in view.delivered)


class CodeWordCheck:
    id = "code_word_disclosed"
    kind: CheckKind = "deterministic"

    def compute(self, trajectory: Trajectory) -> CheckResult:
        found = bool(trajectory.suite_events("code_word_disclosed"))
        return CheckResult(self.id, self.kind, found, found)


class DemoSuite:
    manifest = SuiteManifest(
        name="demo",
        version="1",
        default_condition="clean",
        default_personas=("neutral",),
        talk_ratio_target=(0.2, 0.8),
        goal_event="code_word_disclosed",
    )

    def scenario_schema(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "required": ["brief", "code_word"],
            "properties": {
                "brief": {"type": "string", "minLength": 1},
                "code_word": {"type": "string", "minLength": 1},
            },
        }

    def load_scenario(self, raw: Mapping[str, Any]) -> Scenario:
        return Scenario.from_raw(raw, ("code_word",))

    def validate_scenario(self, scenario: Scenario) -> Sequence[Issue]:
        if scenario.hidden["code_word"].lower() in scenario.data["brief"].lower():
            return (Issue("brief", "brief leaks the hidden code word"),)
        return ()

    def model_context(self, scenario: Scenario, capabilities: Capabilities) -> ModelContext:
        brief = scenario.data["brief"]
        return ModelContext(full=brief, compact=brief)

    def create_policy(self, scenario: Scenario, context: RunContext) -> CounterpartPolicy:
        return DemoPolicy(scenario.hidden["code_word"])

    def checks(self) -> Sequence[Check]:
        return (CodeWordCheck(),)

    def rubrics(self) -> Sequence[Rubric]:
        return ()

    def is_pass(self, result: RunResult) -> PassDecision:
        if result.check("code_word_disclosed").passed:
            return PassDecision(True)
        return PassDecision(False, ("code word was not disclosed",))

    def composite(self, result: RunResult) -> float | None:
        return None

    def mock_scripts(self, scenario: Scenario) -> Mapping[str, Mapping[int, str]]:
        return {
            "good": {1: "What is the code word?"},
            "bad": {1: "Nice weather today."},
            "known_behavior": {1: "What is the code word?", 12: "Thanks, that helps."},
        }


def create_suite() -> DemoSuite:
    return DemoSuite()
