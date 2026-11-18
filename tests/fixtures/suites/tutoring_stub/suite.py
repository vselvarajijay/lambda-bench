"""Interface stub for tutoring with changing hidden state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
class TutoringPolicy:
    target: int
    mastery: int = 0

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        if "because" not in utterance.text.lower():
            return PolicyOutput((SpeechSegment("I am lost."),), (SuiteEvent("learner_lost"),))
        self.mastery += 1
        return PolicyOutput(
            (SpeechSegment("That helps."),),
            (SuiteEvent("mastery_changed", {"mastery": self.mastery}),),
        )

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        return PolicyOutput((SpeechSegment("Why is the sky blue?"),)) if view.tick == 0 else None

    def on_delivered(self, span: DeliveredSpan) -> Sequence[SuiteEvent]:
        return ()

    def is_finished(self, view: ConversationView) -> bool:
        return self.mastery >= self.target


class MasteryCheck:
    id = "mastery"
    kind: CheckKind = "deterministic"

    def compute(self, trajectory: Trajectory) -> CheckResult:
        values = [event.data["mastery"] for event in trajectory.suite_events("mastery_changed")]
        mastery = max(values, default=0)
        return CheckResult(
            self.id, self.kind, mastery, mastery >= trajectory.scenario.hidden["target"]
        )


class TutoringSuite:
    manifest = SuiteManifest("tutoring_stub", "1", "clean", ("learner",), (0.55, 0.75))

    def scenario_schema(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "required": ["target"],
            "properties": {"target": {"type": "integer", "minimum": 1}},
        }

    def load_scenario(self, raw: Mapping[str, Any]) -> Scenario:
        return Scenario.from_raw(raw, ("target",))

    def validate_scenario(self, scenario: Scenario) -> Sequence[Issue]:
        return ()

    def model_context(self, scenario: Scenario, capabilities: Capabilities) -> ModelContext:
        return ModelContext("Teach the learner.", "Teach the learner.")

    def create_policy(self, scenario: Scenario, context: RunContext) -> CounterpartPolicy:
        return TutoringPolicy(scenario.hidden["target"])

    def checks(self) -> Sequence[Check]:
        return (MasteryCheck(),)

    def rubrics(self) -> Sequence[Rubric]:
        return ()

    def is_pass(self, result: RunResult) -> PassDecision:
        return PassDecision(bool(result.check("mastery").passed))

    def composite(self, result: RunResult) -> float | None:
        return None

    def mock_scripts(self, scenario: Scenario) -> Mapping[str, Mapping[int, str]]:
        return {
            "good": {
                1: "It scatters because of air.",
                4: "Blue scatters more because it is short.",
            },
            "bad": {1: "Just memorize it."},
        }


def create_suite() -> TutoringSuite:
    return TutoringSuite()
