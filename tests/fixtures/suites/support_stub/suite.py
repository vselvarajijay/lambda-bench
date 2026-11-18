"""Interface stub for an outcome-graded support call."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lambda_duplex.adapters.base import Capabilities, ModelContext, ToolSpec
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
class SupportPolicy:
    target_status: str
    done: bool = False

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        if "refund issued" not in utterance.text.lower():
            return PolicyOutput((SpeechSegment("I still need the refund."),))
        self.done = True
        return PolicyOutput(
            (SpeechSegment("Thanks."),),
            (SuiteEvent("db_row_changed", {"status": self.target_status}),),
        )

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        return PolicyOutput((SpeechSegment("I need a refund."),)) if view.tick == 0 else None

    def on_delivered(self, span: DeliveredSpan) -> Sequence[SuiteEvent]:
        return ()

    def is_finished(self, view: ConversationView) -> bool:
        return self.done


class FinalStateCheck:
    id = "final_state"
    kind: CheckKind = "deterministic"

    def compute(self, trajectory: Trajectory) -> CheckResult:
        expected = trajectory.scenario.hidden["expected_status"]
        states = [event.data["status"] for event in trajectory.suite_events("db_row_changed")]
        passed = bool(states) and states[-1] == expected
        return CheckResult(self.id, self.kind, passed, passed)


class SupportSuite:
    manifest = SuiteManifest("support_stub", "1", "clean", ("caller",), (0.35, 0.6))

    def scenario_schema(self) -> Mapping[str, Any]:
        return {"type": "object", "required": ["expected_status"]}

    def load_scenario(self, raw: Mapping[str, Any]) -> Scenario:
        return Scenario.from_raw(raw, ("expected_status",))

    def validate_scenario(self, scenario: Scenario) -> Sequence[Issue]:
        return ()

    def model_context(self, scenario: Scenario, capabilities: Capabilities) -> ModelContext:
        tools = (ToolSpec("issue_refund", "Issue a refund"),) if capabilities.tools else ()
        return ModelContext("Help the caller.", "Help the caller.", tools)

    def create_policy(self, scenario: Scenario, context: RunContext) -> CounterpartPolicy:
        return SupportPolicy(scenario.hidden["expected_status"])

    def checks(self) -> Sequence[Check]:
        return (FinalStateCheck(),)

    def rubrics(self) -> Sequence[Rubric]:
        return ()

    def is_pass(self, result: RunResult) -> PassDecision:
        return PassDecision(bool(result.check("final_state").passed))

    def composite(self, result: RunResult) -> float | None:
        return None

    def mock_scripts(self, scenario: Scenario) -> Mapping[str, Mapping[int, str]]:
        return {"good": {1: "Refund issued."}, "bad": {1: "Please hold."}}


def create_suite() -> SupportSuite:
    return SupportSuite()
