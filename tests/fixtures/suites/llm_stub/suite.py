"""Stub suite whose counterpart depends on an LLM and a seeded stream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lambda_duplex.adapters.base import Capabilities, ModelContext
from lambda_duplex.llm import LLMGateway, LLMRequest
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
class LLMPolicy:
    llm: LLMGateway
    token: int
    done: bool = False

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        request = LLMRequest(
            "fake", ({"role": "user", "content": utterance.text},), "reply", {"token": self.token}
        )
        response = await self.llm.complete(request)
        self.done = True
        return PolicyOutput(
            (SpeechSegment(response.text),), (SuiteEvent("replied", {"token": self.token}),)
        )

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        return None

    def on_delivered(self, span: DeliveredSpan) -> Sequence[SuiteEvent]:
        return ()

    def is_finished(self, view: ConversationView) -> bool:
        return self.done


class RepliedCheck:
    id = "replied"
    kind: CheckKind = "deterministic"

    def compute(self, trajectory: Trajectory) -> CheckResult:
        passed = bool(trajectory.suite_events("replied"))
        return CheckResult(self.id, self.kind, passed, passed)


class LLMSuite:
    manifest = SuiteManifest("llm_stub", "1", "clean", (), (0.0, 1.0))

    def scenario_schema(self) -> Mapping[str, object]:
        return {"type": "object"}

    def load_scenario(self, raw: Mapping[str, object]) -> Scenario:
        return Scenario.from_raw(raw, ())

    def validate_scenario(self, scenario: Scenario) -> Sequence[Issue]:
        return ()

    def model_context(self, scenario: Scenario, capabilities: Capabilities) -> ModelContext:
        return ModelContext("Talk.", "Talk.")

    def create_policy(self, scenario: Scenario, context: RunContext) -> CounterpartPolicy:
        return LLMPolicy(context.llm, context.rng("policy").randrange(1_000_000))

    def checks(self) -> Sequence[Check]:
        return (RepliedCheck(),)

    def rubrics(self) -> Sequence[Rubric]:
        return ()

    def is_pass(self, result: RunResult) -> PassDecision:
        return PassDecision(bool(result.check("replied").passed))

    def composite(self, result: RunResult) -> float | None:
        return None

    def mock_scripts(self, scenario: Scenario) -> Mapping[str, Mapping[int, str]]:
        return {"good": {0: "Hello."}}


def create_suite() -> LLMSuite:
    return LLMSuite()
