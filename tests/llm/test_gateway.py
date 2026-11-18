from __future__ import annotations

import asyncio

import pytest

from lambda_duplex.llm import (
    FakeProvider,
    LLMDisabledError,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    ReplayMissError,
)

REQUEST = LLMRequest("fake", ({"role": "user", "content": "hi"},), "test", {"temperature": 0})


def test_cache_mode_serves_known_calls_and_records_new_ones() -> None:
    first = LLMGateway("cache", FakeProvider(lambda request, call: "stored"))
    assert asyncio.run(first.complete(REQUEST)).text == "stored"
    provider = FakeProvider(lambda request, call: "new")

    second = LLMGateway("cache", provider, first.entries)

    assert asyncio.run(second.complete(REQUEST)).text == "stored"
    assert provider.calls == 0
    assert asyncio.run(second.complete(REQUEST)).text == "new"
    assert [entry["occurrence"] for entry in second.entries] == [0, 1]


def test_cache_mode_requires_a_provider() -> None:
    with pytest.raises(ValueError, match="cache mode"):
        LLMGateway("cache")


def test_failed_call_does_not_use_up_an_occurrence() -> None:
    def respond(request: LLMRequest, call: int) -> str:
        if not attempts:
            attempts.append(1)
            raise OSError("reset")
        return "ok"

    attempts: list[int] = []
    gateway = LLMGateway("record", FakeProvider(respond))
    with pytest.raises(OSError, match="reset"):
        asyncio.run(gateway.complete(REQUEST))

    asyncio.run(gateway.complete(REQUEST))

    assert gateway.entries[0]["occurrence"] == 0


def test_record_then_replay_keeps_repeated_call_order() -> None:
    provider = FakeProvider(lambda request, call: f"answer {call}")
    recorder = LLMGateway("record", provider)
    first = asyncio.run(recorder.complete(REQUEST))
    second = asyncio.run(recorder.complete(REQUEST))

    replay = LLMGateway("replay", entries=recorder.entries)

    assert [first.text, second.text] == ["answer 0", "answer 1"]
    assert asyncio.run(replay.complete(REQUEST)) == first
    assert asyncio.run(replay.complete(REQUEST)) == second
    assert provider.calls == 2


def test_replay_miss_fails_loudly() -> None:
    with pytest.raises(ReplayMissError, match="test"):
        asyncio.run(LLMGateway("replay").complete(REQUEST))


def test_replay_exhaustion_fails_loudly() -> None:
    recorder = LLMGateway("record", FakeProvider(lambda request, call: "once"))
    asyncio.run(recorder.complete(REQUEST))
    replay = LLMGateway("replay", entries=recorder.entries)
    asyncio.run(replay.complete(REQUEST))

    with pytest.raises(ReplayMissError):
        asyncio.run(replay.complete(REQUEST))


def test_off_mode_rejects_calls() -> None:
    with pytest.raises(LLMDisabledError):
        asyncio.run(LLMGateway().complete(REQUEST))


def test_record_requires_provider() -> None:
    with pytest.raises(ValueError, match="provider"):
        LLMGateway("record")


def test_request_key_changes_with_parameters() -> None:
    other = LLMRequest(REQUEST.model, REQUEST.messages, REQUEST.purpose, {"temperature": 1})

    assert REQUEST.key() != other.key()


class SlowProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        text = f"reply-{self.calls}"
        await asyncio.sleep(0)
        return LLMResponse(text, "slow-1")


async def _pair(gateway: LLMGateway) -> list[str]:
    first, second = await asyncio.gather(gateway.complete(REQUEST), gateway.complete(REQUEST))
    return [first.text, second.text]


def test_concurrent_identical_calls_replay_in_order() -> None:
    recorder = LLMGateway("record", SlowProvider())
    recorded = asyncio.run(_pair(recorder))

    assert [entry["occurrence"] for entry in recorder.entries] == [0, 1]
    assert asyncio.run(_pair(LLMGateway("replay", entries=recorder.entries))) == recorded


def test_unstorable_response_does_not_use_up_an_occurrence() -> None:
    class BadThenGood(SlowProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            response = await super().complete(request)
            if self.calls == 1:
                return LLMResponse("bad", "slow-1", {"value": float("nan")})
            return response

    gateway = LLMGateway("record", BadThenGood())
    with pytest.raises(ValueError):
        asyncio.run(gateway.complete(REQUEST))

    asyncio.run(gateway.complete(REQUEST))

    assert [entry["occurrence"] for entry in gateway.entries] == [0]
