from __future__ import annotations

import asyncio

import pytest

from lambda_duplex.llm import (
    FakeProvider,
    LLMDisabledError,
    LLMGateway,
    LLMRequest,
    ReplayMissError,
)

REQUEST = LLMRequest("fake", ({"role": "user", "content": "hi"},), "test", {"temperature": 0})


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
