from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lambda_duplex import cli
from lambda_duplex.judges import (
    BlindingError,
    JudgeConfig,
    JudgeError,
    JudgeResponseError,
    RubricError,
    create_provider,
    fake_judge_response,
    judge_run,
    load_judgments,
    load_rubric,
    resolve_rubric,
    shuffled_runs,
    validate_response,
)
from lambda_duplex.judges.blinding import (
    Turn,
    _fingerprint,
    assert_blinded,
    blinded_transcript,
    identity_terms,
    redact,
    transcript_fingerprint,
)
from lambda_duplex.judges.rubric import parse_rubric
from lambda_duplex.llm import FakeProvider, LLMRequest
from lambda_duplex.trajectory.replay import verify_run
from lambda_duplex.trajectory.store import find_run_dirs, load_run

ROOT = Path(__file__).resolve().parents[2]
RUBRIC = resolve_rubric("question_quality", None)
TURNS = (
    Turn(0, "B", "Hello there."),
    Turn(1, "A", "What brought you here today?"),
)
Respond = Callable[[LLMRequest, int], str]


def demo_run(out: Path, model: str = "mock") -> Path:
    arguments = ["run", "--suite", str(ROOT / "suites/demo"), "--model", model, "--out", str(out)]
    assert cli.main(arguments) == 0
    return find_run_dirs(out)[0]


def config(respond: Respond = fake_judge_response, **values: Any) -> JudgeConfig:
    values.setdefault("backoff_s", 0)
    return JudgeConfig(RUBRIC, "fake", FakeProvider(respond), "fake-judge", **values)


def judge(path: Path, settings: JudgeConfig) -> Any:
    return asyncio.run(judge_run(path, settings))


def reply(**changes: Any) -> dict[str, Any]:
    items = [
        {
            "id": item.id,
            "score": 3,
            "rationale": "Clear.",
            "evidence": [{"turn": 1, "quote": "what brought you"}],
        }
        for item in RUBRIC.items
    ]
    items[0].update(changes)
    return {"items": items}


def rejected(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    with pytest.raises(JudgeResponseError) as caught:
        validate_response(text, RUBRIC, TURNS)
    return str(caught.value)


def test_valid_reply_keeps_scores_and_evidence() -> None:
    items = validate_response(json.dumps(reply()), RUBRIC, TURNS)

    assert [item["id"] for item in items] == ["open_questions", "neutrality"]
    assert items[0]["score"] == 3
    assert items[0]["evidence"] == [{"turn": 1, "speaker": "A", "quote": "what brought you"}]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"items": [NaN]}', "wrong shape"),
        ({"items": []}, "exactly these items"),
        ({"items": [], "extra": 1}, "wrong shape"),
        (reply(score=6), "outside the scale"),
        (reply(score=0), "outside the scale"),
        (reply(score=3.0), "outside the scale"),
        (reply(score=True), "wrong shape"),
        (reply(score="3"), "wrong shape"),
        (reply(rationale=""), "wrong shape"),
        (reply(id="unknown"), "exactly these items"),
        (reply(evidence=[]), "needs 1 quotes"),
        (reply(evidence=[{"turn": 9, "quote": "what"}]), "does not exist"),
        (reply(evidence=[{"turn": 1.0, "quote": "what"}]), "does not exist"),
        (reply(evidence=[{"turn": 1, "quote": "never said"}]), "not in turn"),
        (reply(evidence=[{"turn": 1, "quote": "wha"}]), "not in turn"),
        (reply(evidence=[{"turn": 1, "quote": "..."}]), "not in turn"),
        (reply(evidence=[{"turn": 0, "quote": "hello"}]), "quote from speaker A"),
        (reply(evidence=[{"turn": 1, "quote": "what"}, {"turn": 1, "quote": "WHAT"}]), "repeats"),
    ],
)
def test_schema_failures_are_rejected(value: Any, message: str) -> None:
    assert message in rejected(value)


def test_duplicate_items_are_rejected() -> None:
    value = reply()
    value["items"][1] = value["items"][0]

    assert "exactly these items" in rejected(value)


def test_not_applicable_needs_permission() -> None:
    assert validate_response(json.dumps(reply(score=None, evidence=[])), RUBRIC, TURNS)
    raw = {
        "id": "strict",
        "version": "1",
        "instructions": "Score.",
        "items": [{"id": "open_questions", "prompt": "P", "scale": {"min": 1, "max": 2}}],
    }
    answer = {"items": [reply(score=None, evidence=[])["items"][0]]}

    with pytest.raises(JudgeResponseError, match="score is required"):
        validate_response(json.dumps(answer), parse_rubric(raw), TURNS)


def test_quotes_ignore_case_and_punctuation_but_not_word_order() -> None:
    assert validate_response(
        json.dumps(reply(evidence=[{"turn": 1, "quote": "WHAT, brought you"}])), RUBRIC, TURNS
    )
    assert "not in turn" in rejected(reply(evidence=[{"turn": 1, "quote": "you brought what"}]))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "Bad/Id"}, "invalid rubric"),
        ({"items": []}, "invalid rubric"),
        ({"surprise": 1}, "invalid rubric"),
        ({"items": [{"id": "a", "prompt": "P", "scale": {"min": 3, "max": 3}}]}, "below max"),
        (
            {
                "items": [
                    {
                        "id": "a",
                        "prompt": "P",
                        "scale": {"min": 1, "max": 3},
                        "anchors": [{"score": 9, "text": "T"}],
                    }
                ]
            },
            "outside the scale",
        ),
        (
            {
                "items": [
                    {"id": "a", "prompt": "P", "scale": {"min": 1, "max": 3}},
                    {"id": "a", "prompt": "Q", "scale": {"min": 1, "max": 3}},
                ]
            },
            "unique",
        ),
    ],
)
def test_invalid_rubrics_are_rejected(change: dict[str, Any], message: str) -> None:
    raw = {
        "id": "r",
        "version": "1",
        "instructions": "Score.",
        "items": [{"id": "a", "prompt": "P", "scale": {"min": 1, "max": 3}}],
    }
    with pytest.raises(RubricError, match=message):
        parse_rubric({**raw, **change})


def test_rubric_lookup_prefers_the_suite_and_checks_the_id(tmp_path: Path) -> None:
    folder = tmp_path / "rubrics"
    folder.mkdir()
    text = (
        'id: question_quality\nversion: "suite"\ninstructions: Score.\n'
        "items:\n  - {id: a, prompt: P, scale: {min: 1, max: 2}}\n"
    )
    (folder / "question_quality.yaml").write_text(text, encoding="utf-8")
    assert resolve_rubric("question_quality", tmp_path).version == "suite"

    (folder / "other.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(RubricError, match="declares id"):
        resolve_rubric("other", tmp_path)
    with pytest.raises(RubricError, match="unknown rubric"):
        resolve_rubric("missing", tmp_path)
    assert load_rubric(folder / "other.yaml").id == "question_quality"


def test_rubric_rejects_yaml_aliases(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("id: &a x\nversion: *a\n", encoding="utf-8")

    with pytest.raises(RubricError, match="cannot read"):
        load_rubric(path)


def test_outputs_are_separate_and_marked_non_deterministic(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    protected = ["result.json", "manifest.json", "events.jsonl", "llm_cache.jsonl", "scenario.json"]
    before = {name: (run / name).read_bytes() for name in protected}

    outcome = judge(run, config())

    record = load_judgments(run)[0]
    assert outcome.record["status"] == "scored"
    assert record["deterministic"] is False
    assert record["kind"] == "judge"
    assert record["calibration"] == "uncalibrated"
    assert record["rubric"]["sha256"] == RUBRIC.sha256
    assert record["judge"]["model_version"] == "fake-1"
    assert {name: (run / name).read_bytes() for name in protected} == before
    report = asyncio.run(verify_run(run))
    assert report.ok, report.problems


def test_stored_judgment_must_stay_marked_non_deterministic(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    record = load_judgments(run)[0]
    record["deterministic"] = True
    (run / "judges.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(JudgeError, match="invalid judgment"):
        load_judgments(run)


def test_cache_avoids_a_second_call_and_keeps_output_identical(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    first = judge(run, config())
    stored = (run / "judges.jsonl").read_bytes()
    settings = config(lambda request, call: pytest.fail("provider must not be called"))

    second = judge(run, settings)

    assert (first.provider_calls, first.cache_hits) == (1, 0)
    assert (second.provider_calls, second.cache_hits) == (0, 1)
    assert (run / "judges.jsonl").read_bytes() == stored


def test_cache_misses_when_rubric_model_or_seed_changes(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    raw = {
        "id": RUBRIC.id,
        "version": "0.2",
        "instructions": RUBRIC.instructions,
        "items": [{**item.public(), "anchors": []} for item in RUBRIC.items],
    }
    provider = FakeProvider(fake_judge_response)
    base_order = load_judgments(run)[0]["blinding"]["item_order"]
    seed = next(
        value
        for value in range(1, 50)
        if judge(run, config(seed=value)).record["blinding"]["item_order"] != base_order
    )
    (run / "judge_cache.jsonl").unlink()
    judge(run, config())

    changed = [
        JudgeConfig(parse_rubric(raw), "fake", provider, "fake-judge"),
        JudgeConfig(RUBRIC, "fake", provider, "other-judge"),
        JudgeConfig(RUBRIC, "fake", provider, "fake-judge", seed=seed),
    ]
    assert [judge(run, item).provider_calls for item in changed] == [1, 1, 1]


def test_run_path_check_ignores_short_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = demo_run(tmp_path / "runs")
    monkeypatch.chdir(run)

    assert judge(Path("."), config()).record["status"] == "scored"


def test_plain_word_settings_are_not_redacted(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["model"]["version"] = "main"
    manifest["model"]["settings"] = {"mode": "text", "checkpoint": "tuned-v3"}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    terms = identity_terms(load_run(run))

    assert "tuned-v3" in terms
    assert "main" not in terms and "text" not in terms
    assert judge(run, config()).record["status"] == "scored"


def test_refresh_replaces_cached_reply(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    provider_reply = json.loads(fake_judge_response(_first_request(run), 0))
    provider_reply["items"][0]["rationale"] = "Fresh."
    settings = config(lambda request, call: json.dumps(provider_reply), refresh=True)

    outcome = judge(run, settings)

    assert outcome.provider_calls == 1
    assert "Fresh." in {item["rationale"] for item in load_judgments(run)[0]["items"]}
    assert len((run / "judge_cache.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert judge(run, config(lambda request, call: pytest.fail("cached"))).provider_calls == 0


def _first_request(run: Path) -> LLMRequest:
    line = (run / "judge_cache.jsonl").read_text(encoding="utf-8").splitlines()[0]
    request = json.loads(line)["request"]
    return LLMRequest(**{**request, "messages": tuple(request["messages"])})


def test_tampered_cache_is_refused(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    entry = json.loads((run / "judge_cache.jsonl").read_text(encoding="utf-8"))
    entry["request"]["model"] = "other"
    (run / "judge_cache.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")

    with pytest.raises(JudgeError, match="does not match"):
        judge(run, config())


def test_invalid_reply_is_retried_with_feedback_then_cached(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    seen: list[LLMRequest] = []

    def respond(request: LLMRequest, call: int) -> str:
        seen.append(request)
        return "nonsense" if call == 0 else fake_judge_response(request, call)

    outcome = judge(run, config(respond))

    assert outcome.record["status"] == "scored"
    assert outcome.record["attempts"] == 2
    assert "not valid JSON" in outcome.record["errors"][0]
    assert seen[1].messages[2]["content"] == "nonsense"
    assert "rejected" in seen[1].messages[3]["content"]
    again = judge(run, config(lambda request, call: pytest.fail("cached")))
    assert again.record == outcome.record
    assert again.cache_hits == 2


def test_exhausted_retries_store_a_failure_without_scores(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")

    outcome = judge(run, config(lambda request, call: "nonsense", retries=1))

    assert outcome.record["status"] == "failed"
    assert outcome.record["items"] == []
    assert outcome.record["attempts"] == 2
    assert outcome.provider_calls == 2


def test_provider_errors_are_retried_and_not_cached(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    calls = {"count": 0}

    def respond(request: LLMRequest, call: int) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("connection reset")
        return fake_judge_response(request, call)

    outcome = judge(run, config(respond))

    assert outcome.record["status"] == "scored"
    assert outcome.record["errors"] == ["provider: connection reset"]
    assert outcome.provider_calls == 1
    assert len((run / "judge_cache.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_replies_are_kept_when_the_provider_crashes(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")

    def respond(request: LLMRequest, call: int) -> str:
        if call == 0:
            return "nonsense"
        raise KeyError("provider bug")

    with pytest.raises(KeyError):
        judge(run, config(respond))

    assert (run / "judge_cache.jsonl").is_file()
    assert not (run / "judges.jsonl").exists()


def _request_for(run: Path, **values: Any) -> tuple[LLMRequest, Any]:
    seen: list[LLMRequest] = []

    def respond(request: LLMRequest, call: int) -> str:
        seen.append(request)
        return fake_judge_response(request, call)

    outcome = judge(run, config(respond, **values))
    return seen[0], outcome


def test_request_hides_identity_results_and_hidden_facts(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs", "mock:bad")
    stored = load_run(run)
    hidden = stored.scenario.hidden["code_word"]

    request, _ = _request_for(run)

    content = json.dumps(request.messages).lower()
    for leaked in ("mock", hidden.lower(), "passed", "code_word", str(run).lower()):
        assert leaked not in content
    payload = json.loads(request.messages[1]["content"])
    assert set(payload) == {"items", "transcript"}
    assert {row["speaker"] for row in payload["transcript"]} == {"A", "B"}


def _rewrite_model_text(run: Path, text: str) -> None:
    lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    for row in rows:
        if row["source"] == "model" and row["type"] == "spoke":
            row["data"]["text"] = text
    (run / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_self_identification_is_redacted(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    _rewrite_model_text(run, "I am MOCK, built by Acme-Labs. What is your name?")

    request, outcome = _request_for(run, blind_terms=("acme labs",))

    content = request.messages[1]["content"]
    assert "MOCK" not in content and "Acme" not in content
    assert "I am [REDACTED], built by [REDACTED]." in content
    assert outcome.record["blinding"]["redactions"] == 2


def test_redaction_keeps_longer_words() -> None:
    assert redact("mockingbird and mock", ("mock",)) == ("mockingbird and [REDACTED]", 1)
    assert redact("unchanged", ()) == ("unchanged", 0)


def test_short_or_numeric_identity_values_are_not_terms(tmp_path: Path) -> None:
    run = load_run(demo_run(tmp_path / "runs"))

    assert identity_terms(run) == ("mock-0", "mock")


def test_blinding_check_blocks_leaks_before_any_call(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    leaking = {
        "id": RUBRIC.id,
        "version": "leak",
        "instructions": "This speaker is the mock system.",
        "items": [{"id": "a", "prompt": "P", "scale": {"min": 1, "max": 2}}],
    }
    provider = FakeProvider(fake_judge_response)
    settings = JudgeConfig(parse_rubric(leaking), "fake", provider, "fake-judge")

    outcome = judge(run, settings)

    assert provider.calls == 0
    assert outcome.record["status"] == "failed"
    assert outcome.record["errors"][0].startswith("blinding:")


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"items": [], "transcript": [], "model": "x"}),
        json.dumps({"items": [], "transcript": [{"turn": 0, "seed": 1}]}),
        json.dumps({"items": [], "transcript": [{"text": "RUN_PATH"}]}),
    ],
)
def test_blinding_check_rejects_unexpected_payloads(tmp_path: Path, content: str) -> None:
    run = load_run(demo_run(tmp_path / "runs"))
    body = content.replace("RUN_PATH", str(run.path))
    request = LLMRequest(
        "j", ({"role": "system", "content": ""}, {"role": "user", "content": body}), "p"
    )

    with pytest.raises(BlindingError):
        assert_blinded(request, (), run)


def test_item_order_is_seeded_and_scores_stay_on_their_items(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    orders = {
        tuple(judge(run, config(seed=seed)).record["blinding"]["item_order"]) for seed in range(8)
    }

    assert orders == {("open_questions", "neutrality"), ("neutrality", "open_questions")}
    assert (
        judge(run, config(seed=1)).record["blinding"]["item_order"]
        == judge(run, config(seed=1)).record["blinding"]["item_order"]
    )


def test_run_order_does_not_follow_directory_order() -> None:
    paths = [Path(f"run{index}") for index in range(20)]

    assert shuffled_runs(paths, 1, "r") == shuffled_runs(list(reversed(paths)), 1, "r")
    assert shuffled_runs(paths, 1, "r") != sorted(paths)
    assert shuffled_runs(paths, 1, "r") != shuffled_runs(paths, 2, "r")


def test_stored_transcript_replaces_adapter_text(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    row = {"start_ms": 100, "text": "Heard from audio?", "source": "whisper"}
    (run / "transcript.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    transcript = blinded_transcript(load_run(run))

    assert transcript.source == "transcript:whisper"
    assert [turn.text for turn in transcript.turns if turn.speaker == "A"] == ["Heard from audio?"]


def test_invalid_transcript_is_refused(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    (run / "transcript.jsonl").write_text('{"source": "x"}\n', encoding="utf-8")

    assert cli.main(["judge", str(run), "--rubric", "question_quality", "--provider", "fake"]) == 2


def test_silent_model_is_unscorable_without_a_call(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    (run / "transcript.jsonl").write_text("", encoding="utf-8")
    provider = FakeProvider(fake_judge_response)

    outcome = judge(run, JudgeConfig(RUBRIC, "fake", provider, "fake-judge"))

    assert provider.calls == 0
    assert outcome.record["status"] == "unscorable"
    assert "no transcribed speech" in outcome.record["errors"][0]


def test_incomplete_run_is_unscorable(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["status"] = "failed"
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")

    assert judge(run, config()).record["status"] == "unscorable"


def test_cut_off_counterpart_turn_is_marked(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    for row in rows:
        if row["type"] == "segment_delivered" and row["data"]["start_tick"] == 0:
            row["data"]["interrupted"] = True
    (run / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    turns = blinded_transcript(load_run(run)).turns

    assert turns[0].cut_off is True
    assert turns[0].text == ""
    assert "cut_off" not in turns[1].public()


def test_cli_judges_then_uses_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = demo_run(tmp_path / "runs")
    capsys.readouterr()
    arguments = ["judge", str(tmp_path / "runs"), "--rubric", "question_quality"]
    arguments += ["--provider", "fake"]

    assert cli.main(arguments) == 0
    first = capsys.readouterr().out
    assert cli.main(arguments) == 0
    second = capsys.readouterr().out

    assert "status=scored" in first and "provider_calls=1 cache_hits=0" in first
    assert "provider_calls=0 cache_hits=1" in second
    assert "non-deterministic" in second
    assert (run / "judges.jsonl").is_file()


def test_cli_reports_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = demo_run(tmp_path / "runs")
    base = ["judge", str(run), "--provider", "fake"]

    assert cli.main([*base, "--rubric", "missing"]) == 2
    assert "unknown rubric" in capsys.readouterr().err
    assert cli.main([*base, "--rubric", "question_quality", "--retries", "-1"]) == 2
    empty = ["judge", str(tmp_path / "none"), "--provider", "fake", "--rubric", "question_quality"]
    assert cli.main(empty) == 2
    (run / "judge_cache.jsonl").write_text("broken\n", encoding="utf-8")
    assert cli.main([*base, "--rubric", "question_quality"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["judge", str(run), "--provider", "real", "--rubric", "question_quality"])


def test_cli_exits_nonzero_when_a_judgment_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = demo_run(tmp_path / "runs")
    provider = FakeProvider(lambda request, call: "nonsense")
    monkeypatch.setattr(cli, "create_provider", lambda name: provider)

    code = cli.main(["judge", str(run), "--rubric", "question_quality", "--provider", "fake"])

    assert code == 1
    assert load_judgments(run)[0]["status"] == "failed"


def test_cli_accepts_a_rubric_file(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    path = tmp_path / "custom.yaml"
    path.write_text(
        'id: custom\nversion: "1"\ninstructions: Score.\n'
        "items:\n  - {id: a, prompt: P, scale: {min: 1, max: 2}}\n",
        encoding="utf-8",
    )

    assert cli.main(["judge", str(run), "--rubric", str(path), "--provider", "fake"]) == 0
    assert load_judgments(run)[0]["rubric"]["id"] == "custom"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown judge provider"):
        create_provider("real")


def _add_event(run: Path, **values: Any) -> None:
    path = run / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.append({**rows[-1], **values})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_injected_transcript(run: Path) -> None:
    row = {"time_ms": 0, "text": "Planned words.", "source": "injected"}
    (run / "transcript.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_heard_speech_outranks_an_injected_transcript(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    _write_injected_transcript(run)
    _add_event(
        run,
        source="counterpart",
        type="model_transcript",
        data={"text": "Heard words.", "start_tick": 1},
    )

    transcript = blinded_transcript(load_run(run))

    assert transcript.source == "asr_events"
    assert [turn.text for turn in transcript.turns if turn.speaker == "A"] == ["Heard words."]


@pytest.mark.parametrize("injected", [False, True])
def test_truncated_adapter_text_is_not_judged(tmp_path: Path, injected: bool) -> None:
    run = demo_run(tmp_path / "runs")
    if injected:
        _write_injected_transcript(run)
    _add_event(run, source="model", type="truncated", data={"ms": 160})

    outcome = judge(run, config(lambda request, call: pytest.fail("no call expected")))

    assert outcome.record["status"] == "unscorable"
    assert "cannot show what was delivered" in outcome.record["errors"][0]


def test_cut_off_words_never_reach_the_judge(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    spoken = next(
        event
        for event in load_run(run).trajectory.events
        if event.source == "counterpart" and event.type == "spoke"
    )
    path = run / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["type"] == "segment_delivered" and row["data"]["start_tick"] == spoken.tick:
            row["data"]["interrupted"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    seen: list[LLMRequest] = []

    def respond(request: LLMRequest, call: int) -> str:
        seen.append(request)
        return fake_judge_response(request, call)

    outcome = judge(run, config(respond))

    assert outcome.record["status"] == "scored"
    assert spoken.data["text"] not in seen[0].messages[1]["content"]
    assert spoken.data["text"] not in (run / "judges.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("said", "quoted"), [("I don\u2019t know", "don't know"), ("I don't know", "don\u2019t know")]
)
def test_quotes_accept_either_apostrophe(said: str, quoted: str) -> None:
    row = reply()
    for item in row["items"]:
        item["evidence"] = [{"turn": 1, "quote": quoted}]

    assert validate_response(json.dumps(row), RUBRIC, (TURNS[0], Turn(1, "A", said)))


def test_cache_is_not_shared_between_providers(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    provider = FakeProvider(fake_judge_response)

    outcome = judge(run, JudgeConfig(RUBRIC, "other", provider, "fake-judge", backoff_s=0))

    assert outcome.provider_calls == 1
    assert provider.calls == 1


def test_failed_pass_asks_again_on_the_next_run(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    assert judge(run, config(lambda r, c: "nonsense", retries=1)).record["status"] == "failed"

    outcome = judge(run, config(retries=1))

    assert outcome.record["status"] == "scored"
    assert outcome.provider_calls == 1


def test_replies_from_a_crash_are_dropped_if_the_next_pass_fails(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")

    def crash(request: LLMRequest, call: int) -> str:
        if call == 0:
            return "nonsense"
        raise KeyError("provider bug")

    with pytest.raises(KeyError):
        judge(run, config(crash))
    judge(run, config(lambda r, c: "still nonsense", retries=1))

    assert "nonsense" not in (run / "judge_cache.jsonl").read_text(encoding="utf-8")

    outcome = judge(run, config(retries=1))

    assert outcome.record["status"] == "scored"
    assert outcome.provider_calls == 1


def test_two_character_model_names_are_hidden(tmp_path: Path) -> None:
    run = load_run(demo_run(tmp_path / "runs"))
    run.manifest["model"]["name"] = "o1"

    assert redact("I am o1 today", identity_terms(run)) == ("I am [REDACTED] today", 1)


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.update(calibration="calibrated"),
        lambda row: row["rubric"].pop("id"),
        lambda row: row["judge"].pop("provider"),
        lambda row: row["blinding"].pop("seed"),
    ],
)
def test_malformed_stored_judgment_is_refused(
    tmp_path: Path, change: Callable[[dict[str, Any]], Any]
) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    path = run / "judges.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    change(row)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(JudgeError, match="invalid judgment"):
        load_judgments(run)


def test_parallel_passes_keep_every_judgment(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda seed: judge(run, config(seed=seed)), range(8)))

    assert len(load_judgments(run)) == 8
    assert not list(run.glob(".judges.jsonl.*"))


def test_cli_finishes_other_runs_after_a_bad_one(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    first = demo_run(root)
    second = first.with_name(first.name + "-copy")
    shutil.copytree(first, second)
    (first / "transcript.jsonl").write_text("not json\n", encoding="utf-8")

    code = cli.main(["judge", str(root), "--rubric", "question_quality", "--provider", "fake"])

    assert code == 2
    assert len(load_judgments(second)) == 1
    assert not (first / "judges.jsonl").exists()


def _set_delivery(run: Path, change: Callable[[dict[str, Any]], bool]) -> str:
    """Edit the first counterpart turn's delivery events and return its text."""
    stored = load_run(run).trajectory.events
    spoken = next(e for e in stored if e.source == "counterpart" and e.type == "spoke")
    path = run / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    kept = []
    for row in rows:
        matches = row["type"] == "segment_delivered" and row["data"]["start_tick"] == spoken.tick
        if not matches or change(row["data"]):
            kept.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
    return str(spoken.data["text"])


def test_turn_still_playing_at_the_end_is_withheld(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    text = _set_delivery(run, lambda data: False)

    outcome = judge(run, config())

    assert outcome.record["transcript"]["turns"][0]["cut_off"] is True
    assert text not in (run / "judges.jsonl").read_text(encoding="utf-8")
    assert text not in (run / "judge_cache.jsonl").read_text(encoding="utf-8")


def test_finished_turn_is_matched_by_its_start(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")

    def lengthen(data: dict[str, Any]) -> bool:
        data["end_tick"] = data["start_tick"] + 5
        return True

    text = _set_delivery(run, lengthen)

    assert blinded_transcript(load_run(run)).turns[0].text == text


def test_interrupted_turn_is_matched_by_its_start(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")

    def interrupt(data: dict[str, Any]) -> bool:
        data["end_tick"] = data["start_tick"] + 5
        data["interrupted"] = True
        return True

    _set_delivery(run, interrupt)

    assert blinded_transcript(load_run(run)).turns[0].text == ""


def test_adapter_text_is_not_judged_in_an_audio_run(tmp_path: Path) -> None:
    run = load_run(demo_run(tmp_path / "runs"))
    audio_run = replace(run, manifest={**run.manifest, "audio": {"tracks": []}})

    assert blinded_transcript(audio_run).delivery_known is False


@pytest.mark.parametrize(
    ("reference", "hidden"),
    [("acme/secretmodel:v2", "secretmodel"), ("mock:secret-tune", "secret-tune")],
)
def test_parts_of_the_model_reference_are_hidden(
    tmp_path: Path, reference: str, hidden: str
) -> None:
    run = load_run(demo_run(tmp_path / "runs"))
    run.manifest["model"]["reference"] = reference

    assert redact(f"I am {hidden}.", identity_terms(run))[0] == "I am [REDACTED]."


def test_short_blind_terms_are_honored_and_common_words_are_not_added(tmp_path: Path) -> None:
    run = load_run(demo_run(tmp_path / "runs"))
    run.manifest["model"]["name"] = "go"
    terms = identity_terms(run, ("HF", "42"))

    assert redact("Let us go to HF and 42", terms) == (
        "Let us go to [REDACTED] and [REDACTED]",
        2,
    )


def test_quote_from_the_wrong_turn_is_rejected() -> None:
    row = reply()
    for item in row["items"]:
        item["evidence"] = [{"turn": 1, "quote": "Hello there"}]

    with pytest.raises(JudgeResponseError, match="not in turn 1"):
        validate_response(json.dumps(row), RUBRIC, TURNS)


def test_failed_refresh_keeps_the_stored_score_and_cache(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    first = judge(run, config())
    cache = (run / "judge_cache.jsonl").read_bytes()

    def outage(request: LLMRequest, call: int) -> str:
        raise OSError("outage")

    outcome = judge(run, config(outage, refresh=True, retries=1))

    assert outcome.record["status"] == "failed"
    assert "kept the earlier scored judgment" in outcome.record["errors"]
    assert load_judgments(run) == [first.record]
    assert (run / "judge_cache.jsonl").read_bytes() == cache


def test_cli_survives_an_unexpected_crash_in_one_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "runs"
    first = demo_run(root)
    second = first.with_name(first.name + "-copy")
    shutil.copytree(first, second)
    (first / "judges.jsonl").mkdir()

    code = cli.main(["judge", str(root), "--rubric", "question_quality", "--provider", "fake"])

    assert code == 2
    assert len(load_judgments(second)) == 1
    assert first.name in capsys.readouterr().err


def _set_reference(run: Path, reference: str) -> None:
    path = run / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model"]["reference"] = reference
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_plain_reference_parts_stay_in_speech(tmp_path: Path) -> None:
    run = load_run(demo_run(tmp_path / "runs"))
    run.manifest["model"]["reference"] = "acme/mock:good"

    assert redact("A good answer from acme", identity_terms(run))[1] == 0


def test_model_named_like_a_prompt_word_can_still_be_judged(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    _set_reference(run, "acme/json:v1")

    assert judge(run, config()).record["status"] == "scored"


def test_finished_text_turns_are_not_marked_cut_off(tmp_path: Path) -> None:
    turns = blinded_transcript(load_run(demo_run(tmp_path / "runs"))).turns

    assert not any(turn.cut_off for turn in turns)
    assert all(turn.text for turn in turns)


def _outage(request: LLMRequest, call: int) -> str:
    raise OSError("outage")


def test_failure_replaces_a_score_of_an_older_transcript(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    row = {"start_ms": 100, "text": "New words here.", "source": "whisper"}
    (run / "transcript.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    outcome = judge(run, config(_outage, retries=0))

    assert "kept the earlier scored judgment" not in outcome.record["errors"]
    assert [row["status"] for row in load_judgments(run)] == ["failed"]


def test_score_from_another_judge_model_is_not_kept_instead(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    provider = FakeProvider(_outage)

    outcome = judge(run, JudgeConfig(RUBRIC, "fake", provider, "other", retries=0, backoff_s=0))

    assert "kept the earlier scored judgment" not in outcome.record["errors"]
    assert sorted(row["status"] for row in load_judgments(run)) == ["failed", "scored"]


def test_earlier_failure_is_not_protected(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config(_outage, retries=0))

    outcome = judge(run, config(_outage, retries=0))

    assert "kept the earlier scored judgment" not in outcome.record["errors"]


def test_cli_warns_when_the_stored_suite_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = demo_run(tmp_path / "runs")
    path = run / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["suite"]["reference"] = "missing-suite"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    code = cli.main(["judge", str(run), "--rubric", "question_quality", "--provider", "fake"])

    assert code == 0
    assert "using the built-in rubric only" in capsys.readouterr().err


def test_cli_refuses_a_missing_explicit_suite(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    arguments = ["judge", str(run), "--rubric", "question_quality", "--provider", "fake"]

    assert cli.main([*arguments, "--suite", "missing-suite"]) == 2
    assert not (run / "judges.jsonl").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_cli_reports_an_unreadable_run_and_judges_the_rest(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    first = demo_run(root)
    second = first.with_name(first.name + "-copy")
    shutil.copytree(first, second)
    first.chmod(0)
    try:
        code = cli.main(["judge", str(root), "--rubric", "question_quality", "--provider", "fake"])
    finally:
        first.chmod(0o755)

    assert code == 2
    assert len(load_judgments(second)) == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
@pytest.mark.parametrize("mode", [0, 0o444])
def test_cli_reports_an_unreadable_named_path(tmp_path: Path, mode: int) -> None:
    run = demo_run(tmp_path / "runs")
    run.chmod(mode)
    try:
        code = cli.main(["judge", str(run), "--rubric", "question_quality", "--provider", "fake"])
    finally:
        run.chmod(0o755)

    assert code == 2


def test_rubric_item_text_is_checked_for_identity(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    leaking = {
        "id": RUBRIC.id,
        "version": "leak",
        "instructions": "Score the speaker.",
        "items": [{"id": "a", "prompt": "Is this mock?", "scale": {"min": 1, "max": 2}}],
    }
    provider = FakeProvider(fake_judge_response)

    outcome = judge(run, JudgeConfig(parse_rubric(leaking), "fake", provider, "fake-judge"))

    assert provider.calls == 0
    assert "identity term" in outcome.record["errors"][0]


@pytest.mark.parametrize(
    ("items", "params"),
    [
        ([{"nested": {"prompt": "mock"}}], {}),
        ([{"anchors": ["mock"]}], {}),
        ([], {"hint": "mock"}),
    ],
)
def test_nested_request_values_are_checked_for_identity(
    tmp_path: Path, items: list[dict[str, Any]], params: dict[str, Any]
) -> None:
    run = load_run(demo_run(tmp_path / "runs"))
    body = json.dumps({"items": items, "transcript": []})
    messages = ({"role": "system", "content": ""}, {"role": "user", "content": body})

    with pytest.raises(BlindingError, match="identity term"):
        assert_blinded(LLMRequest("j", messages, "p", params), ("mock",), run)


@pytest.mark.parametrize(
    ("reference", "spoken"),
    [
        ("acme/secretmodel/", "secretmodel"),
        ("acme/secretmodel:v2", "acme/secretmodel"),
        ("acme/secretmodel:v2", "v2"),
        ("acme/secretmodel@sha-1", "sha-1"),
        ("acme-2/model:v2", "acme-2"),
        ("acme/team-7/model:v2", "team-7"),
    ],
)
def test_reference_forms_are_hidden(tmp_path: Path, reference: str, spoken: str) -> None:
    run = load_run(demo_run(tmp_path / "runs"))
    run.manifest["model"]["reference"] = reference

    assert redact(f"I am {spoken} now", identity_terms(run))[0] == "I am [REDACTED] now"


def test_failed_pass_with_new_blind_term_keeps_the_score(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    first = judge(run, config())

    outcome = judge(run, config(_outage, retries=0, blind_terms=("hello",)))

    assert "kept the earlier scored judgment" in outcome.record["errors"]
    assert load_judgments(run) == [first.record]


def test_judge_refuses_a_special_file_without_hanging(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    os.mkfifo(run / "judges.jsonl")

    with pytest.raises(JudgeError, match="not a regular file"):
        load_judgments(run)


def test_failed_pass_keeps_a_score_written_before_fingerprints(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    path = run / "judges.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    del row["transcript"]["input_sha256"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cache = (run / "judge_cache.jsonl").read_bytes()

    outcome = judge(run, config(_outage, refresh=True, retries=0))

    assert "kept the earlier scored judgment" in outcome.record["errors"]
    assert load_judgments(run) == [row]
    assert (run / "judge_cache.jsonl").read_bytes() == cache


def test_failed_pass_replaces_an_old_score_of_other_words(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    judge(run, config())
    path = run / "judges.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    del row["transcript"]["input_sha256"]
    row["transcript"]["sha256"] = "other"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    judge(run, config(_outage, refresh=True, retries=0))

    assert [stored["status"] for stored in load_judgments(run)] == ["failed"]


FINGERPRINT_ROWS = [(0.0, 0, "B", "Hello.", False), (160.0, 1, "A", "Why?", False)]


@pytest.mark.parametrize(
    ("rows", "known"),
    [
        (FINGERPRINT_ROWS, False),
        ([(0.0, 0, "A", "Hello.", False), FINGERPRINT_ROWS[1]], True),
        ([(0.0, 0, "B", "Hello.", True), FINGERPRINT_ROWS[1]], True),
        ([*FINGERPRINT_ROWS, (320.0, 0, "B", "", True)], True),
        (list(reversed(FINGERPRINT_ROWS)), True),
        ([FINGERPRINT_ROWS[0], (160.0, 1, "A", "How?", False)], True),
    ],
)
def test_fingerprint_changes_with_what_a_judge_reads(rows: list[Any], known: bool) -> None:
    assert _fingerprint(rows, known) != _fingerprint(FINGERPRINT_ROWS, True)


@pytest.mark.parametrize(
    "rows",
    [
        [(5.0, 0, "B", "Hello.", False), (900.0, 1, "A", "Why?", False)],
        [(0.0, 0, "B", "  Hello. ", False), FINGERPRINT_ROWS[1]],
        [*FINGERPRINT_ROWS, (320.0, 0, "B", "  ", False)],
    ],
)
def test_fingerprint_ignores_what_a_judge_cannot_see(rows: list[Any]) -> None:
    assert _fingerprint(rows, True) == _fingerprint(FINGERPRINT_ROWS, True)


def test_stored_and_current_fingerprints_both_track_delivery(tmp_path: Path) -> None:
    run = demo_run(tmp_path / "runs")
    before = transcript_fingerprint(load_run(run))
    _add_event(run, source="model", type="truncated", data={"ms": 160})
    stored = load_run(run)

    assert transcript_fingerprint(stored) != before
    assert blinded_transcript(stored).input_sha256 == transcript_fingerprint(stored)
