# Judges

Judges score what was said against a rubric. Their scores are opinions from a language model, so they stay out of `result.json` and never affect pass or fail.

```bash
uv run lambda-duplex judge runs/fixture --rubric question_quality --provider fake
```

`--provider fake` returns placeholder scores for testing the pipeline. They carry no meaning. It is the only provider today.

## Output

Each run directory gains two files:

| File | Contents |
| --- | --- |
| `judges.jsonl` | One record per rubric, judge model and seed. Every record has `"kind": "judge"`, `"deterministic": false` and `"calibration": "uncalibrated"`. |
| `judge_cache.jsonl` | Every judge reply, keyed by the exact request. |

A record's `status` is one of:

- `scored`: every item passed validation.
- `failed`: retries ran out or blinding failed. No scores are stored, and the command exits with 1.
- `unscorable`: the run did not complete, the model said nothing, or the only model text comes from the adapter in an audio run or after a truncation. No judge call is made. Run `transcribe` with a speech recognizer to judge such a run.

## Rubrics

`--rubric` takes an id or a YAML file. An id is looked up in the suite's `rubrics/` folder, then among the built-in rubrics.

```yaml
id: question_quality
version: "0.1"
instructions: Score how well speaker A asks questions.
items:
  - id: open_questions
    prompt: Speaker A asks open questions.
    scale: {min: 1, max: 5}
    anchors:
      - {score: 1, text: Almost all yes-or-no.}
    allow_na: true    # the judge may answer null
    min_evidence: 1   # quotes required for a score
```

## Evidence

Every score needs quotes from numbered turns, including one from the evaluated speaker. A quote must match consecutive whole words in that turn, ignoring case and punctuation. A reply that fails any check is sent back with the reason, up to `--retries` times.

## Blinding

The judge sees only the rubric and a transcript with speakers `A` and `B`. It does not receive the model identity, run path, seed, scenario data, metrics or pass result.

- The model's name, full reference, the model part of that reference, and session ids are replaced with `[REDACTED]` wherever they are spoken. So are its version, string settings and other reference parts when they look like identifiers, such as `tuned-v3`. Plain words such as `main` or an organization name are kept, because removing them would damage ordinary speech.
- Add other names with `--blind-term`, for example the vendor. Redaction cannot catch identity expressed any other way, such as a distinctive phrase.
- Rubric items and runs are shuffled using `--seed`.
- Each request's transcript and rubric text are checked before sending. If an identity term remains, no call is made and the record is `failed`.

## What the judge reads

Model speech comes from the first available source: a transcript of the model audio, live transcription events, then adapter text. A counterpart turn that did not finish playing, because it was interrupted or the run ended, appears with `cut_off` and no words, since the run does not record which words were played. This keeps speech nobody heard away from the judge, at the cost of the part that was heard.

## Cache

A repeated request uses the stored reply, so a second pass makes no new provider call. Its record can differ if the first pass recorded a temporary provider error. Changing the transcript, rubric or judge model changes the request. A new seed changes it only when it produces a different item order. The provider is part of the request key. Replies from a pass that ends `failed` are dropped, so the next pass asks again. `--refresh` also asks the provider again. A failed pass never replaces a stored score for the same transcript: the command reports the failure and keeps the score and its cache.

## Reports

`lambda-duplex report <runs> --format html` adds judge scores in their own section, labeled non-deterministic and uncalibrated. Pass rates and pass^k use only stored pass decisions, so judge files cannot change them. Scored, failed, unscorable, malformed and missing judgments are counted separately, and a run with an unreadable `judges.jsonl` is listed by name.

A suite composite appears only when the suite is available at the stored version and declares weights for parts that have values. Each part is shown beside it. Otherwise the report says why no composite is shown.
