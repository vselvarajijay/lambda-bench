# 0001: Suite interface

**Status:** Approved with required changes, which are applied. **Date:** 2026-11-18.

An independent read-only review examined slice 1.3 before merge. The `architect` agent does not exist until slice 1.7, so that review stands in for its sign-off.

## Decisions

- `Scenario` separates model-safe `data` from simulator-only `hidden`. Model-context code receives `hidden={}`, and validation rejects hidden text found in the context, with tools on and off.
- `decide_pass` is the only way to reach a pass decision. Failed runs and runs ended by the tick limit never pass.
- A suite ends a run by finishing the conversation, including when its time budget expires.
- Delivery credit comes from `DeliveredSpan.tags`, which are empty unless the tagged segment played in full. `Event.origin` separates `policy` events from `delivery` events.
- Event data is copied to JSON form when recorded, so live scoring and rescoring read the same values.
- Policies receive a deep copy of the scenario.
- `PolicyOutput.preempt` lets scripted moments replace queued speech.

## Rules for later changes

- New suite hooks must be optional `runtime_checkable` protocols, such as `MockScripted`, so existing suites keep loading.
- New fields on core-built types such as `ConversationView`, `RunContext` and `Trajectory` need defaults or must be set only by the core.
- Tool execution must not sit on `ModelContext`, because that object reaches the adapter.
- Suite schemas should avoid `additionalProperties: false`, which rejects core fields. Use `unevaluatedProperties` when strictness is needed.
