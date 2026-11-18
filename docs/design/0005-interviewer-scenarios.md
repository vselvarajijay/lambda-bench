# 0005: Interviewer scenario rules

**Status:** Revised after independent scenario and validator review. **Date:** 2026-11-18.

## Unlock meaning

- Conditions classified inside one `all` must hold for the same model utterance.
- `asked_about(topic)` means the utterance asks about that brief topic.
- `followed_up_on(fact)` means the utterance refers to that disclosed fact's specific content, or to a line from a scripted moment triggered by that fact. A generic prompt such as "tell me more" is false.
- `silence_after_trailoff(fact, min_s)` is a rule predicate. When every other condition in an unlock clause containing it is met, the persona's reply ends with a trail-off segment recorded in the event log and discloses no locked fact. The timer starts when that segment finishes playing. A substantive onset, as defined in `docs/metrics.md`, cancels it; a backchannel does not. At `min_s`, the condition holds and the persona continues on that tick without another model utterance. A scripted moment pause does not count.
- `acknowledged(event)` means a substantive model turn acknowledges that fact or scripted moment. A backchannel is not enough.
- Event references are `{disclosed: fact}` or `{moment: id}`.
- A negated timing condition on an event, such as `not elapsed_since`, is false until that event has happened.
- Silence stops at the start of substantive model speech. While audio is being classified, timers hold from the detected speech start. A backchannel releases the hold.
- Scripted pause time is added to running silence timers when the pause begins.
- Engine times are integer milliseconds. Boundary times count as elapsed, and an interruption at the same instant as an event counts as after it.

## Tiers

- Surface facts have no fact dependency.
- Every usable path to a probe fact needs a specific follow-up.
- Every usable path to a deep fact builds on a probe or deep fact and needs `followed_up_on` or `acknowledged`.
- A usable path can contain at most one silence condition.
- Every usable path needs at least one positive condition. Facts with no possible path, and authored `any` branches that contradict themselves, are rejected. Moment dependencies count when applying these tier rules.

## Scripted moments

- A mid-thought pause must be a hard moment. A substantive onset through the pause grace period fails it. A backchannel does not.
- For contradictions, `first` plays immediately after the triggering fact's span in the same reply, and `second` plays inside the trigger's elapsed-time window, measured from the end of `first`. Scenario-declared contradiction lines are exempt from simulator drift checks.
- `near` means the window ends before 90 seconds. `far` means it begins after 180 seconds. A moment whose actual delivery misses that window is void and is excluded from MomentScore and MomentReach.
- Scripted moments never overlap. A moment that comes due during another moment's delivery or scoring window waits until that window ends.
- `probed_contradiction` passes only when, within `within_turns` model turns after `second` finishes, the model refers to the conflict between the two statements and neutrally asks the persona to reconcile them, without asserting which is true.
- A scripted line is a locked-fact leak only if it states a locked fact's content. Slice 3.6 fixtures must confirm that the `m1` line in `ad_feedback_0007` is not an `f3` leak.
- A contradiction cannot fire between a deep disclosure and the end of its emotional moment's scoring window.
- `acknowledged_before_moving_on` passes only when the model acknowledges the disclosure and does not press for further medical detail.
- A topic marked `key` is one where a leading question is a hard violation.

## Reference interviewers

`tell_me_more` may ask each brief topic once, then uses only generic prompts. The worked scenario must not disclose its deep fact to that strategy.
