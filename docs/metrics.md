# Conversation metrics

All times are milliseconds on the conversation clock. Counterpart turns come from the simulator plan. Model speech comes from Silero VAD on the isolated model track.

## Shared definitions

- **Model turn:** model speech segments separated by less than 500 ms.
- **Backchannel:** a model turn with less than 1000 ms of speech. When word counts exist, it must also have at most 3 words.
- **Substantive onset:** the start of a model turn that is not a backchannel.
- **Eligible turn:** a counterpart turn that expects a response.
- **Response window:** from the start of an eligible turn up to, but not including, the next counterpart turn. For the last turn, it runs through the end of the conversation.
- **Supporting count:** a value with fewer than 10 supporting events is marked `sufficient: false`.

## Timing and balance

| Metric | Definition | Alignment |
|---|---|---|
| `response_latency_*_ms` | For answered turns whose first substantive onset is at or after the turn end, onset minus turn end. Reports mean, p50 and p90, using linear interpolation for percentiles. | τ-voice `L_R`; Full-Duplex-Bench smooth turn-taking latency, except early starts are not clamped to zero |
| `response_rate` | Eligible turns with a substantive onset in their response window, divided by eligible turns. | τ-voice `R_R` |
| `early_start_rate` | Eligible turns whose first substantive onset comes before the turn ends, divided by eligible turns. | Project definition |
| `talk_time_ratio` | Model speech divided by model plus counterpart speech. Counterpart pauses are excluded, and backchannels count as model speech. | Project definition |
| `talk_time_in_target` | Whether the ratio is inside the suite's inclusive target range. | Project definition |
| `dead_air_ratio` | Time with neither side speaking, outside scripted pauses, divided by conversation time outside scripted pauses. | Project definition |
| `goal_reached` | Whether the suite's goal event happened. | Project definition |
| `time_to_goal_ms` | Goal time, or conversation duration if the goal was not reached. | Project definition |
| `model_turns_to_goal` | Substantive model turns starting at or before that time. | Project definition |

## Overlap behavior

A **counterpart event** is a backchannel, vocal tic, or non-directed remark. It counts as counterpart speech for balance and dead air, but it is not a conversational turn.

| Metric | Definition | Alignment |
|---|---|---|
| `interruption_benign_*` | Substantive onsets during the last 300 ms of a counterpart turn, plus model backchannels overlapping counterpart speech. | Project definition |
| `interruption_pause_cut_in_*` | Substantive onsets inside a scripted pause, or inside another mid-turn silence of at least 600 ms. | Full-Duplex-Bench pause handling, as takeover |
| `interruption_mid_sentence_*` | Other substantive onsets inside a counterpart turn, including inside silences shorter than 600 ms. | τ-voice `I_A`, subdivided |
| `*_rate`, `*_per_minute` | Count divided by eligible turns, or by minutes of counterpart turn speech. | τ-voice `I_A` |
| `yield_latency_mean_ms` | When a counterpart turn starts during a substantive model turn, model turn end minus that start, averaged over yields within 2000 ms. | τ-voice `L_Y`; Full-Duplex-Bench v1.5 stop latency |
| `yield_rate` | Yields within 2000 ms divided by those interruptions. Exactly 2000 ms counts as a yield. | τ-voice `R_Y` |
| `resume_rate` | Yielded interruptions followed by a new substantive onset after the yield and no later than 5000 ms after the interrupting turn ends. | Project definition |
| `pause_tolerance` | Scripted pauses with no substantive onset from the pause start through 200 ms after its end, divided by scripted pauses. | Full-Duplex-Bench take-over rate, inverted |
| `backchannel_per_minute` | Model backchannels per minute of counterpart turn speech. | Full-Duplex-Bench backchannel task |
| `backchannel_well_placed_rate` | Backchannels starting inside a counterpart pause, or within 300 ms of a pause edge or turn end, divided by backchannels. | Project definition |
| `selectivity_backchannel` | Counterpart backchannels during a substantive model turn where the model keeps speaking for more than 1000 ms after the event starts. | τ-voice `S_BC` |
| `selectivity_vocal_tic`, `selectivity_non_directed` | While the model speaks, the same rule as backchannels. While it is silent outside any counterpart turn, the event is handled correctly if no substantive onset occurs from its start through 2000 ms after its end. | τ-voice `S_VT`, `S_ND` |

Interruption rates and per-minute values use eligible turns as their supporting count. Onsets inside ineligible turns are still classified, and ineligible turns still count as counterpart speech. `resume_rate` uses yielded interruptions as its denominator. Speech that overlaps is counted once in balance and dead-air calculations. Time inside scripted pauses is excluded from dead air, even when a counterpart event or model speech occurs there. A counterpart backchannel while the model is silent is not scored by `selectivity_backchannel`.

Selectivity follows τ-voice and cannot tell a reaction from a turn that happened to end within the window. Natural endings can therefore lower these scores, so compare them across models on the same scenarios and do not read them as absolute reaction rates.

Counterpart segments that were cut off, and the simulator's injected interruption line, are not eligible turns because they do not ask the model to answer.

Whether the model recovers the thread of the conversation is a judge item and is not measured here.

## Edge cases

- `talk_time_in_target` is null when nobody spoke, because the ratio is undefined.
- If only some segments in a merged model turn have word counts, the backchannel rule uses duration only.
- A substantive model onset during a scripted pause counts as an early start for response timing. Pause tolerance reports the same event as a pause failure.
- Supporting counts are eligible turns for rates, answered non-early turns for latency, model plus counterpart turns for talk balance, and one conversation for dead air and goal metrics. Per-conversation goal metrics are always marked insufficient on their own and become meaningful when aggregated.
