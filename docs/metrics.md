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

## Edge cases

- `talk_time_in_target` is null when nobody spoke, because the ratio is undefined.
- If only some segments in a merged model turn have word counts, the backchannel rule uses duration only.
- A substantive model onset during a scripted pause counts as an early start for response timing. Pause tolerance reports the same event as a pause failure.
- Supporting counts are eligible turns for rates, answered non-early turns for latency, model plus counterpart turns for talk balance, and one conversation for dead air and goal metrics. Per-conversation goal metrics are always marked insufficient on their own and become meaningful when aggregated.
