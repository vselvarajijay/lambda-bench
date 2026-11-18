# 0002: Audio timing

**Status:** Approved after independent review. **Date:** 2026-11-18.

- The orchestrator clock is authoritative. Each audio frame is exactly one tick at 24 kHz.
- Model audio produced on tick `n` is audible to the counterpart on tick `n+1`. `model.flac` stores it at `n`, and `mix.flac` stores it at `n+1`. Metrics must name which reference they use.
- In audio mode the adapter receives degraded PCM only. Counterpart text and clean audio are for the simulator and recorder.
- Counterpart credit is granted on the tick after the final audio frame reached the model. `DeliveredSpan.end_tick` is the last tick that played.
- A preempted span ends on the tick before its replacement, carries no credit tags, and its `text` may include unplayed words.
- Mock model text is released when its final sample plays. Slice 2.8 replaces text trust with transcription of the model audio.
- `replay --verify` checks each stored track against the manifest as well as regenerated audio.
- Overlap metrics must use the separate tracks and tick offsets, not `mix.flac`. A reply can start on the tick that the mix still carries the previous model audio.
- Truncated mock text is dropped whole. Slice 2.8 must cut transcripts to the audio that was actually heard.
