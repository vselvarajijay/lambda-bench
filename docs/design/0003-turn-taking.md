# 0003: Turn-taking

**Status:** Approved after independent review, with the last fixes applied from that review. **Date:** 2026-11-18.

- Free-form turn-taking applies only to conditions that request it. Strict conditions keep scripted behavior.
- Model speech is detected from audio with a level and variation gate. Silences shorter than the 500 ms model-turn gap do not end the model's turn.
- Interruptions and backchannels require model speech on the current tick. Overlap for yielding counts time through the last speech actually heard.
- After yielding, the simulator does not interrupt again until that model run ends.
- Interrupted segments and the injected interruption line are not eligible turns for response metrics.
- Random streams are derived from the run seed and scenario ID, so model comparisons stay paired and scenarios stay distinct.
