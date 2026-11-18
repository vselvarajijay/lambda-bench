# Audio fixtures

`speech_labeled.flac` is built by `scripts/make_vad_fixture.py` from `jfk.flac` in the faster-whisper test data, an excerpt of President John F. Kennedy's 1961 inaugural address. Speech chunks are spliced between low-level noise, so `speech_labeled.json` gives exact boundaries. The clip tests timing, not naturalness.

`../speech.wav` is the first four seconds of that clip at 24 kHz, used to preview audio effects.
