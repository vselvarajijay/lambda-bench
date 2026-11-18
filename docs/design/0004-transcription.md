# 0004: Transcription

**Status:** Approved after independent review. **Date:** 2026-11-18.

- With `--transcript whisper`, the counterpart reacts to speech recognized from model audio. Adapter text is ignored.
- A model utterance is transcribed after 500 ms without detected speech, after 15 seconds of continuous speech, or when the conversation ends. Each clip keeps about 320 ms of context on each side.
- Interrupt decisions that use an LLM receive a partial transcript of the current utterance.
- Very quiet speech can fall below the live speech gate and go untranscribed.
- Whisper transcription is limited to `lockstep` runs. `realtime` runs must use injected text until a non-blocking transcription path is measured.
- Replay uses the transcripts recorded in the run, so it does not need Whisper or repeat its output.
- `lambda-duplex transcribe` writes `transcript.jsonl`, using either Whisper on VAD segments or adapter text labeled `injected`.

## Local measurement

Measured on the DGX Spark CPU with `int8` models on the 11-second public-domain JFK clip, while GPU evaluation jobs were running:

| Model | Compute time / audio time | Word error rate |
|---|---:|---:|
| `tiny.en` | 0.065 | 0.000 |
| `base.en` | 0.102 | 0.091 |
| `small.en` | 0.292 | 0.000 |

`small.en` is the default for lockstep runs. These batch figures do not show that transcription can keep up in real-time mode, and one clean clip does not establish accuracy on noisy or synthetic model speech.
