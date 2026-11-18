# Model adapters

An adapter connects one model to the benchmark. It implements `ModelAdapter` in `src/lambda_duplex/adapters/base.py`:

- `describe()` returns the exact model name, version, and settings.
- `start()` opens a session with model-visible context.
- `step()` receives one tick of degraded counterpart audio and returns exactly one tick of model audio.
- `stop()` closes the session and reports how many steps ran.

Audio is 16-bit mono PCM at 24 kHz. The default tick is 160 ms, or 3,840 samples. Declare capabilities truthfully, because they affect how results are interpreted.

Run the conformance check before using an adapter:

```bash
uv run lambda-duplex adapter-check mock
```

The check covers session lifecycle, frame timing and size, silence, repeated speech during model output, restart, and declared capabilities. A pass shows that the adapter follows the protocol. It does not measure conversation quality.
