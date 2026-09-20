# AGENTS.md

Instructions for AI agents working on **extract-speech** (repo: `sybil`).

---

## What this is

A CLI that transcribes and optionally diarizes speech from video/audio files.
It runs Whisper **in-process** (not via a server, not via a cloud API) and uses
pyannote for speaker attribution.

Pipeline: `ffmpeg` extract → optional denoise → Whisper transcribe → pyannote
diarize → align words to speaker turns → format.

> **Naming:** the GitHub repo and directory are `sybil`; the distribution is
> `extract-speech`; the import package is `extract_speech`. These deliberately
> differ — `sybil` is an unrelated, actively maintained package on PyPI, so the
> import package must not use that name.

---

## Commands

All commands run from the repo root.

| Task | Command |
|---|---|
| Install | `make install` |
| Run tests | `make test` |
| Format | `make format` |
| Lint | `make lint` |
| Type-check | `make type-check` |
| CI-style gate (no rewriting) | `make check` |
| Everything | `make all` |
| Run the tool | `uv run extract-speech <file> [options]` |

Never invoke `pytest`, `ruff` or `pyright` directly — go through `make` (or
`uv run`) so the correct environment is used.

---

## Conventions

- **Package manager:** `uv`. Dependencies live in `pyproject.toml`; dev tools in
  `[dependency-groups] dev`. Never edit `uv.lock` by hand.
- **Python:** `>=3.12` (see `.python-version`). Do not use syntax newer than that.
- **Layout:** `src/` layout. Code in `src/extract_speech/`, tests in `tests/`.
- **Formatter/linter:** ruff, line length 120.
- **Type checker:** pyright, `standard` mode.
- **Types:** annotate everything. The module uses `from __future__ import
  annotations`, so `X | None` is fine throughout.
- **Structured data:** dataclasses, not dicts. See `TranscribeConfig`,
  `ConfigOverrides`, `Utterance`, `SpeakerTurn`.
- **Errors:** raise `TranscriptionError` for external-step failures (ffmpeg,
  diarization). Do not swallow exceptions.
- **Heavy imports** (`whisper`, `torch`, `pyannote`) are imported *inside* the
  functions that need them, not at module top level. This keeps `--help` and the
  test suite fast. Preserve this pattern.

---

## Testing

Tests exercise **pure logic only** — config resolution, speaker alignment,
formatting, `.env` handling. They never load Whisper or pyannote, never touch
audio, and never hit the network. Keep it that way: it is why the suite runs in
well under a second.

When adding a feature, put the decidable logic in a pure function and test that
directly, rather than reaching for mocks of the ML stack.

---

## Known gotchas (learned the hard way — do not regress these)

1. **Whisper hallucinates repetition loops on sparse/quiet audio.** A 56-minute
   phone call produced ~5 minutes of one phrase repeated hundreds of times. The
   fix, now in the `clean` profile: `condition_on_previous_text=False` (breaks
   the feedback loop) plus the temperature ladder and `logprob_threshold` so
   poisoned segments get retried instead of accepted.
2. **Beam search causes a *separate* repetition cascade** when
   `word_timestamps=True` on clean audio. This is why `clean` decodes greedily
   (`beam_size=None`). Do not "optimise" this by turning beam search on.
3. **Diarization must run on the raw audio, never the denoised audio.**
   Denoising distorts voiceprints and degrades speaker attribution. See
   `main()` — `run_diarization(pipeline, raw_wav, ...)` while Whisper gets
   `whisper_input`.
4. **Never hand pyannote a file path.** Pass a waveform mapping —
   `{"waveform": tensor, "sample_rate": rate}` — as `run_diarization` does.
   The path form makes pyannote decode the file with torchcodec, which links
   FFmpeg's C libraries by exact soname and breaks on every FFmpeg major
   upgrade (ffmpeg 9 ships `libavutil.61`; torchcodec 0.13 wants 56-60). The
   waveform form is a documented pyannote input and also avoids decoding the
   same audio twice.
5. **Load the diarization pipeline before transcribing.** Loading validates the
   token and model licence in seconds; transcription takes tens of minutes.
   `main()` deliberately calls `load_diarization_pipeline` up front.
6. **Whisper segment boundaries do not align with speaker changes.** Speaker
   attribution is done per *word*, then merged — see `group_words_by_speaker`.
   Attributing whole segments produces visibly wrong transcripts.
7. **Auto speaker detection over-splits.** Short backchannels ("sí", "vale") get
   assigned their own speaker. Pass `--speakers N` when the count is known.

---

## Diarization requires a token

`HF_TOKEN` must be set, via `.env` (gitignored) or the environment, and the
model conditions must be accepted at
<https://hf.co/pyannote/speaker-diarization-community-1>.
See `.env.example`. Without it, use `--no-diarize`.

Never commit `.env`. Never paste a token into a command line, a commit message,
or a log — it ends up in shell history and agent transcripts.

---

## Before reporting work as done

Run `make check` and paste the real output. Required:

- tests pass (report the count),
- ruff format check and lint clean,
- pyright clean, or each remaining diagnostic explained.

If you changed transcription behaviour, also smoke-test on a real file and
compare against a known-good transcript. "Tests pass" is not evidence that
transcription quality held up — the suite deliberately never runs Whisper.
