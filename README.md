# extract-speech

Transcribe **and diarize** speech from video/audio files, tagging each speaker
as `Persona 1`, `Persona 2`, … Uses `ffmpeg` for audio extraction, OpenAI
**Whisper** for transcription, and **pyannote.audio** for speaker diarization —
all running locally after the models are downloaded.

> The GitHub repo and directory are named `sybil`; the installed command and
> distribution are `extract-speech`, and the import package is `extract_speech`.
> The names differ deliberately: `sybil` is an unrelated, actively maintained
> package on PyPI.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **ffmpeg** — audio extraction (also required by pyannote's audio decoding)
- A **HuggingFace token** — required only for diarization (see below)

```bash
# macOS
brew install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
```

## Setup

```bash
make install     # creates .venv and installs runtime + dev dependencies
```

Run `make help` to see every available target.

### HuggingFace token (for diarization)

Diarization uses the gated `pyannote/speaker-diarization-community-1` model.
One-time setup:

1. Accept the model conditions (logged in to HuggingFace):
   - <https://hf.co/pyannote/speaker-diarization-community-1>
   - <https://hf.co/pyannote/segmentation-3.0>
2. Create a **Read** token at <https://hf.co/settings/tokens>.
3. Provide it in **any** of these ways (precedence: real env var > `.env` > `--hf-token`):

```bash
# Option A: export it
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Option B: put it in a .env file (auto-loaded, git-ignored)
cp .env.example .env   # then edit .env and paste your token

# Option C: pass it on the command line
uv run extract-speech video.mp4 --hf-token hf_xxxx
```

The `.env` file is listed in `.gitignore` and is never committed.

If you only want a plain transcript, use `--no-diarize` and no token is needed.

## Usage

```bash
# Clean phone call, diarized, Spanish — all defaults
uv run extract-speech conversation.mp4

# Force the number of speakers (recommended for clean 2-person calls;
# auto-detect can over-segment short backchannel utterances)
uv run extract-speech conversation.mp4 --speakers 2

# Noisy / far-field recording, force 3 speakers, save to file
uv run extract-speech meeting.mp4 --profile noisy --speakers 3 -o out.txt

# Transcription only (no speaker tags), larger model, English
uv run extract-speech talk.mp4 --no-diarize --whisper-model large --language en
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | (required) | Path to the video or audio file |
| `--profile` | `clean` | Knob bundle: `clean` or `noisy` (see below) |
| `--whisper-model` (alias `--model`) | `medium` | `tiny`, `base`, `small`, `medium`, `large` |
| `--language` | `es` | Language code (`es`, `en`, `fr`, `de`, …) |
| `--diarize` / `--no-diarize` | `--diarize` | Tag speakers as `Persona N` (on by default) |
| `--speakers` | auto | Force an exact number of speakers |
| `--hf-token` | env `HF_TOKEN` | HuggingFace token for diarization |
| `--output` / `-o` | stdout | Save transcript to a file |
| `--denoise` | (profile) | Override denoise method: `loudnorm`, `spectral`, `ffmpeg` |
| `--no-denoise` | (profile) | Override: skip denoising entirely |
| `--condition-on-previous` / `--no-condition-on-previous` | (profile) | Override Whisper's condition-on-previous-text |

## Profiles

A profile bundles the audio + Whisper decoding knobs that differ between clean
and noisy recordings. Individual flags override the bundle.

| Knob | `clean` (default) | `noisy` |
|------|-------------------|---------|
| Denoising | off | `loudnorm` |
| `temperature` | fallback ladder `0.0 … 1.0` | fallback ladder `0.0 … 1.0` |
| `no_speech_threshold` | `0.6` | `0.4` |
| `condition_on_previous_text` | off | off |
| `logprob_threshold` | `-1.0` | `-1.0` |
| `beam_size` / `best_of` | greedy (none) | `5` / `5` |

- **`clean`** (default) — close-mic / phone audio. No denoising and greedy
  decoding. It keeps Whisper's anti-hallucination guards on (temperature
  fallback ladder, `condition_on_previous_text=False`, log-prob and compression
  gates): on real recordings, a sparse/quiet opening (people connecting on a
  call) otherwise sends greedy decoding into a word-repetition loop that
  cascades and truncates the transcript. Beam search is *not* used here because
  it triggered a separate repetition cascade with word-level timestamps.
- **`noisy`** — faint / far-field voices in background noise. Same guards plus
  loudness normalization, an aggressive no-speech threshold, and beam search to
  recover hard-to-hear speech.

## How it works

1. **Audio extraction** — `ffmpeg` extracts 16 kHz mono WAV.
2. **Denoising** — profile-dependent; applied only to the Whisper input.
   Diarization always uses the **raw** audio, since denoising can distort
   speaker voiceprints.
3. **Transcription** — Whisper produces word-level timestamps (when diarizing).
4. **Diarization + alignment** — pyannote produces speaker turns; each word is
   assigned to the overlapping turn, and consecutive same-speaker words are
   merged into utterances. This splits speaker changes that occur *inside* a
   Whisper segment. Speakers are numbered by first appearance.

## Output format

Diarized:

```
[00:00:03] Persona 1: Sí, dígame.
[00:00:20] Persona 2: Sí, sí, envíamelas.
```

Plain (`--no-diarize`):

```
[00:00:01 --> 00:00:04] Hola, buenos días.
```

## Development

```bash
make test         # run tests (fast; no models/network/audio)
make lint         # ruff check
make format       # ruff format
make type-check   # pyright
make check        # lint + type-check + tests, without rewriting files
make all          # install + format + lint + type-check + test
```

Layout: the package lives in `src/extract_speech/`, tests in `tests/`.

The tests cover the pure logic — profile resolution and override precedence,
speaker alignment (including the "speaker change inside one Whisper segment"
case), and formatting. They do not load Whisper or pyannote.

## Notes on large / long files

- Diarization memory and time scale with duration. For long recordings prefer
  `--profile clean` (no denoise pass) and a smaller `--whisper-model` if needed.
- Very large source files (multi-GB video) work, but audio extraction and
  transcription dominate runtime; consider `--whisper-model small` first to
  validate output, then re-run with `medium`/`large` if desired.

## Troubleshooting

- **"Diarization requires a HuggingFace token"** — set `HF_TOKEN` / `--hf-token`,
  or use `--no-diarize`.
- **"Failed to load pyannote pipeline"** — ensure you accepted the model
  conditions (see Setup) and the token has read access.
- **Auto-detect finds too many speakers** — short backchannels ("sí", "vale")
  can be split off as extra speakers. Pass `--speakers N` to fix the count.
- **Repeated words / hallucinations** — use `--profile clean` (greedy decoding).
  On genuinely noisy audio try `--profile noisy` and a larger model.
- **Out of memory** — use a smaller `--whisper-model` (`small`/`tiny`).
