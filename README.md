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

### Batch: a folder in, a folder out

```bash
uv run extract-speech --source ~/Downloads/Tapo --target ~/transcripts --profile noisy
```

The `--source` tree is searched recursively and its structure mirrored into
`--target`, with each file becoming `<name>_transcript.txt`:

```
Tapo/20260522/Living Room/1779486332469_0.mp4
  -> transcripts/20260522/Living Room/1779486332469_0_transcript.txt
```

Mirroring matters: the same filename can appear under several subdirectories, so
flattening would silently overwrite one transcript with another.

- Files whose transcript already exists are **skipped**, so an interrupted batch
  can simply be re-run. `--overwrite` forces a redo.
- Models are loaded **once** for the whole batch, not per file.
- A file that fails is reported and the batch **continues**; the summary lists
  every failure and the exit code is non-zero if there was any.
- Non-media files are ignored.

### Remote execution

Offload the work to a machine you have SSH access to — useful when a large model
or a long recording is too much for this laptop:

```bash
uv run extract-speech --source ~/Downloads/Tapo --target ~/transcripts \
  --run-in-remote my-box --whisper-model large
```

Works for a single file too. The remote runs **the same CLI on the same original
files**, so its output matches a local run; nothing is converted beforehand.

Phases: preflight (`uv` and `ffmpeg` present) → provision (`rsync` the project,
`uv sync`) → upload sources → run remotely → download transcripts.

- `--remote-dir` (default `~/.sybil`) is kept between runs, so re-provisioning
  takes seconds and a batch can resume.
- **Interrupted runs keep their work.** Transcripts are downloaded even when the
  remote run fails; re-running the same command resumes, because the remote
  target persists and finished files are skipped.
- Only `pyproject.toml`, `uv.lock`, `.python-version`, `src/` and `tests/` are
  uploaded — an allowlist, so `.env` can never be shipped by accident.
- The HuggingFace token is piped over the SSH channel into the remote process
  environment. It is never written to the remote disk, nor placed on a command
  line where the remote's `ps` would show it.
- Symlinked sources are followed (`rsync -L`), so a library of symlinks
  transfers the real files rather than dangling pointers.

Requirements on the remote: `uv`, `ffmpeg`, and SSH access. Python is supplied by
`uv` per `.python-version`, so the remote's own Python does not matter.

### Single file

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
| `input` | — | Path to a single video or audio file (omit when using `--source`) |
| `--source` | — | Directory to transcribe recursively; requires `--target` |
| `--target` | — | Output directory for batch mode; mirrors the `--source` tree |
| `--overwrite` | off | Re-transcribe files that already have a transcript |
| `--run-in-remote` | — | SSH host to run the transcription on |
| `--remote-dir` | `~/.sybil` | Working directory on the remote host |
| `--profile` | `clean` | Knob bundle: `clean` or `noisy` (see below) |
| `--whisper-model` (alias `--model`) | `medium` | `tiny`, `base`, `small`, `medium`, `large` |
| `--language` | `es` | Language code (`es`, `en`, `fr`, `de`, …) |
| `--diarize` / `--no-diarize` | `--diarize` | Tag speakers as `Persona N` (on by default) |
| `--speakers` | auto | Force an exact number of speakers |
| `--hf-token` | env `HF_TOKEN` | HuggingFace token for diarization |
| `--output` / `-o` | stdout | Save transcript to a file |
| `--denoise` | (profile) | Override denoise method: `loudnorm`, `spectral`, `ffmpeg`, `demucs` |
| `--demucs-model` | `htdemucs` | Demucs model for `--denoise demucs` |
| `--no-denoise` | (profile) | Override: skip denoising entirely |
| `--condition-on-previous` / `--no-condition-on-previous` | (profile) | Override Whisper's condition-on-previous-text |

## Profiles

A profile bundles the audio + Whisper decoding knobs that differ between clean
and noisy recordings. Individual flags override the bundle.

| Knob | `clean` (default) | `noisy` |
|------|-------------------|---------|
| Denoising | off | **`demucs`** (vocal isolation) |
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
  **Demucs vocal isolation**, an aggressive no-speech threshold, and beam search
  to recover hard-to-hear speech.

  Measured on real far-field recordings, Demucs removed 34–43% of non-vocal
  energy and largely eliminated Whisper's hallucinations. On one security-camera
  clip the transcript went from **14 segments to 1**: `loudnorm` produced six
  `...` segments, a repeated "Y escapillada" loop, and one line duplicated
  verbatim, while Demucs left only the single real utterance.

  It is not free: **~0.15× realtime and ~1.4 GB RAM** (about +8½ minutes on a
  56-minute recording). Use `--denoise loudnorm` for the old cheap behaviour, and
  note it is pointless on clean audio — on a clean phone call it removed 0.1% of
  energy. Pick the model with `--demucs-model` (default `htdemucs`; `htdemucs_ft`
  is a bag of four networks, better but ~4× slower).

## How it works

1. **Audio extraction** — `ffmpeg` extracts 16 kHz mono WAV.
2. **Preflight** — the extracted audio is verified readable, and the pyannote
   pipeline is loaded (validating your token and model licence) *before*
   transcription starts, so those failures surface in seconds rather than after
   a long transcription has already run.
3. **Denoising** — profile-dependent; applied only to the Whisper input.
   Diarization always uses the **raw** audio, since denoising can distort
   speaker voiceprints. `loudnorm`, `spectral` and `ffmpeg` filter the extracted
   16 kHz mono WAV; `demucs` instead re-extracts 44.1 kHz stereo from the
   original source (see below).
4. **Transcription** — Whisper produces word-level timestamps (when diarizing).
5. **Diarization + alignment** — the raw audio is read into memory and passed to
   pyannote as a waveform, not as a file path (see below). pyannote produces
   speaker turns; each word is assigned to the overlapping turn, and consecutive
   same-speaker words are merged into utterances. This splits speaker changes
   that occur *inside* a Whisper segment. Speakers are numbered by first
   appearance.

### Why Demucs re-extracts from the source

Demucs requires **44,100 Hz stereo** (`model.samplerate` / `model.audio_channels`),
but step 1 produces 16 kHz mono for Whisper. Reusing that file would hand Demucs
audio already stripped of everything above 8 kHz *and* of the inter-channel
differences separation models depend on. So `denoise_demucs` runs its own
extraction from `AudioInputs.source`.

A single ffmpeg invocation covers both cases: it upsamples and duplicates a
16 kHz mono source (no loss — there was nothing above 8 kHz to keep), and
properly downsamples a 48 kHz stereo one while keeping the channels distinct.
The extra decode costs ~3 s even on a 6 GB, 56-minute video, since `-vn` skips
the video stream entirely.

### Why audio is passed to pyannote in memory

pyannote accepts either a file path or a `{"waveform": ..., "sample_rate": ...}`
mapping. Given a path, it decodes the file itself using **torchcodec**, which
links FFmpeg's C libraries by exact version: torchcodec 0.13 requires
`libavutil.56`–`60` (FFmpeg 4–8), so an upgrade to FFmpeg 9 — which ships
`libavutil.61` — breaks diarization with `torchcodec is not available`.

Since `ffmpeg` has already produced plain PCM in step 1, this tool reads that
WAV with `soundfile` and hands pyannote the waveform directly. That keeps
diarization working regardless of the installed FFmpeg version, and avoids
decoding the same audio twice.

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
- The raw audio is held in memory for diarization: roughly **3.8 MB per minute**
  at 16 kHz mono float32, so about 215 MB for a 56-minute recording.
- Very large source files (multi-GB video) work, but audio extraction and
  transcription dominate runtime; consider `--whisper-model small` first to
  validate output, then re-run with `medium`/`large` if desired.

## Troubleshooting

- **"Diarization requires a HuggingFace token"** — set `HF_TOKEN` / `--hf-token`,
  or use `--no-diarize`.
- **"Failed to load pyannote pipeline"** — ensure you accepted the model
  conditions (see Setup) and the token has read access.
- **"torchcodec is not available"** — this tool avoids torchcodec for
  diarization, so you should not hit it there. If another library raises it
  (for example anything going through `torchaudio.load`), your FFmpeg major
  version is newer than torchcodec supports. Check with
  `ffmpeg -version` and compare against
  `ls .venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core*.dylib`
  (one shim per supported FFmpeg major). Installing a supported FFmpeg
  alongside, or upgrading torchcodec, are the options.
- **Auto-detect finds too many speakers** — short backchannels ("sí", "vale")
  can be split off as extra speakers. Pass `--speakers N` to fix the count.
- **Repeated words / hallucinations** — use `--profile clean` (greedy decoding).
  On genuinely noisy audio try `--profile noisy` and a larger model.
- **Out of memory** — use a smaller `--whisper-model` (`small`/`tiny`).
