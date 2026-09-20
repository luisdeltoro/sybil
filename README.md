# extract-speech

Transcribe speech from video or audio files, optionally tagging who spoke.
Runs entirely on your machine: `ffmpeg` for audio, **Whisper** for transcription,
**pyannote** for speaker diarization, **Demucs** for noise separation.

> Names differ deliberately. Repo and directory: `sybil`. Command and
> distribution: `extract-speech`. Import package: `extract_speech`.
> (`sybil` is an unrelated, actively maintained package on PyPI.)

## Quick start

```bash
make install                                              # create .venv, install deps

uv run extract-speech call.mp4                            # one file, diarized, Spanish
uv run extract-speech --source ~/videos --target ~/txt    # a whole folder
uv run extract-speech --source ~/videos --target ~/txt \
  --run-in-remote my-box --whisper-model large            # offload to another host
```

`make help` lists every target.

## Prerequisites

| | |
|---|---|
| **Python 3.12+** | supplied by `uv`; your system Python is not used |
| **[uv](https://docs.astral.sh/uv/)** | package and environment manager |
| **ffmpeg** | audio extraction (used by this tool and by Whisper) |
| **HuggingFace token** | only for diarization — see below |

```bash
brew install ffmpeg                                # macOS
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is missing
```

## HuggingFace token

Needed only for diarization. Skip this entirely with `--no-diarize`.

1. Accept the model conditions while logged in to HuggingFace:
   - <https://hf.co/pyannote/speaker-diarization-community-1>
   - <https://hf.co/pyannote/segmentation-3.0>
2. Create a **Read** token at <https://hf.co/settings/tokens>.
3. Supply it — precedence: real env var → `.env` → `--hf-token`.

```bash
export HF_TOKEN=hf_xxxx                  # A: environment
cp .env.example .env                     # B: .env file (git-ignored, auto-loaded)
uv run extract-speech v.mp4 --hf-token hf_xxxx   # C: flag
```

---

# What it does

## Transcription

OpenAI Whisper, running locally. Model weights download once and are cached.

| Flag | Default | Notes |
|---|---|---|
| `--whisper-model` (alias `--model`) | `medium` | `tiny`, `base`, `small`, `medium`, `large` |
| `--language` | `es` | any Whisper language code (`en`, `fr`, `de`, …) |

Bigger models are slower and more accurate. Start with `small` to validate a
pipeline, then re-run with `medium`/`large`.

## Speaker diarization

Identifies *who* speaks when, labelling utterances `Persona 1`, `Persona 2`, …
numbered by first appearance. **On by default.**

| Flag | Default | Notes |
|---|---|---|
| `--diarize` / `--no-diarize` | on | `--no-diarize` gives a plain transcript and needs no token |
| `--speakers N` | auto-detect | force an exact speaker count |

Two things worth knowing:

- **Auto-detect over-splits.** Short backchannels ("sí", "vale") can be assigned
  their own speaker. If you know the count, pass `--speakers N`.
- **Attribution is per word, not per segment.** Whisper's segment boundaries do
  not align with speaker changes, so a speaker switch *inside* one segment is
  split correctly.

## Noise handling

Four methods. Choose with `--denoise <method>` (which also switches denoising
on), or `--no-denoise` to skip. The default comes from `--profile`.

| Method | What it does | Use when | Cost |
|---|---|---|---|
| `loudnorm` | EBU R128 loudness normalisation; lifts quiet speech to broadcast level | recording is clean but too quiet | negligible |
| `spectral` | Spectral gating (`noisereduce`); subtracts a stationary noise profile | constant hum, hiss, fan | low |
| `ffmpeg` | Bandpass 300–3500 Hz + FFT denoise + dynamic normalisation | telephone-band speech inside broadband noise | negligible |
| `demucs` | ML source separation; discards everything but the vocal stem | far-field or noisy rooms; suppresses hallucinated text | ~0.15× realtime, ~1.4 GB RAM |

`--demucs-model` selects the separation model (default `htdemucs`, one network;
`htdemucs_ft` is four networks — better, roughly 4× slower).

Denoising is applied **only to the transcription input**. Diarization always uses
the untouched audio, because denoising distorts speaker voiceprints.

## Profiles

A profile bundles the audio and Whisper decoding settings that differ between
clean and noisy recordings. Individual flags override it.

| Setting | `clean` (default) | `noisy` |
|---|---|---|
| Denoising | off | `demucs` |
| `temperature` | ladder `0.0 … 1.0` | ladder `0.0 … 1.0` |
| `no_speech_threshold` | `0.6` | `0.4` |
| `condition_on_previous_text` | off | off |
| `logprob_threshold` | `-1.0` | `-1.0` |
| `beam_size` / `best_of` | greedy | `5` / `5` |

- **`clean`** — close-mic and phone audio. Greedy decoding with Whisper's
  anti-hallucination guards on.
- **`noisy`** — faint or far-field voices in background noise. Adds Demucs
  separation, a lower no-speech threshold, and beam search. Considerably slower;
  use `--denoise loudnorm` for a cheap alternative. Pointless on clean audio.

---

# Processing modes

## Single file

```bash
uv run extract-speech call.mp4                                  # stdout
uv run extract-speech call.mp4 --speakers 2 -o out.txt          # to a file
uv run extract-speech talk.mp4 --no-diarize --language en
uv run extract-speech room.mp4 --profile noisy --whisper-model large
```

Without `--output`/`-o` the transcript is printed to stdout.

## Batch — folder in, folder out

```bash
uv run extract-speech --source ~/Downloads/Tapo --target ~/transcripts --profile noisy
```

`--source` is searched recursively; its structure is mirrored into `--target`,
each file becoming `<name>_transcript.txt`:

```
Tapo/20260522/Living Room/1779486332469_0.mp4
  -> transcripts/20260522/Living Room/1779486332469_0_transcript.txt
```

| Behaviour | |
|---|---|
| **Resumable** | files that already have a transcript are skipped; `--overwrite` forces a redo |
| **Fault tolerant** | a failing file is reported and the batch continues; the summary lists every failure |
| **Efficient** | models load once for the whole batch, not per file |
| **Selective** | non-media files are ignored |

Structure is mirrored rather than flattened because the same filename may appear
in several subdirectories — flattening would overwrite one transcript with
another.

## Remote execution

Offload the work to any host you can SSH into.

```bash
uv run extract-speech --source ~/videos --target ~/txt \
  --run-in-remote my-box --whisper-model large
```

Works for a single file too. The remote runs **the same CLI on the same original
files**, so output matches a local run — nothing is converted beforehand.

```
preflight → provision → upload → run → download
```

| Behaviour | |
|---|---|
| **Resumable** | transcripts are downloaded even if the remote run fails; re-run the same command to continue where it stopped |
| **Fast re-runs** | `--remote-dir` (default `~/.sybil`) persists, so re-provisioning takes seconds |
| **Secrets stay local** | only an allowlist is uploaded (`pyproject.toml`, `uv.lock`, `.python-version`, `src/`, `tests/`), so `.env` cannot be shipped by accident. The token is piped over SSH into the remote process environment — never written to remote disk, never on a command line |
| **Symlink safe** | sources are followed (`rsync -L`), so a symlinked media library transfers real files |

Remote requirements: `uv`, `ffmpeg`, SSH access. Python comes from `uv`.

---

# CLI reference

**Mode**: `both` · `batch` = requires `--source` · `single` = single-file only.

### Input / output

| Flag | Default | Mode | Description |
|---|---|---|---|
| `input` | — | single | Path to one video or audio file |
| `--source` | — | batch | Directory to transcribe recursively; requires `--target` |
| `--target` | — | batch | Output directory; mirrors the `--source` tree |
| `--output`, `-o` | stdout | single | Write the transcript to this file |
| `--overwrite` | off | batch | Re-transcribe files that already have a transcript |

### Transcription

| Flag | Default | Description |
|---|---|---|
| `--whisper-model`, `--model` | `medium` | `tiny`, `base`, `small`, `medium`, `large` |
| `--language` | `es` | Whisper language code |

### Diarization

| Flag | Default | Description |
|---|---|---|
| `--diarize` / `--no-diarize` | on | Tag speakers as `Persona N` |
| `--speakers` | auto | Force an exact speaker count |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token |

### Noise handling

| Flag | Default | Description |
|---|---|---|
| `--denoise` | from profile | `loudnorm`, `spectral`, `ffmpeg`, `demucs` — implies denoising on |
| `--no-denoise` | from profile | Skip denoising entirely |
| `--demucs-model` | `htdemucs` | Only used with `--denoise demucs` |

### Tuning

| Flag | Default | Description |
|---|---|---|
| `--profile` | `clean` | `clean` or `noisy` |
| `--condition-on-previous` / `--no-…` | from profile | Override Whisper's condition-on-previous-text |

### Remote

| Flag | Default | Description |
|---|---|---|
| `--run-in-remote` | — | SSH host (alias or `user@host`) to run on |
| `--remote-dir` | `~/.sybil` | Working directory on that host |

---

# Output

Diarized (default):

```
[00:00:03] Persona 1: Sí, dígame.
[00:00:20] Persona 2: Sí, sí, envíamelas.
```

Plain (`--no-diarize`):

```
[00:00:01 --> 00:00:04] Hola, buenos días.
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | success, including "no speech detected" |
| `1` | an error occurred, or **any** file in a batch failed |

---

# Troubleshooting

| Symptom | Fix |
|---|---|
| `Diarization requires a HuggingFace token` | Set `HF_TOKEN` / `--hf-token`, or use `--no-diarize` |
| `Failed to load pyannote pipeline` | Accept both model licences (see above) and check the token has read access |
| Too many speakers detected | Short backchannels get their own speaker — pass `--speakers N` |
| Repeated words, invented sentences | Use `--profile noisy` (Demucs separation) and a larger model. On clean audio keep `--profile clean` |
| Out of memory | Smaller `--whisper-model`; avoid `--denoise demucs` |
| `torchcodec is not available` | Not raised by this tool — diarization avoids torchcodec. If another library raises it, your ffmpeg major version is newer than torchcodec supports. Compare `ffmpeg -version` against `ls .venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core*` (one shim per supported ffmpeg major; `.dylib` on macOS, `.so` on Linux) |

### Long recordings

- Diarization holds the audio in memory: ~3.8 MB per minute, so ~215 MB for a
  56-minute recording.
- Demucs adds ~0.15× realtime and ~1.4 GB RAM. For long clean recordings prefer
  `--profile clean`.
- Multi-GB video is fine — `ffmpeg` reads only the audio stream.

---

# Development

```bash
make test         # tests (fast: no models, network or audio)
make lint         # ruff check
make format       # ruff format
make type-check   # pyright
make check        # lint + type-check + test, without rewriting files
make all          # install + format + lint + type-check + test
```

Package in `src/extract_speech/`, tests in `tests/`. Tests cover the pure logic —
profile resolution, speaker alignment, path mapping, remote command construction
— and never load Whisper, pyannote or Demucs.

See `AGENTS.md` for conventions and hard-won gotchas before changing the
pipeline.

---

# Design notes

**Why Demucs re-extracts from the source.** Demucs needs 44.1 kHz stereo, but the
pipeline extracts 16 kHz mono for Whisper. Reusing that file would hand Demucs
audio already stripped of everything above 8 kHz *and* of the inter-channel
differences separation relies on, degrading it silently. One ffmpeg call handles
both source shapes, and costs ~3 s even on a 6 GB video since only the audio
stream is read.

**Why pyannote receives a waveform, not a file path.** Given a path, pyannote
decodes the file with **torchcodec**, which links ffmpeg's C libraries by exact
version — torchcodec 0.13 needs `libavutil.56`–`60` (ffmpeg 4–8), so ffmpeg 9
(`libavutil.61`) breaks diarization outright. Since `ffmpeg` has already produced
plain PCM, the tool reads it with `soundfile` and passes the samples directly.
Diarization then works regardless of installed ffmpeg version, and the same audio
is not decoded twice.

**Why the `clean` profile decodes greedily.** Beam search with word-level
timestamps triggered a repetition cascade. Separately, a sparse or quiet opening
sends Whisper into a word-repetition loop that truncates the transcript, which is
why `condition_on_previous_text` is off and the temperature ladder plus log-prob
gate are on.

**What Demucs measurably buys.** On far-field recordings it removed 34–43% of
non-vocal energy and largely eliminated fabricated text: one security-camera clip
went from 14 segments — six bare `...`, a repeated nonsense loop, one line
duplicated verbatim — down to the single real utterance. It does **not** improve
word accuracy; it stops Whisper inventing content. On a clean phone call it
removes ~0.1% of energy and is not worth running.

**Fail-fast ordering.** Models load and audio is validated *before* transcription
starts, so a bad token, an unaccepted licence or an unreadable file surfaces in
seconds rather than after a long run.
