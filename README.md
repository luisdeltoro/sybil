# extract_speech

Transcribe speech from noisy video/audio files using OpenAI Whisper with noise reduction.

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** - Python package manager
- **ffmpeg** - for audio extraction from video files

```bash
# Install ffmpeg (macOS)
brew install ffmpeg

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage

No manual installation needed. `uv run` handles the isolated environment and dependencies automatically on first run.

```bash
cd extract_speech

# Basic usage (Spanish audio, medium model)
uv run python transcribe.py /path/to/video.mp4

# Use a larger model for better accuracy (slower, needs ~10GB RAM)
uv run python transcribe.py /path/to/video.mp4 --model large

# Save output to a file
uv run python transcribe.py /path/to/video.mp4 --output transcript.txt

# Skip noise reduction (if audio is already clean)
uv run python transcribe.py /path/to/video.mp4 --no-denoise

# Transcribe in a different language
uv run python transcribe.py /path/to/video.mp4 --language en
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | (required) | Path to video or audio file |
| `--model` | `medium` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `--language` | `es` | Language code (e.g., `es`, `en`, `fr`, `de`) |
| `--output` / `-o` | stdout | Save transcript to a text file |
| `--no-denoise` | off | Skip the noise reduction step |

## Model Recommendations

| Model | RAM/VRAM | Speed | Best For |
|-------|----------|-------|----------|
| `tiny` | ~1 GB | Very fast | Quick previews, clean audio |
| `small` | ~2 GB | Fast | Decent quality, low resources |
| `medium` | ~5 GB | Moderate | Good balance (default) |
| `large` | ~10 GB | Slow | Maximum accuracy, noisy audio |

On Apple Silicon Macs, Whisper uses MPS (Metal) acceleration automatically when available.

## Output Format

Transcript is output with timestamps per segment:

```
[00:00:01 --> 00:00:04] Hola, buenos dias.
[00:00:05 --> 00:00:08] Necesitamos revisar el documento.
```

## How It Works

1. **Audio extraction** - ffmpeg extracts audio from the input file as 16kHz mono WAV
2. **Noise reduction** - `noisereduce` applies spectral gating to suppress stationary background noise (fans, hum, traffic)
3. **Transcription** - OpenAI Whisper processes the cleaned audio and outputs timestamped text segments

## Troubleshooting

- **"ffmpeg not found"** - Install with `brew install ffmpeg`
- **Out of memory** - Use a smaller model: `--model small` or `--model tiny`
- **Poor accuracy** - Try `--model large`, or ensure `--language es` matches the spoken language
- **Noise reduction making it worse** - Try `--no-denoise` if the audio has non-stationary noise (e.g., music, multiple overlapping sounds)
