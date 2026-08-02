# extract_speech

Transcribe speech from noisy video/audio files using OpenAI Whisper with configurable noise reduction.

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

# Basic usage (Spanish audio, medium model, loudnorm denoising)
uv run python transcribe.py /path/to/video.mp4

# Use a larger model for better accuracy (slower, needs ~10GB RAM)
uv run python transcribe.py /path/to/video.mp4 --model large

# Choose a different noise reduction method
uv run python transcribe.py /path/to/video.mp4 --denoise spectral
uv run python transcribe.py /path/to/video.mp4 --denoise ffmpeg

# Skip noise reduction entirely (if audio is already clean)
uv run python transcribe.py /path/to/video.mp4 --no-denoise

# Save output to a file
uv run python transcribe.py /path/to/video.mp4 --output transcript.txt

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
| `--denoise` | `loudnorm` | Noise reduction method: `loudnorm`, `spectral`, `ffmpeg` |
| `--no-denoise` | off | Skip noise reduction entirely |

## Noise Reduction Methods

| Method | Best For | How It Works |
|--------|----------|--------------|
| `loudnorm` (default) | Faint voices buried in noise | EBU R128 loudness normalization. Boosts quiet speech to broadcast level without aggressive filtering that might remove speech. |
| `spectral` | Constant background noise (fans, hum, hiss) | Spectral gating via `noisereduce`. Estimates noise profile and subtracts it. Can be too aggressive on very noisy audio. |
| `ffmpeg` | Broadband noise with speech in mid frequencies | Bandpass filter (300-3500 Hz) + FFT denoise + dynamic normalization. Isolates speech frequencies and boosts quiet passages. |

### When to use which

- **Start with `loudnorm`** (default) - works best for most noisy recordings where speech is faint.
- **Try `spectral`** if there's a constant drone/hum you want removed (e.g., air conditioning, electrical hum).
- **Try `ffmpeg`** if there's lots of low-frequency rumble or high-frequency hiss and speech is in the middle.
- **Use `--no-denoise`** if the audio is already clean or if all methods make it worse.

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
2. **Noise reduction** - Configurable method processes the audio to improve speech clarity
3. **Transcription** - OpenAI Whisper processes the audio with tuned parameters (beam search, temperature fallback, hallucination prevention) and outputs timestamped text segments

## Troubleshooting

- **"ffmpeg not found"** - Install with `brew install ffmpeg`
- **Out of memory** - Use a smaller model: `--model small` or `--model tiny`
- **Poor accuracy** - Try `--model large`, or ensure `--language es` matches the spoken language
- **Hallucinations (repeated words like "Gracias", "Si, si")** - The audio SNR is too low. Try `--model large` and different `--denoise` methods, or accept that the audio may be too noisy for reliable transcription.
- **Noise reduction removing speech** - Try `--denoise loudnorm` (gentlest) or `--no-denoise`
