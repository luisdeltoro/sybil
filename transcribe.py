#!/usr/bin/env python3
"""
Transcribe speech from noisy video/audio files.

Uses ffmpeg for audio extraction, configurable noise reduction,
and OpenAI Whisper for speech-to-text transcription.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_ffmpeg():
    """Verify ffmpeg is available on the system."""
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg is not installed.", file=sys.stderr)
        print("Install it with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)


def extract_audio(input_path: Path, output_wav: Path):
    """Extract audio from video/audio file as 16kHz mono WAV."""
    print(f"Extracting audio from: {input_path.name}")
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ar", "16000",         # 16kHz sample rate (Whisper native)
        "-ac", "1",             # mono
        "-y",                   # overwrite
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error extracting audio:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  Audio extracted: {output_wav.name}")


def denoise_loudnorm(input_wav: Path, output_wav: Path):
    """Apply EBU R128 loudness normalization via ffmpeg.

    This approach boosts quiet speech to a standard broadcast level without
    aggressive spectral processing that might remove speech along with noise.
    Works best for audio where voices are faint relative to background noise.
    """
    print("Applying loudness normalization (EBU R128)...")
    cmd = [
        "ffmpeg",
        "-i", str(input_wav),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during loudnorm:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("  Loudness normalization complete.")


def denoise_spectral(input_wav: Path, output_wav: Path):
    """Apply spectral noise reduction using noisereduce.

    Uses spectral gating to suppress stationary background noise (fans, hum,
    constant hiss). Can be aggressive and may remove speech in very noisy audio.
    """
    import numpy as np
    import noisereduce as nr
    import soundfile as sf

    print("Applying spectral noise reduction...")
    data, rate = sf.read(str(input_wav))

    reduced = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=True,
        prop_decrease=0.75,
    )

    sf.write(str(output_wav), reduced, rate)
    print("  Spectral noise reduction complete.")


def denoise_ffmpeg_filters(input_wav: Path, output_wav: Path):
    """Apply ffmpeg audio filters: bandpass + FFT denoise + dynamic normalization.

    Combines a speech-focused frequency bandpass (300-3500 Hz), FFT-based noise
    reduction, and dynamic normalization. Good for isolating speech from broadband
    noise while boosting quiet passages.
    """
    print("Applying ffmpeg filters (bandpass + afftdn + dynaudnorm)...")
    filters = ",".join([
        "highpass=f=300",
        "lowpass=f=3500",
        "afftdn=nf=-25:nr=20:nt=w",
        "dynaudnorm=f=150:g=15",
    ])
    cmd = [
        "ffmpeg",
        "-i", str(input_wav),
        "-af", filters,
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during ffmpeg filters:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("  FFmpeg filter processing complete.")


DENOISE_METHODS = {
    "loudnorm": denoise_loudnorm,
    "spectral": denoise_spectral,
    "ffmpeg": denoise_ffmpeg_filters,
}


def transcribe_audio(
    audio_path: Path, model_name: str, language: str, condition_on_previous: bool
) -> list[dict]:
    """Transcribe audio using Whisper and return segments."""
    import whisper

    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    print(f"Transcribing (language={language}, condition_on_previous={condition_on_previous})...")
    result = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.4,
        condition_on_previous_text=condition_on_previous,
        verbose=False,
    )

    return result["segments"]


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_output(segments: list[dict]) -> str:
    """Format transcription segments with timestamps."""
    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"[{start} --> {end}] {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe speech from noisy video/audio files using Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  uv run python transcribe.py video.mp4
  uv run python transcribe.py video.mp4 --model large
  uv run python transcribe.py video.mp4 --denoise spectral
  uv run python transcribe.py video.mp4 --denoise ffmpeg --output transcript.txt
  uv run python transcribe.py video.mp4 --no-denoise --language en

Denoise methods:
  loudnorm  - EBU R128 loudness normalization (default). Boosts quiet speech
              without aggressive filtering. Best for faint voices in noise.
  spectral  - Spectral gating via noisereduce. Suppresses stationary noise
              (fans, hum). Can be too aggressive on very noisy audio.
  ffmpeg    - Bandpass (300-3500Hz) + FFT denoise + dynamic normalization.
              Isolates speech frequencies and boosts quiet passages.
        """,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the video or audio file to transcribe",
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Language code for transcription (default: es)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save transcript to file (default: print to stdout)",
    )
    parser.add_argument(
        "--denoise",
        default="loudnorm",
        choices=list(DENOISE_METHODS.keys()),
        help="Noise reduction method (default: loudnorm)",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip noise reduction entirely",
    )
    parser.add_argument(
        "--condition-on-previous",
        action="store_true",
        help="Condition each segment on previously decoded text. Improves "
        "coherence on clean audio but can cause hallucination cascades on "
        "noisy audio (default: off)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    check_ffmpeg()

    # Work in a temp directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        raw_wav = tmp / "raw_audio.wav"
        clean_wav = tmp / "clean_audio.wav"

        # Step 1: Extract audio
        extract_audio(args.input, raw_wav)

        # Step 2: Noise reduction (configurable)
        if args.no_denoise:
            transcribe_path = raw_wav
            print("Skipping noise reduction (--no-denoise).")
        else:
            denoise_fn = DENOISE_METHODS[args.denoise]
            denoise_fn(raw_wav, clean_wav)
            transcribe_path = clean_wav

        # Step 3: Transcribe
        segments = transcribe_audio(
            transcribe_path, args.model, args.language, args.condition_on_previous
        )

    # Step 4: Format and output
    if not segments:
        print("No speech detected in the audio.")
        sys.exit(0)

    transcript = format_output(segments)

    if args.output:
        args.output.write_text(transcript, encoding="utf-8")
        print(f"\nTranscript saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("TRANSCRIPT")
        print("=" * 60)
        print(transcript)
        print("=" * 60)


if __name__ == "__main__":
    main()
