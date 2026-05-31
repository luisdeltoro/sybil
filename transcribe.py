#!/usr/bin/env python3
"""
Transcribe speech from noisy video/audio files.

Uses ffmpeg for audio extraction, noisereduce for denoising,
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


def denoise_audio(input_wav: Path, output_wav: Path):
    """Apply spectral noise reduction to the audio."""
    import numpy as np
    import noisereduce as nr
    import soundfile as sf

    print("Applying noise reduction...")
    data, rate = sf.read(str(input_wav))

    # noisereduce uses the first portion of audio to estimate noise profile
    # stationary=True assumes consistent background noise (fans, hum, etc.)
    reduced = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=True,
        prop_decrease=0.75,  # how much to reduce noise (0=none, 1=full)
    )

    sf.write(str(output_wav), reduced, rate)
    print("  Noise reduction complete.")


def transcribe_audio(audio_path: Path, model_name: str, language: str) -> list[dict]:
    """Transcribe audio using Whisper and return segments."""
    import whisper

    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    print(f"Transcribing (language={language})...")
    result = model.transcribe(
        str(audio_path),
        language=language,
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
  uv run python transcribe.py audio.wav --model large --output transcript.txt
  uv run python transcribe.py video.mp4 --no-denoise --language en
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
        "--no-denoise",
        action="store_true",
        help="Skip noise reduction step",
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

        # Step 2: Noise reduction (optional)
        if args.no_denoise:
            transcribe_path = raw_wav
            print("Skipping noise reduction (--no-denoise).")
        else:
            denoise_audio(raw_wav, clean_wav)
            transcribe_path = clean_wav

        # Step 3: Transcribe
        segments = transcribe_audio(transcribe_path, args.model, args.language)

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
