#!/usr/bin/env python3
"""
Transcribe and (optionally) diarize speech from video/audio files.

Pipeline:
  1. Extract audio with ffmpeg (16 kHz mono WAV).
  2. Optionally denoise, according to the selected profile.
  3. Transcribe with OpenAI Whisper (word-level timestamps when diarizing).
  4. Optionally diarize with pyannote.audio and tag each utterance
     as "Persona 1", "Persona 2", ...

Profiles bundle the audio + Whisper knobs that differ between clean recordings
(e.g. phone calls) and noisy recordings (faint / far-field voices):

  clean  - no denoise, greedy decoding, standard no-speech threshold, with
           Whisper's anti-hallucination guards on (temperature ladder,
           condition-on-previous OFF, log-prob gate). Best for close-mic /
           phone audio.
  noisy  - loudnorm denoise, aggressive no-speech threshold, beam search, same
           anti-hallucination guards. Best for faint voices in background noise.

Individual flags (--denoise, --no-denoise, --condition-on-previous / --no-...)
override the profile defaults.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration model + profiles
# ---------------------------------------------------------------------------

# Whisper's default temperature fallback ladder (hallucination recovery).
_TEMP_LADDER: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class TranscribeConfig:
    """Resolved configuration for the transcription pipeline.

    A profile provides the defaults; explicit CLI flags override them.
    """

    denoise_enabled: bool
    denoise_method: str  # one of DENOISE_METHODS keys; ignored if denoise disabled
    temperature: tuple[float, ...]
    no_speech_threshold: float
    condition_on_previous_text: bool
    # None => greedy decoding (Whisper default). Beam search can trigger a
    # word-repetition hallucination cascade with word_timestamps on clean audio
    # (verified empirically), so the clean profile decodes greedily.
    beam_size: int | None = None
    best_of: int | None = None
    compression_ratio_threshold: float = 2.4
    # Whisper's average-log-probability gate. Segments below this are retried at
    # a higher temperature. Wiring it in (with the temperature ladder) is part
    # of the anti-hallucination guard.
    logprob_threshold: float = -1.0


# The two profiles differ in the audio + Whisper decoding knobs below.
PROFILES: dict[str, TranscribeConfig] = {
    "clean": TranscribeConfig(
        denoise_enabled=False,
        denoise_method="loudnorm",
        # condition_on_previous_text=False breaks the repetition feedback loop
        # that Whisper falls into on sparse/quiet openings (verified on a real
        # 56-min recording). The temperature ladder + logprob/compression gates
        # let it retry poisoned segments instead of cascading.
        temperature=_TEMP_LADDER,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        beam_size=None,  # greedy: correct + fast for clean audio
        best_of=None,
    ),
    "noisy": TranscribeConfig(
        denoise_enabled=True,
        denoise_method="loudnorm",
        temperature=_TEMP_LADDER,
        no_speech_threshold=0.4,
        condition_on_previous_text=False,
        beam_size=5,  # beam search aids recovery on faint/noisy audio
        best_of=5,
    ),
}


@dataclass
class ConfigOverrides:
    """CLI-provided overrides applied on top of a profile.

    ``None`` means "leave the profile value untouched".
    """

    denoise_enabled: bool | None = None
    denoise_method: str | None = None
    condition_on_previous_text: bool | None = None


def resolve_config(profile: str, overrides: ConfigOverrides) -> TranscribeConfig:
    """Combine a named profile with explicit CLI overrides.

    Args:
        profile: Key into :data:`PROFILES`.
        overrides: Fields set to non-``None`` replace the profile value.

    Returns:
        The effective :class:`TranscribeConfig`.

    Raises:
        ValueError: If ``profile`` is unknown.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile!r}. Choose from {sorted(PROFILES)}.")

    base = PROFILES[profile]
    return TranscribeConfig(
        denoise_enabled=(
            base.denoise_enabled if overrides.denoise_enabled is None else overrides.denoise_enabled
        ),
        denoise_method=(
            base.denoise_method if overrides.denoise_method is None else overrides.denoise_method
        ),
        temperature=base.temperature,
        no_speech_threshold=base.no_speech_threshold,
        condition_on_previous_text=(
            base.condition_on_previous_text
            if overrides.condition_on_previous_text is None
            else overrides.condition_on_previous_text
        ),
        compression_ratio_threshold=base.compression_ratio_threshold,
        logprob_threshold=base.logprob_threshold,
        beam_size=base.beam_size,
        best_of=base.best_of,
    )


# ---------------------------------------------------------------------------
# Data model for diarized output
# ---------------------------------------------------------------------------


@dataclass
class Utterance:
    """A contiguous span of speech attributed to a single speaker."""

    start: float
    end: float
    speaker: int  # 1-based "Persona N"
    text: str


@dataclass
class SpeakerTurn:
    """A speaker-homogeneous time span produced by diarization."""

    start: float
    end: float
    speaker: str  # raw pyannote label, e.g. "SPEAKER_00"


# ---------------------------------------------------------------------------
# System checks + audio extraction / denoising
# ---------------------------------------------------------------------------


class TranscriptionError(RuntimeError):
    """Raised when an external step (ffmpeg, diarization) fails."""


def check_ffmpeg() -> None:
    """Verify ffmpeg is available on the system."""
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg is not installed.", file=sys.stderr)
        print("Install it with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)


def _run_ffmpeg(cmd: list[str], step: str) -> None:
    """Run an ffmpeg command, raising :class:`TranscriptionError` on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TranscriptionError(f"{step} failed:\n{result.stderr}")


def extract_audio(input_path: Path, output_wav: Path) -> None:
    """Extract audio from a video/audio file as 16 kHz mono WAV."""
    print(f"Extracting audio from: {input_path.name}")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vn",  # no video
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-ar",
            "16000",  # 16 kHz (Whisper native)
            "-ac",
            "1",  # mono
            "-y",
            str(output_wav),
        ],
        "Audio extraction",
    )
    print(f"  Audio extracted: {output_wav.name}")


def denoise_loudnorm(input_wav: Path, output_wav: Path) -> None:
    """Apply EBU R128 loudness normalization via ffmpeg.

    Boosts quiet speech to a standard broadcast level without aggressive
    spectral processing that might remove speech along with noise.
    """
    print("Applying loudness normalization (EBU R128)...")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-i",
            str(input_wav),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(output_wav),
        ],
        "Loudness normalization",
    )
    print("  Loudness normalization complete.")


def denoise_spectral(input_wav: Path, output_wav: Path) -> None:
    """Apply spectral noise reduction using ``noisereduce`` (spectral gating)."""
    import noisereduce as nr
    import soundfile as sf

    print("Applying spectral noise reduction...")
    data, rate = sf.read(str(input_wav))
    reduced = nr.reduce_noise(y=data, sr=rate, stationary=True, prop_decrease=0.75)
    sf.write(str(output_wav), reduced, rate)
    print("  Spectral noise reduction complete.")


def denoise_ffmpeg_filters(input_wav: Path, output_wav: Path) -> None:
    """Bandpass (300-3500 Hz) + FFT denoise + dynamic normalization via ffmpeg."""
    print("Applying ffmpeg filters (bandpass + afftdn + dynaudnorm)...")
    filters = ",".join(
        [
            "highpass=f=300",
            "lowpass=f=3500",
            "afftdn=nf=-25:nr=20:nt=w",
            "dynaudnorm=f=150:g=15",
        ]
    )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-i",
            str(input_wav),
            "-af",
            filters,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(output_wav),
        ],
        "FFmpeg filter processing",
    )
    print("  FFmpeg filter processing complete.")


DENOISE_METHODS = {
    "loudnorm": denoise_loudnorm,
    "spectral": denoise_spectral,
    "ffmpeg": denoise_ffmpeg_filters,
}


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe_audio(
    audio_path: Path,
    whisper_model: str,
    language: str,
    config: TranscribeConfig,
    word_timestamps: bool,
) -> list[dict]:
    """Transcribe audio with Whisper and return its segment dictionaries."""
    import whisper

    print(f"Loading Whisper model: {whisper_model}")
    model = whisper.load_model(whisper_model)

    print(
        f"Transcribing (language={language}, "
        f"condition_on_previous={config.condition_on_previous_text}, "
        f"word_timestamps={word_timestamps})..."
    )
    result = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=config.beam_size,
        best_of=config.best_of,
        temperature=config.temperature,
        compression_ratio_threshold=config.compression_ratio_threshold,
        logprob_threshold=config.logprob_threshold,
        no_speech_threshold=config.no_speech_threshold,
        condition_on_previous_text=config.condition_on_previous_text,
        word_timestamps=word_timestamps,
        verbose=False,
    )
    return result["segments"]


# ---------------------------------------------------------------------------
# Diarization + alignment (pure logic separated for testability)
# ---------------------------------------------------------------------------


def diarize_audio(audio_path: Path, hf_token: str, num_speakers: int | None) -> list[SpeakerTurn]:
    """Run pyannote speaker diarization and return speaker turns.

    Note: diarization uses the *raw* extracted audio, never the denoised
    version, because denoising can distort speaker voiceprints.

    Raises:
        TranscriptionError: If the pipeline cannot be loaded or run.
    """
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    print("Loading pyannote diarization pipeline (community-1)...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token
        )
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        raise TranscriptionError(
            "Failed to load pyannote pipeline. Ensure your HF token is valid and "
            "you accepted the model conditions at "
            "https://hf.co/pyannote/speaker-diarization-community-1\n"
            f"Underlying error: {exc}"
        ) from exc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        pipeline.to(torch.device(device))
    except Exception:  # noqa: BLE001 - MPS support in pyannote can be partial
        print(f"  ({device} unavailable for pyannote; falling back to cpu)")
        pipeline.to(torch.device("cpu"))

    print("Running diarization...")
    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    with ProgressHook() as hook:
        output = pipeline(str(audio_path), hook=hook, **kwargs)

    return [
        SpeakerTurn(start=segment.start, end=segment.end, speaker=label)
        for segment, _, label in output.speaker_diarization.itertracks(yield_label=True)
    ]


def build_speaker_map(turns: list[SpeakerTurn]) -> dict[str, int]:
    """Map raw pyannote labels to stable 1-based Persona numbers.

    Speakers are numbered by first appearance (earliest turn start), so
    "Persona 1" is whoever speaks first.
    """
    order: list[str] = []
    for turn in sorted(turns, key=lambda t: t.start):
        if turn.speaker not in order:
            order.append(turn.speaker)
    return {label: i + 1 for i, label in enumerate(order)}


def speaker_at(turns: list[SpeakerTurn], time: float) -> str | None:
    """Return the raw speaker label active at ``time``.

    Chooses the turn with the greatest overlap around ``time``; if ``time``
    falls in no turn, returns the nearest turn's speaker. Returns ``None``
    only when there are no turns at all.
    """
    if not turns:
        return None

    best_label: str | None = None
    best_overlap = 0.0
    for turn in turns:
        # Overlap of an instant is measured against a tiny window to break ties
        # deterministically in favour of a turn that actually contains ``time``.
        overlap = min(turn.end, time + 1e-6) - max(turn.start, time)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = turn.speaker

    if best_label is not None:
        return best_label

    nearest = min(turns, key=lambda t: min(abs(t.start - time), abs(t.end - time)))
    return nearest.speaker


def _iter_words(segments: list[dict]):
    """Yield ``(word_text, start, end)`` tuples from Whisper segments.

    Falls back to segment-level granularity when word timestamps are absent.
    """
    for segment in segments:
        words = segment.get("words")
        if words:
            for word in words:
                yield word["word"], word["start"], word["end"]
        else:
            yield segment["text"], segment["start"], segment["end"]


def group_words_by_speaker(
    segments: list[dict],
    turns: list[SpeakerTurn],
    speaker_map: dict[str, int],
) -> list[Utterance]:
    """Assign each word to a speaker and merge consecutive same-speaker words.

    This splits speaker turns *within* a Whisper segment, which is essential
    because Whisper segment boundaries do not align with speaker changes.
    """
    utterances: list[Utterance] = []
    cur_speaker: int | None = None
    cur_start: float | None = None
    cur_end: float | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf and cur_speaker is not None and cur_start is not None:
            utterances.append(
                Utterance(
                    start=cur_start,
                    end=cur_end if cur_end is not None else cur_start,
                    speaker=cur_speaker,
                    text="".join(buf).strip(),
                )
            )
        buf = []

    for text, start, end in _iter_words(segments):
        mid = (start + end) / 2
        label = speaker_at(turns, mid)
        persona = speaker_map.get(label, 1) if label is not None else 1

        if cur_speaker is None:
            cur_speaker, cur_start = persona, start
        elif persona != cur_speaker:
            flush()
            cur_speaker, cur_start = persona, start
        cur_end = end
        buf.append(text)

    flush()
    return utterances


# ---------------------------------------------------------------------------
# Formatting / output
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_plain(segments: list[dict]) -> str:
    """Format transcription segments with timestamps (no speaker labels)."""
    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"[{start} --> {end}] {seg['text'].strip()}")
    return "\n".join(lines)


def format_diarized(utterances: list[Utterance]) -> str:
    """Format diarized utterances as ``[HH:MM:SS] Persona N: text``."""
    return "\n".join(
        f"[{format_timestamp(u.start)}] Persona {u.speaker}: {u.text}" for u in utterances
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize speech from video/audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Clean phone call, diarized, Spanish (all defaults)
  uv run python transcribe.py conversation.mp4

  # Noisy far-field recording, force 3 speakers, save to file
  uv run python transcribe.py meeting.mp4 --profile noisy --speakers 3 -o out.txt

  # Transcription only (no speaker tags), larger model, English
  uv run python transcribe.py talk.mp4 --no-diarize --whisper-model large --language en

Profiles bundle audio + Whisper decoding knobs:
  clean  (default) - no denoise, greedy decoding, condition-on-previous ON.
                     Best for close-mic / phone audio.
  noisy            - loudnorm denoise, temperature fallback ladder,
                     aggressive no-speech threshold, condition-on-previous OFF.
                     Best for faint voices in background noise.

Diarization requires a HuggingFace token (env HF_TOKEN or --hf-token) and
acceptance of the model conditions at
https://hf.co/pyannote/speaker-diarization-community-1
""",
    )
    parser.add_argument("input", type=Path, help="Video or audio file to process")

    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="clean",
        help="Preset knob bundle (default: clean)",
    )
    parser.add_argument(
        "--whisper-model",
        "--model",  # backward-compatible alias
        dest="whisper_model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument("--language", default="es", help="Language code (default: es)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Save transcript to a file")

    # Diarization (on by default)
    parser.add_argument(
        "--diarize",
        dest="diarize",
        action="store_true",
        default=True,
        help="Tag speakers as Persona N (default: on)",
    )
    parser.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="Disable speaker tagging (transcription only)",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Force an exact number of speakers (default: auto-detect)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for diarization (default: env HF_TOKEN)",
    )

    # Profile overrides
    parser.add_argument(
        "--denoise",
        choices=list(DENOISE_METHODS),
        default=None,
        help="Override denoise method (implies denoising ON)",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Override: skip noise reduction entirely",
    )
    parser.add_argument(
        "--condition-on-previous",
        dest="condition_on_previous",
        action="store_true",
        default=None,
        help="Override: condition each segment on previously decoded text",
    )
    parser.add_argument(
        "--no-condition-on-previous",
        dest="condition_on_previous",
        action="store_false",
        help="Override: do not condition on previously decoded text",
    )
    return parser


def overrides_from_args(args: argparse.Namespace) -> ConfigOverrides:
    """Translate CLI flags into a :class:`ConfigOverrides`."""
    denoise_enabled: bool | None = None
    denoise_method: str | None = None
    if args.no_denoise:
        denoise_enabled = False
    elif args.denoise is not None:
        denoise_enabled = True
        denoise_method = args.denoise

    return ConfigOverrides(
        denoise_enabled=denoise_enabled,
        denoise_method=denoise_method,
        condition_on_previous_text=args.condition_on_previous,
    )


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    A minimal, dependency-free parser: blank lines and ``#`` comments are
    ignored, surrounding quotes on values are stripped, and existing
    environment variables are never overwritten (real env vars win).
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_hf_token(cli_token: str | None) -> str:
    """Resolve the HF token from the CLI flag or the environment.

    Raises:
        TranscriptionError: If no token is available.
    """
    token = cli_token or os.environ.get("HF_TOKEN")
    if not token:
        raise TranscriptionError(
            "Diarization requires a HuggingFace token. Set the HF_TOKEN "
            "environment variable, add it to a .env file, pass --hf-token, "
            "or use --no-diarize."
        )
    return token


def _write_output(text: str, output: Path | None, header: str) -> None:
    """Write the transcript to a file or stdout."""
    if output:
        output.write_text(text, encoding="utf-8")
        print(f"\nTranscript saved to: {output}")
    else:
        print("\n" + "=" * 60)
        print(header)
        print("=" * 60)
        print(text)
        print("=" * 60)


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.speakers is not None and args.speakers < 1:
        print("Error: --speakers must be >= 1", file=sys.stderr)
        sys.exit(1)

    check_ffmpeg()
    config = resolve_config(args.profile, overrides_from_args(args))

    # Resolve the token early so we fail fast before heavy work.
    hf_token = resolve_hf_token(args.hf_token) if args.diarize else ""

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_wav = tmp / "raw_audio.wav"
            clean_wav = tmp / "clean_audio.wav"

            # Step 1: extract audio
            extract_audio(args.input, raw_wav)

            # Step 2: denoise (Whisper input only; diarization uses raw audio)
            if config.denoise_enabled:
                DENOISE_METHODS[config.denoise_method](raw_wav, clean_wav)
                whisper_input = clean_wav
            else:
                print("Denoising disabled (profile/override).")
                whisper_input = raw_wav

            # Step 3: transcribe
            segments = transcribe_audio(
                whisper_input,
                args.whisper_model,
                args.language,
                config,
                word_timestamps=args.diarize,
            )

            if not segments:
                print("No speech detected in the audio.")
                sys.exit(0)

            # Step 4: diarize + align (optional)
            if args.diarize:
                turns = diarize_audio(raw_wav, hf_token, args.speakers)
                if not turns:
                    print(
                        "Warning: no speakers detected; emitting plain transcript.",
                        file=sys.stderr,
                    )
                    _write_output(format_plain(segments), args.output, "TRANSCRIPT")
                    return
                speaker_map = build_speaker_map(turns)
                utterances = group_words_by_speaker(segments, turns, speaker_map)
                print(f"\nDetected {len(speaker_map)} speaker(s).")
                _write_output(format_diarized(utterances), args.output, "DIARIZED TRANSCRIPT")
            else:
                _write_output(format_plain(segments), args.output, "TRANSCRIPT")

    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
