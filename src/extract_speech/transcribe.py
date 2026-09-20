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
  noisy  - Demucs vocal isolation, aggressive no-speech threshold, beam search,
           same anti-hallucination guards. Best for faint voices in background
           noise. Demucs costs ~0.15x realtime and ~1.4 GB RAM; use
           "--denoise loudnorm" for the cheap variant.

Individual flags (--denoise, --no-denoise, --condition-on-previous / --no-...)
override the profile defaults.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    # Imported for annotations only; the runtime import lives in read_waveform
    # so that --help and the test suite stay free of heavy imports.
    import numpy as np

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
        # Demucs vocal isolation rather than loudnorm. Measured on real
        # far-field recordings: 34-43% of non-vocal energy removed, and
        # Whisper's repetition hallucinations largely eliminated (one clip went
        # from 17 segments, including a "si si si si..." loop, to 1). It costs
        # ~0.15x realtime and ~1.4 GB of RAM, so `--denoise loudnorm` is still
        # there when that trade is not wanted.
        denoise_method="demucs",
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
        denoise_enabled=(base.denoise_enabled if overrides.denoise_enabled is None else overrides.denoise_enabled),
        denoise_method=(base.denoise_method if overrides.denoise_method is None else overrides.denoise_method),
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


@dataclass(frozen=True)
class AudioInputs:
    """The audio a denoise step may draw on.

    Filter-based methods read ``extracted``, the 16 kHz mono WAV that Whisper
    and pyannote also consume. Source separation reads ``source`` instead: it
    requires 44.1 kHz stereo, and feeding it ``extracted`` would hand it audio
    that had already been downsampled to 16 kHz and downmixed to mono, throwing
    away both the >8 kHz band and the inter-channel differences that separation
    models rely on.
    """

    source: Path
    extracted: Path


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


def read_waveform(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file into memory as a ``(channel, time)`` float32 array.

    Diarization is handed this waveform instead of a file path, so pyannote
    never invokes its own audio decoder. That decoder (torchcodec) links
    FFmpeg's C libraries by exact version and breaks whenever FFmpeg is
    upgraded; ffmpeg has already produced plain PCM by this point, so decoding
    it a second time is redundant anyway.

    The ``(channel, time)`` orientation and 2-D shape are required by
    pyannote's own input validation.

    Args:
        wav_path: A WAV file, normally the output of :func:`extract_audio`.

    Returns:
        The samples as a ``(channel, time)`` float32 array, and the sample rate.

    Raises:
        TranscriptionError: If the file cannot be read, or contains no samples.
    """
    import soundfile as sf

    try:
        # always_2d gives (time, channel) uniformly, for mono and multichannel.
        data, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - soundfile raises several types
        raise TranscriptionError(f"Could not read audio from {wav_path}: {exc}") from exc

    if data.shape[0] == 0:
        raise TranscriptionError(f"{wav_path} contains no audio samples. The source file may have no audio stream.")

    # soundfile is time-major; pyannote wants channel-major.
    return data.T, int(rate)


def denoise_loudnorm(inputs: AudioInputs, output_wav: Path) -> None:
    """Apply EBU R128 loudness normalization via ffmpeg.

    Boosts quiet speech to a standard broadcast level without aggressive
    spectral processing that might remove speech along with noise.
    """
    print("Applying loudness normalization (EBU R128)...")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-i",
            str(inputs.extracted),
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


def denoise_spectral(inputs: AudioInputs, output_wav: Path) -> None:
    """Apply spectral noise reduction using ``noisereduce`` (spectral gating)."""
    import noisereduce as nr
    import soundfile as sf

    print("Applying spectral noise reduction...")
    data, rate = sf.read(str(inputs.extracted))
    reduced = nr.reduce_noise(y=data, sr=rate, stationary=True, prop_decrease=0.75)
    sf.write(str(output_wav), reduced, rate)
    print("  Spectral noise reduction complete.")


def denoise_ffmpeg_filters(inputs: AudioInputs, output_wav: Path) -> None:
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
            str(inputs.extracted),
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


# ---------------------------------------------------------------------------
# Demucs source separation (vocal isolation)
# ---------------------------------------------------------------------------

# What the htdemucs family is trained on; apply_model does not resample.
DEMUCS_SAMPLE_RATE = 44100
DEMUCS_CHANNELS = 2
DEFAULT_DEMUCS_MODEL = "htdemucs"


def available_demucs_models() -> list[str]:
    """List the bag names demucs ships, without loading any weights."""
    import demucs

    remote = Path(demucs.__file__).parent / "remote"
    return sorted(p.stem for p in remote.glob("*.yaml"))


def validate_demucs_model(name: str, available: list[str]) -> str:
    """Check a Demucs model name against the installed bags.

    Raises:
        TranscriptionError: If ``name`` is not one of ``available``.
    """
    if name not in available:
        raise TranscriptionError(f"Unknown Demucs model {name!r}. Available: {', '.join(available)}")
    return name


def normalise_for_demucs(waveform: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Centre and scale a waveform the way demucs' own reference code does.

    demucs derives the statistics from the *channel average* rather than per
    channel (``ref = wav.mean(0)``), so the two channels stay on a common scale.

    Args:
        waveform: A ``(channel, time)`` array.

    Returns:
        The normalised waveform, plus the mean and standard deviation needed to
        undo it. A zero standard deviation (silent or constant audio) is
        reported as ``1.0`` so the scaling stays a no-op instead of producing
        NaNs that would silently poison the separation.
    """
    ref = waveform.mean(axis=0)
    mean = float(ref.mean())
    std = float(ref.std())
    if std == 0.0:
        std = 1.0
    return (waveform - mean) / std, mean, std


def denormalise_stems(stems: Any, mean: float, std: float) -> Any:
    """Invert :func:`normalise_for_demucs` on the separated stems."""
    return stems * std + mean


def stem_index(sources: list[str], stem: str) -> int:
    """Return the position of ``stem`` in a model's source list.

    Raises:
        TranscriptionError: If the model does not produce that stem.
    """
    if stem not in sources:
        raise TranscriptionError(f"Model does not produce a {stem!r} stem. It produces: {', '.join(sources)}")
    return sources.index(stem)


def load_demucs_model(model_name: str = DEFAULT_DEMUCS_MODEL) -> Any:
    """Load a Demucs model onto the best available device.

    Separate from :func:`denoise_demucs` so a batch loads the 80 MB bag once
    rather than once per file.

    Raises:
        TranscriptionError: If ``model_name`` is not an installed bag.
    """
    import torch
    from demucs.pretrained import get_model

    validate_demucs_model(model_name, available_demucs_models())
    print(f"Loading Demucs model: {model_name}")
    model = get_model(model_name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model


def denoise_demucs(model: Any, inputs: AudioInputs, output_wav: Path) -> None:
    """Isolate the vocal stem with Demucs and write it as 16 kHz mono.

    Unlike the filter-based methods this reads ``inputs.source`` rather than the
    extracted 16 kHz mono WAV, because Demucs needs 44.1 kHz stereo and must not
    be fed audio that has already been downsampled and downmixed.

    On noisy recordings this removed 34-43% of non-vocal energy and sharply
    reduced Whisper's repetition hallucinations. On clean close-mic audio it
    removes almost nothing (~0.1%), so it is not worth its cost there.
    """
    import torch
    from demucs.apply import apply_model

    device = next(model.parameters()).device

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        demucs_in = tmp / "demucs_in.wav"

        # Extract straight from the source at the model's native format. One
        # ffmpeg call covers both cases: it upsamples and duplicates a 16 kHz
        # mono source, and properly downsamples a 48 kHz stereo one while
        # keeping the two channels distinct.
        print(f"Extracting {DEMUCS_SAMPLE_RATE} Hz stereo audio for Demucs...")
        _run_ffmpeg(
            [
                "ffmpeg",
                "-i",
                str(inputs.source),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(DEMUCS_SAMPLE_RATE),
                "-ac",
                str(DEMUCS_CHANNELS),
                "-y",
                str(demucs_in),
            ],
            "Demucs audio extraction",
        )

        waveform, rate = read_waveform(demucs_in)
        if waveform.shape[0] != DEMUCS_CHANNELS:
            raise TranscriptionError(
                f"Demucs needs {DEMUCS_CHANNELS} channels but got {waveform.shape[0]} from {demucs_in}."
            )

        normalised, mean, std = normalise_for_demucs(waveform)

        print(f"Separating vocals on {device} (this is the slow step)...")
        with torch.no_grad():
            stems = apply_model(model, torch.from_numpy(normalised)[None], device=device, progress=False)[0]
        stems = denormalise_stems(stems, mean, std)

        vocals = stems[stem_index(list(model.sources), "vocals")]

        # Back to the 16 kHz mono that Whisper expects, via ffmpeg so the
        # resampling matches the rest of the pipeline.
        vocals_wav = tmp / "vocals.wav"
        import soundfile as sf

        sf.write(str(vocals_wav), vocals.mean(0).cpu().numpy(), rate)
        _run_ffmpeg(
            [
                "ffmpeg",
                "-i",
                str(vocals_wav),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(output_wav),
            ],
            "Vocal stem downmix",
        )
        print("  Vocal isolation complete.")


# Filter-based methods: uniform (inputs, output_wav) signature, no model needed.
DENOISE_METHODS = {
    "loudnorm": denoise_loudnorm,
    "spectral": denoise_spectral,
    "ffmpeg": denoise_ffmpeg_filters,
}

# Source separation is deliberately not in that registry: it needs a loaded
# model, so its signature differs. Both sets together form the valid choices.
DEMUCS_METHOD = "demucs"
ALL_DENOISE_METHODS = frozenset(DENOISE_METHODS) | {DEMUCS_METHOD}


def apply_denoise(
    method: str,
    inputs: AudioInputs,
    output_wav: Path,
    demucs_model: Any = None,
) -> None:
    """Dispatch to the named denoise method.

    Raises:
        TranscriptionError: If the method is unknown, or Demucs was selected
            without a loaded model.
    """
    if method == DEMUCS_METHOD:
        if demucs_model is None:
            raise TranscriptionError("Demucs denoising requires a loaded model.")
        denoise_demucs(demucs_model, inputs, output_wav)
        return
    if method not in DENOISE_METHODS:
        raise TranscriptionError(
            f"Unknown denoise method {method!r}. Available: {', '.join(sorted(ALL_DENOISE_METHODS))}"
        )
    DENOISE_METHODS[method](inputs, output_wav)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def load_whisper_model(whisper_model: str) -> Any:
    """Load a Whisper model.

    Separate from :func:`transcribe_audio` so a batch loads the weights once
    rather than once per file (medium takes ~6 s, large considerably longer).
    """
    import whisper

    print(f"Loading Whisper model: {whisper_model}")
    return whisper.load_model(whisper_model)


def transcribe_audio(
    model: Any,
    audio_path: Path,
    language: str,
    config: TranscribeConfig,
    word_timestamps: bool,
) -> list[dict]:
    """Transcribe audio with an already-loaded Whisper model."""
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
    # Whisper annotates its return as dict[str, str | list], so indexing it
    # widens to "str | list". "segments" is always a list of segment dicts.
    return cast("list[dict]", result["segments"])


# ---------------------------------------------------------------------------
# Diarization + alignment (pure logic separated for testability)
# ---------------------------------------------------------------------------


def load_diarization_pipeline(hf_token: str) -> Any:
    """Load the pyannote diarization pipeline and move it to the best device.

    Kept separate from :func:`run_diarization` so the caller can load the
    pipeline *before* transcription starts. Loading validates the token and the
    model licence, and those are the failures worth surfacing in seconds rather
    than after a long transcription has already run.

    Args:
        hf_token: A HuggingFace token with access to the pyannote models.

    Returns:
        The ready-to-use pipeline. Untyped because pyannote's own annotations
        provide no usable information.

    Raises:
        TranscriptionError: If the pipeline cannot be loaded.
    """
    import torch
    from pyannote.audio import Pipeline

    print("Loading pyannote diarization pipeline (community-1)...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        raise TranscriptionError(
            "Failed to load pyannote pipeline. Ensure your HF token is valid and "
            "you accepted the model conditions at "
            "https://hf.co/pyannote/speaker-diarization-community-1\n"
            f"Underlying error: {exc}"
        ) from exc

    # from_pretrained returns Optional[Pipeline]: it yields None (rather than
    # raising) when the checkpoint cannot be resolved, e.g. the model conditions
    # have not been accepted for this token.
    if pipeline is None:
        raise TranscriptionError(
            "pyannote returned no pipeline for 'pyannote/speaker-diarization-community-1'. "
            "This usually means the token lacks access: accept the model conditions at "
            "https://hf.co/pyannote/speaker-diarization-community-1 and "
            "https://hf.co/pyannote/segmentation-3.0, then retry."
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        pipeline.to(torch.device(device))
    except Exception:  # noqa: BLE001 - MPS support in pyannote can be partial
        print(f"  ({device} unavailable for pyannote; falling back to cpu)")
        pipeline.to(torch.device("cpu"))

    return pipeline


def turns_from_diarization(diarization: Any) -> list[SpeakerTurn]:
    """Convert a pyannote ``Annotation`` into :class:`SpeakerTurn` objects.

    ``yield_label=True`` is required: without it ``itertracks`` yields
    ``(segment, track_id)`` pairs and the label is lost.
    """
    return [
        SpeakerTurn(start=segment.start, end=segment.end, speaker=label)
        for segment, _, label in diarization.itertracks(yield_label=True)
    ]


def run_diarization(pipeline: Any, audio_path: Path, num_speakers: int | None) -> list[SpeakerTurn]:
    """Diarize ``audio_path`` with an already-loaded pipeline.

    The audio is read into memory and handed to pyannote as a waveform mapping
    rather than as a path. pyannote accepts either (its ``AudioFile`` type is
    ``str | Path | IOBase | Mapping``), but the path form makes it decode the
    file with torchcodec, which links FFmpeg's C libraries by exact version and
    fails whenever FFmpeg is upgraded. ffmpeg has already produced plain PCM at
    this point, so the waveform form also avoids decoding the same audio twice.

    Note: diarization uses the *raw* extracted audio, never the denoised
    version, because denoising can distort speaker voiceprints.

    Raises:
        TranscriptionError: If the audio cannot be read.
    """
    import torch
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    waveform, sample_rate = read_waveform(audio_path)

    print("Running diarization...")
    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    with ProgressHook() as hook:
        output: Any = pipeline(
            {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate},
            hook=hook,
            **kwargs,
        )

    return turns_from_diarization(output.speaker_diarization)


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
    return "\n".join(f"[{format_timestamp(u.start)}] Persona {u.speaker}: {u.text}" for u in utterances)


# ---------------------------------------------------------------------------
# Batch mode: folder in, folder out
# ---------------------------------------------------------------------------

# Containers ffmpeg can extract audio from. Lowercase and dotted; matching is
# case-insensitive.
MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        # video
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
        # audio
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".aac",
        ".ogg",
        ".opus",
        ".wma",
    }
)

TRANSCRIPT_SUFFIX = "_transcript.txt"


def discover_inputs(source: Path) -> list[Path]:
    """Find every media file under ``source``, recursively.

    Returns:
        Matching files sorted by path, so a batch is reproducible and progress
        counters are stable between runs.

    Raises:
        TranscriptionError: If ``source`` is not an existing directory.
    """
    if not source.is_dir():
        raise TranscriptionError(f"{source} is not a directory.")
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)


def output_path_for(input_path: Path, source_root: Path, target_root: Path) -> Path:
    """Map an input file to its transcript path under ``target_root``.

    The directory structure below ``source_root`` is mirrored, because the same
    filename can appear in several subdirectories (the Tapo corpus has
    identical names under ``Bedroom/`` and ``Living Room/``); flattening would
    silently overwrite one transcript with another.

    Raises:
        TranscriptionError: If ``input_path`` is not below ``source_root``.
    """
    try:
        relative = input_path.relative_to(source_root)
    except ValueError as exc:
        raise TranscriptionError(f"{input_path} is outside the source directory {source_root}.") from exc
    return target_root / relative.with_name(relative.stem + TRANSCRIPT_SUFFIX)


def plan_batch(
    inputs: list[Path],
    source_root: Path,
    target_root: Path,
    overwrite: bool,
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Pair each input with its output, separating work from what already exists.

    Skipping completed files makes a long batch resumable: an interrupted run
    can simply be repeated.

    Returns:
        ``(pending, skipped)`` where ``pending`` holds ``(input, output)`` pairs.
    """
    pending: list[tuple[Path, Path]] = []
    skipped: list[Path] = []
    for item in inputs:
        destination = output_path_for(item, source_root, target_root)
        if destination.exists() and not overwrite:
            skipped.append(item)
        else:
            pending.append((item, destination))
    return pending, skipped


def format_batch_summary(succeeded: int, skipped: int, failed: list[tuple[str, str]]) -> str:
    """Render the end-of-batch report.

    Each failure is collapsed to a single line: ffmpeg and torch errors are
    multi-line, and letting them through turns the summary of a large batch into
    a wall of text. The full error was already printed when the file failed.
    """
    lines = [
        "",
        "=" * 60,
        f"Batch complete: {succeeded} transcribed, {skipped} skipped, {len(failed)} failed",
        "=" * 60,
    ]
    for name, reason in failed:
        first_line = reason.strip().splitlines()[0] if reason.strip() else "unknown error"
        lines.append(f"  FAILED  {name}: {first_line}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Remote execution
# ---------------------------------------------------------------------------

DEFAULT_REMOTE_DIR = "~/.sybil"

# Everything the remote needs in order to run "uv sync" and the CLI. An
# allowlist rather than rsync --exclude, because excludes fail open: forgetting
# one silently ships it. .env must never reach the remote -- the HuggingFace
# token is forwarded per session over stdin instead.
PROVISION_ALLOWLIST: tuple[str, ...] = ("pyproject.toml", "uv.lock", ".python-version", "src", "tests")


@dataclass(frozen=True)
class RemoteConfig:
    """Where work happens on the remote host.

    ``root`` is a remote path and may be ``~``-relative; it is never resolved
    locally. Paths below it are built with POSIX separators because the remote
    is a POSIX host regardless of what this client runs on.
    """

    host: str
    root: str

    def _under(self, name: str) -> str:
        return f"{self.root.rstrip('/')}/{name}"

    @property
    def project(self) -> str:
        """Where the checkout and its virtualenv live."""
        return self._under("project")

    @property
    def inputs(self) -> str:
        """Where source media is uploaded."""
        return self._under("inputs")

    @property
    def outputs(self) -> str:
        """Where transcripts are written, and kept between runs so an
        interrupted batch can resume."""
        return self._under("outputs")


def parse_remote_config(host: str, root: str) -> RemoteConfig:
    """Validate the remote destination.

    Raises:
        TranscriptionError: If either value is blank.
    """
    if not host.strip():
        raise TranscriptionError("--run-in-remote needs a host (an ssh alias or user@host).")
    if not root.strip():
        raise TranscriptionError("--remote-dir needs a directory.")
    return RemoteConfig(host=host.strip(), root=root.strip())


def expand_remote_root(root: str, home: str) -> str:
    """Resolve a leading ``~`` against the remote home directory.

    Quoting is mandatory when sending a command through ssh, but a quoted ``~``
    is a literal directory name to the remote shell -- it creates a folder
    called ``~`` in the login directory instead of using ``$HOME``. Resolving it
    up front keeps every later path absolute and safely quotable.

    ``~user`` forms are left untouched: they refer to a different user's home,
    and rewriting them against ours would silently point elsewhere.
    """
    if root == "~":
        return home
    if root.startswith("~/"):
        return f"{home.rstrip('/')}/{root[2:]}"
    return root


def join_remote_command(argv: list[str]) -> str:
    """Join an argv into a string safe for the remote shell.

    ``ssh`` always hands a *string* to the remote shell, so this is the one
    place shell quoting is unavoidable. Paths such as ``Living Room`` and any
    metacharacters must survive intact.
    """
    return shlex.join(argv)


def build_remote_argv(args: argparse.Namespace, cfg: RemoteConfig) -> list[str]:
    """Translate the local invocation into the argv the remote should run.

    The remote runs the identical CLI on the identical input, so its behaviour
    matches a local run. Local paths are rewritten to their remote equivalents,
    and the remote-execution flags are dropped so the remote does not try to
    delegate onwards.

    The HuggingFace token is deliberately excluded: it is passed through the
    process environment via stdin, never on a command line where the remote's
    ``ps`` would expose it.
    """
    argv: list[str] = ["extract-speech"]

    if args.source is not None:
        argv += ["--source", cfg.inputs, "--target", cfg.outputs]
        if args.overwrite:
            argv.append("--overwrite")
    else:
        # A single file is staged under inputs/ keeping its original name.
        argv.append(f"{cfg.inputs}/{args.input.name}")
        argv += ["--output", f"{cfg.outputs}/{args.input.stem}{TRANSCRIPT_SUFFIX}"]

    argv += ["--profile", args.profile, "--whisper-model", args.whisper_model, "--language", args.language]

    if args.diarize:
        if args.speakers is not None:
            argv += ["--speakers", str(args.speakers)]
    else:
        argv.append("--no-diarize")

    if args.no_denoise:
        argv.append("--no-denoise")
    elif args.denoise is not None:
        argv += ["--denoise", args.denoise]

    if args.condition_on_previous is True:
        argv.append("--condition-on-previous")
    elif args.condition_on_previous is False:
        argv.append("--no-condition-on-previous")

    argv += ["--demucs-model", args.demucs_model]
    return argv


def _run_local(cmd: list[str], step: str, stdin_text: str | None = None) -> None:
    """Run a local helper process (ssh/rsync), streaming its output.

    A list argv means no local shell, so paths with spaces need no escaping
    here. Remote quoting is handled by :func:`join_remote_command`.

    Raises:
        TranscriptionError: If the command is missing or exits non-zero.
    """
    try:
        result = subprocess.run(cmd, input=stdin_text, text=True, check=False)
    except FileNotFoundError as exc:
        raise TranscriptionError(f"{step} failed: {cmd[0]} is not installed.") from exc
    if result.returncode != 0:
        raise TranscriptionError(f"{step} failed (exit {result.returncode}).")


def _ssh(cfg: RemoteConfig, command: str, stdin_text: str | None = None) -> None:
    """Run a shell command on the remote host."""
    _run_local(["ssh", cfg.host, command], f"Remote command on {cfg.host}", stdin_text)


def _capture_remote(cfg: RemoteConfig, command: str, step: str) -> str:
    """Run a remote command and return its stdout, stripped."""
    try:
        result = subprocess.run(["ssh", cfg.host, command], capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise TranscriptionError(f"{step} failed: ssh is not installed.") from exc
    if result.returncode != 0:
        raise TranscriptionError(f"{step} failed: {result.stderr.strip() or f'exit {result.returncode}'}")
    return result.stdout.strip()


def remote_preflight(cfg: RemoteConfig) -> RemoteConfig:
    """Check the remote can do the work, before uploading anything.

    Returns:
        The config with its root resolved to an absolute remote path.

    Raises:
        TranscriptionError: If the host is unreachable or a tool is missing.
    """
    print(f"Checking {cfg.host}...")
    home = _capture_remote(cfg, 'printf %s "$HOME"', f"Connecting to {cfg.host}")
    if not home:
        raise TranscriptionError(f"Could not determine the home directory on {cfg.host}.")
    _ssh(cfg, "command -v uv >/dev/null || { echo 'uv is not installed on the remote' >&2; exit 1; }")
    _ssh(cfg, "command -v ffmpeg >/dev/null || { echo 'ffmpeg is not installed on the remote' >&2; exit 1; }")
    print(f"  uv and ffmpeg present; home is {home}.")
    return RemoteConfig(host=cfg.host, root=expand_remote_root(cfg.root, home))


def remote_provision(cfg: RemoteConfig, project_root: Path) -> None:
    """Copy the project to the remote and install its dependencies."""
    print(f"Provisioning {cfg.host}:{cfg.project}...")
    _ssh(cfg, join_remote_command(["mkdir", "-p", cfg.project, cfg.inputs, cfg.outputs]))

    present = [name for name in PROVISION_ALLOWLIST if (project_root / name).exists()]
    _run_local(
        ["rsync", "-az", "--delete", *[str(project_root / name) for name in present], f"{cfg.host}:{cfg.project}/"],
        "Uploading project",
    )
    _ssh(cfg, join_remote_command(["sh", "-lc", f"cd {shlex.quote(cfg.project)} && uv sync --all-groups"]))
    print("  Dependencies installed.")


def build_upload_args(cfg: RemoteConfig, local: Path, is_directory: bool) -> list[str]:
    """Build the rsync argv that stages sources on the remote.

    A directory upload uses ``--delete`` so the remote staging area mirrors the
    local source: otherwise a file deleted locally lingers remotely and is
    rediscovered on every later run, producing phantom work (and phantom
    failures, if it was the broken file you just removed).

    A single-file upload does not delete, since it is adding one file to a
    staging area that may hold inputs for a batch being resumed.

    ``-L`` resolves symlinks and sends the file they point at. Without it the
    link itself is copied, and a media library built from symlinks (as the
    Conversaciones archive is) arrives on the remote as dangling pointers to
    local paths -- the upload reports success and the run then fails with
    "File not found".
    """
    source = f"{local}/" if is_directory else str(local)
    args = ["rsync", "-az", "-L", "--partial"]
    if is_directory:
        args.append("--delete")
    return [*args, source, f"{cfg.host}:{cfg.inputs}/"]


def build_download_args(cfg: RemoteConfig, target: Path) -> list[str]:
    """Build the rsync argv that retrieves transcripts.

    Never uses ``--delete``: that would destroy transcripts, defeating the
    guarantee that whatever finished is retrieved.
    """
    return ["rsync", "-az", "--partial", f"{cfg.host}:{cfg.outputs}/", f"{target}/"]


def remote_upload_inputs(cfg: RemoteConfig, args: argparse.Namespace) -> None:
    """Upload the sources. Originals are sent, so the remote runs exactly as local would."""
    is_directory = args.source is not None
    local = args.source if is_directory else args.input
    print(f"Uploading {local} -> {cfg.host}:{cfg.inputs}...")
    _run_local(build_upload_args(cfg, local, is_directory), "Uploading sources")


def remote_download_outputs(cfg: RemoteConfig, args: argparse.Namespace) -> None:
    """Retrieve whatever transcripts exist, including after a partial run."""
    if args.source is not None:
        args.target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading transcripts -> {args.target}...")
        _run_local(build_download_args(cfg, args.target), "Downloading")
    else:
        destination = args.output or Path(f"{args.input.stem}{TRANSCRIPT_SUFFIX}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        remote_file = f"{cfg.outputs}/{args.input.stem}{TRANSCRIPT_SUFFIX}"
        print(f"Downloading transcript -> {destination}...")
        _run_local(["rsync", "-az", f"{cfg.host}:{remote_file}", str(destination)], "Downloading")


def run_remote(args: argparse.Namespace, hf_token: str, project_root: Path) -> int:
    """Orchestrate a remote run and return a process exit code.

    Transcripts are downloaded even when the remote run fails, so an
    interrupted batch keeps the files it completed. Because the remote's output
    directory persists and batch mode skips existing transcripts, repeating the
    command resumes where it stopped.
    """
    cfg = parse_remote_config(args.run_in_remote, args.remote_dir)

    cfg = remote_preflight(cfg)
    remote_provision(cfg, project_root)
    remote_upload_inputs(cfg, args)

    # The token is piped over the encrypted channel and exported inside the
    # remote shell, so it never lands on the remote disk or in a command line.
    command = join_remote_command(["uv", "run", *build_remote_argv(args, cfg)])
    wrapper = (
        f"cd {shlex.quote(cfg.project)} && "
        + ("IFS= read -r HF_TOKEN && export HF_TOKEN && " if hf_token else "")
        + command
    )

    print(f"\nRunning on {cfg.host}:\n  {command}\n")
    failure: TranscriptionError | None = None
    try:
        _ssh(cfg, f"sh -lc {shlex.quote(wrapper)}", stdin_text=f"{hf_token}\n" if hf_token else None)
    except TranscriptionError as exc:
        failure = exc
        print(f"Remote run failed: {exc}", file=sys.stderr)
        print("Retrieving any transcripts it completed...", file=sys.stderr)

    remote_download_outputs(cfg, args)

    if failure is not None:
        print(
            f"\nRe-run the same command to resume: finished transcripts are kept on {cfg.host} and will be skipped.",
            file=sys.stderr,
        )
        return 1
    print("Done.")
    return 0


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
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Video or audio file to process (omit when using --source)",
    )

    # Batch mode
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Directory of media files to transcribe (recursive); requires --target",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help=(
            "Directory for batch transcripts. The --source tree is mirrored and "
            f"each file becomes <name>{TRANSCRIPT_SUFFIX}"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe files whose transcript already exists (default: skip them)",
    )

    # Remote execution
    parser.add_argument(
        "--run-in-remote",
        metavar="HOST",
        default=None,
        help=(
            "Run the transcription on HOST over SSH (an ssh alias or user@host). "
            "Sources are uploaded, the same CLI runs there, and transcripts are "
            "downloaded back. Works for a single file or with --source/--target"
        ),
    )
    parser.add_argument(
        "--remote-dir",
        default=DEFAULT_REMOTE_DIR,
        help=(
            f"Working directory on the remote host (default: {DEFAULT_REMOTE_DIR}). "
            "Kept between runs so re-provisioning is fast and an interrupted "
            "batch can resume"
        ),
    )

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
        choices=sorted(ALL_DENOISE_METHODS),
        default=None,
        help="Override denoise method (implies denoising ON)",
    )
    parser.add_argument(
        "--demucs-model",
        default=DEFAULT_DEMUCS_MODEL,
        help=(
            f"Demucs model for --denoise demucs (default: {DEFAULT_DEMUCS_MODEL}). "
            "htdemucs is a single network; htdemucs_ft is a bag of four, "
            "higher quality but roughly 4x slower"
        ),
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


@dataclass(frozen=True)
class LoadedModels:
    """Models loaded once and reused for every file in a run."""

    whisper: Any
    pipeline: Any = None  # pyannote; None when diarization is off
    demucs: Any = None  # None unless Demucs denoising is selected


def transcribe_to_text(
    source: Path,
    models: LoadedModels,
    config: TranscribeConfig,
    language: str,
    speakers: int | None,
) -> str:
    """Run the full pipeline for one file and return the formatted transcript.

    Shared by single-file and batch mode so both behave identically. Models are
    passed in already loaded; this function never loads weights.

    Raises:
        TranscriptionError: If any external step fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        raw_wav = tmp / "raw_audio.wav"
        clean_wav = tmp / "clean_audio.wav"

        extract_audio(source, raw_wav)
        inputs = AudioInputs(source=source, extracted=raw_wav)

        # Reject unreadable or empty audio before any expensive work.
        read_waveform(raw_wav)

        if config.denoise_enabled:
            apply_denoise(config.denoise_method, inputs, clean_wav, models.demucs)
            whisper_input = clean_wav
        else:
            print("Denoising disabled (profile/override).")
            whisper_input = raw_wav

        segments = transcribe_audio(
            models.whisper,
            whisper_input,
            language,
            config,
            word_timestamps=models.pipeline is not None,
        )
        if not segments:
            return ""

        if models.pipeline is None:
            return format_plain(segments)

        # Diarization uses the raw audio: denoising distorts voiceprints.
        turns = run_diarization(models.pipeline, raw_wav, speakers)
        if not turns:
            print("Warning: no speakers detected; emitting plain transcript.", file=sys.stderr)
            return format_plain(segments)

        speaker_map = build_speaker_map(turns)
        utterances = group_words_by_speaker(segments, turns, speaker_map)
        print(f"\nDetected {len(speaker_map)} speaker(s).")
        return format_diarized(utterances)


def _load_models(args: argparse.Namespace, config: TranscribeConfig, hf_token: str) -> LoadedModels:
    """Load every model the run needs, before any transcription starts.

    Loading first means a bad token, an unaccepted model licence or a mistyped
    Demucs model surfaces in seconds rather than after hours of transcription.
    """
    demucs_model = (
        load_demucs_model(args.demucs_model)
        if config.denoise_enabled and config.denoise_method == DEMUCS_METHOD
        else None
    )
    pipeline = load_diarization_pipeline(hf_token) if args.diarize else None
    return LoadedModels(
        whisper=load_whisper_model(args.whisper_model),
        pipeline=pipeline,
        demucs=demucs_model,
    )


def _run_single(args: argparse.Namespace, config: TranscribeConfig, hf_token: str) -> int:
    """Transcribe one file. Returns a process exit code."""
    models = _load_models(args, config, hf_token)
    text = transcribe_to_text(args.input, models, config, args.language, args.speakers)
    if not text:
        print("No speech detected in the audio.")
        return 0
    header = "DIARIZED TRANSCRIPT" if models.pipeline is not None else "TRANSCRIPT"
    _write_output(text, args.output, header)
    return 0


def _run_batch(args: argparse.Namespace, config: TranscribeConfig, hf_token: str) -> int:
    """Transcribe every media file under --source into --target.

    A failure on one file is reported and the batch continues: one unreadable
    recording should not discard the work already done on the others.
    """
    inputs = discover_inputs(args.source)
    if not inputs:
        print(f"No media files found under {args.source}")
        print(f"Recognised extensions: {', '.join(sorted(MEDIA_EXTENSIONS))}")
        return 0

    pending, skipped = plan_batch(inputs, args.source, args.target, args.overwrite)
    print(f"Found {len(inputs)} media file(s): {len(pending)} to transcribe, {len(skipped)} already done.")
    for item in skipped:
        print(f"  SKIP  {item.relative_to(args.source)} (transcript exists; --overwrite to redo)")
    if not pending:
        return 0

    models = _load_models(args, config, hf_token)

    succeeded = 0
    failed: list[tuple[str, str]] = []
    for index, (source, destination) in enumerate(pending, start=1):
        relative = source.relative_to(args.source)
        print(f"\n[{index}/{len(pending)}] {relative}")
        try:
            text = transcribe_to_text(source, models, config, args.language, args.speakers)
        except TranscriptionError as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed.append((str(relative), str(exc)))
            continue
        if not text:
            print("  No speech detected; writing an empty transcript.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(f"  -> {destination.relative_to(args.target)}")
        succeeded += 1

    print(format_batch_summary(succeeded, len(skipped), failed))
    return 1 if failed else 0


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    batch = args.source is not None
    if batch:
        if args.target is None:
            print("Error: --target is required with --source", file=sys.stderr)
            sys.exit(1)
        if args.output is not None:
            print("Error: --output applies to a single file; use --target with --source", file=sys.stderr)
            sys.exit(1)
    elif args.input is None:
        print("Error: provide an input file, or --source and --target for a folder", file=sys.stderr)
        sys.exit(1)
    elif not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.speakers is not None and args.speakers < 1:
        print("Error: --speakers must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Resolve the token early so we fail fast before heavy work. Remote runs
    # need it too, to forward to the host.
    hf_token = resolve_hf_token(args.hf_token) if args.diarize else ""

    if args.run_in_remote:
        # ffmpeg and the models are the remote's concern, not this machine's.
        try:
            sys.exit(run_remote(args, hf_token, Path(__file__).resolve().parents[2]))
        except TranscriptionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    check_ffmpeg()
    config = resolve_config(args.profile, overrides_from_args(args))

    try:
        sys.exit(_run_batch(args, config, hf_token) if batch else _run_single(args, config, hf_token))
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
