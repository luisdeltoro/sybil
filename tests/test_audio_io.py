"""Tests for reading extracted audio into an in-memory waveform.

Diarization is handed a waveform rather than a file path, so pyannote never
needs its own audio decoder (torchcodec). These tests pin the contract that
pyannote's ``validate_file`` enforces: a 2-D ``(channel, time)`` array whose
channel count does not exceed its sample count.

They write real WAV files via soundfile but never import torch, whisper or
pyannote, so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from extract_speech.transcribe import TranscriptionError, read_waveform


def write_wav(path, data: np.ndarray, rate: int = 16000) -> None:
    """Write ``data`` to ``path`` as a WAV file (soundfile wants time-major)."""
    sf.write(str(path), data, rate)


def test_mono_returns_channel_first_2d_shape(tmp_path):
    wav = tmp_path / "mono.wav"
    write_wav(wav, np.zeros(8000, dtype=np.float32))

    waveform, _ = read_waveform(wav)

    assert waveform.ndim == 2, "pyannote requires a 2-D (channel, time) array"
    assert waveform.shape == (1, 8000)


def test_returns_the_files_sample_rate(tmp_path):
    wav = tmp_path / "rate.wav"
    write_wav(wav, np.zeros(2205, dtype=np.float32), rate=22050)

    _, rate = read_waveform(wav)

    assert rate == 22050


def test_returns_float32(tmp_path):
    # The file is 16-bit PCM on disk; it must come back as float32 for torch.
    wav = tmp_path / "pcm16.wav"
    sf.write(str(wav), np.zeros(1000, dtype=np.int16), 16000, subtype="PCM_16")

    waveform, _ = read_waveform(wav)

    assert waveform.dtype == np.float32


def test_stereo_is_returned_channel_first(tmp_path):
    # soundfile reads stereo as (time, channel); pyannote needs it transposed.
    wav = tmp_path / "stereo.wav"
    write_wav(wav, np.zeros((4000, 2), dtype=np.float32))

    waveform, _ = read_waveform(wav)

    assert waveform.shape == (2, 4000)


def test_channel_count_never_exceeds_sample_count(tmp_path):
    # pyannote rejects any array where shape[0] > shape[1]. A very short
    # stereo clip is the realistic way to trip that, so verify orientation
    # holds even when the sample count is small.
    wav = tmp_path / "short_stereo.wav"
    write_wav(wav, np.zeros((3, 2), dtype=np.float32))

    waveform, _ = read_waveform(wav)

    assert waveform.shape[0] <= waveform.shape[1]
    assert waveform.shape == (2, 3)


def test_preserves_sample_values(tmp_path):
    wav = tmp_path / "values.wav"
    samples = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    write_wav(wav, samples)

    waveform, _ = read_waveform(wav)

    np.testing.assert_allclose(waveform[0], samples, atol=1e-6)


def test_missing_file_raises_transcription_error(tmp_path):
    with pytest.raises(TranscriptionError, match="Could not read audio"):
        read_waveform(tmp_path / "does_not_exist.wav")


def test_empty_audio_raises_transcription_error(tmp_path):
    # A header-only WAV decodes fine but yields no samples; diarization on it
    # would fail deep inside pyannote, so reject it early with a clear message.
    wav = tmp_path / "empty.wav"
    write_wav(wav, np.zeros(0, dtype=np.float32))

    with pytest.raises(TranscriptionError, match="no audio samples"):
        read_waveform(wav)
