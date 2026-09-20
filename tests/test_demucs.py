"""Tests for the pure logic behind Demucs vocal isolation.

The separation itself (an 80 MB model on the GPU) is not unit-tested; the
normalisation, stem selection and model-name validation around it are, since
those are where the bugs would be silent.

Normalisation mirrors demucs' own reference implementation in ``separate.py``:

    ref = wav.mean(0); wav -= ref.mean(); wav /= ref.std()
    ...  sources *= ref.std(); sources += ref.mean()

Getting it wrong does not raise - it quietly changes the model's input scale
and degrades separation, so it is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from extract_speech.transcribe import (
    AudioInputs,
    TranscriptionError,
    denormalise_stems,
    normalise_for_demucs,
    stem_index,
    validate_demucs_model,
)

DEMUCS_SOURCES = ["drums", "bass", "other", "vocals"]


# --- AudioInputs -----------------------------------------------------------


def test_audio_inputs_holds_source_and_extracted(tmp_path):
    inputs = AudioInputs(source=tmp_path / "video.mp4", extracted=tmp_path / "raw.wav")

    assert inputs.source.name == "video.mp4"
    assert inputs.extracted.name == "raw.wav"


def test_audio_inputs_is_immutable(tmp_path):
    inputs = AudioInputs(source=tmp_path / "a.mp4", extracted=tmp_path / "b.wav")

    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        inputs.source = tmp_path / "c.mp4"  # type: ignore[misc]


# --- normalisation ---------------------------------------------------------


def test_normalise_centres_and_scales():
    # Identical channels, so the channel average equals the channel itself and
    # the expected scale is unambiguous. (See the next test for why independent
    # channels would not be.)
    rng = np.random.default_rng(0)
    mono = rng.normal(loc=3.0, scale=7.0, size=5000).astype(np.float32)
    wav = np.stack([mono, mono])

    normalised, mean, std = normalise_for_demucs(wav)

    # The reference signal is the channel average, so that is what becomes
    # zero-mean / unit-variance -- not each channel independently.
    ref = normalised.mean(axis=0)
    assert ref.mean() == pytest.approx(0.0, abs=1e-4)
    assert ref.std() == pytest.approx(1.0, abs=1e-3)
    assert mean == pytest.approx(3.0, abs=0.2)
    assert std == pytest.approx(7.0, abs=0.3)


def test_statistics_come_from_the_channel_average_not_per_channel():
    # demucs computes ref = wav.mean(0) before taking std, so two *independent*
    # channels of scale s average to scale s/sqrt(2). Pinned because it is easy
    # to assume the scalars describe a single channel -- they do not.
    rng = np.random.default_rng(0)
    wav = rng.normal(loc=0.0, scale=7.0, size=(2, 20000)).astype(np.float32)

    _, _, std = normalise_for_demucs(wav)

    assert std == pytest.approx(7.0 / np.sqrt(2), rel=0.05)


def test_normalise_then_denormalise_round_trips():
    rng = np.random.default_rng(1)
    wav = rng.normal(loc=-2.0, scale=0.5, size=(2, 4000)).astype(np.float32)

    normalised, mean, std = normalise_for_demucs(wav)
    restored = denormalise_stems(normalised, mean, std)

    np.testing.assert_allclose(restored, wav, rtol=1e-4, atol=1e-4)


def test_normalise_returns_plain_floats():
    # The scalars are reapplied to a torch tensor later; numpy scalars would
    # silently promote dtypes there.
    wav = np.ones((2, 100), dtype=np.float32)

    _, mean, std = normalise_for_demucs(wav)

    assert isinstance(mean, float)
    assert isinstance(std, float)


def test_silent_audio_does_not_divide_by_zero():
    # The Tapo corpus contains near-silent clips; std == 0 would yield NaNs and
    # demucs would return garbage rather than fail loudly.
    wav = np.zeros((2, 1000), dtype=np.float32)

    normalised, mean, std = normalise_for_demucs(wav)

    assert np.all(np.isfinite(normalised))
    assert std == 1.0
    assert mean == 0.0


def test_constant_signal_does_not_divide_by_zero():
    wav = np.full((2, 1000), 0.25, dtype=np.float32)

    normalised, _, std = normalise_for_demucs(wav)

    assert np.all(np.isfinite(normalised))
    assert std == 1.0


def test_denormalise_is_a_no_op_for_identity_scalars():
    stems = np.arange(12, dtype=np.float32).reshape(3, 4)

    np.testing.assert_allclose(denormalise_stems(stems, 0.0, 1.0), stems)


# --- stem selection --------------------------------------------------------


def test_stem_index_finds_vocals():
    assert stem_index(DEMUCS_SOURCES, "vocals") == 3


def test_stem_index_finds_first_stem():
    assert stem_index(DEMUCS_SOURCES, "drums") == 0


def test_stem_index_unknown_stem_raises_and_lists_options():
    with pytest.raises(TranscriptionError, match="vocals"):
        stem_index(DEMUCS_SOURCES, "accordion")


# --- model name validation -------------------------------------------------


def test_validate_demucs_model_accepts_known_name():
    assert validate_demucs_model("htdemucs", ["htdemucs", "htdemucs_ft"]) == "htdemucs"


def test_validate_demucs_model_rejects_unknown_name():
    # Catching a typo before transcription matters: otherwise it surfaces only
    # after Whisper has already run.
    with pytest.raises(TranscriptionError, match="htdemucs_ft"):
        validate_demucs_model("htdemuc", ["htdemucs", "htdemucs_ft"])


def test_validate_demucs_model_error_names_the_bad_value():
    with pytest.raises(TranscriptionError, match="nope"):
        validate_demucs_model("nope", ["htdemucs"])
