"""Tests for profile resolution and CLI-override precedence."""

from __future__ import annotations

import pytest

from transcribe import (
    PROFILES,
    ConfigOverrides,
    build_parser,
    overrides_from_args,
    resolve_config,
)


def test_clean_profile_defaults():
    cfg = resolve_config("clean", ConfigOverrides())
    assert cfg.denoise_enabled is False
    assert cfg.temperature == (0.0,)
    assert cfg.no_speech_threshold == 0.6
    assert cfg.condition_on_previous_text is True
    # Greedy decoding: beam search caused a hallucination cascade on clean audio.
    assert cfg.beam_size is None
    assert cfg.best_of is None


def test_noisy_profile_defaults():
    cfg = resolve_config("noisy", ConfigOverrides())
    assert cfg.denoise_enabled is True
    assert cfg.denoise_method == "loudnorm"
    assert cfg.temperature == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert cfg.no_speech_threshold == 0.4
    assert cfg.condition_on_previous_text is False
    assert cfg.beam_size == 5
    assert cfg.best_of == 5


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="Unknown profile"):
        resolve_config("bogus", ConfigOverrides())


def test_no_denoise_overrides_noisy_profile():
    cfg = resolve_config("noisy", ConfigOverrides(denoise_enabled=False))
    assert cfg.denoise_enabled is False
    # Other noisy knobs are untouched by a denoise override.
    assert cfg.no_speech_threshold == 0.4


def test_denoise_method_override_enables_denoise_on_clean():
    cfg = resolve_config("clean", ConfigOverrides(denoise_enabled=True, denoise_method="ffmpeg"))
    assert cfg.denoise_enabled is True
    assert cfg.denoise_method == "ffmpeg"


def test_condition_override_true_on_noisy():
    cfg = resolve_config("noisy", ConfigOverrides(condition_on_previous_text=True))
    assert cfg.condition_on_previous_text is True


def test_none_overrides_leave_profile_untouched():
    base = PROFILES["clean"]
    cfg = resolve_config("clean", ConfigOverrides())
    assert cfg == base


# --- CLI parsing / override translation ---


def test_cli_defaults():
    args = build_parser().parse_args(["video.mp4"])
    assert args.profile == "clean"
    assert args.whisper_model == "medium"
    assert args.language == "es"
    assert args.diarize is True
    assert args.speakers is None


def test_cli_no_diarize():
    args = build_parser().parse_args(["video.mp4", "--no-diarize"])
    assert args.diarize is False


def test_cli_model_alias_maps_to_whisper_model():
    args = build_parser().parse_args(["video.mp4", "--model", "small"])
    assert args.whisper_model == "small"


def test_cli_whisper_model_flag():
    args = build_parser().parse_args(["video.mp4", "--whisper-model", "large"])
    assert args.whisper_model == "large"


def test_cli_speakers_forced():
    args = build_parser().parse_args(["video.mp4", "--speakers", "3"])
    assert args.speakers == 3


def test_overrides_no_denoise_wins():
    args = build_parser().parse_args(["video.mp4", "--no-denoise"])
    ov = overrides_from_args(args)
    assert ov.denoise_enabled is False


def test_overrides_denoise_method_implies_enabled():
    args = build_parser().parse_args(["video.mp4", "--denoise", "spectral"])
    ov = overrides_from_args(args)
    assert ov.denoise_enabled is True
    assert ov.denoise_method == "spectral"


def test_overrides_no_denoise_beats_denoise_method():
    args = build_parser().parse_args(["video.mp4", "--denoise", "spectral", "--no-denoise"])
    ov = overrides_from_args(args)
    assert ov.denoise_enabled is False


def test_overrides_condition_flags():
    args = build_parser().parse_args(["video.mp4", "--no-condition-on-previous"])
    assert overrides_from_args(args).condition_on_previous_text is False
    args = build_parser().parse_args(["video.mp4", "--condition-on-previous"])
    assert overrides_from_args(args).condition_on_previous_text is True
    args = build_parser().parse_args(["video.mp4"])
    assert overrides_from_args(args).condition_on_previous_text is None
