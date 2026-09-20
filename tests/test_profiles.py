"""Tests for profile resolution and CLI-override precedence."""

from __future__ import annotations

import pytest

from extract_speech.transcribe import (
    ALL_DENOISE_METHODS,
    PROFILES,
    ConfigOverrides,
    build_parser,
    overrides_from_args,
    resolve_config,
)


def test_clean_profile_defaults():
    cfg = resolve_config("clean", ConfigOverrides())
    assert cfg.denoise_enabled is False
    # Anti-hallucination guards: temperature ladder + condition OFF + logprob gate.
    # (condition=True caused a repetition cascade on sparse/quiet openings.)
    assert cfg.temperature == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert cfg.no_speech_threshold == 0.6
    assert cfg.condition_on_previous_text is False
    assert cfg.logprob_threshold == -1.0
    # Greedy decoding: beam search caused a hallucination cascade on clean audio.
    assert cfg.beam_size is None
    assert cfg.best_of is None


def test_noisy_profile_defaults():
    cfg = resolve_config("noisy", ConfigOverrides())
    assert cfg.denoise_enabled is True
    # Demucs vocal isolation, not loudnorm: on noisy recordings it removed
    # 34-43% of non-vocal energy and collapsed hallucinated segment runs
    # (17 -> 1 on a Tapo clip). It costs ~0.15x realtime, so `--denoise
    # loudnorm` remains available when that trade is not wanted.
    assert cfg.denoise_method == "demucs"
    assert cfg.temperature == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert cfg.no_speech_threshold == 0.4
    assert cfg.condition_on_previous_text is False
    assert cfg.beam_size == 5
    assert cfg.best_of == 5


def test_all_denoise_methods_are_selectable():
    assert set(ALL_DENOISE_METHODS) == {"loudnorm", "spectral", "ffmpeg", "demucs"}


def test_every_profile_names_a_real_denoise_method():
    for name, cfg in PROFILES.items():
        assert cfg.denoise_method in ALL_DENOISE_METHODS, name


def test_cli_denoise_choices_match_the_registry():
    # A method that exists but is not offered on the CLI is unreachable.
    action = next(a for a in build_parser()._actions if a.dest == "denoise")
    assert set(action.choices or []) == set(ALL_DENOISE_METHODS)


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


def test_cli_demucs_model_defaults_to_htdemucs():
    # htdemucs is a single network; htdemucs_ft is a bag of four and ~4x slower.
    args = build_parser().parse_args(["video.mp4"])
    assert args.demucs_model == "htdemucs"


def test_cli_demucs_model_can_be_overridden():
    args = build_parser().parse_args(["video.mp4", "--demucs-model", "htdemucs_ft"])
    assert args.demucs_model == "htdemucs_ft"


def test_cli_denoise_accepts_demucs():
    args = build_parser().parse_args(["video.mp4", "--denoise", "demucs"])
    ov = overrides_from_args(args)
    assert ov.denoise_enabled is True
    assert ov.denoise_method == "demucs"


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
