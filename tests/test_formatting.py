"""Tests for timestamp and transcript formatting."""

from __future__ import annotations

from extract_speech.transcribe import Utterance, format_diarized, format_plain, format_timestamp


def test_format_timestamp_zero():
    assert format_timestamp(0) == "00:00:00"


def test_format_timestamp_minutes_seconds():
    assert format_timestamp(75) == "00:01:15"


def test_format_timestamp_hours():
    assert format_timestamp(3661) == "01:01:01"


def test_format_timestamp_truncates_fractions():
    assert format_timestamp(59.9) == "00:00:59"


def test_format_plain():
    segments = [
        {"text": " Hola ", "start": 1.0, "end": 3.0},
        {"text": "Adios", "start": 3.0, "end": 5.0},
    ]
    out = format_plain(segments)
    assert out == "[00:00:01 --> 00:00:03] Hola\n[00:00:03 --> 00:00:05] Adios"


def test_format_diarized():
    utts = [
        Utterance(start=1.0, end=3.0, speaker=1, text="Hola"),
        Utterance(start=3.0, end=5.0, speaker=2, text="Adios"),
    ]
    out = format_diarized(utts)
    assert out == "[00:00:01] Persona 1: Hola\n[00:00:03] Persona 2: Adios"


def test_format_diarized_empty():
    assert format_diarized([]) == ""
