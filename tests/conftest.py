"""Shared fixtures and helpers for the test suite.

Tests exercise the pure logic (config resolution, alignment, formatting) and
never load Whisper, pyannote, audio, or the network.
"""

from __future__ import annotations

import pytest

from extract_speech.transcribe import SpeakerTurn


def make_word(word: str, start: float, end: float) -> dict:
    """Build a Whisper word-timestamp dict."""
    return {"word": word, "start": start, "end": end}


def make_segment(text: str, start: float, end: float, words: list[dict] | None = None) -> dict:
    """Build a Whisper segment dict, optionally with word timestamps."""
    seg = {"text": text, "start": start, "end": end}
    if words is not None:
        seg["words"] = words
    return seg


@pytest.fixture
def two_speaker_turns() -> list[SpeakerTurn]:
    """Turns: A speaks 0-10s, B speaks 10-20s, A again 20-30s."""
    return [
        SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=10.0, end=20.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=20.0, end=30.0, speaker="SPEAKER_00"),
    ]
