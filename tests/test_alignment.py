"""Tests for diarization alignment logic (speaker_at, grouping, speaker map)."""

from __future__ import annotations

from conftest import make_segment, make_word

from extract_speech.transcribe import (
    SpeakerTurn,
    build_speaker_map,
    group_words_by_speaker,
    speaker_at,
)

# --- speaker_at ---


def test_speaker_at_inside_turn(two_speaker_turns):
    assert speaker_at(two_speaker_turns, 5.0) == "SPEAKER_00"
    assert speaker_at(two_speaker_turns, 15.0) == "SPEAKER_01"
    assert speaker_at(two_speaker_turns, 25.0) == "SPEAKER_00"


def test_speaker_at_boundary_prefers_containing_turn(two_speaker_turns):
    # At exactly 10.0, the second turn [10, 20] contains the instant.
    assert speaker_at(two_speaker_turns, 10.0) == "SPEAKER_01"


def test_speaker_at_in_gap_uses_nearest():
    turns = [
        SpeakerTurn(0.0, 5.0, "SPEAKER_00"),
        SpeakerTurn(10.0, 15.0, "SPEAKER_01"),
    ]
    # 6.0 is nearer to the first turn's end (5.0) than the second's start (10.0).
    assert speaker_at(turns, 6.0) == "SPEAKER_00"
    # 9.0 is nearer to the second turn's start.
    assert speaker_at(turns, 9.0) == "SPEAKER_01"


def test_speaker_at_empty_turns_returns_none():
    assert speaker_at([], 3.0) is None


def test_speaker_at_overlapping_turns_picks_greater_overlap():
    # Overlapping turns; instant 4.0 sits only inside the second turn.
    turns = [
        SpeakerTurn(0.0, 3.0, "SPEAKER_00"),
        SpeakerTurn(3.5, 8.0, "SPEAKER_01"),
    ]
    assert speaker_at(turns, 4.0) == "SPEAKER_01"


# --- build_speaker_map ---


def test_speaker_map_numbers_by_first_appearance():
    turns = [
        SpeakerTurn(10.0, 20.0, "SPEAKER_01"),
        SpeakerTurn(0.0, 10.0, "SPEAKER_00"),
        SpeakerTurn(20.0, 30.0, "SPEAKER_01"),
    ]
    mapping = build_speaker_map(turns)
    # SPEAKER_00 starts earliest -> Persona 1.
    assert mapping == {"SPEAKER_00": 1, "SPEAKER_01": 2}


def test_speaker_map_three_speakers():
    turns = [
        SpeakerTurn(0.0, 1.0, "A"),
        SpeakerTurn(1.0, 2.0, "B"),
        SpeakerTurn(2.0, 3.0, "C"),
    ]
    assert build_speaker_map(turns) == {"A": 1, "B": 2, "C": 3}


# --- group_words_by_speaker ---


def test_grouping_merges_consecutive_same_speaker(two_speaker_turns):
    words = [make_word(" Hola", 1.0, 2.0), make_word(" Luis", 2.0, 3.0)]
    segments = [make_segment("Hola Luis", 1.0, 3.0, words)]
    smap = build_speaker_map(two_speaker_turns)

    utts = group_words_by_speaker(segments, two_speaker_turns, smap)
    assert len(utts) == 1
    assert utts[0].speaker == 1
    assert utts[0].text == "Hola Luis"


def test_grouping_splits_within_a_single_whisper_segment(two_speaker_turns):
    # One Whisper segment spans a speaker change at t=10 (the PoC failure mode).
    words = [
        make_word(" Si", 8.0, 9.0),  # SPEAKER_00 -> Persona 1
        make_word(" dime", 9.0, 9.8),  # Persona 1
        make_word(" Claro", 12.0, 13.0),  # SPEAKER_01 -> Persona 2
        make_word(" vale", 13.0, 14.0),  # Persona 2
    ]
    segments = [make_segment("Si dime Claro vale", 8.0, 14.0, words)]
    smap = build_speaker_map(two_speaker_turns)

    utts = group_words_by_speaker(segments, two_speaker_turns, smap)
    assert [u.speaker for u in utts] == [1, 2]
    assert utts[0].text == "Si dime"
    assert utts[1].text == "Claro vale"


def test_grouping_alternating_speakers(two_speaker_turns):
    words = [
        make_word(" A1", 1.0, 2.0),  # P1
        make_word(" B1", 11.0, 12.0),  # P2
        make_word(" A2", 21.0, 22.0),  # P1 (third turn)
    ]
    segments = [make_segment("A1 B1 A2", 1.0, 22.0, words)]
    smap = build_speaker_map(two_speaker_turns)

    utts = group_words_by_speaker(segments, two_speaker_turns, smap)
    assert [u.speaker for u in utts] == [1, 2, 1]


def test_grouping_falls_back_to_segment_level_without_words(two_speaker_turns):
    # No word timestamps -> whole segment treated as one unit at its midpoint.
    segments = [make_segment("bloque entero", 11.0, 19.0)]  # midpoint 15 -> P2
    smap = build_speaker_map(two_speaker_turns)

    utts = group_words_by_speaker(segments, two_speaker_turns, smap)
    assert len(utts) == 1
    assert utts[0].speaker == 2
    assert utts[0].text == "bloque entero"


def test_grouping_empty_segments():
    assert group_words_by_speaker([], [], {}) == []


def test_grouping_records_span():
    words = [make_word(" uno", 1.0, 2.0), make_word(" dos", 2.0, 4.0)]
    segments = [make_segment("uno dos", 1.0, 4.0, words)]
    smap = {"SPEAKER_00": 1}
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00")]

    utts = group_words_by_speaker(segments, turns, smap)
    assert utts[0].start == 1.0
    assert utts[0].end == 4.0
