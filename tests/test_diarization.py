"""Tests for turning pyannote's diarization output into SpeakerTurn objects.

pyannote returns an Annotation whose ``itertracks`` yields
``(segment, track_id, label)`` triples. These tests use a stub with that shape,
so the conversion logic is covered without loading pyannote or torch.
"""

from __future__ import annotations

from extract_speech.transcribe import SpeakerTurn, turns_from_diarization


class StubSegment:
    """Stands in for pyannote's Segment, which exposes .start and .end."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class StubDiarization:
    """Stands in for pyannote's Annotation."""

    def __init__(self, tracks: list[tuple[StubSegment, str, str]]) -> None:
        self._tracks = tracks
        self.yield_label_arg: bool | None = None

    def itertracks(self, yield_label: bool = False):
        self.yield_label_arg = yield_label
        return iter(self._tracks)


def test_converts_tracks_to_speaker_turns():
    stub = StubDiarization(
        [
            (StubSegment(0.0, 1.5), "_", "SPEAKER_00"),
            (StubSegment(1.5, 3.0), "_", "SPEAKER_01"),
        ]
    )

    turns = turns_from_diarization(stub)

    assert turns == [
        SpeakerTurn(start=0.0, end=1.5, speaker="SPEAKER_00"),
        SpeakerTurn(start=1.5, end=3.0, speaker="SPEAKER_01"),
    ]


def test_requests_labels_from_pyannote():
    # Without yield_label=True, itertracks yields 2-tuples and unpacking breaks.
    stub = StubDiarization([])

    turns_from_diarization(stub)

    assert stub.yield_label_arg is True


def test_empty_diarization_yields_no_turns():
    assert turns_from_diarization(StubDiarization([])) == []


def test_preserves_input_order_including_repeated_speakers():
    stub = StubDiarization(
        [
            (StubSegment(0.0, 1.0), "_", "SPEAKER_00"),
            (StubSegment(1.0, 2.0), "_", "SPEAKER_01"),
            (StubSegment(2.0, 3.0), "_", "SPEAKER_00"),
        ]
    )

    turns = turns_from_diarization(stub)

    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [t.start for t in turns] == [0.0, 1.0, 2.0]
