"""Tests for batch discovery, output path mapping and run planning.

These are the parts of batch mode where a bug is silent rather than loud: a
wrong path mapping overwrites a transcript instead of failing, and a wrong
skip decision quietly does nothing. The Tapo corpus is the motivating shape --
the same filename appears under several subdirectories.
"""

from __future__ import annotations

import pytest

from extract_speech.transcribe import (
    MEDIA_EXTENSIONS,
    TranscriptionError,
    discover_inputs,
    format_batch_summary,
    output_path_for,
    plan_batch,
)


def touch(path, name: str):
    """Create ``name`` (with parents) under ``path`` and return it."""
    f = path / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x00")
    return f


# --- discovery -------------------------------------------------------------


def test_discovers_media_recursively(tmp_path):
    touch(tmp_path, "20260522/Bedroom/a.mp4")
    touch(tmp_path, "20260522/Living Room/b.mp4")
    touch(tmp_path, "20260529/Bedroom/c.mkv")

    found = discover_inputs(tmp_path)

    assert [p.name for p in found] == ["a.mp4", "b.mp4", "c.mkv"]


def test_discovery_is_sorted_for_deterministic_batches(tmp_path):
    for name in ("z.mp4", "a.mp4", "m.mp4"):
        touch(tmp_path, name)

    assert [p.name for p in discover_inputs(tmp_path)] == ["a.mp4", "m.mp4", "z.mp4"]


def test_discovery_ignores_non_media_files(tmp_path):
    touch(tmp_path, "keep.mp4")
    touch(tmp_path, "notes.txt")
    touch(tmp_path, "cover.jpg")
    touch(tmp_path, "transcript_transcript.txt")

    assert [p.name for p in discover_inputs(tmp_path)] == ["keep.mp4"]


def test_discovery_accepts_audio_as_well_as_video(tmp_path):
    touch(tmp_path, "a.wav")
    touch(tmp_path, "b.m4a")
    touch(tmp_path, "c.mp4")

    assert len(discover_inputs(tmp_path)) == 3


def test_discovery_is_case_insensitive_on_extension(tmp_path):
    touch(tmp_path, "SHOUTY.MP4")

    assert [p.name for p in discover_inputs(tmp_path)] == ["SHOUTY.MP4"]


def test_discovery_of_empty_tree_returns_empty_list(tmp_path):
    assert discover_inputs(tmp_path) == []


def test_discovery_rejects_missing_directory(tmp_path):
    with pytest.raises(TranscriptionError, match="not a directory"):
        discover_inputs(tmp_path / "nope")


def test_media_extensions_are_lowercase_and_dotted():
    assert all(e.startswith(".") and e == e.lower() for e in MEDIA_EXTENSIONS)


# --- output path mapping ---------------------------------------------------


def test_output_mirrors_subdirectories(tmp_path):
    src, tgt = tmp_path / "src", tmp_path / "out"
    inp = touch(src, "20260522/Bedroom/clip.mp4")

    assert output_path_for(inp, src, tgt) == tgt / "20260522/Bedroom/clip_transcript.txt"


def test_output_at_source_root_has_no_extra_nesting(tmp_path):
    src, tgt = tmp_path / "src", tmp_path / "out"
    inp = touch(src, "clip.mp4")

    assert output_path_for(inp, src, tgt) == tgt / "clip_transcript.txt"


def test_identical_stems_in_different_subdirs_do_not_collide(tmp_path):
    # The real corpus has the same filename under Bedroom/ and Living Room/.
    # Flattening would silently overwrite one transcript with the other.
    src, tgt = tmp_path / "src", tmp_path / "out"
    a = touch(src, "Bedroom/1779486332469_0.mp4")
    b = touch(src, "Living Room/1779486332469_0.mp4")

    assert output_path_for(a, src, tgt) != output_path_for(b, src, tgt)


def test_output_preserves_spaces_in_directory_names(tmp_path):
    src, tgt = tmp_path / "src", tmp_path / "out"
    inp = touch(src, "Living Room/clip.mp4")

    assert output_path_for(inp, src, tgt).parent.name == "Living Room"


def test_output_replaces_only_the_final_extension(tmp_path):
    src, tgt = tmp_path / "src", tmp_path / "out"
    inp = touch(src, "holiday.2026.mp4")

    assert output_path_for(inp, src, tgt).name == "holiday.2026_transcript.txt"


def test_output_rejects_input_outside_the_source_root(tmp_path):
    src, tgt = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    outside = touch(tmp_path, "elsewhere/clip.mp4")

    with pytest.raises(TranscriptionError, match="outside"):
        output_path_for(outside, src, tgt)


# --- planning --------------------------------------------------------------


def test_plan_skips_inputs_whose_transcript_exists(tmp_path):
    src, tgt = tmp_path / "src", tmp_path / "out"
    done = touch(src, "done.mp4")
    todo = touch(src, "todo.mp4")
    out = output_path_for(done, src, tgt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("already transcribed", encoding="utf-8")

    pending, skipped = plan_batch([done, todo], src, tgt, overwrite=False)

    assert [i.name for i, _ in pending] == ["todo.mp4"]
    assert [i.name for i in skipped] == ["done.mp4"]


def test_plan_with_overwrite_skips_nothing(tmp_path):
    src, tgt = tmp_path / "src", tmp_path / "out"
    done = touch(src, "done.mp4")
    out = output_path_for(done, src, tgt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("stale", encoding="utf-8")

    pending, skipped = plan_batch([done], src, tgt, overwrite=True)

    assert len(pending) == 1
    assert skipped == []


def test_plan_pairs_each_input_with_its_output(tmp_path):
    src, tgt = tmp_path / "src", tmp_path / "out"
    inp = touch(src, "Bedroom/clip.mp4")

    pending, _ = plan_batch([inp], src, tgt, overwrite=False)

    assert pending == [(inp, tgt / "Bedroom/clip_transcript.txt")]


def test_plan_of_nothing_is_empty(tmp_path):
    assert plan_batch([], tmp_path, tmp_path / "out", overwrite=False) == ([], [])


# --- summary ---------------------------------------------------------------


def test_summary_counts_each_outcome():
    summary = format_batch_summary(succeeded=5, skipped=2, failed=[])

    assert "5" in summary and "2" in summary


def test_summary_names_failed_inputs():
    summary = format_batch_summary(succeeded=1, skipped=0, failed=[("bad.mp4", "boom")])

    assert "bad.mp4" in summary
    assert "boom" in summary


def test_summary_keeps_each_failure_on_one_line():
    # ffmpeg errors are multi-line; letting them through makes a summary of a
    # 16-file batch unreadable, which defeats the point of having one. Assert on
    # the total line count, since checking only the line containing the filename
    # would pass even when the extra lines leak through underneath it.
    reason = "Audio extraction failed:\nmoov atom not found\nError opening input"
    summary = format_batch_summary(succeeded=0, skipped=0, failed=[("bad.mp4", reason)])

    body = [line for line in summary.splitlines() if line.strip()]
    # 3 header lines (rule, counts, rule) + exactly 1 line per failure.
    assert len(body) == 4, f"expected one line per failure, got:\n{summary}"
    assert "bad.mp4" in body[-1]
    assert "Audio extraction failed:" in body[-1]
    assert "moov atom not found" not in summary
