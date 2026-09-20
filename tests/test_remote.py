"""Tests for remote-execution orchestration.

Only the pure parts are covered: path derivation, the provisioning allowlist,
and translating local arguments into the command the remote will run. No SSH,
no network.

The riskiest piece is argument translation. Two failure modes are silent rather
than loud: forgetting to strip --run-in-remote makes the remote invoke itself
recursively, and dropping a flag like --whisper-model makes the remote quietly
transcribe with different settings than asked for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from extract_speech.transcribe import (
    PROVISION_ALLOWLIST,
    RemoteConfig,
    TranscriptionError,
    build_download_args,
    build_parser,
    build_remote_argv,
    build_upload_args,
    expand_remote_root,
    join_remote_command,
    parse_remote_config,
)


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


# --- config / path derivation ----------------------------------------------


def test_remote_paths_are_derived_from_the_root():
    cfg = RemoteConfig(host="myhost", root="~/.sybil")

    assert cfg.project == "~/.sybil/project"
    assert cfg.inputs == "~/.sybil/inputs"
    assert cfg.outputs == "~/.sybil/outputs"


def test_remote_paths_stay_posix_even_though_the_client_may_not_be():
    # The remote is Linux; building its paths with the local separator would
    # break if this ever ran from Windows.
    cfg = RemoteConfig(host="h", root="/home/me/.sybil")

    assert "\\" not in cfg.inputs
    assert cfg.inputs.startswith("/home/me/.sybil")


def test_parse_remote_config_accepts_a_bare_alias():
    cfg = parse_remote_config("myhost", "~/.sybil")

    assert cfg.host == "myhost"


def test_parse_remote_config_accepts_user_at_host():
    assert parse_remote_config("me@box.example.com", "~/.sybil").host == "me@box.example.com"


def test_parse_remote_config_rejects_empty_host():
    with pytest.raises(TranscriptionError, match="host"):
        parse_remote_config("   ", "~/.sybil")


def test_parse_remote_config_rejects_empty_root():
    with pytest.raises(TranscriptionError, match="directory"):
        parse_remote_config("myhost", "  ")


# --- provisioning allowlist ------------------------------------------------


def test_provision_allowlist_carries_what_uv_sync_needs():
    assert "pyproject.toml" in PROVISION_ALLOWLIST
    assert "uv.lock" in PROVISION_ALLOWLIST
    assert "src" in PROVISION_ALLOWLIST


def test_provision_allowlist_never_carries_secrets_or_local_state():
    # An allowlist is used rather than rsync --exclude because excludes fail
    # open: forgetting one silently ships it. .env in particular must never
    # reach the remote -- the token is forwarded per-session instead.
    forbidden = {".env", ".venv", ".git", "output"}
    assert forbidden.isdisjoint(PROVISION_ALLOWLIST)


# --- argument translation: batch -------------------------------------------


def test_remote_argv_rewrites_source_and_target_to_remote_paths():
    cfg = RemoteConfig(host="h", root="~/.sybil")
    args = parse(["--source", "/local/in", "--target", "/local/out", "--run-in-remote", "h"])

    argv = build_remote_argv(args, cfg)

    assert "--source" in argv
    assert cfg.inputs in argv
    assert "--target" in argv
    assert cfg.outputs in argv
    assert "/local/in" not in argv
    assert "/local/out" not in argv


def test_remote_argv_strips_the_remote_flags_to_avoid_recursion():
    cfg = RemoteConfig(host="h", root="~/.sybil")
    args = parse(["--source", "/in", "--target", "/out", "--run-in-remote", "h", "--remote-dir", "~/x"])

    argv = build_remote_argv(args, cfg)

    assert "--run-in-remote" not in argv
    assert "--remote-dir" not in argv


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--whisper-model", "large"),
        ("--profile", "noisy"),
        ("--language", "en"),
        ("--speakers", "3"),
        ("--denoise", "demucs"),
        ("--demucs-model", "htdemucs_ft"),
    ],
)
def test_remote_argv_preserves_pipeline_flags(flag, value):
    # Losing any of these means the remote silently transcribes with different
    # settings than the user asked for.
    cfg = RemoteConfig(host="h", root="~/.sybil")
    args = parse(["--source", "/in", "--target", "/out", "--run-in-remote", "h", flag, value])

    argv = build_remote_argv(args, cfg)

    assert flag in argv
    assert value in argv


def test_remote_argv_preserves_no_diarize():
    cfg = RemoteConfig(host="h", root="~/.sybil")
    args = parse(["--source", "/in", "--target", "/out", "--run-in-remote", "h", "--no-diarize"])

    assert "--no-diarize" in build_remote_argv(args, cfg)


def test_remote_argv_preserves_overwrite():
    cfg = RemoteConfig(host="h", root="~/.sybil")
    args = parse(["--source", "/in", "--target", "/out", "--run-in-remote", "h", "--overwrite"])

    assert "--overwrite" in build_remote_argv(args, cfg)


def test_remote_argv_omits_overwrite_when_not_requested():
    # Batch resume depends on the remote skipping finished files, which
    # --overwrite would defeat.
    cfg = RemoteConfig(host="h", root="~/.sybil")
    args = parse(["--source", "/in", "--target", "/out", "--run-in-remote", "h"])

    assert "--overwrite" not in build_remote_argv(args, cfg)


def test_remote_argv_never_carries_the_token():
    cfg = RemoteConfig(host="h", root="~/.sybil")
    args = parse(["--source", "/in", "--target", "/out", "--run-in-remote", "h", "--hf-token", "hf_secret"])

    argv = build_remote_argv(args, cfg)

    # The token goes over stdin into the process environment, never into a
    # command line where the remote's `ps` would expose it.
    assert "hf_secret" not in argv
    assert "--hf-token" not in argv


# --- argument translation: single file --------------------------------------


def test_remote_argv_for_a_single_file_uses_the_staged_input_name():
    cfg = RemoteConfig(host="h", root="~/.sybil")
    args = parse(["/local/videos/My Clip.mp4", "--run-in-remote", "h"])

    argv = build_remote_argv(args, cfg)

    assert f"{cfg.inputs}/My Clip.mp4" in argv
    assert "/local/videos/My Clip.mp4" not in argv


def test_remote_argv_for_a_single_file_redirects_output_into_the_remote_dir():
    cfg = RemoteConfig(host="h", root="~/.sybil")
    args = parse(["/local/clip.mp4", "-o", "/local/out.txt", "--run-in-remote", "h"])

    argv = build_remote_argv(args, cfg)

    assert "--output" in argv or "-o" in argv
    assert "/local/out.txt" not in argv
    assert any(a.startswith(cfg.outputs) for a in argv)


# --- shell quoting ---------------------------------------------------------


def test_join_quotes_paths_containing_spaces():
    # ssh passes a command string to the remote shell, so spaces must survive.
    # The real corpus has a "Living Room" directory.
    command = join_remote_command(["extract-speech", "--source", "~/.sybil/inputs/Living Room"])

    assert "Living Room" in command
    assert "'~/.sybil/inputs/Living Room'" in command or '"~/.sybil/inputs/Living Room"' in command


def test_join_leaves_simple_tokens_unquoted():
    assert join_remote_command(["extract-speech", "--profile", "noisy"]) == "extract-speech --profile noisy"


def test_join_neutralises_shell_metacharacters():
    command = join_remote_command(["extract-speech", "--source", "/tmp/a; rm -rf ~"])

    assert "; rm -rf ~" not in command.replace("'/tmp/a; rm -rf ~'", "")


# --- remote home expansion -------------------------------------------------


def test_tilde_root_is_expanded_to_the_remote_home():
    # Quoting is mandatory for ssh, but quoting "~" makes the remote shell treat
    # it as a literal directory name, so a "~" folder is created in the login
    # directory instead of using $HOME. Resolve it before any quoting happens.
    assert expand_remote_root("~/.sybil", "/home/ldeltoro") == "/home/ldeltoro/.sybil"


def test_bare_tilde_root_becomes_the_home_itself():
    assert expand_remote_root("~", "/home/me") == "/home/me"


def test_absolute_root_is_left_alone():
    assert expand_remote_root("/opt/sybil", "/home/me") == "/opt/sybil"


def test_relative_root_is_left_alone():
    # A relative path is resolved by the remote shell against the login dir,
    # which is the documented behaviour of ssh commands.
    assert expand_remote_root("work/sybil", "/home/me") == "work/sybil"


def test_tilde_user_form_is_not_mangled():
    # "~other/dir" means another user's home; rewriting it against our own home
    # would silently point somewhere wrong.
    assert expand_remote_root("~other/dir", "/home/me") == "~other/dir"


def test_expanded_root_produces_quotable_paths():
    cfg = RemoteConfig(host="h", root=expand_remote_root("~/.sybil", "/home/me"))

    assert join_remote_command(["mkdir", "-p", cfg.project]) == "mkdir -p /home/me/.sybil/project"


# --- transfer argument construction ----------------------------------------


def test_input_upload_mirrors_local_by_deleting_stale_remote_files():
    # Without --delete, a file removed locally lingers on the remote and keeps
    # being "found" on every subsequent run -- phantom work, and phantom
    # failures if it was the bad file you just removed.
    cfg = RemoteConfig(host="h", root="/home/me/.sybil")

    args = build_upload_args(cfg, Path("/local/src"), is_directory=True)

    assert "--delete" in args
    assert f"h:{cfg.inputs}/" in args


def test_single_file_upload_does_not_delete():
    # A lone file is added to the staging dir; deleting its siblings would
    # discard inputs the user may still be using for a resumed batch.
    cfg = RemoteConfig(host="h", root="/home/me/.sybil")

    args = build_upload_args(cfg, Path("/local/clip.mp4"), is_directory=False)

    assert "--delete" not in args


def test_output_download_never_deletes():
    # --delete here would destroy transcripts, which is the opposite of the
    # "retrieve whatever finished" guarantee.
    cfg = RemoteConfig(host="h", root="/home/me/.sybil")

    args = build_download_args(cfg, Path("/local/out"))

    assert "--delete" not in args


def test_uploads_follow_symlinks_rather_than_copying_the_link():
    # The Conversaciones directory is entirely symlinks into an archive folder.
    # Plain "rsync -a" preserves the link, so the remote receives a pointer to a
    # macOS path that does not exist there and the run fails with
    # "File not found" after the upload has already reported success.
    cfg = RemoteConfig(host="h", root="/home/me/.sybil")

    for is_directory in (True, False):
        args = build_upload_args(cfg, Path("/local/src"), is_directory=is_directory)
        assert "-L" in args or "--copy-links" in args, f"is_directory={is_directory}"
