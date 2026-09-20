"""Tests for .env loading and HF token resolution."""

from __future__ import annotations

import pytest

from extract_speech.transcribe import TranscriptionError, load_dotenv, resolve_hf_token


def test_load_dotenv_missing_file_is_noop(tmp_path):
    load_dotenv(tmp_path / "does_not_exist.env")  # must not raise


def test_load_dotenv_sets_new_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_DOTENV_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("PYTEST_DOTENV_KEY=abc123\n", encoding="utf-8")
    load_dotenv(env)
    assert __import__("os").environ["PYTEST_DOTENV_KEY"] == "abc123"


def test_load_dotenv_ignores_comments_and_blanks(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_DOTENV_REAL", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n\n   \nPYTEST_DOTENV_REAL=value\n# trailing comment\n",
        encoding="utf-8",
    )
    load_dotenv(env)
    assert __import__("os").environ["PYTEST_DOTENV_REAL"] == "value"


def test_load_dotenv_strips_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_DOTENV_QUOTED", raising=False)
    env = tmp_path / ".env"
    env.write_text('PYTEST_DOTENV_QUOTED="quoted value"\n', encoding="utf-8")
    load_dotenv(env)
    assert __import__("os").environ["PYTEST_DOTENV_QUOTED"] == "quoted value"


def test_load_dotenv_does_not_overwrite_existing(tmp_path, monkeypatch):
    # Real environment variables must win over .env.
    monkeypatch.setenv("PYTEST_DOTENV_EXISTING", "from_env")
    env = tmp_path / ".env"
    env.write_text("PYTEST_DOTENV_EXISTING=from_file\n", encoding="utf-8")
    load_dotenv(env)
    assert __import__("os").environ["PYTEST_DOTENV_EXISTING"] == "from_env"


def test_resolve_hf_token_prefers_cli(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env_tok")
    assert resolve_hf_token("cli_tok") == "cli_tok"


def test_resolve_hf_token_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env_tok")
    assert resolve_hf_token(None) == "env_tok"


def test_resolve_hf_token_missing_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(TranscriptionError, match="HuggingFace token"):
        resolve_hf_token(None)
