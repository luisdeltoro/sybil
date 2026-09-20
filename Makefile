# Makefile for extract-speech (repo: sybil)

.PHONY: help install test format lint type-check check clean all

# Default target
all: install format lint type-check test

help:
	@echo "Available targets:"
	@echo "  make install       - Install dependencies using uv"
	@echo "  make test          - Run tests with pytest"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make type-check    - Run type checking with pyright"
	@echo "  make check         - lint + type-check + test (no reformatting)"
	@echo "  make clean         - Clean up generated files and the venv"
	@echo "  make all           - install + format + lint + type-check + test (default)"
	@echo ""
	@echo "Running the tool:"
	@echo "  uv run extract-speech <video-or-audio-file> [options]"
	@echo "  uv run extract-speech --help"
	@echo ""
	@echo "Examples:"
	@echo "  uv run extract-speech call.mp4 --speakers 2"
	@echo "  uv run extract-speech meeting.mp4 --profile noisy --whisper-model large -o out.txt"
	@echo ""
	@echo "Diarization needs HF_TOKEN in .env (see .env.example)."

install:
	uv sync --all-groups

test: install
	uv run python -m pytest

format: install
	uv run ruff format src tests

lint: install
	uv run ruff check src tests

type-check: install
	uv run pyright src

# CI-style gate: verifies formatting rather than rewriting it.
check: install
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run pyright src
	uv run python -m pytest

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pyright/
	rm -rf .venv/
