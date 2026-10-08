.PHONY: setup check format

setup:
	uv sync --locked

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked python -m compileall -q src

format:
	uv run --locked ruff format .
