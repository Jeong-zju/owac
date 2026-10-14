.PHONY: setup check format research-hub research-hub-serve

HUB_PORT ?= 8876

setup:
	uv sync --locked

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked python -m compileall -q src

format:
	uv run --locked ruff format .

research-hub:
	uv run --locked python -m owac.runtime.research_hub build

research-hub-serve:
	uv run --locked python -m owac.runtime.research_hub serve --port $(HUB_PORT)
