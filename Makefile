.PHONY: install test test-cov lint format run eval

install:
	uv sync --extra dev

test:
	uv run pytest --tb=short -q

test-cov:
	uv run pytest --cov=src --cov-report=term-missing

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

run:
	uv run uvicorn src.api.main:app --reload --port 8001

eval:
	uv run python evals/run_eval.py
