.PHONY: run lint format check test install

run:
	.venv/bin/python -m kai

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

check: lint
	.venv/bin/ruff format --check .

test:
	.venv/bin/python -m pytest tests/ -v

install:
	.venv/bin/pip install -e '.[dev]'
