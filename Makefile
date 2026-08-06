.PHONY: help install test lint fmt typecheck check

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Sync the environment with project + dev dependencies
	uv sync

test: ## Run the test suite
	uv run pytest

lint: ## Check code style and imports
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Auto-format code
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Run zuban type checker
	uv run zuban check webpower

check: lint typecheck test ## Run lint, typecheck, and tests
