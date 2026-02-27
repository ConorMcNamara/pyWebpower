.PHONY: help install test lint fmt typecheck check

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install package with dev dependencies
	pip3 install -e ".[dev]"

test: ## Run the test suite
	python3 -m pytest

lint: ## Check code style and imports
	python3 -m ruff check .
	python3 -m ruff format --check .

fmt: ## Auto-format code
	python3 -m ruff format .
	python3 -m ruff check --fix .

typecheck: ## Run mypy type checker
	python3 -m mypy webpower --ignore-missing-imports

check: lint typecheck test ## Run lint, typecheck, and tests
