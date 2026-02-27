# Contributing to pyWebpower

Thank you for your interest in contributing! This document covers how to set up a development environment, run the test suite, and submit changes.

## Development Setup

1. **Fork and clone** the repository, then create a branch for your change:

   ```bash
   git checkout -b my-feature
   ```

2. **Install the package with dev dependencies:**

   ```bash
   pip install -e ".[dev]"
   ```

## Running Checks

A `Makefile` is provided as a convenience wrapper. Run `make help` to see all targets. The most common ones:

```bash
make test      # run the test suite
make lint      # ruff check + ruff format --check
make fmt       # auto-format with ruff
make typecheck # mypy
make check     # lint + typecheck + test
```

You can also invoke the tools directly:

```bash
python3 -m pytest
python3 -m ruff check .
python3 -m ruff format .
python3 -m mypy webpower --ignore-missing-imports
```

## Coding Guidelines

- **Style:** code is formatted with [ruff](https://docs.astral.sh/ruff/) (`line-length = 120`). Run `make fmt` before committing.
- **Types:** use builtin generics (`list[str]`, `X | None`) rather than the `typing` module equivalents.
- **Naming:** follow the existing naming conventions, which mirror the original R package where practical.
- **Tests:** add or update tests in `test/` for any behaviour you change or add. Tests use `pytest`.

## Submitting a Pull Request

1. Ensure `make check` passes locally.
2. Write a clear PR description explaining *what* changed and *why*.
3. Keep commits focused — one logical change per commit.

## Reporting Issues

Please open a [GitHub issue](../../issues) with a minimal reproducible example and the output you observed versus the output you expected.
