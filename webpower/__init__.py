"""Evaluating statistical power through Webpower."""

__version__ = "0.1.0"

from webpower import power_tests

__all__: list[str] = ["power_tests"]


def __dir__() -> list[str]:
    return __all__
