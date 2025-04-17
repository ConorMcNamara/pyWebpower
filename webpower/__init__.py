"""Evaluating statistical power through Webpower."""

__version__ = "0.1.0"

from typing import List

from webpower import power_tests

__all__: List[str] = ["power_tests"]


def __dir__() -> List[str]:
    return __all__
