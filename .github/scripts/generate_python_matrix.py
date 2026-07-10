"""Generate the CI Python-version matrix from ``requires-python``.

Reads the minimum supported Python version from ``pyproject.toml`` and queries
endoflife.date for every currently-supported (non-EOL, already-released) CPython
release at or above that minimum. The resulting JSON list is written to
``$GITHUB_OUTPUT`` as ``python-versions`` so the ``test`` job can consume it as a
matrix. New Python releases are picked up automatically; end-of-life versions
drop off on their own.

Run locally to preview:  ``python .github/scripts/generate_python_matrix.py``
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ENDOFLIFE_URL = "https://endoflife.date/api/python.json"

# Used only if endoflife.date is unreachable, so CI degrades gracefully instead
# of failing. Bump the ceiling if you ever need to outrun a network outage.
FALLBACK_LATEST = (3, 14)


def read_minimum_version() -> tuple[int, int]:
    """Parse the lower bound of ``requires-python`` from ``pyproject.toml``."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)

    spec = data["project"]["requires-python"]  # e.g. ">=3.10"
    digits = spec.lstrip(">=~^ ").split(",")[0].strip()
    major, minor = digits.split(".")[:2]
    return int(major), int(minor)


def fetch_supported_versions(minimum: tuple[int, int]) -> list[tuple[int, int]]:
    """Return released, non-EOL CPython cycles at or above ``minimum``."""
    with urllib.request.urlopen(ENDOFLIFE_URL, timeout=30) as resp:
        cycles = json.load(resp)

    today = datetime.date.today()
    versions: list[tuple[int, int]] = []
    for cycle in cycles:
        major, minor = (int(part) for part in cycle["cycle"].split("."))
        if (major, minor) < minimum:
            continue

        # Skip cycles that have not been released yet (stable-only policy).
        release_date = cycle.get("releaseDate")
        if release_date and datetime.date.fromisoformat(release_date) > today:
            continue

        # ``eol`` is either False (never) or an ISO date. Skip if already EOL.
        eol = cycle.get("eol")
        if isinstance(eol, str) and datetime.date.fromisoformat(eol) <= today:
            continue

        versions.append((major, minor))

    return versions


def fallback_versions(minimum: tuple[int, int]) -> list[tuple[int, int]]:
    """Enumerate ``minimum``..``FALLBACK_LATEST`` when the network is unavailable."""
    if minimum[0] != FALLBACK_LATEST[0]:
        return [minimum]
    return [(minimum[0], minor) for minor in range(minimum[1], FALLBACK_LATEST[1] + 1)]


def main() -> None:
    """Resolve the supported versions and write the matrix to ``$GITHUB_OUTPUT``."""
    minimum = read_minimum_version()
    try:
        versions = fetch_supported_versions(minimum)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
        print(f"::warning::endoflife.date lookup failed ({exc}); using fallback range", file=sys.stderr)
        versions = fallback_versions(minimum)

    versions.sort()
    formatted = [f"{major}.{minor}" for major, minor in versions]
    payload = json.dumps(formatted)
    latest = formatted[-1] if formatted else ""

    print(f"Supported Python versions: {formatted}", file=sys.stderr)
    print(f"Highest version: {latest}", file=sys.stderr)

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"python-versions={payload}\n")
            f.write(f"python-latest={latest}\n")
    else:
        # Local invocation: just print the matrix payload.
        print(payload)


if __name__ == "__main__":
    main()
