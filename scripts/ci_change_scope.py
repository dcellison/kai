"""Conservatively classify changed paths for Kai CI."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

_CLIENT_PREFIXES = ("workshop-client/", "src/kai/workshop/static/")
_DOCUMENTATION_PREFIXES = ("docs/", "home/docs/", ".github/ISSUE_TEMPLATE/")
_DOCUMENTATION_FILES = {
    "AGENTS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
}
_DEPENDENCY_FILES = {
    "Makefile",
    "pyproject.toml",
    ".github/workflows/dependency-audit.yml",
    "scripts/ci_change_scope.py",
}


@dataclass(frozen=True, slots=True)
class ChangeScope:
    client: bool
    full: bool
    dependency: bool


def _normalized_path(raw_path: str) -> str:
    path = raw_path.strip().replace("\\", "/")
    if not path:
        return ""
    normalized = str(PurePosixPath(path))
    if normalized == "." or normalized.startswith("../") or normalized.startswith("/"):
        return ""
    return normalized


def _is_documentation(path: str) -> bool:
    return path in _DOCUMENTATION_FILES or path.startswith(_DOCUMENTATION_PREFIXES)


def _is_dependency_input(path: str) -> bool:
    return path in _DEPENDENCY_FILES or path.startswith("requirements/")


def classify_paths(paths: Iterable[str], *, force_full: bool = False) -> ChangeScope:
    """Return a fail-closed CI scope for the supplied repository paths."""
    if force_full:
        return ChangeScope(client=True, full=True, dependency=True)

    client = False
    full = False
    dependency = False
    saw_path = False
    for raw_path in paths:
        path = _normalized_path(raw_path)
        if not path:
            continue
        saw_path = True
        if _is_dependency_input(path):
            dependency = True
        if path.startswith(_CLIENT_PREFIXES):
            client = True
        elif not _is_documentation(path):
            full = True

    if not saw_path:
        full = True
    if full:
        client = True
    return ChangeScope(client=client, full=full, dependency=dependency)


def _write_outputs(scope: ChangeScope, output_path: str | None) -> None:
    lines = (
        f"client={str(scope.client).lower()}",
        f"full={str(scope.full).lower()}",
        f"dependency={str(scope.dependency).lower()}",
    )
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Changed repository paths; stdin is used when omitted")
    parser.add_argument("--force-full", action="store_true", help="Select every validation lane")
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="Optional GitHub Actions output file",
    )
    args = parser.parse_args(argv)
    paths = args.paths if args.paths else sys.stdin.read().splitlines()
    _write_outputs(classify_paths(paths, force_full=args.force_full), args.github_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
