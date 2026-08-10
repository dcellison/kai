#!/usr/bin/env python3
"""Report large Python modules by physical line count.

This is a lightweight maintainability guardrail for trust-boundary review.
It intentionally reports by default instead of failing: Kai currently has
known oversized modules, and the immediate goal is a stable decomposition
baseline rather than blocking unrelated security fixes.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("src/kai")
DEFAULT_LIMIT = 2000


@dataclass(frozen=True, order=True)
class ModuleSize:
    """Line-count measurement for one Python source file."""

    lines: int
    path: Path


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield Python source files below `root`, excluding generated/cache trees."""
    for path in sorted(root.rglob("*.py")):
        parts = set(path.parts)
        if "__pycache__" in parts or ".venv" in parts or ".git" in parts:
            continue
        yield path


def count_lines(path: Path) -> int:
    """Return the number of physical lines in `path`."""
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def measure_modules(root: Path) -> list[ModuleSize]:
    """Return module sizes below `root`, largest first."""
    sizes = [ModuleSize(lines=count_lines(path), path=path) for path in iter_python_files(root)]
    return sorted(sizes, reverse=True)


def render_report(sizes: Iterable[ModuleSize], *, limit: int, top: int) -> str:
    """Render a compact line-count report."""
    rows = list(sizes)
    oversized = [row for row in rows if row.lines >= limit]
    visible = rows[:top]

    lines = [
        f"Python module size report (threshold: {limit} lines)",
        f"Modules at or above threshold: {len(oversized)}",
        "",
        "lines  path",
        "-----  ----",
    ]
    lines.extend(f"{row.lines:5d}  {row.path}" for row in visible)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help=f"source root to scan (default: {DEFAULT_ROOT})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"line-count threshold for oversized modules (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument("--top", type=int, default=20, help="number of largest modules to print (default: 20)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(render_report(measure_modules(args.root), limit=args.limit, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
