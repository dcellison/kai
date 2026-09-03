"""Select and run one exhaustive, deterministic Kai pytest shard."""

from __future__ import annotations

import argparse
from pathlib import Path

CI_TEST_SHARDS = ("core", "memory", "workshop")
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def classify_test_file(path: Path) -> str:
    """Assign one test module to exactly one stable CI shard."""
    name = path.name
    if name.startswith("test_workshop_"):
        return "workshop"
    if name.startswith(("test_memory", "test_eval_")):
        return "memory"
    return "core"


def discover_test_files(repository_root: Path = _REPOSITORY_ROOT) -> tuple[Path, ...]:
    """Return every pytest module in deterministic repository-relative order."""
    tests_root = repository_root / "tests"
    return tuple(
        sorted(
            (path.relative_to(repository_root) for path in tests_root.rglob("test_*.py")),
            key=lambda path: path.as_posix(),
        )
    )


def select_test_files(shard: str, repository_root: Path = _REPOSITORY_ROOT) -> tuple[Path, ...]:
    """Return the complete non-overlapping module set for one known shard."""
    if shard not in CI_TEST_SHARDS:
        raise ValueError(f"Unknown CI test shard: {shard}")
    return tuple(path for path in discover_test_files(repository_root) if classify_test_file(path) == shard)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", choices=CI_TEST_SHARDS)
    parser.add_argument("--list", action="store_true", help="List selected modules without running pytest")
    args = parser.parse_args(argv)
    selected = select_test_files(args.shard)
    if not selected:
        parser.error(f"CI test shard {args.shard!r} selected no test modules")
    if args.list:
        for path in selected:
            print(path.as_posix())
        return 0

    import pytest

    return int(pytest.main(["-v", *(path.as_posix() for path in selected)]))


if __name__ == "__main__":
    raise SystemExit(main())
