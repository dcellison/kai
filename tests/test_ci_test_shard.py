from pathlib import Path

import pytest

from scripts.ci_test_shard import (
    CI_TEST_SHARDS,
    classify_test_file,
    discover_test_files,
    select_test_files,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("test_bot.py", "core"),
        ("test_memory.py", "memory"),
        ("test_memory_extraction.py", "memory"),
        ("test_eval_behavioral.py", "memory"),
        ("test_workshop_foundation.py", "workshop"),
    ],
)
def test_test_module_classification_is_stable(name: str, expected: str) -> None:
    assert classify_test_file(Path("tests") / name) == expected


def test_shards_are_disjoint_and_cover_every_test_module() -> None:
    discovered = set(discover_test_files())
    selected = {shard: set(select_test_files(shard)) for shard in CI_TEST_SHARDS}

    assert discovered
    assert set().union(*selected.values()) == discovered
    for shard in CI_TEST_SHARDS:
        other_files = set().union(*(files for other, files in selected.items() if other != shard))
        assert selected[shard].isdisjoint(other_files)


def test_unknown_shard_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown CI test shard"):
        select_test_files("unknown")
