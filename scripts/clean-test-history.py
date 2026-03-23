#!/usr/bin/env python3
"""
Clean test-contaminated entries from conversation history files.

Tests that ran before PR #117 could write entries with chat_id 12345
(the test fixture default) to production history files. This script
removes those entries.

Usage:
    python scripts/clean-test-history.py [history_dir]

If no directory is specified, defaults to workspace/.claude/history/
relative to the script's location.

Safe to run multiple times - idempotent.
"""

import json
import sys
from pathlib import Path

# Test fixture chat_id that leaked into production history
TEST_CHAT_ID = 12345


def clean_history_dir(history_dir: Path) -> None:
    if not history_dir.is_dir():
        print(f"History directory not found: {history_dir}")
        print("Nothing to clean.")
        return

    total_removed = 0
    files_cleaned = 0

    for path in sorted(history_dir.glob("*.jsonl")):
        try:
            with open(path) as f:
                lines = f.readlines()
        except OSError as e:
            print(f"  Cannot read {path.name}: {e}")
            continue

        clean = []
        removed = 0
        for line in lines:
            try:
                record = json.loads(line)
                if record.get("chat_id") == TEST_CHAT_ID:
                    removed += 1
                else:
                    clean.append(line)
            except json.JSONDecodeError:
                clean.append(line)  # keep malformed lines

        if removed > 0:
            # Atomic write: write to temp file, then rename. If the
            # process is killed mid-write, the original file is intact.
            tmp = path.with_suffix(".tmp")
            try:
                with open(tmp, "w") as f:
                    f.writelines(clean)
                tmp.replace(path)
            except OSError as e:
                print(f"  Cannot write {path.name}: {e}")
                tmp.unlink(missing_ok=True)
                continue
            print(f"  {path.name}: removed {removed} test entries, kept {len(clean)}")
            total_removed += removed
            files_cleaned += 1

    if total_removed == 0:
        print("No test contamination found. History files are clean.")
    else:
        print(f"\nDone: removed {total_removed} entries from {files_cleaned} files.")


def main() -> None:
    if len(sys.argv) > 1:
        history_dir = Path(sys.argv[1])
    else:
        # Default: workspace/.claude/history/ relative to repo root
        script_dir = Path(__file__).resolve().parent
        history_dir = script_dir.parent / "workspace" / ".claude" / "history"

    print(f"Scanning: {history_dir}")
    clean_history_dir(history_dir)


if __name__ == "__main__":
    main()
