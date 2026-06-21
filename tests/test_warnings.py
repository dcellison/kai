"""Regression guard for the async-mock warning bucket from issue 532.

When this test fails, a new test has reintroduced the
`AsyncMockMixin._execute_mock_call was never awaited` pattern. Either
fix the mock setup at the site that emits the warning, or remove this
test if the maintenance burden is no longer worthwhile.

The guard runs a focused pytest subprocess against the two files that
historically carried the warning bucket. The subprocess form keeps the
warning capture independent of the parent pytest's warning filters,
which is necessary because pytest's `-W` flag at the outer level can
otherwise mask the inner warnings.
"""

import subprocess
import sys


def test_async_mock_warning_bucket_stays_empty():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-W",
            "default",
            "--quiet",
            "tests/test_triage.py",
            "tests/test_claude.py",
        ],
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    # The returncode check guards against a nested pytest invocation
    # that exits early (collection error, import failure, unrelated
    # test failure) and therefore never emits the warnings the count
    # below is meant to catch. Without this gate, a broken nested run
    # would report zero leaks and the regression guard would pass for
    # the wrong reason.
    assert proc.returncode == 0, (
        f"Subprocess pytest exited {proc.returncode}; warning count "
        f"cannot be trusted. Captured output:\n{combined}"
    )
    leaks = combined.count("AsyncMockMixin._execute_mock_call")
    assert leaks == 0, (
        f"{leaks} async-mock RuntimeWarning(s) detected. Each one "
        "points at a test whose mock setup discards a coroutine.\n"
        f"Captured output:\n{combined}"
    )
