"""Regression guard for the async-mock warning bucket from issues 532 and 712.

When this test fails, a new test has reintroduced the
`AsyncMockMixin._execute_mock_call was never awaited` pattern in one
of the files this guard covers. Either fix the mock setup at the site
that emits the warning, or remove this test if the maintenance burden
is no longer worthwhile.

The guard is scoped to the test files where the bucket was identified
and cleared (PR #711 for issue 532; the PR closing issue 712 for the
remaining files). New files added to the suite that misuse AsyncMock
will not be caught here automatically; extend the file list below
when adding the same shape of fix elsewhere.

The two recurring root shapes the guard catches:

1. AsyncMock-where-MagicMock-was-needed: a sync interface (e.g. a
   Config dataclass, an aiohttp.ClientSession's `.post(...)` call)
   mocked with AsyncMock, so calling it returns an unawaited
   coroutine that production then misuses (assigns to a value,
   passes to `async with`, etc.).
2. wait_for-leaks-coroutine: a patched `asyncio.wait_for` with a
   bare `side_effect=TimeoutError` that raises without closing the
   already-constructed coroutine production passed to it. The fix is
   a coroutine-closing helper.

The guard runs a focused pytest subprocess so the warning capture
stays independent of the parent pytest's warning filters, which is
necessary because pytest's `-W` flag at the outer level can otherwise
mask the inner warnings.
"""

import subprocess
import sys

# Files covered by the regression guard. Add a file here when a new
# test in it gets fixed for the same async-mock leak pattern, so future
# regressions surface immediately.
_GUARDED_FILES = (
    "tests/test_triage.py",
    "tests/test_claude.py",
    "tests/test_webhook.py",
    "tests/test_review.py",
    "tests/test_tts.py",
    "tests/test_transcribe.py",
)


def test_async_mock_warning_bucket_empty_across_guarded_files():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-W",
            "default",
            "--quiet",
            *_GUARDED_FILES,
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
        f"Subprocess pytest exited {proc.returncode}; warning count cannot be trusted. Captured output:\n{combined}"
    )
    leaks = combined.count("AsyncMockMixin._execute_mock_call")
    assert leaks == 0, (
        f"{leaks} async-mock RuntimeWarning(s) detected. Each one "
        "points at a test whose mock setup discards a coroutine.\n"
        f"Captured output:\n{combined}"
    )
