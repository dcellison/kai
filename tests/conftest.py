"""
Global test fixtures.

Fixtures here apply to ALL tests automatically, providing safety
guarantees that individual tests don't need to remember to set up.
"""

# ── Session-wide Mem0 isolation ─────────────────────────────────────
#
# Mem0 opens a telemetry/migration-tracking Qdrant client at
# `$MEM0_DIR/migrations_qdrant` (see mem0/memory/main.py:378-379 and
# the parallel construction at line 1819). That path is hardcoded
# outside the user's `vector_store.config` and defaults to
# `$HOME/.mem0/migrations_qdrant`. If the production Kai service is
# running on the same machine, it holds a portalocker lock on that
# folder and any test-suite run racing against it dies with
# `Storage folder ... is already accessed by another instance of
# Qdrant client`. See issue #357.
#
# The path is resolved at module-import time in
# `mem0/memory/setup.py:8` (and again, independently, in
# `mem0/configs/base.py:13`), which means the override must be in
# `os.environ` BEFORE any test module imports `mem0`.
# `tests/test_memory.py:80` does `import mem0` at module top to
# detect availability, so a fixture-level override is too late -
# pytest collects the test module after conftest runs but before
# any fixture body executes. conftest.py module-top is the only
# hook that fires before test collection, making this the right
# place for the env-var set.
import atexit
import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

# Hard override (NOT setdefault). A dev who happens to run the
# suite with `MEM0_DIR=$HOME/.mem0` exported in their shell is the
# exact scenario this fix targets; respecting an inherited value
# would silently reopen the bug. tempfile.mkdtemp is used instead
# of pytest's tmp_path_factory because that factory is only
# accessible from inside fixtures - module-top code has no clean
# way to reach it without poking at private pytest internals.
_MEM0_TEST_DIR = tempfile.mkdtemp(prefix="kai-test-mem0-")
os.environ["MEM0_DIR"] = _MEM0_TEST_DIR


@atexit.register
def _cleanup_mem0_test_dir() -> None:
    """Best-effort removal of the per-run Mem0 telemetry root.

    `tempfile.mkdtemp` does not auto-clean; without this hook every
    `make test` invocation leaks one directory under /tmp/. Errors
    are swallowed (ignore_errors=True) so a stray open file handle
    from a crashed worker cannot mask a real test failure with a
    secondary traceback at interpreter shutdown.
    """
    shutil.rmtree(_MEM0_TEST_DIR, ignore_errors=True)


# ── Per-test history isolation ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_history_dir(tmp_path):
    """Redirect history logging to a temp directory for ALL tests.

    Without this, any test that calls a handler function (handle_photo,
    _job_callback, etc.) without patching log_message will write test
    data (chat_id 12345, MagicMock paths) to the REAL production
    history files in workspace/.claude/history/. That contaminated
    data gets injected into Claude's session context, confusing the
    inner Claude with fake conversations.

    This fixture patches _LOG_DIR globally so no test can ever write
    to the real history directory, regardless of whether individual
    tests remember to patch log_message.
    """
    with patch("kai.history._LOG_DIR", tmp_path):
        yield


@pytest.fixture(autouse=True)
def _isolate_backend_data_dir(tmp_path):
    """Redirect kai.backend.DATA_DIR to tmp_path for ALL tests.

    Mirrors _isolate_history_dir in shape (autouse, redirects a
    module-level path constant via unittest.mock.patch) and in
    intent (prevent test runs from writing into the real DATA_DIR).
    Without this guarantee, any test that exercises
    resolve_home_workspace or ensure_user_home without explicitly
    redirecting backend.DATA_DIR creates a per-user home directory
    under the real DATA_DIR. In dev DATA_DIR equals PROJECT_ROOT,
    so leaked chat_ids from test fixtures (e.g. 111, 222, 12345)
    materialize as empty directories in the working tree.

    Tests that need a specific DATA_DIR can still override with
    their own patch / monkeypatch; the autouse fixture only sets
    the safe default.
    """
    with patch("kai.backend.DATA_DIR", tmp_path):
        yield
