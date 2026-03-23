"""
Global test fixtures.

Fixtures here apply to ALL tests automatically, providing safety
guarantees that individual tests don't need to remember to set up.
"""

from unittest.mock import patch

import pytest


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
