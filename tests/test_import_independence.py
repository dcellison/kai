"""First-import order independence for cycle-prone kai modules.

kai.workshop's package initialization eagerly imports the execution stack,
which reaches back into several top-level modules. Each entry below must
therefore import cleanly as the very first kai import of a process, which
only a fresh interpreter per module can prove: importing them in-process
would inherit whatever sys.modules state earlier tests created.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# When breaking a new import cycle, add its entry module here so the fix
# stays proven; the list only covers what it names.
_ENTRY_MODULES = (
    "kai.bot",
    "kai.conversation_compatibility",
    "kai.pool",
    "kai.sessions",
    "kai.workshop",
    "kai.workshop.client_commands",
    "kai.workshop.protected_execution",
    "kai.workshop.runtime_pool",
)


@pytest.mark.parametrize("module", _ENTRY_MODULES)
def test_module_imports_first(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
