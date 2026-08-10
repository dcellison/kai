"""Metadata validation for protected Kai configuration files."""

from __future__ import annotations

import stat
from pathlib import Path


class ProtectedConfigError(RuntimeError):
    """Raised when a protected config file cannot be trusted."""


_DEFAULT_MAX_MODE = 0o600
_MAX_MODES_BY_PATH = {
    "/etc/kai/backends.yaml": 0o644,
}


def max_mode_for_protected_file(path: str | Path) -> int:
    """Return the most permissive allowed mode for a protected config file."""
    return _MAX_MODES_BY_PATH.get(str(path), _DEFAULT_MAX_MODE)


def validate_protected_file_metadata(
    path: str | Path,
    *,
    max_mode: int | None = None,
    require_root_owner: bool = True,
    missing_ok: bool = False,
) -> bool:
    """
    Validate ownership and mode for a protected config file.

    Returns True when the file exists and passes validation. Returns False
    only when ``missing_ok`` is true and the file is absent.
    """
    protected_path = Path(path)
    try:
        stat_result = protected_path.stat()
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    except OSError as e:
        raise ProtectedConfigError(f"could not stat protected config {protected_path}: {e}") from e

    if not stat.S_ISREG(stat_result.st_mode):
        raise ProtectedConfigError(f"protected config {protected_path} is not a regular file")

    mode = stat.S_IMODE(stat_result.st_mode)
    allowed_mode = max_mode_for_protected_file(protected_path) if max_mode is None else max_mode
    if mode & ~allowed_mode:
        raise ProtectedConfigError(
            f"protected config {protected_path} has unsafe permissions {mode:#04o}; "
            f"expected no more than {allowed_mode:#04o}"
        )
    if require_root_owner and stat_result.st_uid != 0:
        raise ProtectedConfigError(
            f"protected config {protected_path} is owned by uid {stat_result.st_uid}; expected root ownership"
        )
    return True
