"""Platform-specific named access for private Kai runtime files."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], *, action: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise OSError(f"{action}: {detail}")


def _grant_command(path: Path, reader_user: str, *, directory: bool) -> list[str]:
    if sys.platform == "darwin":
        permissions = (
            "list,search,readattr,readextattr,readsecurity" if directory else "read,readattr,readextattr,readsecurity"
        )
        return [
            "/bin/chmod",
            "+a",
            f"user:{reader_user} allow {permissions}",
            str(path),
        ]
    if sys.platform.startswith("linux"):
        setfacl = shutil.which("setfacl")
        if setfacl is None:
            raise OSError("setfacl is required for isolated named-file access on Linux")
        permissions = "r-x" if directory else "r--"
        return [setfacl, "-m", f"u:{reader_user}:{permissions}", str(path)]
    raise OSError(f"isolated named-file access is unsupported on {sys.platform}")


def grant_named_read_access(path: Path, reader_user: str, *, directory: bool) -> None:
    """Grant one OS user read/traversal access to an otherwise private path."""
    _run(
        _grant_command(path, reader_user, directory=directory),
        action=f"could not grant access to {reader_user} for {path}",
    )


def replace_named_read_access(
    path: Path,
    reader_user: str | None,
    *,
    directory: bool,
) -> None:
    """Replace Kai-managed extended ACLs with at most one named reader.

    Protected history and upload paths are Kai-owned, so the installer owns
    their complete extended-ACL contract. Clearing first prevents an old
    ``os_user`` from retaining access after an operator changes users.yaml.
    """
    if sys.platform == "darwin":
        clear_command = ["/bin/chmod", "-N", str(path)]
    elif sys.platform.startswith("linux"):
        setfacl = shutil.which("setfacl")
        if setfacl is None:
            raise OSError("setfacl is required for isolated named-file access on Linux")
        # Directories can carry default ACLs that silently grant access to
        # future children. Remove both access (-b) and default (-k) entries.
        clear_command = [setfacl, "-b", "-k", str(path)] if directory else [setfacl, "-b", str(path)]
    else:
        raise OSError(f"isolated named-file access is unsupported on {sys.platform}")

    _run(clear_command, action=f"could not clear stale access for {path}")
    if reader_user:
        grant_named_read_access(path, reader_user, directory=directory)


def replace_named_inherited_read_access(path: Path, reader_user: str) -> None:
    """Grant one reader access to a private directory and future children.

    Principal-owned outbound staging directories need a two-sided boundary:
    the agent OS user creates files, while Kai's service user must be able to
    read those files for canonical artifact publication.  A directory-only
    access ACL is insufficient because newly created ``0600`` files would
    still be unreadable.  Replace the complete extended ACL and install an
    inheritable read/traversal entry for the service user.
    """
    if not reader_user:
        raise ValueError("reader_user must be non-empty")
    if sys.platform == "darwin":
        clear_command = ["/bin/chmod", "-N", str(path)]
        grant_commands = [
            [
                "/bin/chmod",
                "+a#",
                "0",
                (f"user:{reader_user} allow list,search,readattr,readextattr,readsecurity,directory_inherit"),
                str(path),
            ],
            [
                "/bin/chmod",
                "+a#",
                "1",
                (f"user:{reader_user} allow read,readattr,readextattr,readsecurity,file_inherit,directory_inherit"),
                str(path),
            ],
        ]
    elif sys.platform.startswith("linux"):
        setfacl = shutil.which("setfacl")
        if setfacl is None:
            raise OSError("setfacl is required for isolated named-file access on Linux")
        clear_command = [setfacl, "-b", "-k", str(path)]
        grant_commands = [
            [
                setfacl,
                "-m",
                f"u:{reader_user}:r-x,d:u:{reader_user}:r-x",
                str(path),
            ]
        ]
    else:
        raise OSError(f"isolated named-file access is unsupported on {sys.platform}")

    _run(clear_command, action=f"could not clear stale access for {path}")
    for grant_command in grant_commands:
        _run(
            grant_command,
            action=f"could not grant inherited access to {reader_user} for {path}",
        )
