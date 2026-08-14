"""Cross-user subprocess launch mechanics for protected agent runtimes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


def wrap_command_for_target_user(
    command: Sequence[str],
    *,
    target_user: str,
    working_directory: Path,
    preserve_env: Iterable[str] = (),
) -> list[str]:
    """Enter a private workspace after sudo changes execution identity.

    ``asyncio.create_subprocess_exec(cwd=...)`` changes directory before it
    executes sudo, while the child still has the Kai service UID. Canonical
    managed homes are intentionally private to the target OS user, so the
    service cannot enter them. ``sudo -D`` performs that transition under the
    target-user execution policy instead.
    """
    preserved = tuple(preserve_env)
    wrapped = [
        "sudo",
        "-H",
        "-D",
        str(working_directory),
        "-u",
        target_user,
    ]
    if preserved:
        wrapped.append(f"--preserve-env={','.join(preserved)}")
    return [*wrapped, "--", *command]


def subprocess_spawn_cwd(
    working_directory: Path,
    *,
    target_user: str | None,
) -> str | None:
    """Return a pre-exec cwd only when no cross-user transition is needed."""
    return None if target_user is not None else str(working_directory)
