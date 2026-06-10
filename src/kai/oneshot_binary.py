"""
Shared agent-binary resolver for the one-shot reasoner family.

Single source of truth for "where is the agent binary" across the
one-shot backends (claude, codex, opencode, goose).
`kai.config.load_config()` uses this to fail-fast at startup when the
memory reasoner backend points at an unreachable binary; the
`kai.oneshot` reasoner classes use it to build argv with the resolved
absolute path; `kai.smoke.memory` reads the resolved binary from
`OneShotResult.raw_metadata["resolved_binary"]` to show the operator
which binary actually ran.

Module is a leaf by design: imports `os`, `shutil`, `pathlib` only.
Neither `kai.config` nor `kai.oneshot` is imported here. The
constraint exists because `kai.oneshot` already imports from
`kai.config`, so putting the resolver in `kai.oneshot` while having
`kai.config.load_config()` call into it would create a circular
import. Keeping the resolver leaf-only sidesteps that entirely and
keeps the dependency direction one-way.

`BinaryResolutionError` is deliberately distinct from
`kai.oneshot.OneShotRoutingError` so config-load and smoke can
distinguish "binary not found" from runtime routing errors when
deciding what to print. The reasoner argv builders catch
`BinaryResolutionError` and convert to `OneShotRoutingError` so
the existing `memory_extraction.py` `except OneShotError` catch
surface is unchanged; the new error type is what `kai.config`
and `kai.smoke.memory` see directly.
"""

import os
import shutil
from pathlib import Path


class BinaryResolutionError(Exception):
    """
    Raised when `resolve_oneshot_binary` cannot resolve the named
    backend's binary. The exception message carries the resolution
    sequence that was tried (e.g., "CODEX_BIN unset, `codex` not on
    PATH") so the operator has a concrete remediation pointer
    without re-deriving the lookup order.

    Distinct from `kai.oneshot.OneShotRoutingError` so the
    config-validation and smoke call sites can catch this leaf
    module's exception without depending on `kai.oneshot`. The
    reasoner argv builders catch and rewrap into
    `OneShotRoutingError` to preserve the existing reasoner-error
    hierarchy `memory_extraction` already handles.
    """


def resolve_oneshot_binary(backend: str) -> str:
    """
    Return the absolute path of the agent binary the OneShotReasoner
    of `backend` will invoke.

    `backend == "claude"`: `shutil.which("claude")`. Resolution
    sequence: claude on PATH only. Failure message names that
    sequence.

    `backend == "codex"`: branch on `CODEX_BIN`.
      - Explicit override: when `CODEX_BIN` is set, resolve to that
        exact path. Validate `Path(p).is_file()` AND
        `os.access(p, os.X_OK)`. On either check failing, raise
        `BinaryResolutionError` naming the override path and the
        specific failure (not-a-file vs not-executable). NO fallback
        to PATH; a bad explicit override is a configuration error
        worth surfacing, not silently recovering from.
      - No override: `shutil.which("codex")`. On no resolution,
        raise `BinaryResolutionError` naming both candidates
        (CODEX_BIN unset, codex not on PATH).

    `backend == "opencode"`: branch on `OPENCODE_BIN` with the same
    is-file plus executable validation pattern as the codex arm. The
    one-shot adapter spawns `opencode acp` as a short-lived
    JSON-RPC server per call (distinct from the persistent
    OpenCodeBackend's conversational session); both paths resolve
    through this function so any future PATH or override semantics
    change applies uniformly.

    Raises `BinaryResolutionError` with a single-line message
    describing the resolution sequence that was tried and the
    specific failure mode that fired. Callers should surface the
    message verbatim where appropriate; it is designed to be
    operator-readable.

    Raises `ValueError` on an unknown backend string. That is
    distinct from a resolution miss: unknown backend means the
    caller's config validation upstream of this function has a
    bug, not the operator's PATH.
    """
    if backend == "claude":
        # Resolution: claude on PATH only. No explicit-override
        # variable for claude (CLAUDE_CONFIG_DIR controls auth state,
        # not binary discovery). A future CLAUDE_BIN override would
        # land here as the second branch.
        resolved = shutil.which("claude")
        if resolved is None:
            raise BinaryResolutionError("could not resolve claude binary: `claude` not on PATH")
        return resolved

    if backend == "codex":
        override = os.environ.get("CODEX_BIN")
        if override:
            # Explicit override: validate as a configuration error,
            # do not fall back. The two checks fire in order so the
            # message names the specific failure mode rather than a
            # composite. Path.is_file resolves symlinks; the
            # executable check is on the resolved target's stat.
            override_path = Path(override)
            if not override_path.is_file():
                raise BinaryResolutionError(f"could not resolve codex binary: CODEX_BIN={override!r} not-a-file")
            if not os.access(str(override_path), os.X_OK):
                raise BinaryResolutionError(f"could not resolve codex binary: CODEX_BIN={override!r} not-executable")
            return str(override_path)
        # No override: try PATH. Failure message names both
        # candidates so the operator does not have to guess which
        # branch fired.
        resolved = shutil.which("codex")
        if resolved is None:
            raise BinaryResolutionError("could not resolve codex binary: CODEX_BIN unset, `codex` not on PATH")
        return resolved

    if backend == "opencode":
        # Structural mirror of the codex arm. OPENCODE_BIN, when set,
        # carries an absolute path the operator validated out of band;
        # the same is-file + executable pair fires so a typo or stale
        # path surfaces as a configuration error rather than a silent
        # PATH-fallback recovery. OpenCode binaries are commonly
        # installed under ~/.local/bin via the upstream installer
        # script, which is already on the bot user's PATH for the
        # conversational backend, so the no-override branch covers the
        # standard install.
        override = os.environ.get("OPENCODE_BIN")
        if override:
            override_path = Path(override)
            if not override_path.is_file():
                raise BinaryResolutionError(f"could not resolve opencode binary: OPENCODE_BIN={override!r} not-a-file")
            if not os.access(str(override_path), os.X_OK):
                raise BinaryResolutionError(
                    f"could not resolve opencode binary: OPENCODE_BIN={override!r} not-executable"
                )
            return str(override_path)
        resolved = shutil.which("opencode")
        if resolved is None:
            raise BinaryResolutionError("could not resolve opencode binary: OPENCODE_BIN unset, `opencode` not on PATH")
        return resolved

    if backend == "goose":
        # Structural mirror of the codex / opencode arms. GOOSE_BIN,
        # when set, is validated as a configuration error with no
        # PATH fallback; unset falls through to PATH discovery, which
        # covers the common Homebrew install where `goose` is already
        # resolvable from the service user's shell.
        override = os.environ.get("GOOSE_BIN")
        if override:
            override_path = Path(override)
            if not override_path.is_file():
                raise BinaryResolutionError(f"could not resolve goose binary: GOOSE_BIN={override!r} not-a-file")
            if not os.access(str(override_path), os.X_OK):
                raise BinaryResolutionError(f"could not resolve goose binary: GOOSE_BIN={override!r} not-executable")
            return str(override_path)
        resolved = shutil.which("goose")
        if resolved is None:
            raise BinaryResolutionError("could not resolve goose binary: GOOSE_BIN unset, `goose` not on PATH")
        return resolved

    # Unknown backend. Unlike a resolution miss this is a caller bug
    # (config validation upstream of this function did not catch the
    # invalid backend string), so the exception type is distinct.
    raise ValueError(f"unknown backend for binary resolution: {backend!r}")
