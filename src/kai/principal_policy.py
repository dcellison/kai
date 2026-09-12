"""Migration helpers for the backend-neutral per-principal policy file."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

PRINCIPAL_POLICY_MIGRATION = "principal-policy-v1"

_LEGACY_TITLE = "# Kai"
_NEUTRAL_TITLE = "# Principal Policy"
_LEGACY_ABOUT = (
    "This file is the bootstrap template for Kai's backend-neutral identity. The installer "
    "copies it to `<DATA_DIR>/home/<principal_id>/AGENTS.md` for each canonical Workshop "
    "human with an assigned runtime; `backend.ensure_user_home` lazily seeds it for profiles "
    "added later in development mode. Claude receives a thin `.claude/CLAUDE.md` import "
    "adapter; all managed identity content remains here. Edit the per-principal `AGENTS.md` "
    "to add operator-personal content; the tracked template ships universal content only. "
    'Once customized, you can delete this "About This File" section from the per-principal copy.'
)
_LEGACY_CHAT_ID_ABOUT = (
    "This file is the bootstrap template for Kai's backend-neutral identity. The installer "
    "copies it to `<DATA_DIR>/home/<chat_id>/AGENTS.md` for every user in `users.yaml` at "
    "install time; `backend.ensure_user_home` lazily seeds it for users added later in "
    "development mode. Claude receives a thin `.claude/CLAUDE.md` import adapter; all managed "
    "identity content remains here. Edit the per-user `AGENTS.md` to add operator-personal "
    "content; the tracked template ships universal content only. Once customized, you can "
    'delete this "About This File" section from the per-user copy.'
)
_NEUTRAL_ABOUT = (
    "This file is the bootstrap template for Kai's backend-neutral principal policy. The "
    "installer copies it to `<DATA_DIR>/home/<principal_id>/AGENTS.md` for each canonical "
    "Workshop human with an assigned runtime; `backend.ensure_user_home` lazily seeds it for "
    "profiles added later in development mode. Claude receives a thin `.claude/CLAUDE.md` "
    "import adapter; all managed policy content remains here. Edit the per-principal "
    "`AGENTS.md` to add operator-personal content; the tracked template ships universal "
    'content only. Once customized, you can delete this "About This File" section from the '
    "per-principal copy. Agent identity belongs exclusively to the active canonical agent "
    "definition."
)
_LEGACY_IDENTITY_SECTION = (
    "## Who You Are\n\n"
    "You're Kai, a personal AI assistant available through configured clients such as "
    "Workshop and Telegram. You run locally on the operator's machine and have access to a "
    "shell, the filesystem, the web, a scheduler, and a per-principal memory store.\n\n"
)
_LEGACY_TELEGRAM_IDENTITY_SECTION = (
    "## Who You Are\n\n"
    "You're Kai, a personal AI assistant accessed via Telegram. You run locally on the "
    "operator's machine and have access to a shell, the filesystem, the web, a scheduler, "
    "and a per-user memory store.\n\n"
)
_LEGACY_OPERATOR_IDENTITY_SECTION = (
    "## Who You Are\n\n"
    "You're Kai, an agentic AI coding assistant who lives in Telegram and runs locally on "
    "your user's machine. You're not a butler or a service. You're a peer who happens to "
    "have access to a shell, the filesystem, the web, and a scheduling API. Act like one.\n\n"
)
_LEGACY_IDENTITY_SECTIONS = (
    _LEGACY_IDENTITY_SECTION,
    _LEGACY_TELEGRAM_IDENTITY_SECTION,
    _LEGACY_OPERATOR_IDENTITY_SECTION,
)
_LEGACY_ABOUT_SECTIONS = (_LEGACY_ABOUT, _LEGACY_CHAT_ID_ABOUT)
_IDENTITY_MARKERS = ("you're kai", "you are kai")


class PrincipalPolicyMigrationConflict(RuntimeError):
    """A managed policy contains identity text that cannot be isolated safely."""


@dataclass(frozen=True, slots=True)
class PrincipalPolicyMigrationPlan:
    content: str
    changed: bool


def plan_principal_policy_migration(content: str) -> PrincipalPolicyMigrationPlan:
    """Remove only the known legacy Kai identity material from one policy file.

    Everything outside the managed title, About paragraph, and exact legacy
    identity section is preserved byte-for-byte. An unfamiliar ``Who You Are``
    section or stray Kai identity assertion is ambiguous and therefore fails
    closed instead of silently rewriting operator-authored content.
    """
    migrated = content
    for legacy_identity in _LEGACY_IDENTITY_SECTIONS:
        if legacy_identity in migrated:
            migrated = migrated.replace(legacy_identity, "", 1)

    identity_scan = migrated.casefold().replace("\u2019", "'")
    if "## who you are" in identity_scan or any(marker in identity_scan for marker in _IDENTITY_MARKERS):
        raise PrincipalPolicyMigrationConflict(
            "Managed AGENTS.md contains a customized or ambiguous Kai identity section. "
            "The original was not changed; move that identity wording into the canonical "
            "Kai agent definition before re-running `make install`."
        )

    if migrated.startswith(_LEGACY_TITLE + "\n"):
        migrated = _NEUTRAL_TITLE + migrated[len(_LEGACY_TITLE) :]
    for legacy_about in _LEGACY_ABOUT_SECTIONS:
        if legacy_about in migrated:
            migrated = migrated.replace(legacy_about, _NEUTRAL_ABOUT, 1)

    return PrincipalPolicyMigrationPlan(content=migrated, changed=migrated != content)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_private_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _migration_receipt(before: str, after: str) -> str:
    receipt = {
        "after_sha256": _sha256(after),
        "backup": "AGENTS.md.before",
        "before_sha256": _sha256(before),
        "migration": PRINCIPAL_POLICY_MIGRATION,
        "version": 1,
    }
    return json.dumps(receipt, indent=2, sort_keys=True) + "\n"


def validate_principal_policy_migration_record(home: Path, before: str, after: str) -> Path:
    """Fail closed on conflicting migration artifacts without writing anything."""
    migration_root = home / ".kai-migrations"
    migration_dir = migration_root / PRINCIPAL_POLICY_MIGRATION
    for path in (migration_root, migration_dir):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise PrincipalPolicyMigrationConflict(f"Invalid principal-policy migration path: {path}")
    backup_path = migration_dir / "AGENTS.md.before"
    receipt_path = migration_dir / "receipt.json"
    for path in (backup_path, receipt_path):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise PrincipalPolicyMigrationConflict(f"Invalid principal-policy migration artifact: {path}")

    if backup_path.is_file() and backup_path.read_text(encoding="utf-8") != before:
        raise PrincipalPolicyMigrationConflict(
            f"Existing principal-policy backup does not match the current legacy document: {backup_path}"
        )
    receipt_text = _migration_receipt(before, after)
    if receipt_path.is_file() and receipt_path.read_text(encoding="utf-8") != receipt_text:
        raise PrincipalPolicyMigrationConflict(
            f"Existing principal-policy migration receipt conflicts with the planned migration: {receipt_path}"
        )
    return migration_dir


def record_principal_policy_migration(home: Path, before: str, after: str) -> Path:
    """Persist a private, deterministic backup and receipt before policy replacement."""
    migration_dir = validate_principal_policy_migration_record(home, before, after)
    migration_root = migration_dir.parent
    migration_root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(migration_root, 0o700)
    backup_path = migration_dir / "AGENTS.md.before"
    receipt_path = migration_dir / "receipt.json"
    if not backup_path.is_file():
        _write_private_atomic(backup_path, before)
    if not receipt_path.is_file():
        receipt_text = _migration_receipt(before, after)
        _write_private_atomic(receipt_path, receipt_text)
    return migration_dir
