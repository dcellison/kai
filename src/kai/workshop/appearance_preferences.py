"""Canonical Workshop appearance preferences owned by human principals."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from kai.workshop.domain import PrincipalId

DEFAULT_WORKSHOP_THEME = "atom-one-dark"


class WorkshopAppearancePreferenceError(RuntimeError):
    """Base failure for canonical Workshop appearance preferences."""


class WorkshopAppearancePreferenceAccessDenied(WorkshopAppearancePreferenceError):
    """The authenticated principal cannot own appearance preferences."""


class WorkshopAppearancePreferenceValidationError(WorkshopAppearancePreferenceError):
    """A requested appearance preference is unsupported."""


class WorkshopAppearancePreferenceConflict(WorkshopAppearancePreferenceError):
    """Appearance preferences changed after the caller loaded them."""


class WorkshopAppearancePreferenceStorageError(WorkshopAppearancePreferenceError):
    """Canonical appearance preference state is unavailable."""


@dataclass(frozen=True, slots=True)
class AppearanceThemeChoice:
    theme_id: str
    display_name: str
    color_scheme: str


WORKSHOP_APPEARANCE_THEMES = (
    AppearanceThemeChoice(DEFAULT_WORKSHOP_THEME, "Atom One Dark", "dark"),
    AppearanceThemeChoice("atom-one-light", "Atom One Light", "light"),
    AppearanceThemeChoice("dracula", "Dracula", "dark"),
    AppearanceThemeChoice("nord", "Nord", "dark"),
    AppearanceThemeChoice("solarized-dark", "Solarized Dark", "dark"),
    AppearanceThemeChoice("solarized-light", "Solarized Light", "light"),
    AppearanceThemeChoice("catppuccin-mocha", "Catppuccin Mocha", "dark"),
    AppearanceThemeChoice("catppuccin-latte", "Catppuccin Latte", "light"),
    AppearanceThemeChoice("github-light-default", "GitHub Light Default", "light"),
    AppearanceThemeChoice("github-dark-default", "GitHub Dark Default", "dark"),
    AppearanceThemeChoice("github-dark-dimmed", "GitHub Dark Dimmed", "dark"),
)
_THEMES_BY_ID = {item.theme_id: item for item in WORKSHOP_APPEARANCE_THEMES}


@dataclass(frozen=True, slots=True)
class AppearancePreferenceAuthority:
    principal_id: PrincipalId


@dataclass(frozen=True, slots=True)
class AppearancePreferenceMutation:
    operation: str
    changed: bool


@dataclass(frozen=True, slots=True)
class AppearancePreferenceSnapshot:
    theme_id: str
    themes: tuple[AppearanceThemeChoice, ...]
    revision: str
    mutation: AppearancePreferenceMutation | None = None


class WorkshopAppearancePreferenceService:
    """Persist an allowlisted theme selection per canonical human principal."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: Path) -> WorkshopAppearancePreferenceService:
        if not path.is_file():
            raise WorkshopAppearancePreferenceStorageError("Appearance preference database is unavailable")
        connection = await aiosqlite.connect(str(path))
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA busy_timeout=5000")
        await connection.execute("PRAGMA foreign_keys=ON")
        return cls(connection)

    async def close(self) -> None:
        await self._connection.close()

    def authority_for_principal(
        self,
        principal_id: str | PrincipalId,
    ) -> AppearancePreferenceAuthority:
        try:
            canonical = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
        except (TypeError, ValueError) as exc:
            raise WorkshopAppearancePreferenceAccessDenied("Appearance preference access denied") from exc
        return AppearancePreferenceAuthority(canonical)

    async def inspect(
        self,
        authority: AppearancePreferenceAuthority,
    ) -> AppearancePreferenceSnapshot:
        async with self._lock:
            return await self._snapshot_locked(authority)

    async def set_theme(
        self,
        authority: AppearancePreferenceAuthority,
        theme_id: str,
        *,
        expected_revision: str,
    ) -> AppearancePreferenceSnapshot:
        normalized = theme_id.strip().lower()
        if normalized not in _THEMES_BY_ID:
            raise WorkshopAppearancePreferenceValidationError("Workshop theme is unsupported")
        async with self._lock:
            try:
                await self._connection.execute("BEGIN IMMEDIATE")
                before = await self._snapshot_locked(authority)
                if before.revision != expected_revision:
                    raise WorkshopAppearancePreferenceConflict("Appearance preferences changed since they were loaded")
                if normalized == DEFAULT_WORKSHOP_THEME:
                    await self._connection.execute(
                        "DELETE FROM principal_appearance_preferences WHERE principal_id = ?",
                        (str(authority.principal_id),),
                    )
                else:
                    await self._connection.execute(
                        "INSERT INTO principal_appearance_preferences "
                        "(principal_id, theme_id) VALUES (?, ?) "
                        "ON CONFLICT(principal_id) DO UPDATE SET theme_id = excluded.theme_id, "
                        "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                        (str(authority.principal_id), normalized),
                    )
                after = await self._snapshot_locked(authority)
                await self._connection.commit()
            except WorkshopAppearancePreferenceError:
                await self._connection.rollback()
                raise
            except Exception as exc:
                await self._connection.rollback()
                raise WorkshopAppearancePreferenceStorageError("Appearance preferences could not be saved") from exc
            return AppearancePreferenceSnapshot(
                after.theme_id,
                after.themes,
                after.revision,
                AppearancePreferenceMutation("set_theme", before.revision != after.revision),
            )

    async def _snapshot_locked(
        self,
        authority: AppearancePreferenceAuthority,
    ) -> AppearancePreferenceSnapshot:
        async with self._connection.execute(
            "SELECT kind FROM principals WHERE id = ?",
            (str(authority.principal_id),),
        ) as cursor:
            principal = await cursor.fetchone()
        if principal is None or str(principal[0]) != "human":
            raise WorkshopAppearancePreferenceAccessDenied("Appearance preference access denied")
        async with self._connection.execute(
            "SELECT theme_id FROM principal_appearance_preferences WHERE principal_id = ?",
            (str(authority.principal_id),),
        ) as cursor:
            row = await cursor.fetchone()
        stored_theme = str(row[0]) if row is not None else None
        effective_theme = stored_theme if stored_theme in _THEMES_BY_ID else DEFAULT_WORKSHOP_THEME
        revision_material = stored_theme if stored_theme is not None else "<default>"
        revision = (
            "apr_"
            + hashlib.sha256(
                f"kai-workshop-appearance:v1:{authority.principal_id}:{revision_material}".encode()
            ).hexdigest()[:32]
        )
        return AppearancePreferenceSnapshot(
            effective_theme,
            WORKSHOP_APPEARANCE_THEMES,
            revision,
        )
