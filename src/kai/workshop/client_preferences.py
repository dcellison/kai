"""Canonical presentation preferences owned by authenticated client bindings."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from kai.workshop.domain import ChannelBindingId, PrincipalId

VOICE_MODE_OFF = "off"
VOICE_MODE_TEXT_AND_VOICE = "text_and_voice"
VOICE_MODE_VOICE_ONLY = "voice_only"
VOICE_MODES = frozenset(
    {
        VOICE_MODE_OFF,
        VOICE_MODE_TEXT_AND_VOICE,
        VOICE_MODE_VOICE_ONLY,
    }
)

_LEGACY_TO_CANONICAL_MODE = {
    "off": VOICE_MODE_OFF,
    "on": VOICE_MODE_TEXT_AND_VOICE,
    "only": VOICE_MODE_VOICE_ONLY,
}
_CANONICAL_TO_LEGACY_MODE = {value: key for key, value in _LEGACY_TO_CANONICAL_MODE.items()}


class WorkshopClientPreferenceError(RuntimeError):
    """Base failure for canonical client-binding preferences."""


class WorkshopClientPreferenceAccessDenied(WorkshopClientPreferenceError):
    """The authenticated principal does not own the selected binding."""


class WorkshopClientPreferenceValidationError(WorkshopClientPreferenceError):
    """A requested client preference is unsupported."""


class WorkshopClientPreferenceConflict(WorkshopClientPreferenceError):
    """Client preferences changed after the caller loaded them."""


class WorkshopClientPreferenceStorageError(WorkshopClientPreferenceError):
    """Canonical client preference state is unavailable or unsafe."""


@dataclass(frozen=True, slots=True)
class ClientVoiceCapability:
    """One adapter-declared voice-output capability."""

    transport: str
    display_name: str
    enabled: bool
    voices: tuple[tuple[str, str], ...]
    default_voice: str

    def __post_init__(self) -> None:
        if not self.transport or not self.display_name:
            raise ValueError("Client voice capability identity must be non-empty")
        voice_keys = {key for key, _label in self.voices}
        if not voice_keys or self.default_voice not in voice_keys:
            raise ValueError("Client voice capability must include its default voice")


@dataclass(frozen=True, slots=True)
class ClientPreferenceAuthority:
    principal_id: PrincipalId


@dataclass(frozen=True, slots=True)
class ClientBindingPreferenceAuthority:
    principal_id: PrincipalId
    binding_id: ChannelBindingId


@dataclass(frozen=True, slots=True)
class VoiceChoice:
    value: str
    display_name: str


@dataclass(frozen=True, slots=True)
class ClientBindingVoicePreference:
    choice_id: str
    client_name: str
    mode: str
    voice: str
    voice_name: str
    editable: bool


@dataclass(frozen=True, slots=True)
class ClientPreferenceMutation:
    operation: str
    changed: bool


@dataclass(frozen=True, slots=True)
class ClientPreferenceSnapshot:
    available: bool
    unavailable_reason: str | None
    bindings: tuple[ClientBindingVoicePreference, ...]
    voices: tuple[VoiceChoice, ...]
    revision: str
    mutation: ClientPreferenceMutation | None = None


@dataclass(frozen=True, slots=True)
class _OwnedBinding:
    binding_id: ChannelBindingId
    principal_id: PrincipalId
    transport: str
    external_subject: str


def _choice_id(principal_id: PrincipalId, binding_id: ChannelBindingId) -> str:
    encoded = f"kai-client-binding-choice:v1:{principal_id}:{binding_id}".encode()
    return "cbd_" + hashlib.sha256(encoded).hexdigest()[:32]


class WorkshopClientPreferenceService:
    """Own voice presentation state by canonical principal and client binding."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        capabilities: tuple[ClientVoiceCapability, ...],
    ) -> None:
        self._connection = connection
        self._capabilities = {item.transport: item for item in capabilities}
        self._lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        path: Path,
        capabilities: tuple[ClientVoiceCapability, ...],
    ) -> WorkshopClientPreferenceService:
        if not path.is_file():
            raise WorkshopClientPreferenceStorageError("Client preference database is unavailable")
        connection = await aiosqlite.connect(str(path))
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA busy_timeout=5000")
        await connection.execute("PRAGMA foreign_keys=ON")
        service = cls(connection, capabilities)
        await service.reconcile_legacy_preferences()
        return service

    async def close(self) -> None:
        await self._connection.close()

    def authority_for_principal(
        self,
        principal_id: str | PrincipalId,
    ) -> ClientPreferenceAuthority:
        try:
            canonical = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
        except (TypeError, ValueError) as exc:
            raise WorkshopClientPreferenceAccessDenied("Client preference access denied") from exc
        return ClientPreferenceAuthority(canonical)

    async def authority_for_transport_binding(
        self,
        *,
        transport: str,
        external_subject: str,
        external_channel_id: str,
    ) -> ClientBindingPreferenceAuthority:
        async with self._lock:
            async with self._connection.execute(
                "SELECT ei.principal_id, cb.id FROM external_identities ei "
                "JOIN channel_bindings cb ON cb.transport = ei.provider "
                "AND cb.external_channel_id = ? "
                "JOIN channels c ON c.id = cb.channel_id AND c.kind = 'direct' "
                "JOIN channel_memberships cm ON cm.channel_id = cb.channel_id "
                "AND cm.principal_id = ei.principal_id "
                "WHERE ei.provider = ? AND ei.external_subject = ?",
                (external_channel_id, transport, external_subject),
            ) as cursor:
                rows = tuple(await cursor.fetchall())
            if len(rows) != 1 or transport not in self._capabilities:
                raise WorkshopClientPreferenceAccessDenied("Client preference access denied")
            return ClientBindingPreferenceAuthority(
                PrincipalId(str(rows[0][0])),
                ChannelBindingId(str(rows[0][1])),
            )

    async def inspect(
        self,
        authority: ClientPreferenceAuthority,
    ) -> ClientPreferenceSnapshot:
        async with self._lock:
            return await self._snapshot_locked(authority)

    async def inspect_binding(
        self,
        authority: ClientBindingPreferenceAuthority,
    ) -> ClientBindingVoicePreference:
        async with self._lock:
            owned = await self._owned_bindings_locked(authority.principal_id)
            binding = next((item for item in owned if item.binding_id == authority.binding_id), None)
            if binding is None:
                raise WorkshopClientPreferenceAccessDenied("Client preference access denied")
            return await self._preference_locked(binding)

    async def set_mode(
        self,
        authority: ClientBindingPreferenceAuthority,
        mode: str,
        *,
        expected_revision: str | None = None,
    ) -> ClientPreferenceSnapshot:
        normalized = mode.strip().lower()
        if normalized not in VOICE_MODES:
            raise WorkshopClientPreferenceValidationError("Voice mode is unsupported")
        return await self._mutate(
            authority,
            field="mode",
            value=normalized,
            expected_revision=expected_revision,
        )

    async def set_voice(
        self,
        authority: ClientBindingPreferenceAuthority,
        voice: str,
        *,
        expected_revision: str | None = None,
        enable_if_off: bool = False,
    ) -> ClientPreferenceSnapshot:
        normalized = voice.strip().lower()
        async with self._lock:
            binding = await self._require_owned_binding_locked(authority)
            capability = self._capabilities[binding.transport]
            if not capability.enabled:
                raise WorkshopClientPreferenceAccessDenied("Client voice output is unavailable")
            if normalized not in {key for key, _label in capability.voices}:
                raise WorkshopClientPreferenceValidationError("Voice is unsupported")
            before = await self._snapshot_locked(ClientPreferenceAuthority(authority.principal_id))
            self._check_revision(before, expected_revision)
            current = await self._preference_locked(binding)
            mode = VOICE_MODE_VOICE_ONLY if enable_if_off and current.mode == VOICE_MODE_OFF else current.mode
            changed = current.voice != normalized or current.mode != mode
            if changed:
                await self._write_locked(binding, mode=mode, voice=normalized)
            after = await self._snapshot_locked(ClientPreferenceAuthority(authority.principal_id))
            return ClientPreferenceSnapshot(
                after.available,
                after.unavailable_reason,
                after.bindings,
                after.voices,
                after.revision,
                ClientPreferenceMutation("set_client_voice", changed),
            )

    async def set_choice_mode(
        self,
        authority: ClientPreferenceAuthority,
        choice_id: str,
        mode: str,
        *,
        expected_revision: str,
    ) -> ClientPreferenceSnapshot:
        binding = await self._authority_for_choice(authority, choice_id)
        return await self.set_mode(binding, mode, expected_revision=expected_revision)

    async def set_choice_voice(
        self,
        authority: ClientPreferenceAuthority,
        choice_id: str,
        voice: str,
        *,
        expected_revision: str,
    ) -> ClientPreferenceSnapshot:
        binding = await self._authority_for_choice(authority, choice_id)
        return await self.set_voice(binding, voice, expected_revision=expected_revision)

    async def reconcile_legacy_preferences(self) -> None:
        """Migrate each eligible binding once while retaining rollback mirrors."""
        async with self._lock:
            try:
                await self._connection.execute("BEGIN IMMEDIATE")
                bindings = await self._all_owned_bindings_locked()
                for binding in bindings:
                    async with self._connection.execute(
                        "SELECT 1 FROM client_binding_voice_migrations WHERE channel_binding_id = ?",
                        (binding.binding_id,),
                    ) as cursor:
                        if await cursor.fetchone() is not None:
                            continue
                    mode_value = await self._legacy_setting_locked(f"voice_mode:{binding.external_subject}")
                    voice_value = await self._legacy_setting_locked(f"voice_name:{binding.external_subject}")
                    capability = self._capabilities[binding.transport]
                    mode = _LEGACY_TO_CANONICAL_MODE.get(mode_value or "", VOICE_MODE_OFF)
                    voice_keys = {key for key, _label in capability.voices}
                    voice = voice_value if voice_value in voice_keys else capability.default_voice
                    await self._connection.execute(
                        "INSERT OR IGNORE INTO client_binding_voice_preferences "
                        "(channel_binding_id, principal_id, mode, voice_name) VALUES (?, ?, ?, ?)",
                        (binding.binding_id, binding.principal_id, mode, voice),
                    )
                    await self._connection.execute(
                        "INSERT INTO client_binding_voice_migrations "
                        "(channel_binding_id, principal_id, mode_migrated, voice_migrated) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            binding.binding_id,
                            binding.principal_id,
                            int(mode_value is not None),
                            int(voice_value is not None),
                        ),
                    )
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise

    async def _mutate(
        self,
        authority: ClientBindingPreferenceAuthority,
        *,
        field: str,
        value: str,
        expected_revision: str | None,
    ) -> ClientPreferenceSnapshot:
        async with self._lock:
            binding = await self._require_owned_binding_locked(authority)
            if not self._capabilities[binding.transport].enabled:
                raise WorkshopClientPreferenceAccessDenied("Client voice output is unavailable")
            before = await self._snapshot_locked(ClientPreferenceAuthority(authority.principal_id))
            self._check_revision(before, expected_revision)
            current = await self._preference_locked(binding)
            changed = current.mode != value
            if changed:
                await self._write_locked(binding, mode=value, voice=current.voice)
            after = await self._snapshot_locked(ClientPreferenceAuthority(authority.principal_id))
            return ClientPreferenceSnapshot(
                after.available,
                after.unavailable_reason,
                after.bindings,
                after.voices,
                after.revision,
                ClientPreferenceMutation(f"set_client_voice_{field}", changed),
            )

    async def _authority_for_choice(
        self,
        authority: ClientPreferenceAuthority,
        choice_id: str,
    ) -> ClientBindingPreferenceAuthority:
        async with self._lock:
            binding = next(
                (
                    item
                    for item in await self._owned_bindings_locked(authority.principal_id)
                    if _choice_id(authority.principal_id, item.binding_id) == choice_id
                ),
                None,
            )
        if binding is None:
            raise WorkshopClientPreferenceAccessDenied("Client preference access denied")
        return ClientBindingPreferenceAuthority(authority.principal_id, binding.binding_id)

    @staticmethod
    def _check_revision(snapshot: ClientPreferenceSnapshot, expected_revision: str | None) -> None:
        if expected_revision is not None and expected_revision != snapshot.revision:
            raise WorkshopClientPreferenceConflict("Client preferences changed since they were loaded")

    async def _require_owned_binding_locked(
        self,
        authority: ClientBindingPreferenceAuthority,
    ) -> _OwnedBinding:
        binding = next(
            (
                item
                for item in await self._owned_bindings_locked(authority.principal_id)
                if item.binding_id == authority.binding_id
            ),
            None,
        )
        if binding is None:
            raise WorkshopClientPreferenceAccessDenied("Client preference access denied")
        return binding

    async def _all_owned_bindings_locked(self) -> tuple[_OwnedBinding, ...]:
        if not self._capabilities:
            return ()
        placeholders = ",".join("?" for _item in self._capabilities)
        async with self._connection.execute(
            "SELECT DISTINCT cb.id, ei.principal_id, cb.transport, ei.external_subject "
            "FROM external_identities ei "
            "JOIN channel_bindings cb ON cb.transport = ei.provider "
            "AND cb.external_channel_id = ei.external_subject "
            "JOIN channels c ON c.id = cb.channel_id AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = cb.channel_id "
            "AND cm.principal_id = ei.principal_id "
            f"WHERE cb.transport IN ({placeholders}) ORDER BY ei.principal_id, cb.id",
            tuple(self._capabilities),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        return tuple(
            _OwnedBinding(
                ChannelBindingId(str(row[0])),
                PrincipalId(str(row[1])),
                str(row[2]),
                str(row[3]),
            )
            for row in rows
        )

    async def _owned_bindings_locked(self, principal_id: PrincipalId) -> tuple[_OwnedBinding, ...]:
        return tuple(item for item in await self._all_owned_bindings_locked() if item.principal_id == principal_id)

    async def _preference_locked(self, binding: _OwnedBinding) -> ClientBindingVoicePreference:
        capability = self._capabilities[binding.transport]
        async with self._connection.execute(
            "SELECT mode, voice_name FROM client_binding_voice_preferences "
            "WHERE channel_binding_id = ? AND principal_id = ?",
            (binding.binding_id, binding.principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopClientPreferenceStorageError("Canonical client preference is missing")
        mode = str(row[0])
        voice = str(row[1])
        labels = dict(capability.voices)
        if mode not in VOICE_MODES or voice not in labels:
            raise WorkshopClientPreferenceStorageError("Canonical client preference is invalid")
        return ClientBindingVoicePreference(
            _choice_id(binding.principal_id, binding.binding_id),
            capability.display_name,
            mode,
            voice,
            labels[voice],
            capability.enabled,
        )

    async def _snapshot_locked(self, authority: ClientPreferenceAuthority) -> ClientPreferenceSnapshot:
        bindings = await self._owned_bindings_locked(authority.principal_id)
        preferences = tuple([await self._preference_locked(item) for item in bindings])
        capabilities = tuple(self._capabilities[item.transport] for item in bindings)
        enabled = any(item.enabled for item in capabilities)
        voices: dict[str, str] = {}
        for capability in capabilities:
            voices.update(capability.voices)
        revision_payload = {
            "principal_id": str(authority.principal_id),
            "bindings": [
                [item.choice_id, item.client_name, item.mode, item.voice, item.editable] for item in preferences
            ],
        }
        revision = (
            "cvp_"
            + hashlib.sha256(json.dumps(revision_payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()[
                :32
            ]
        )
        return ClientPreferenceSnapshot(
            enabled,
            None if enabled else "Voice output is not enabled for an eligible client.",
            preferences,
            tuple(VoiceChoice(key, label) for key, label in sorted(voices.items())),
            revision,
        )

    async def _write_locked(self, binding: _OwnedBinding, *, mode: str, voice: str) -> None:
        legacy_mode = _CANONICAL_TO_LEGACY_MODE[mode]
        try:
            await self._connection.execute("BEGIN IMMEDIATE")
            await self._connection.execute(
                "UPDATE client_binding_voice_preferences SET mode = ?, voice_name = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE channel_binding_id = ? AND principal_id = ?",
                (mode, voice, binding.binding_id, binding.principal_id),
            )
            await self._connection.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (f"voice_mode:{binding.external_subject}", legacy_mode),
            )
            await self._connection.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (f"voice_name:{binding.external_subject}", voice),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    async def _legacy_setting_locked(self, key: str) -> str | None:
        async with self._connection.execute(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()
        return None if row is None else str(row[0])
