"""Core-owned eligibility policy for optional delivery adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kai.workshop.domain import ChannelBindingId, ChannelId, PrincipalId, WorkshopId
from kai.workshop.store import WorkshopEventStore

_TRANSPORT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


@dataclass(frozen=True, slots=True)
class DeliveryAdapterCapabilities:
    """Immutable delivery features declared by one enabled adapter."""

    transport: str
    final_text: bool = True
    preview_streaming: bool = False
    message_editing: bool = False
    replies: bool = False
    threads: bool = False
    attachments: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.transport, str) or not _TRANSPORT_PATTERN.fullmatch(self.transport):
            raise ValueError("transport must be a lowercase transport identifier")
        for field_name in (
            "final_text",
            "preview_streaming",
            "message_editing",
            "replies",
            "threads",
            "attachments",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")
        if self.preview_streaming and (not self.final_text or not self.message_editing):
            raise ValueError("preview_streaming requires final_text and message_editing")


@dataclass(frozen=True, slots=True)
class EligibleDeliveryBinding:
    """One persisted destination paired with its current adapter capabilities."""

    binding_id: ChannelBindingId
    transport: str
    capabilities: DeliveryAdapterCapabilities


@dataclass(frozen=True, slots=True)
class WorkshopDeliveryBindingPolicy:
    """Resolve persisted bindings only for currently enabled transports.

    A binding records a possible external destination. It does not by itself
    authorize delivery while its adapter is disabled.
    """

    enabled_transports: frozenset[str]
    adapter_capabilities: tuple[DeliveryAdapterCapabilities, ...] = ()
    notification_deep_link: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled_transports, frozenset):
            raise TypeError("enabled_transports must be a frozenset")
        if any(
            not isinstance(transport, str) or not _TRANSPORT_PATTERN.fullmatch(transport)
            for transport in self.enabled_transports
        ):
            raise ValueError("enabled_transports must contain lowercase transport identifiers")
        if not isinstance(self.adapter_capabilities, tuple) or any(
            not isinstance(capabilities, DeliveryAdapterCapabilities) for capabilities in self.adapter_capabilities
        ):
            raise TypeError("adapter_capabilities must contain DeliveryAdapterCapabilities values")
        transports = tuple(capabilities.transport for capabilities in self.adapter_capabilities)
        if len(set(transports)) != len(transports):
            raise ValueError("adapter_capabilities must identify unique transports")
        declared = set(transports)
        if declared - self.enabled_transports:
            raise ValueError("adapter_capabilities cannot describe a disabled transport")
        if self.enabled_transports - declared:
            raise ValueError("every enabled transport must declare adapter capabilities")
        if self.notification_deep_link is not None and (
            not isinstance(self.notification_deep_link, str)
            or not self.notification_deep_link.startswith(("http://", "https://"))
            or len(self.notification_deep_link) > 1000
        ):
            raise ValueError("notification_deep_link must be a bounded HTTP(S) URL when supplied")

    @classmethod
    def disabled(cls) -> WorkshopDeliveryBindingPolicy:
        return cls(frozenset())

    def is_enabled(self, transport: str) -> bool:
        if not isinstance(transport, str) or not _TRANSPORT_PATTERN.fullmatch(transport):
            raise ValueError("transport must be a lowercase transport identifier")
        return transport in self.enabled_transports

    def capabilities_for(self, transport: str) -> DeliveryAdapterCapabilities | None:
        """Return declared capabilities for one enabled transport."""
        if not self.is_enabled(transport):
            return None
        for capabilities in self.adapter_capabilities:
            if capabilities.transport == transport:
                return capabilities
        raise RuntimeError("Enabled adapter has no capability declaration")

    async def bindings(
        self,
        store: WorkshopEventStore,
        channel_id: ChannelId,
        *,
        principal_id: PrincipalId | None = None,
    ) -> tuple[EligibleDeliveryBinding, ...]:
        """Return binding destinations currently eligible for publication."""
        if not isinstance(channel_id, ChannelId):
            raise ValueError("channel_id must be a ChannelId")
        if principal_id is not None and not isinstance(principal_id, PrincipalId):
            raise ValueError("principal_id must be a PrincipalId when supplied")
        if not self.enabled_transports:
            return ()
        placeholders = ", ".join("?" for _ in self.enabled_transports)
        identity_filter = (
            "AND EXISTS (SELECT 1 FROM external_identities ei "
            "WHERE ei.principal_id = ? AND ei.provider = cb.transport "
            "AND ei.external_subject = cb.external_channel_id) "
            if principal_id is not None
            else ""
        )
        parameters: tuple[object, ...] = (
            channel_id,
            *sorted(self.enabled_transports),
            *((principal_id,) if principal_id is not None else ()),
        )
        async with store.connection.execute(
            "SELECT cb.id, cb.transport FROM channel_bindings cb "
            f"WHERE cb.channel_id = ? AND cb.transport IN ({placeholders}) "
            f"{identity_filter}ORDER BY cb.id",
            parameters,
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        bindings: list[EligibleDeliveryBinding] = []
        for row in rows:
            transport = str(row[1])
            capabilities = self.capabilities_for(transport)
            assert capabilities is not None
            bindings.append(
                EligibleDeliveryBinding(
                    binding_id=ChannelBindingId(str(row[0])),
                    transport=transport,
                    capabilities=capabilities,
                )
            )
        return tuple(bindings)

    async def binding_ids(
        self,
        store: WorkshopEventStore,
        channel_id: ChannelId,
        *,
        transport: str | None = None,
    ) -> tuple[ChannelBindingId, ...]:
        """Return stable bindings eligible under the current adapter policy."""
        if not isinstance(channel_id, ChannelId):
            raise ValueError("channel_id must be a ChannelId")
        bindings = await self.bindings(store, channel_id)
        return tuple(binding.binding_id for binding in bindings if transport is None or binding.transport == transport)

    async def principal_bindings(
        self,
        store: WorkshopEventStore,
        workshop_id: WorkshopId,
        principal_id: PrincipalId,
    ) -> tuple[EligibleDeliveryBinding, ...]:
        """Resolve enabled adapter destinations from a canonical recipient.

        The caller supplies no transport identity. A destination is eligible
        only while the principal's current external identity, direct channel
        binding, and channel membership all agree.
        """
        if not isinstance(workshop_id, WorkshopId):
            raise ValueError("workshop_id must be a WorkshopId")
        if not isinstance(principal_id, PrincipalId):
            raise ValueError("principal_id must be a PrincipalId")
        if not self.enabled_transports:
            return ()
        placeholders = ", ".join("?" for _ in self.enabled_transports)
        async with store.connection.execute(
            "SELECT DISTINCT cb.id, cb.transport FROM external_identities ei "
            "JOIN channel_bindings cb ON cb.transport = ei.provider "
            "AND cb.external_channel_id = ei.external_subject "
            "JOIN channels c ON c.id = cb.channel_id AND c.kind = 'direct' "
            "AND c.workshop_id = ? "
            "JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = ei.principal_id "
            "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
            "AND wm.principal_id = ei.principal_id "
            f"WHERE ei.principal_id = ? AND cb.transport IN ({placeholders}) "
            "ORDER BY cb.transport, cb.id",
            (workshop_id, principal_id, *sorted(self.enabled_transports)),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        bindings: list[EligibleDeliveryBinding] = []
        for row in rows:
            transport = str(row[1])
            capabilities = self.capabilities_for(transport)
            assert capabilities is not None
            bindings.append(
                EligibleDeliveryBinding(
                    binding_id=ChannelBindingId(str(row[0])),
                    transport=transport,
                    capabilities=capabilities,
                )
            )
        return tuple(bindings)
