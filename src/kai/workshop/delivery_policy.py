"""Core-owned eligibility policy for optional delivery adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kai.workshop.domain import ChannelBindingId, ChannelId
from kai.workshop.store import WorkshopEventStore

_TRANSPORT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


@dataclass(frozen=True, slots=True)
class WorkshopDeliveryBindingPolicy:
    """Resolve persisted bindings only for currently enabled transports.

    A binding records a possible external destination. It does not by itself
    authorize delivery while its adapter is disabled.
    """

    enabled_transports: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled_transports, frozenset):
            raise TypeError("enabled_transports must be a frozenset")
        if any(
            not isinstance(transport, str) or not _TRANSPORT_PATTERN.fullmatch(transport)
            for transport in self.enabled_transports
        ):
            raise ValueError("enabled_transports must contain lowercase transport identifiers")

    @classmethod
    def disabled(cls) -> WorkshopDeliveryBindingPolicy:
        return cls(frozenset())

    def is_enabled(self, transport: str) -> bool:
        if not isinstance(transport, str) or not _TRANSPORT_PATTERN.fullmatch(transport):
            raise ValueError("transport must be a lowercase transport identifier")
        return transport in self.enabled_transports

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
        transports = self.enabled_transports
        if transport is not None:
            transports = frozenset({transport}) if self.is_enabled(transport) else frozenset()
        if not transports:
            return ()
        placeholders = ", ".join("?" for _ in transports)
        async with store.connection.execute(
            f"SELECT id FROM channel_bindings WHERE channel_id = ? AND transport IN ({placeholders}) ORDER BY id",
            (channel_id, *sorted(transports)),
        ) as cursor:
            return tuple(ChannelBindingId(str(row[0])) for row in await cursor.fetchall())
