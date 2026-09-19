"""Checked adapter entry-point bindings to canonical capability operations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kai.capability_boundaries import CANONICAL_SERVICE_BOUNDARIES
from kai.capability_registry import AdapterDisposition, AdapterId, capability_by_id


@dataclass(frozen=True, slots=True)
class AdapterOperationBinding:
    """Checked association between one adapter entry point and an operation."""

    adapter: AdapterId
    operation_id: str
    entrypoint: str
    disposition: AdapterDisposition
    canonical_service: str


_ENTRYPOINT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_WORKSHOP_ROUTE_PREFIX = "capability__"


def bind_adapter_operation(
    adapter: AdapterId,
    operation_id: str,
    entrypoint: str,
) -> AdapterOperationBinding:
    """Validate and return an adapter-to-canonical-operation binding."""
    if _ENTRYPOINT_PATTERN.fullmatch(entrypoint) is None:
        raise ValueError("Adapter operation entry points must be stable lowercase identifiers")
    capability = capability_by_id(operation_id)
    presentation = capability.presentations[adapter]
    if not presentation.implemented or presentation.disposition == AdapterDisposition.UNSUPPORTED_BY_DESIGN:
        raise ValueError(f"Capability {operation_id} is not implemented by {adapter.value}")
    if capability.canonical_service not in CANONICAL_SERVICE_BOUNDARIES:
        raise ValueError(f"Capability {operation_id} names an unregistered canonical service")
    return AdapterOperationBinding(
        adapter,
        operation_id,
        entrypoint,
        presentation.disposition,
        capability.canonical_service,
    )


def workshop_route_name(operation_id: str, entrypoint: str) -> str:
    """Return a unique aiohttp route name carrying its operation identity."""
    binding = bind_adapter_operation(AdapterId.WORKSHOP, operation_id, entrypoint)
    encoded_operation = binding.operation_id.replace(".", "_")
    return f"{_WORKSHOP_ROUTE_PREFIX}{encoded_operation}__{binding.entrypoint}"
