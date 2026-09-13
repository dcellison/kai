"""Low-level internal API scope vocabulary without Workshop dependencies."""

from __future__ import annotations

from enum import StrEnum


class InternalAPIScope(StrEnum):
    """Operations an internal API credential may perform."""

    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    SERVICES_CALL = "services:call"
    MESSAGES_SEND = "messages:send"
    FILES_SEND = "files:send"
    MEMORY_READ = "memory:read"
    MEMORY_ADD = "memory:add"
    MEMORY_DELETE_ALL = "memory:delete-all"
    COLLABORATION_INVOKE = "collaboration:invoke"


# Construct this explicitly so a future scope is never silently granted.
PERSISTENT_AGENT_BASE_SCOPES = frozenset(
    {
        InternalAPIScope.JOBS_READ,
        InternalAPIScope.JOBS_WRITE,
        InternalAPIScope.MESSAGES_SEND,
        InternalAPIScope.FILES_SEND,
        InternalAPIScope.MEMORY_READ,
        InternalAPIScope.MEMORY_ADD,
        InternalAPIScope.COLLABORATION_INVOKE,
    }
)
