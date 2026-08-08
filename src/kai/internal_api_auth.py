"""Process-local authentication and authorization for Kai's internal HTTP API.

Persistent and one-shot agents need a narrow way to call the loopback API, but
must never receive an external webhook signing secret or choose another user's
identity. This module issues random, process-lifetime credentials that resolve
to a server-owned principal and an explicit set of API scopes.
"""

from __future__ import annotations

import hmac
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum


class InternalAPIScope(StrEnum):
    """Operations an internal API credential may perform."""

    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    SERVICES_CALL = "services:call"
    MESSAGES_SEND = "messages:send"
    FILES_SEND = "files:send"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"


_AGENT_SCOPES = frozenset(InternalAPIScope)
_NOTIFICATION_SCOPES = frozenset({InternalAPIScope.MESSAGES_SEND})


@dataclass(frozen=True)
class InternalAPIPrincipal:
    """Identity and authority resolved from an internal API credential."""

    chat_id: int
    scopes: frozenset[InternalAPIScope]

    def allows(self, scope: InternalAPIScope) -> bool:
        """Return whether this principal has the requested API scope."""
        return scope in self.scopes


class InternalAPIAuth:
    """Issue and resolve random credentials for internal API principals.

    Credentials live only for the lifetime of the outer Kai process. Persistent
    backends are lazy children of that same process, so they receive the current
    credential whenever they start and do not require an at-rest token store.

    Args:
        agent_credentials: Optional fixed per-user credentials. Production code
            leaves this unset and receives cryptographically random tokens; the
            explicit form keeps direct handler tests deterministic.
    """

    def __init__(self, agent_credentials: Mapping[int, str] | None = None) -> None:
        self._principals_by_credential: dict[str, InternalAPIPrincipal] = {}
        self._credentials_by_principal: dict[InternalAPIPrincipal, str] = {}

        for chat_id, credential in (agent_credentials or {}).items():
            self._register(
                credential,
                InternalAPIPrincipal(chat_id=chat_id, scopes=_AGENT_SCOPES),
            )

    @classmethod
    def for_users(cls, user_ids: Iterable[int]) -> InternalAPIAuth:
        """Create an auth store with a full agent credential for each user."""
        auth = cls()
        for chat_id in sorted(set(user_ids)):
            auth.agent_credential_for(chat_id)
        return auth

    def agent_credential_for(self, chat_id: int) -> str:
        """Return the full internal API credential for a persistent user agent."""
        return self._credential_for(InternalAPIPrincipal(chat_id=chat_id, scopes=_AGENT_SCOPES))

    def notification_credential_for(self, chat_id: int) -> str:
        """Return a send-message-only credential for a notification agent."""
        return self._credential_for(InternalAPIPrincipal(chat_id=chat_id, scopes=_NOTIFICATION_SCOPES))

    def authenticate(self, credential: str) -> InternalAPIPrincipal | None:
        """Resolve a credential to its server-owned principal, if valid.

        Every candidate is compared with ``hmac.compare_digest``. The store is
        intentionally small (normally one or two credentials per configured
        user), so avoiding a normal dictionary lookup keeps the comparison
        behavior independent of secret prefix or string hash behavior.
        """
        if not credential:
            return None

        matched: InternalAPIPrincipal | None = None
        for expected, principal in self._principals_by_credential.items():
            if hmac.compare_digest(credential, expected):
                matched = principal
        return matched

    def _credential_for(self, principal: InternalAPIPrincipal) -> str:
        """Return an existing credential for a principal or issue a new one."""
        existing = self._credentials_by_principal.get(principal)
        if existing is not None:
            return existing

        while True:
            credential = secrets.token_urlsafe(32)
            if credential not in self._principals_by_credential:
                break
        self._register(credential, principal)
        return credential

    def _register(self, credential: str, principal: InternalAPIPrincipal) -> None:
        """Register one non-empty, unique credential/principal pair."""
        if not credential:
            raise ValueError("Internal API credentials must not be empty")
        if credential in self._principals_by_credential:
            raise ValueError("Internal API credentials must be unique")
        if principal in self._credentials_by_principal:
            raise ValueError("Each internal API principal must have one credential")

        self._principals_by_credential[credential] = principal
        self._credentials_by_principal[principal] = credential
