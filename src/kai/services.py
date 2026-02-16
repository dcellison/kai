"""
Declarative HTTP service layer for external API integrations.

Provides functionality to:
1. Load service definitions from a YAML config file (services.yaml)
2. Validate service configs and check that required env vars are set
3. Inject authentication (bearer, header, query param) into outbound requests
4. Proxy HTTP calls to external APIs on behalf of the inner Claude process

Services are defined as inert data — URL, method, auth type, headers. No code
execution, no plugins, no middleware. Kai's own process makes the HTTP call
directly and returns the result. API keys live in .env and are never exposed
to the inner Claude process or included in conversation context.

The proxy flow:
    Inner Claude  --curl-->  localhost:8080/api/services/{name}  --HTTP-->  External API
                             (webhook.py injects auth from .env)

This follows the same local-proxy pattern as the scheduling API — inner Claude
already uses curl to hit local endpoints. The service layer extends this pattern
to external API calls with authentication injection.

Auth injection by type:
    bearer  ->  Authorization: Bearer $KEY
    header  ->  {auth.name}: $KEY
    query   ->  ?{auth.name}=$KEY
    none    ->  no auth
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import yaml

log = logging.getLogger(__name__)

# Valid authentication types for service definitions
_VALID_AUTH_TYPES = {"bearer", "header", "query", "none"}

# Module-level service registry, populated by load_services() at startup
_services: dict[str, "ServiceDef"] | None = None


# ── Data classes ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthConfig:
    """
    Authentication configuration for an external service.

    Attributes:
        type: Auth injection method — "bearer", "header", "query", or "none".
        env: Environment variable name containing the API key (e.g., "PERPLEXITY_API_KEY").
        name: Header or query parameter name for "header"/"query" types (e.g., "X-Subscription-Token").
        optional: If True, the service remains available even when the env var is unset.
    """

    type: str
    env: str = ""
    name: str = ""
    optional: bool = False


@dataclass(frozen=True)
class ServiceDef:
    """
    A complete external service definition loaded from services.yaml.

    Each service describes a single HTTP endpoint with its URL, method,
    authentication, and optional static headers/params. The description
    and notes fields are injected into inner Claude's context so it knows
    how to use the service.

    Attributes:
        name: Unique identifier used in the proxy URL (e.g., "perplexity").
        url: Base URL for the API endpoint.
        method: HTTP method — "GET" or "POST".
        auth: Authentication configuration.
        headers: Static headers to include on every request.
        params: Static query parameters to include on every request.
        description: Human-readable description of what the service does.
        notes: Usage hints for inner Claude (e.g., expected request body format).
    """

    name: str
    url: str
    method: str
    auth: AuthConfig
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    description: str = ""
    notes: str = ""


@dataclass
class ServiceResponse:
    """
    Result from calling an external service.

    Attributes:
        success: True if the HTTP call completed (even with non-2xx status).
        status: HTTP status code from the external API.
        body: Response body as a string.
        error: Error message if the call failed entirely (timeout, connection error, etc.).
    """

    success: bool
    status: int = 0
    body: str = ""
    error: str = ""


# ── Service loading and validation ──────────────────────────────────


def load_services(config_path: Path) -> dict[str, ServiceDef]:
    """
    Parse a services.yaml file, validate each entry, and populate the module registry.

    Missing file is not an error — returns an empty dict (graceful degradation).
    Invalid YAML syntax is a fatal error — raises SystemExit to prevent startup
    with a silently broken config.

    For each service entry:
    - Validates URL is present and auth type is recognized
    - Checks that the referenced env var is set (skips the service with a warning
      if missing, unless auth.optional is True)
    - Normalizes the HTTP method to uppercase

    Args:
        config_path: Path to the services.yaml file.

    Returns:
        Dict mapping service names to their validated ServiceDef objects.

    Raises:
        SystemExit: If the YAML file exists but contains invalid syntax.
    """
    global _services

    if not config_path.exists():
        log.info("No services config at %s — external services disabled", config_path)
        _services = {}
        return _services

    try:
        raw = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as e:
        log.critical("Invalid YAML in %s: %s", config_path, e)
        raise SystemExit(1) from e

    # Handle empty file or non-dict top level
    if not isinstance(raw, dict):
        log.warning("Services config is not a YAML mapping — ignoring")
        _services = {}
        return _services

    services_raw = raw.get("services", {})
    if not isinstance(services_raw, dict):
        log.warning("'services' key is not a mapping — ignoring")
        _services = {}
        return _services

    result: dict[str, ServiceDef] = {}

    for name, entry in services_raw.items():
        if not isinstance(entry, dict):
            log.warning("Service '%s': entry is not a mapping — skipping", name)
            continue

        # URL is required
        url = entry.get("url")
        if not url:
            log.warning("Service '%s': missing 'url' — skipping", name)
            continue

        # Parse and validate auth config
        auth_raw = entry.get("auth", {})
        if not isinstance(auth_raw, dict):
            log.warning("Service '%s': 'auth' is not a mapping — skipping", name)
            continue

        auth_type = auth_raw.get("type", "none")
        if auth_type not in _VALID_AUTH_TYPES:
            log.warning("Service '%s': invalid auth type '%s' — skipping", name, auth_type)
            continue

        auth = AuthConfig(
            type=auth_type,
            env=auth_raw.get("env", ""),
            name=auth_raw.get("name", ""),
            optional=bool(auth_raw.get("optional", False)),
        )

        # Check that the referenced env var is actually set
        if auth.type != "none" and auth.env and not os.environ.get(auth.env):
            if auth.optional:
                log.info("Service '%s': env var %s not set (optional) — available without auth", name, auth.env)
            else:
                log.warning("Service '%s': env var %s not set — skipping", name, auth.env)
                continue

        # Parse static headers and params
        headers = entry.get("headers", {})
        if not isinstance(headers, dict):
            headers = {}

        params = entry.get("params", {})
        if not isinstance(params, dict):
            params = {}

        method = str(entry.get("method", "GET")).upper()

        result[name] = ServiceDef(
            name=name,
            url=url,
            method=method,
            auth=auth,
            headers={str(k): str(v) for k, v in headers.items()},
            params={str(k): str(v) for k, v in params.items()},
            description=entry.get("description", ""),
            notes=entry.get("notes", ""),
        )

    _services = result
    return _services


# ── Module-level accessors ──────────────────────────────────────────


def get_services() -> dict[str, ServiceDef]:
    """Return the loaded service registry, raising if load_services() hasn't been called."""
    assert _services is not None, "Services not loaded — call load_services() first"
    return _services


def get_available_services() -> list[dict]:
    """
    Return metadata for all loaded services (for context injection).

    Returns a list of dicts with name, description, method, and notes —
    the information inner Claude needs to know what services are available
    and how to use them. No auth details are included.
    """
    assert _services is not None, "Services not loaded — call load_services() first"
    return [
        {
            "name": s.name,
            "description": s.description,
            "method": s.method,
            "notes": s.notes,
        }
        for s in _services.values()
    ]


# ── HTTP proxy call ─────────────────────────────────────────────────


async def call_service(
    name: str,
    *,
    body: dict | None = None,
    params: dict[str, str] | None = None,
    path_suffix: str = "",
) -> ServiceResponse:
    """
    Call an external service by name, injecting authentication from env vars.

    Constructs the HTTP request from the service definition, resolves the API
    key from the environment at call time (not at load time — supports key
    rotation without restart), and makes the request via aiohttp.

    A new ClientSession is created per call. This is simple and adequate for
    the expected call frequency (a few per conversation, not hundreds per second).

    Args:
        name: Service name as defined in services.yaml.
        body: Optional JSON body for POST requests.
        params: Optional query parameters (merged with static params from config).
        path_suffix: Optional path appended to the base URL (e.g., for Jina Reader URLs).

    Returns:
        ServiceResponse with success status, HTTP status code, and response body.
    """
    services = get_services()

    if name not in services:
        return ServiceResponse(success=False, error=f"Unknown service: {name}")

    svc = services[name]

    # Resolve the API key from env at call time (supports rotation without restart)
    api_key = ""
    if svc.auth.type != "none" and svc.auth.env:
        api_key = os.environ.get(svc.auth.env, "")
        if not api_key and not svc.auth.optional:
            return ServiceResponse(success=False, error=f"Env var {svc.auth.env} not set")

    # Build request headers — start with static headers from config
    headers = dict(svc.headers)

    # Inject auth into headers based on type
    if api_key:
        if svc.auth.type == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif svc.auth.type == "header" and svc.auth.name:
            headers[svc.auth.name] = api_key

    # Build query params — merge static config params with caller-provided params
    merged_params = dict(svc.params)
    if params:
        merged_params.update(params)

    # Inject auth into query params if applicable
    if api_key and svc.auth.type == "query" and svc.auth.name:
        merged_params[svc.auth.name] = api_key

    # Construct the full URL (base + optional path suffix)
    url = svc.url + path_suffix

    log.info("Calling service '%s': %s %s", name, svc.method, url)

    try:
        async with aiohttp.ClientSession() as session:
            # 30-second timeout — generous enough for slow APIs, short enough
            # to avoid blocking the proxy endpoint indefinitely
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.request(
                method=svc.method,
                url=url,
                headers=headers,
                params=merged_params if merged_params else None,
                json=body if body else None,
                timeout=timeout,
            ) as resp:
                response_body = await resp.text()
                log.info("Service '%s' responded: %d (%d bytes)", name, resp.status, len(response_body))
                return ServiceResponse(
                    success=True,
                    status=resp.status,
                    body=response_body,
                )
    except TimeoutError:
        log.warning("Service '%s' timed out", name)
        return ServiceResponse(success=False, error="Request timed out (30s)")
    except aiohttp.ClientError as e:
        log.warning("Service '%s' connection error: %s", name, e)
        return ServiceResponse(success=False, error=f"Connection error: {e}")
