"""Metadata-only Claude model discovery and subscription fallback policy.

Protocol contract verified 2026-08-28 against Anthropic's Models API and
Claude Code model-configuration documentation. API-key contexts enumerate
with ``GET /v1/models``. Claude subscription authentication has no documented
machine-readable listing command, so those lanes deliberately report
unsupported and retain Kai's curated aliases plus canonical operator entries.
No message or model-generation endpoint is used.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp

from kai.workshop.model_catalogue import (
    ModelCatalogueValidationError,
    ModelDiscoveryAuthenticationError,
    ModelDiscoveryBatch,
    ModelDiscoveryCandidate,
    ModelDiscoveryUnsupported,
)
from kai.workshop.model_discovery_inventory import (
    ModelDiscoveryAuthMode,
    ModelDiscoveryBackendInventory,
    ModelDiscoveryReadiness,
)

_SOURCE = "anthropic-models-api:v1/models"
_DEFAULT_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"
_TTL_SECONDS = 21_600
_PAGE_LIMIT = 1000
_MAX_PAGES = 20
_MAX_MODELS = _PAGE_LIMIT * _MAX_PAGES
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_REQUEST_TIMEOUT_SECONDS = 20.0


class _ClaudeDiscoveryError(RuntimeError):
    """A sanitized Models API transport failure."""


class ClaudeModelDiscoveryAdapter:
    """Enumerate models visible to one canonical Claude API-key context.

    Subscription-backed Claude Code credentials cannot be reused against the
    public Models API and Claude Code documents no metadata-only listing
    command. Returning ``ModelDiscoveryUnsupported`` for that lane makes the
    catalogue expose curated and operator-managed entries without probing an
    interactive picker or invoking a model.
    """

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = os.environ if environment is None else environment

    async def discover(self, lane: ModelDiscoveryBackendInventory) -> ModelDiscoveryBatch:
        api_key, endpoint = self._discovery_context(lane)
        models: list[ModelDiscoveryCandidate] = []
        after_id: str | None = None
        seen_cursors: set[str] = set()
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
        headers = {
            "anthropic-version": _ANTHROPIC_VERSION,
            "x-api-key": api_key,
        }
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for _page in range(_MAX_PAGES):
                    query: dict[str, str] = {"limit": str(_PAGE_LIMIT)}
                    if after_id is not None:
                        query["after_id"] = after_id
                    url = f"{endpoint}?{urlencode(query)}"
                    # Do not forward the API key across redirects. Operators
                    # must configure the final metadata endpoint explicitly.
                    async with session.get(url, headers=headers, allow_redirects=False) as response:
                        if response.status in {401, 403}:
                            raise ModelDiscoveryAuthenticationError
                        if response.status < 200 or response.status >= 300:
                            raise _ClaudeDiscoveryError("Anthropic Models API metadata request failed")
                        body = await response.content.read(_MAX_RESPONSE_BYTES + 1)
                    if len(body) > _MAX_RESPONSE_BYTES:
                        raise ModelCatalogueValidationError(
                            "Anthropic model catalogue response exceeds the safety limit"
                        )
                    page, next_cursor = _parse_page(body)
                    models.extend(page)
                    if len(models) > _MAX_MODELS:
                        raise ModelCatalogueValidationError("Anthropic model catalogue exceeds the safety limit")
                    if next_cursor is None:
                        return ModelDiscoveryBatch(_SOURCE, tuple(models), ttl_seconds=_TTL_SECONDS)
                    if next_cursor in seen_cursors:
                        raise ModelCatalogueValidationError("Anthropic model catalogue repeated a pagination cursor")
                    seen_cursors.add(next_cursor)
                    after_id = next_cursor
        except (ModelCatalogueValidationError, ModelDiscoveryAuthenticationError):
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise _ClaudeDiscoveryError("Anthropic Models API metadata request failed") from exc
        raise ModelCatalogueValidationError("Anthropic model catalogue pagination did not terminate")

    def _discovery_context(self, lane: ModelDiscoveryBackendInventory) -> tuple[str, str]:
        if lane.backend != "claude" or lane.provider != "anthropic":
            raise ModelDiscoveryUnsupported
        if lane.executable.readiness != ModelDiscoveryReadiness.READY:
            raise ModelDiscoveryUnsupported
        if lane.auth.mode == ModelDiscoveryAuthMode.SUBSCRIPTION:
            raise ModelDiscoveryUnsupported
        if lane.auth.mode != ModelDiscoveryAuthMode.API_KEY or lane.auth.configured is not True:
            raise ModelDiscoveryAuthenticationError
        api_key = self._environment.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ModelDiscoveryAuthenticationError
        base_url = self._environment.get("ANTHROPIC_BASE_URL", _DEFAULT_BASE_URL).strip()
        return api_key, _models_endpoint(base_url)


def _models_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ModelCatalogueValidationError("Anthropic base URL is invalid for model discovery")
    path = f"{parsed.path.rstrip('/')}/v1/models"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _parse_page(body: bytes) -> tuple[tuple[ModelDiscoveryCandidate, ...], str | None]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCatalogueValidationError("Anthropic model catalogue response is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ModelCatalogueValidationError("Anthropic model catalogue response must be an object")
    raw_models = value.get("data")
    has_more = value.get("has_more")
    last_id = value.get("last_id")
    if not isinstance(raw_models, list):
        raise ModelCatalogueValidationError("Anthropic model catalogue data must be a list")
    if not isinstance(has_more, bool):
        raise ModelCatalogueValidationError("Anthropic model catalogue has_more must be boolean")
    if last_id is not None and (not isinstance(last_id, str) or not last_id.strip()):
        raise ModelCatalogueValidationError("Anthropic model catalogue last_id is invalid")
    if has_more and last_id is None:
        raise ModelCatalogueValidationError("Anthropic model catalogue omitted its next cursor")
    models = tuple(_parse_model(item) for item in raw_models)
    return models, last_id if has_more else None


def _parse_model(value: object) -> ModelDiscoveryCandidate:
    if not isinstance(value, dict):
        raise ModelCatalogueValidationError("Anthropic model entry must be an object")
    model_id = _required_text(value, "id")
    display_label = _required_text(value, "display_name")
    if value.get("type") != "model":
        raise ModelCatalogueValidationError("Anthropic model type must be model")
    capabilities = value.get("capabilities")
    if capabilities is not None and not isinstance(capabilities, dict):
        raise ModelCatalogueValidationError("Anthropic model capabilities must be an object or null")
    try:
        json.dumps(capabilities, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ModelCatalogueValidationError("Anthropic model capabilities are not JSON-safe") from exc
    return ModelDiscoveryCandidate(
        model_id,
        display_label,
        {
            "created_at": _required_text(value, "created_at"),
            "max_input_tokens": _optional_integer(value, "max_input_tokens"),
            "max_output_tokens": _optional_integer(value, "max_tokens"),
            "provider_capabilities": capabilities or {},
        },
    )


def _required_text(value: Mapping[str, object], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ModelCatalogueValidationError(f"Anthropic model field {field} must be text")
    return item


def _optional_integer(value: Mapping[str, object], field: str) -> int | None:
    item = value.get(field)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ModelCatalogueValidationError(f"Anthropic model field {field} must be a non-negative integer or null")
    return item
