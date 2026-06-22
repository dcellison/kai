"""
Provider model discovery: cached on disk, refreshed in the background,
falls back gracefully when a provider catalog is unreachable.

Phase 1 supports a single discoverable provider (`openrouter`) and the
synchronous open-ended fallback for the rest. Other providers are still
served by the curated `PROVIDER_MODELS` / `CODEX_MODELS` constants in
`kai.config`; this module is the place new fetchers land in later
phases without touching the curated surfaces.

The runtime contract:

1. `get_provider_model_source(provider)` is the synchronous read. It
   never blocks on HTTP. On an absent or stale cache for a
   discoverable provider it schedules a background refresh and
   returns the current (possibly empty) cache contents tagged with
   `kind` so the UI can label it.
2. `refresh_provider_models(provider)` is the async writer. It is
   called from the background scheduler and from
   `python -m kai.refresh_models`. On failure it raises
   `RefreshError`; the on-disk cache is untouched so a transient
   outage cannot corrupt the operator-visible model list.

Validation lives in `kai.config.validate_model_for_backend`; this
module never decides whether a model id is acceptable, only what the
catalog says is on offer right now.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import aiohttp

from kai.config import DATA_DIR

log = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────


# Bump when the cache file shape changes in a way the reader cannot
# tolerate. Cache files with a mismatched version are treated as
# absent and overwritten by the next refresh; no hand-migration step.
_SCHEMA_VERSION = 1

# 24 hours. OpenRouter publishes new models on the order of days, not
# seconds; a sub-day TTL would burn API budget without UX benefit.
_DISCOVERY_TTL_SECONDS = 86_400

# Per-call timeout for a provider catalog fetch. The fetchers are
# called from a background task; an unresponsive provider should not
# pile up tasks waiting for a timeout that never arrives.
_REFRESH_TIMEOUT_SECONDS = 10.0

# Hex characters of the cache snapshot hash that get embedded in
# Telegram callback_data. 8 hex chars = 32 bits of entropy, plenty
# for catching a between-render-and-click cache change, and short
# enough to fit comfortably under Telegram's 64-byte callback limit
# alongside the `model_pick:` prefix and a 3-digit index.
_CACHE_GEN_LEN = 8


# ── Public types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderModelSource:
    """
    Resolved model surface for one provider at the time of read.

    `kind` tells the UI how to render:
        - "discovered": cached models within TTL.
        - "discovered_stale": cached models past TTL; a background
          refresh is in flight (or was just scheduled) and the next
          read after it completes will return "discovered".
        - "curated": static list from PROVIDER_MODELS / CODEX_MODELS.
        - "open_ended": no list available; UI should prompt for free
          text and the validator will accept any string.

    `cache_gen` is a short hash over the `models` map, used by the
    /models keyboard to bind callback_data to a specific catalog
    snapshot so a refresh that lands between render and click can be
    detected and rejected with a "catalogue changed" message.

    Attributes:
        kind: One of "discovered", "discovered_stale", "curated",
            "open_ended".
        models: id -> display label. Empty for "open_ended".
        refreshed_at: Unix timestamp of the cached fetch, or None for
            non-discovered kinds.
        cache_gen: 8-hex-char snapshot hash, or None for non-discovered
            kinds.
    """

    kind: Literal["discovered", "discovered_stale", "curated", "open_ended"]
    models: Mapping[str, str]
    refreshed_at: float | None
    cache_gen: str | None


@dataclass(frozen=True)
class RefreshResult:
    """
    Successful outcome of `refresh_provider_models`.

    Returned only on success. On failure the function raises
    `RefreshError`; callers can rely on this dataclass never carrying
    an error state.

    Attributes:
        provider: Provider whose cache was refreshed.
        models_added: Ids present in the new fetch and absent from the
            prior cache (empty if no prior cache existed; the entire
            new list lives in `models` on disk).
        models_removed: Ids present in the prior cache and absent from
            the new fetch.
        total: Total ids in the new fetch.
        refreshed_at: Unix timestamp written to the new cache file.
    """

    provider: str
    models_added: list[str]
    models_removed: list[str]
    total: int
    refreshed_at: float


class RefreshError(Exception):
    """
    Raised by `refresh_provider_models` on any failure path.

    Wraps network errors, timeouts, non-2xx responses, and response
    parse failures into a single exception class so callers do not
    need to enumerate every aiohttp / asyncio error type. The on-disk
    cache is guaranteed untouched when this is raised.
    """


# ── Module state ─────────────────────────────────────────────────────


# Per-provider fetcher dispatch. Populated below as each fetcher is
# defined. A provider absent from this map is non-discoverable: reads
# get the synchronous fallback and no refresh is ever scheduled.
_FETCHERS: dict[str, Callable[[float], Awaitable[dict[str, str]]]] = {}


# Background-refresh dedupe. A `/models` spam from one chat would
# otherwise stack identical fetches against the upstream provider.
# Tasks self-evict in `_evict` so a poisoned entry cannot block the
# scheduler forever.
_refresh_tasks: dict[str, asyncio.Task[RefreshResult]] = {}


# ── Cache file IO ────────────────────────────────────────────────────


def _cache_dir() -> Path:
    """
    Return the discovery cache directory, creating it on first use.

    Mode 0o755 so the bot (running as the `kai` service user) can read
    and write, and other users on the host can read for debugging.
    """
    d = DATA_DIR / "discovery"
    d.mkdir(mode=0o755, parents=True, exist_ok=True)
    return d


def _cache_path(provider: str) -> Path:
    """Filesystem path for one provider's cache file."""
    return _cache_dir() / f"{provider}.json"


def _compute_cache_gen(models: Mapping[str, str]) -> str:
    """
    Short hash that binds a rendered keyboard to a catalog snapshot.

    Sorting the items before serialization is load-bearing: dict
    iteration order is insertion order, so two semantically identical
    caches built from different fetcher orderings would otherwise
    produce different hashes and cause spurious cache-gen mismatches.
    """
    canonical = json.dumps(dict(sorted(models.items())), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_CACHE_GEN_LEN]


def _read_cache(provider: str) -> tuple[Mapping[str, str], float, str] | None:
    """
    Return (models, refreshed_at, cache_gen) on a usable cache, None on:
    - Missing file.
    - JSON parse failure (a malformed cache must not escape as
      JSONDecodeError to /models; treat as absent and let the caller
      schedule a refresh).
    - Schema version mismatch.
    - Structural mismatch (missing or wrong-typed required fields).

    All id and display values are coerced to str so a hand-edited
    cache cannot leak non-string values to the keyboard renderer.
    """
    path = _cache_path(provider)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("discovery: malformed cache for %s; treating as absent", provider)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        return None
    models = payload.get("models")
    refreshed_at = payload.get("refreshed_at")
    if not isinstance(models, dict) or not isinstance(refreshed_at, (int, float)):
        return None
    cleaned = {str(k): str(v) for k, v in models.items()}
    return cleaned, float(refreshed_at), _compute_cache_gen(cleaned)


def _write_cache_atomic(provider: str, models: Mapping[str, str], refreshed_at: float) -> None:
    """
    Write the cache via tempfile + os.replace so a mid-write failure
    leaves the prior cache file byte-for-byte unchanged.

    The temp file lives in the cache directory so os.replace is a
    same-filesystem rename (atomic on POSIX). A NamedTemporaryFile in
    /tmp would risk EXDEV on hosts where /var/lib and /tmp live on
    different filesystems.
    """
    path = _cache_path(provider)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "provider": provider,
        "refreshed_at": refreshed_at,
        "models": dict(models),
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{provider}.",
        suffix=".json.tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
        os.replace(tmp_name, path)
        # mkstemp creates with mode 0o600; widen to 0o644 so admin
        # debugging from a different OS user can read the file.
        os.chmod(path, 0o644)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


# ── Refresh scheduling ───────────────────────────────────────────────


def _is_discoverable(provider: str) -> bool:
    """A provider is discoverable iff a fetcher is registered for it."""
    return provider in _FETCHERS


def _schedule_refresh(provider: str) -> None:
    """
    Schedule a background refresh for `provider` on the running loop.

    No-ops in three cases:
    - No event loop is running. Synchronous callers (admin CLI, tests)
      cannot drive an async task, so silently skip rather than crash.
    - A refresh for this provider is already in flight. Dedupes the
      common `/models` spam pattern.
    - The provider is not discoverable. The fetcher dispatch would
      raise inside the task, so don't start it.
    """
    if not _is_discoverable(provider):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if provider in _refresh_tasks:
        return
    task = loop.create_task(refresh_provider_models(provider))
    _refresh_tasks[provider] = task
    task.add_done_callback(lambda t, p=provider: _evict_task(p, t))


def _evict_task(provider: str, task: asyncio.Task[RefreshResult]) -> None:
    """
    Drop the finished task from the dedupe map and log any failure.

    Pop unconditionally even if the task raised so a poisoned entry
    cannot block the next refresh from being scheduled.
    """
    _refresh_tasks.pop(provider, None)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning("discovery: background refresh for %s failed: %s", provider, exc)


# ── Read path ────────────────────────────────────────────────────────


def get_provider_model_source(
    provider: str,
    *,
    schedule_refresh: bool = True,
) -> ProviderModelSource:
    """
    Synchronously resolve the current model surface for `provider`.

    Reads the on-disk cache, decides fresh / stale / fallback, and on
    an absent or stale cache for a discoverable provider schedules a
    background refresh (when `schedule_refresh=True` AND an event loop
    is running). The read path itself never blocks on HTTP: the worst
    case is a single JSON parse of a small cache file and a cheap
    sha256.

    Phase 1 only knows `openrouter` as a discoverable provider; other
    providers always return `kind="open_ended"` here. Callers that
    need the curated dict (installer wizard, validation) keep going
    through `models_for_backend` in `kai.config`.

    Args:
        provider: Provider name (e.g. "openrouter").
        schedule_refresh: When True and a loop is running, an
            absent/stale read may schedule a background refresh. Set
            False from the admin CLI, which drives the refresh itself.

    Returns:
        ProviderModelSource describing the read.
    """
    cache = _read_cache(provider)
    if cache is None:
        if schedule_refresh:
            _schedule_refresh(provider)
        return ProviderModelSource(
            kind="open_ended",
            models={},
            refreshed_at=None,
            cache_gen=None,
        )
    models, refreshed_at, cache_gen = cache
    if (time.time() - refreshed_at) < _DISCOVERY_TTL_SECONDS:
        return ProviderModelSource(
            kind="discovered",
            models=models,
            refreshed_at=refreshed_at,
            cache_gen=cache_gen,
        )
    if schedule_refresh:
        _schedule_refresh(provider)
    return ProviderModelSource(
        kind="discovered_stale",
        models=models,
        refreshed_at=refreshed_at,
        cache_gen=cache_gen,
    )


# ── Fetchers ─────────────────────────────────────────────────────────


async def _fetch_openrouter(timeout: float) -> dict[str, str]:
    """
    Fetch the OpenRouter model catalog.

    `GET https://openrouter.ai/api/v1/models` returns
    `{"data": [{"id": "...", "name": "...", ...}, ...]}`. The endpoint
    does not require an API key. Phase 1 surfaces every catalog row
    without modality filtering; a Phase 2 follow-up may layer a
    chat-suitable-only filter on top.
    """
    url = "https://openrouter.ai/api/v1/models"
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with (
        aiohttp.ClientSession() as session,
        session.get(url, timeout=client_timeout) as resp,
    ):
        resp.raise_for_status()
        payload = await resp.json()
    if not isinstance(payload, dict):
        raise RefreshError(f"openrouter: unexpected response shape (root is {type(payload).__name__})")
    data = payload.get("data")
    if not isinstance(data, list):
        raise RefreshError("openrouter: response missing or non-list 'data' field")
    out: dict[str, str] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        name = row.get("name")
        display = name if isinstance(name, str) and name else model_id
        out[model_id] = display
    return out


_FETCHERS["openrouter"] = _fetch_openrouter


# ── Refresh ──────────────────────────────────────────────────────────


async def refresh_provider_models(
    provider: str,
    *,
    timeout: float = _REFRESH_TIMEOUT_SECONDS,
) -> RefreshResult:
    """
    Refresh one provider's cache and return the diff against the prior
    contents.

    On any failure path (no fetcher, network error, timeout, non-2xx,
    parse error) raises `RefreshError` and leaves the on-disk cache
    untouched. On success writes the cache atomically and returns
    `RefreshResult` for the caller's diff output.

    Args:
        provider: Provider name. Must have a fetcher registered.
        timeout: Per-request timeout in seconds.

    Raises:
        RefreshError: On any fetch or parse failure.
    """
    fetcher = _FETCHERS.get(provider)
    if fetcher is None:
        raise RefreshError(f"{provider}: no fetcher registered; not a discoverable provider")
    try:
        new_models = await fetcher(timeout)
    except RefreshError:
        raise
    except Exception as exc:
        raise RefreshError(f"{provider}: fetch failed ({type(exc).__name__}: {exc})") from exc

    prior = _read_cache(provider)
    prior_models = prior[0] if prior is not None else {}

    # An empty fetch result is treated as an upstream fault, never as
    # a steady state. Two failure modes this rejects:
    #   - Non-empty prior cache: writing `{}` would destroy a working
    #     catalog the operator was still selecting from.
    #   - No prior cache (first run): writing `{}` would lock in
    #     `kind="discovered"` against an empty map for one TTL,
    #     rendering an empty keyboard instead of the open-ended
    #     fallback that lets the operator type a model id while the
    #     provider recovers.
    # Discoverable providers in P1 (openrouter) never legitimately
    # report zero models; raise so the cache stays absent or unchanged
    # and the next scheduled refresh has a chance to recover.
    if not new_models:
        raise RefreshError(f"{provider}: fetch returned an empty catalog; refusing to write")

    added = sorted(set(new_models) - set(prior_models))
    removed = sorted(set(prior_models) - set(new_models))

    refreshed_at = time.time()
    _write_cache_atomic(provider, new_models, refreshed_at)

    return RefreshResult(
        provider=provider,
        models_added=added,
        models_removed=removed,
        total=len(new_models),
        refreshed_at=refreshed_at,
    )
