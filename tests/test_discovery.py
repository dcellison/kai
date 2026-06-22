"""
Tests for the provider model discovery layer.

Covers the synchronous read path (cache hit / stale / absent /
malformed / schema-mismatch), the background-refresh scheduler and
its dedupe, the OpenRouter fetcher's parse rules, atomic writes, and
the import-cycle smoke that pins the topology between
`kai.config.model_source_for_backend` and `kai.discovery`.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from kai import discovery
from kai.discovery import (
    _DISCOVERY_TTL_SECONDS,
    _SCHEMA_VERSION,
    ProviderModelSource,
    RefreshError,
    RefreshResult,
    _cache_path,
    _compute_cache_gen,
    _fetch_openrouter,
    get_provider_model_source,
    refresh_provider_models,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_discovery_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Isolate each test from the others.

    - DATA_DIR is redirected to a per-test tmp directory so cache
      reads and writes touch only that tree.
    - The module-level dedupe map is cleared so a leftover task from
      a prior test cannot poison the current schedule.
    - Any in-flight tasks are cancelled at teardown.
    """
    monkeypatch.setattr(discovery, "DATA_DIR", tmp_path)
    discovery._refresh_tasks.clear()
    yield
    for task in list(discovery._refresh_tasks.values()):
        task.cancel()
    discovery._refresh_tasks.clear()


def _write_cache(provider: str, models: dict[str, str], refreshed_at: float) -> None:
    """Helper: write a well-formed cache file directly without going
    through the discovery layer's write path. Used to seed the cache
    in tests for the read path."""
    path = _cache_path(provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "provider": provider,
        "refreshed_at": refreshed_at,
        "models": models,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _register_fetcher(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returns: dict[str, str] | None = None,
    raises: BaseException | None = None,
    event: asyncio.Event | None = None,
    block_until: asyncio.Event | None = None,
    counter: list[int] | None = None,
) -> None:
    """Helper: swap the openrouter fetcher for a controllable async
    function. Restored at test teardown by monkeypatch.

    `returns`/`raises` decide the outcome. `event` (set just before
    return) and `block_until` (awaited before return) give tests
    deterministic control over the call ordering. `counter` (a single-
    element list used as a mutable counter) lets the dedupe tests
    count call entries without smuggling a closure variable.
    """

    async def fetcher(_timeout: float) -> dict[str, str]:
        if counter is not None:
            counter[0] += 1
        if event is not None:
            event.set()
        if block_until is not None:
            await block_until.wait()
        if raises is not None:
            raise raises
        return dict(returns or {})

    monkeypatch.setitem(discovery._FETCHERS, "openrouter", fetcher)


# ── Read path: absent cache ──────────────────────────────────────────


async def test_absent_cache_for_discoverable_provider_schedules_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First /models on a fresh install returns the open-ended
    fallback AND kicks off a background refresh so the cache
    bootstraps. Without this the first read goes dark and the cache
    is never populated by the runtime path."""
    fetched = asyncio.Event()
    _register_fetcher(monkeypatch, returns={"a/b": "Ay Bee"}, event=fetched)

    source = get_provider_model_source("openrouter")
    assert source.kind == "open_ended"
    assert source.models == {}
    # Refresh task is registered in the dedupe map under the running
    # loop's lifetime; await its completion to keep the fixture's
    # teardown loop clean.
    assert "openrouter" in discovery._refresh_tasks
    await asyncio.wait_for(fetched.wait(), timeout=1.0)
    await asyncio.gather(*discovery._refresh_tasks.values(), return_exceptions=True)


def test_absent_cache_for_discoverable_provider_no_loop_does_not_crash() -> None:
    """In synchronous test contexts there is no running loop. The
    scheduler is a no-op rather than raising, so the discovery layer
    stays callable from any context (admin CLI, sync tests)."""
    source = get_provider_model_source("openrouter")
    assert source.kind == "open_ended"
    assert "openrouter" not in discovery._refresh_tasks


def test_absent_cache_for_non_discoverable_provider_does_not_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider without a fetcher cannot be refreshed; the scheduler
    must not enqueue a task that would raise inside the fetcher
    dispatch. Verified under a fresh dedupe map."""
    source = get_provider_model_source("not-a-real-provider")
    assert source.kind == "open_ended"
    assert "not-a-real-provider" not in discovery._refresh_tasks


# ── Read path: schema and parse failures ─────────────────────────────


async def test_schema_version_mismatch_invalidates_cache_and_schedules_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache file from a future schema cannot be read safely; treat
    as absent and schedule a refresh that will overwrite it."""
    path = _cache_path("openrouter")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 0, "provider": "openrouter", "refreshed_at": 0.0, "models": {}}),
        encoding="utf-8",
    )
    fetched = asyncio.Event()
    _register_fetcher(monkeypatch, returns={"new/model": "New"}, event=fetched)

    source = get_provider_model_source("openrouter")
    assert source.kind == "open_ended"
    assert "openrouter" in discovery._refresh_tasks
    await asyncio.wait_for(fetched.wait(), timeout=1.0)
    await asyncio.gather(*discovery._refresh_tasks.values(), return_exceptions=True)


async def test_malformed_cache_invalidates_and_schedules_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt cache file (invalid JSON) must not let
    JSONDecodeError escape to /models. Treat as absent, schedule a
    refresh, return the open-ended fallback synchronously."""
    path = _cache_path("openrouter")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{this is not json", encoding="utf-8")
    fetched = asyncio.Event()
    _register_fetcher(monkeypatch, returns={"new/model": "New"}, event=fetched)

    source = get_provider_model_source("openrouter")
    assert source.kind == "open_ended"
    assert "openrouter" in discovery._refresh_tasks
    await asyncio.wait_for(fetched.wait(), timeout=1.0)
    await asyncio.gather(*discovery._refresh_tasks.values(), return_exceptions=True)


# ── Read path: fresh and stale cache ─────────────────────────────────


def test_cache_fresh_returns_discovered_without_refresh() -> None:
    """A cache within TTL returns kind=discovered and the scheduler
    does NOT enqueue a refresh; the data is good."""
    _write_cache("openrouter", {"a/b": "Ay Bee"}, time.time() - 60)
    source = get_provider_model_source("openrouter")
    assert source.kind == "discovered"
    assert source.models == {"a/b": "Ay Bee"}
    assert source.cache_gen is not None
    assert "openrouter" not in discovery._refresh_tasks


async def test_cache_stale_returns_discovered_stale_and_schedules_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache past TTL is still served (kind=discovered_stale) so
    the keyboard does not go dark, but a refresh is scheduled in the
    background to bring the cache forward."""
    _write_cache("openrouter", {"a/b": "Ay Bee"}, time.time() - _DISCOVERY_TTL_SECONDS - 60)
    fetched = asyncio.Event()
    _register_fetcher(monkeypatch, returns={"a/b": "Ay Bee"}, event=fetched)

    source = get_provider_model_source("openrouter")
    assert source.kind == "discovered_stale"
    assert "openrouter" in discovery._refresh_tasks
    await asyncio.wait_for(fetched.wait(), timeout=1.0)
    await asyncio.gather(*discovery._refresh_tasks.values(), return_exceptions=True)


# ── Refresh dedupe ───────────────────────────────────────────────────


async def test_concurrent_stale_reads_dedupe_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chat-spam of /models against a stale cache must collapse to a
    single in-flight refresh per provider; otherwise OpenRouter sees
    one request per click."""
    _write_cache("openrouter", {"a/b": "Ay Bee"}, time.time() - _DISCOVERY_TTL_SECONDS - 60)
    block = asyncio.Event()
    counter = [0]
    _register_fetcher(monkeypatch, returns={"a/b": "Ay Bee"}, block_until=block, counter=counter)

    get_provider_model_source("openrouter")
    get_provider_model_source("openrouter")
    get_provider_model_source("openrouter")
    # All three reads share one task.
    assert len(discovery._refresh_tasks) == 1
    block.set()
    await asyncio.gather(*discovery._refresh_tasks.values(), return_exceptions=True)
    assert counter[0] == 1


async def test_concurrent_absent_reads_dedupe_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same dedupe property must hold for the absent-cache bootstrap
    path: two simultaneous first /models calls do not stack two
    network fetches against an upstream that just went live."""
    block = asyncio.Event()
    counter = [0]
    _register_fetcher(monkeypatch, returns={"a/b": "Ay Bee"}, block_until=block, counter=counter)

    get_provider_model_source("openrouter")
    get_provider_model_source("openrouter")
    assert len(discovery._refresh_tasks) == 1
    block.set()
    await asyncio.gather(*discovery._refresh_tasks.values(), return_exceptions=True)
    assert counter[0] == 1


# ── Refresh write path ───────────────────────────────────────────────


async def test_refresh_writes_cache_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful refresh writes via tempfile + os.replace. The
    atomicity property: a write that completes leaves the new cache
    in place; no .tmp detritus survives."""
    _register_fetcher(monkeypatch, returns={"a/b": "Ay Bee", "c/d": "See Dee"})
    result = await refresh_provider_models("openrouter")
    assert isinstance(result, RefreshResult)
    assert result.total == 2

    path = _cache_path("openrouter")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == _SCHEMA_VERSION
    assert payload["models"] == {"a/b": "Ay Bee", "c/d": "See Dee"}
    # No leftover temp files from the atomic-write dance.
    assert not list(path.parent.glob("*.tmp"))


async def test_refresh_with_empty_result_and_nonempty_prior_raises_refresh_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upstream incident that returns a shape-valid but semantically
    empty catalog (e.g. `{"data": []}`) must NOT overwrite a working
    prior cache with `{}`. Without this guard, the next /models read
    would see `kind="discovered"` against an empty map and render a
    keyboard with zero buttons, taking the operator from "stale but
    usable" to "no usable surface" on a transient upstream fault.
    """
    _write_cache("openrouter", {"a/b": "Ay Bee", "c/d": "See Dee"}, time.time() - 60)
    path = _cache_path("openrouter")
    prior_bytes = path.read_bytes()

    _register_fetcher(monkeypatch, returns={})
    with pytest.raises(RefreshError):
        await refresh_provider_models("openrouter")
    assert path.read_bytes() == prior_bytes


async def test_refresh_with_empty_result_and_no_prior_raises_refresh_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a first-run install, an empty fetch must also raise rather
    than write an empty cache. Otherwise the next read would lock in
    `kind="discovered"` against an empty map for one TTL, rendering
    an empty keyboard instead of the open-ended fallback that lets
    the operator type a model id while OpenRouter recovers.
    """
    path = _cache_path("openrouter")
    assert not path.exists()

    _register_fetcher(monkeypatch, returns={})
    with pytest.raises(RefreshError):
        await refresh_provider_models("openrouter")
    assert not path.exists()

    # Next read still takes the open-ended fallback (which the bot
    # surfaces as a text prompt), not an empty discovered keyboard.
    source = get_provider_model_source("openrouter", schedule_refresh=False)
    assert source.kind == "open_ended"


async def test_refresh_failure_raises_RefreshError_and_preserves_prior_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network failure mid-refresh must leave the cache byte-for-
    byte unchanged so the next /models still works against the prior
    snapshot. The exception type is RefreshError so the admin CLI
    has a single class to catch."""
    _write_cache("openrouter", {"a/b": "Ay Bee"}, time.time() - 60)
    path = _cache_path("openrouter")
    prior_bytes = path.read_bytes()

    _register_fetcher(monkeypatch, raises=OSError("simulated network drop"))
    with pytest.raises(RefreshError):
        await refresh_provider_models("openrouter")
    assert path.read_bytes() == prior_bytes

    # Next read still sees the prior cache as a usable snapshot.
    source = get_provider_model_source("openrouter", schedule_refresh=False)
    assert source.kind == "discovered"
    assert source.models == {"a/b": "Ay Bee"}


# ── OpenRouter fetcher ───────────────────────────────────────────────


class _MockResponse:
    """Minimal context-manager response stand-in for aiohttp's
    ClientSession.get(...). raise_for_status() is a no-op; json() is
    awaitable. The session/request shape is mirrored by _MockSession
    below."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def __aenter__(self) -> _MockResponse:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> Any:
        return self._payload


class _MockSession:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def __aenter__(self) -> _MockSession:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    def get(self, *_args: Any, **_kwargs: Any) -> _MockResponse:
        return _MockResponse(self._payload)


def _patch_aiohttp(monkeypatch: pytest.MonkeyPatch, payload: Any) -> None:
    """Replace aiohttp.ClientSession with a stub that returns `payload`
    from session.get().json()."""
    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", lambda: _MockSession(payload))


async def test_openrouter_fetcher_parses_id_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows with a `name` use it as display; rows without `name` fall
    back to `id` as both key and display."""
    _patch_aiohttp(
        monkeypatch,
        {
            "data": [
                {"id": "a/one", "name": "First"},
                {"id": "b/two", "name": "Second"},
                {"id": "c/three"},
            ]
        },
    )
    result = await _fetch_openrouter(10.0)
    assert result == {
        "a/one": "First",
        "b/two": "Second",
        "c/three": "c/three",
    }


async def test_openrouter_fetcher_skips_rows_without_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows lacking an `id` field are dropped silently. A future
    OpenRouter schema change that adds non-model rows must not break
    Kai's read."""
    _patch_aiohttp(
        monkeypatch,
        {
            "data": [
                {"id": "a/one", "name": "First"},
                {"name": "No id"},
                {"id": "", "name": "Empty id"},
                "not a dict",
            ]
        },
    )
    result = await _fetch_openrouter(10.0)
    assert result == {"a/one": "First"}


# ── Cache snapshot identity ──────────────────────────────────────────


def test_cache_gen_changes_when_models_change() -> None:
    """cache_gen binds a keyboard to a specific catalog snapshot. Two
    caches with the same models must produce the same hash; adding a
    model must change it. Otherwise the callback-rejection mechanism
    on background refresh cannot distinguish snapshots."""
    a = {"x/y": "Display"}
    b = {"x/y": "Display"}
    c = {"x/y": "Display", "p/q": "Other"}
    assert _compute_cache_gen(a) == _compute_cache_gen(b)
    assert _compute_cache_gen(a) != _compute_cache_gen(c)


# ── schedule_refresh=False contract ──────────────────────────────────


def test_schedule_refresh_false_never_schedules_on_absent_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin CLI calls with schedule_refresh=False because it drives
    the refresh itself. The read must NOT also schedule a background
    task on the bot's loop; otherwise concurrent refreshes race."""
    _register_fetcher(monkeypatch, returns={"a/b": "Ay Bee"})
    source = get_provider_model_source("openrouter", schedule_refresh=False)
    assert source.kind == "open_ended"
    assert "openrouter" not in discovery._refresh_tasks


def test_schedule_refresh_false_never_schedules_on_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract on the stale-cache branch."""
    _write_cache("openrouter", {"a/b": "Ay Bee"}, time.time() - _DISCOVERY_TTL_SECONDS - 60)
    _register_fetcher(monkeypatch, returns={"a/b": "Ay Bee"})
    source = get_provider_model_source("openrouter", schedule_refresh=False)
    assert source.kind == "discovered_stale"
    assert "openrouter" not in discovery._refresh_tasks


# ── Cache directory permissions ──────────────────────────────────────


async def test_cache_dir_created_with_correct_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """First write under a fresh DATA_DIR creates the discovery
    directory at mode 0o755 so admin debugging from another OS user
    can read the cache."""
    _register_fetcher(monkeypatch, returns={"a/b": "Ay Bee"})
    await refresh_provider_models("openrouter")
    cache_dir = tmp_path / "discovery"
    assert cache_dir.is_dir()
    assert (cache_dir.stat().st_mode & 0o777) == 0o755


# ── Import topology smoke ────────────────────────────────────────────


def test_import_cycle_smoke() -> None:
    """Pin the no-cycle property between kai.config and kai.discovery.

    A future contributor who adds a top-level `from kai.discovery
    import ProviderModelSource` to config.py would re-introduce the
    cycle this PR carefully avoids. This smoke imports both modules
    fresh (well, observes them already imported in the test process)
    and calls the cross-module entry point: a successful return with
    a usable kind proves the topology holds."""
    import kai.config
    import kai.discovery

    assert kai.config is not None
    assert kai.discovery is not None
    source = kai.config.model_source_for_backend("goose", "openrouter", schedule_refresh=False)
    assert isinstance(source, ProviderModelSource)
    assert source.kind in ("discovered", "discovered_stale", "open_ended")
