"""
Tests for the `python -m kai.refresh_models` admin command.

Covers the OpenRouter branch (writes the discovery cache, prints the
diff, maps fetcher failures to exit 2 with prior-cache preservation)
and pins that the existing curated-provider fetchers still produce
their PROVIDER_MODELS diff output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kai import discovery, refresh_models
from kai.discovery import _SCHEMA_VERSION, _cache_path


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the discovery cache to a per-test tmp directory and
    clear any leftover refresh tasks between tests."""
    monkeypatch.setattr(discovery, "DATA_DIR", tmp_path)
    discovery._refresh_tasks.clear()
    yield
    discovery._refresh_tasks.clear()


def _register_openrouter_fetcher(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returns: dict[str, str] | None = None,
    raises: BaseException | None = None,
) -> None:
    """Swap the discovery layer's openrouter fetcher for a controllable
    async function. The admin CLI calls discovery.refresh_provider_models
    which dispatches through the same _FETCHERS map the runtime uses,
    so monkeypatching there exercises the real CLI path."""

    async def fetcher(_timeout: float) -> dict[str, str]:
        if raises is not None:
            raise raises
        return dict(returns or {})

    monkeypatch.setitem(discovery._FETCHERS, "openrouter", fetcher)


def _isolate_curated_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear API keys for the curated providers so their branches skip
    with 'no API key' instead of attempting a real network call.
    Tests that care about the openrouter branch only do not want noise
    from the anthropic/openai/google paths."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


async def test_refresh_models_openrouter_writes_discovery_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The admin CLI's openrouter branch writes the discovery cache
    file as a side effect of the refresh. Pins that calling the CLI
    is sufficient to bootstrap the cache (no separate setup step
    needed)."""
    _isolate_curated_providers(monkeypatch)
    _register_openrouter_fetcher(monkeypatch, returns={"a/one": "First", "b/two": "Second"})

    status = await refresh_models._main([])
    cache_path = _cache_path("openrouter")
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == _SCHEMA_VERSION
    assert payload["models"] == {"a/one": "First", "b/two": "Second"}
    # First fetch with no prior cache produces an "added" diff so the
    # status code reflects "diff present" (1), not "no change" (0).
    assert status == 1


async def test_refresh_models_openrouter_prints_cache_diff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When a prior cache exists, the diff section names every added
    and retired model. Operators rely on this output to spot upstream
    catalog churn during an audit."""
    _isolate_curated_providers(monkeypatch)
    # Seed a prior cache.
    cache_path = _cache_path("openrouter")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "provider": "openrouter",
                "refreshed_at": 0.0,
                "models": {"old/keeper": "Keeper", "old/removed": "To Be Removed"},
            }
        ),
        encoding="utf-8",
    )
    _register_openrouter_fetcher(
        monkeypatch,
        returns={"old/keeper": "Keeper", "new/added": "Freshly Added"},
    )

    status = await refresh_models._main([])
    out = capsys.readouterr().out
    assert "openrouter: 2 models, 1 new, 1 retired" in out
    assert "  +  new/added" in out
    assert "  -  old/removed" in out
    assert status == 1


async def test_refresh_models_openrouter_failure_maps_to_exit_2_and_preserves_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fetcher exception during the openrouter branch must:
    1. Map to exit code 2 (matching the failure semantics of the
       curated providers' /v1/models error path).
    2. Leave the prior on-disk cache byte-for-byte unchanged so the
       runtime /models keeps working against the prior snapshot.
    3. Print an `openrouter: ERROR (...)` line carrying the wrapped
       exception's type so operators can spot what failed."""
    _isolate_curated_providers(monkeypatch)
    cache_path = _cache_path("openrouter")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "provider": "openrouter",
                "refreshed_at": 0.0,
                "models": {"prior/intact": "Intact"},
            }
        ),
        encoding="utf-8",
    )
    prior_bytes = cache_path.read_bytes()

    _register_openrouter_fetcher(monkeypatch, raises=OSError("simulated outage"))
    status = await refresh_models._main([])
    out = capsys.readouterr().out

    assert status == 2
    assert cache_path.read_bytes() == prior_bytes
    assert "openrouter: ERROR" in out
    assert "OSError" in out


async def test_refresh_models_existing_provider_fetchers_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The curated provider audit (anthropic/openai/google) still
    produces its PROVIDER_MODELS diff output. Pins that the
    openrouter addition did NOT change the curated audit behaviour;
    Phase 2b will absorb these into discovery as a separate change."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Avoid the openrouter network call; a successful no-op response
    # keeps the test focused on the curated branch.
    _register_openrouter_fetcher(monkeypatch, returns={})

    async def fake_anthropic(_api_key: str) -> list[str]:
        # Return one model the in-tree PROVIDER_MODELS["anthropic"]
        # does NOT carry so the diff line names a "new" model and the
        # status code rises to 1.
        return ["claude-future-model"]

    monkeypatch.setitem(refresh_models._provider_fetchers, "anthropic", fake_anthropic)

    status = await refresh_models._main([])
    out = capsys.readouterr().out
    # Diff line for anthropic appears with the new-model name.
    assert "anthropic:" in out
    assert "claude-future-model" in out
    # Diff in any provider means status >= 1.
    assert status >= 1
