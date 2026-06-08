"""Opt-in refresh helper for `PROVIDER_MODELS`.

Operator runs `python -m kai.refresh_models` (or `make refresh-models`)
to query each curated provider's `/v1/models` endpoint and print a
unified diff against the in-tree `PROVIDER_MODELS[provider]` keys.
The command never writes source files; operator review of the diff
is the trust boundary. Pass `--write-snippet` to emit a paste-able
Python fragment to stdout that the operator copies into
`src/kai/config.py` by hand.

Exit codes:
- 0: every queried provider responded; no diff against the in-tree
     list (or every queried provider was skipped for missing auth).
- 1: every queried provider responded; at least one provider has a
     diff (new and / or retired models).
- 2: at least one provider failed to respond (network error, 5xx,
     unexpected response shape). Other providers' diffs still print.

Per-provider auth comes from `PROVIDER_KEY_VARS`. A provider whose
API key is unset in env skips with a `"skipped: no API key"` notice;
this is not counted as a failure (exit 2 reserves for actual remote
faults). OpenRouter and Ollama are absent from `PROVIDER_MODELS`
today (open-ended providers) and skip with a one-line notice.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Awaitable, Callable

import aiohttp

from kai.config import PROVIDER_KEY_VARS, PROVIDER_MODELS

log = logging.getLogger(__name__)

# Per-provider request timeout in seconds. Total worst-case wait
# scales as len(providers) * _PROVIDER_TIMEOUT because the providers
# are queried sequentially (the diff-printing is a developer surface,
# not a hot path; sequential keeps the output ordered without an
# extra sort pass).
_PROVIDER_TIMEOUT = 10.0

# Default Ollama host. Operators running ollama on a non-default port
# or remote host set OLLAMA_HOST in env before running this command.
_OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"


async def _fetch_anthropic(api_key: str) -> list[str]:
    """Anthropic `/v1/models` returns a `data` array of objects with
    an `id` field naming each available model."""
    url = "https://api.anthropic.com/v1/models"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    timeout = aiohttp.ClientTimeout(total=_PROVIDER_TIMEOUT)
    async with (
        aiohttp.ClientSession() as session,
        session.get(url, headers=headers, timeout=timeout) as resp,
    ):
        resp.raise_for_status()
        payload = await resp.json()
    return [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]


async def _fetch_openai(api_key: str) -> list[str]:
    """OpenAI `/v1/models` returns the same `data` array shape as Anthropic."""
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = aiohttp.ClientTimeout(total=_PROVIDER_TIMEOUT)
    async with (
        aiohttp.ClientSession() as session,
        session.get(url, headers=headers, timeout=timeout) as resp,
    ):
        resp.raise_for_status()
        payload = await resp.json()
    return [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]


async def _fetch_google(api_key: str) -> list[str]:
    """Google Gemini API `models?key=<key>` returns a `models` array
    of objects with a `name` field (prefixed `models/`). Strip the
    prefix so the returned IDs match the bare model name shape
    PROVIDER_MODELS["google"] keys use."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    timeout = aiohttp.ClientTimeout(total=_PROVIDER_TIMEOUT)
    async with (
        aiohttp.ClientSession() as session,
        session.get(url, timeout=timeout) as resp,
    ):
        resp.raise_for_status()
        payload = await resp.json()
    out: list[str] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        if name.startswith("models/"):
            name = name[len("models/") :]
        if name:
            out.append(name)
    return out


# Per-provider fetcher dispatch. Only providers with entries in
# PROVIDER_MODELS appear; OPEN_ENDED_PROVIDERS (openrouter, ollama)
# render a "skipped: open-ended" line via the dispatch path's
# default-miss branch.
_provider_fetchers: dict[str, Callable[[str], Awaitable[list[str]]]] = {
    "anthropic": _fetch_anthropic,
    "openai": _fetch_openai,
    "google": _fetch_google,
}


def _render_diff(provider: str, remote_models: list[str], curated_models: list[str]) -> tuple[bool, list[str]]:
    """Render a per-provider diff against the in-tree curated list.

    Returns `(has_diff, lines)` where `has_diff` indicates a non-empty
    set-difference in either direction and `lines` is the list of
    output strings ready to print. The diff is symmetric (additions
    AND retirements); the operator decides which side to act on.
    """
    remote_set = set(remote_models)
    curated_set = set(curated_models)
    new_models = sorted(remote_set - curated_set)
    retired_models = sorted(curated_set - remote_set)
    total = len(remote_set)
    if not new_models and not retired_models:
        return False, [f"{provider}: {total} models (no change)"]
    parts: list[str] = []
    summary_bits = [f"{total} models"]
    if new_models:
        summary_bits.append(f"{len(new_models)} new")
    if retired_models:
        summary_bits.append(f"{len(retired_models)} retired")
    parts.append(f"{provider}: " + ", ".join(summary_bits))
    for m in new_models:
        parts.append(f"  +  {m}")
    for m in retired_models:
        parts.append(f"  -  {m}")
    return True, parts


def _render_snippet(provider: str, remote_models: list[str]) -> str:
    """Emit a paste-able Python fragment for src/kai/config.py.

    Display names default to the raw model ID; the operator edits
    them after pasting. Skipped providers (no fetcher, missing auth)
    do not produce a snippet.
    """
    lines = [f'PROVIDER_MODELS["{provider}"] = {{']
    for m in sorted(remote_models):
        lines.append(f'    "{m}": "{m}",')
    lines.append("}")
    return "\n".join(lines)


async def _refresh_one(provider: str, *, write_snippet: bool) -> tuple[int, list[str]]:
    """Refresh one provider's model list.

    Returns `(status, lines)` where `status` is:
        0 = no diff (or skipped)
        1 = diff present
        2 = fetch failed
    """
    if provider not in _provider_fetchers:
        return 0, [f"{provider}: skipped (open-ended provider; no curated list to diff)"]
    key_var = PROVIDER_KEY_VARS.get(provider, "")
    if not key_var:
        return 0, [f"{provider}: skipped (no PROVIDER_KEY_VARS entry; cannot authenticate)"]
    api_key = os.environ.get(key_var, "").strip()
    if not api_key:
        return 0, [f"{provider}: skipped (no API key in env: {key_var})"]

    fetcher = _provider_fetchers[provider]
    try:
        remote_models = await fetcher(api_key)
    except Exception as exc:
        return 2, [f"{provider}: ERROR ({type(exc).__name__}: {exc})"]

    curated_models = list(PROVIDER_MODELS.get(provider, {}).keys())
    has_diff, diff_lines = _render_diff(provider, remote_models, curated_models)
    if write_snippet and (has_diff or not curated_models):
        diff_lines.append("")
        diff_lines.append(_render_snippet(provider, remote_models))
        diff_lines.append("")
    return (1 if has_diff else 0), diff_lines


async def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kai.refresh_models",
        description=(
            "Query each curated provider's /v1/models endpoint and print a "
            "diff against PROVIDER_MODELS in src/kai/config.py. Never writes "
            "source files; operator hand-edits config.py after reviewing the diff."
        ),
    )
    parser.add_argument(
        "--write-snippet",
        action="store_true",
        help=(
            "Emit a paste-able Python fragment per provider with a non-empty "
            "diff. Operator pastes the fragment into src/kai/config.py and "
            "edits display names by hand. Default off."
        ),
    )
    args = parser.parse_args(argv)

    # Walk providers in alphabetical order so the diff output is
    # deterministic across runs (set iteration order would otherwise
    # render differently under PYTHONHASHSEED randomization).
    providers = sorted(set(PROVIDER_MODELS.keys()) | {"openrouter", "ollama"})

    overall_status = 0
    for provider in providers:
        status, lines = await _refresh_one(provider, write_snippet=args.write_snippet)
        for line in lines:
            print(line)
        # Promote to the higher status code; once 2 (failure) is set,
        # it cannot demote back to 1 (diff) or 0 (no diff).
        if status > overall_status:
            overall_status = status

    if args.write_snippet:
        print()
        print(
            "Snippets above are paste-able into PROVIDER_MODELS in "
            "src/kai/config.py. Edit display names by hand before "
            "committing; the snippet's default copies the raw model ID."
        )
    return overall_status


def main() -> None:
    """Argparse entry point. Wraps `_main` in asyncio.run."""
    try:
        rc = asyncio.run(_main(sys.argv[1:]))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
