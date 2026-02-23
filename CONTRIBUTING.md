# Contributing to Kai

Thanks for your interest in contributing to Kai. This document covers
what you need to know before opening a pull request.

## Before You Start

**Open an issue first.** Before writing code, open a GitHub issue describing
what you want to change and why. This lets us discuss the approach before
you invest time in implementation. Small bug fixes (typos, off-by-one errors)
can skip this step.

**Kai is a personal assistant.** It runs locally on one machine for one user.
Contributions should respect that scope -- multi-user features, cloud
deployment support, and similar changes are out of scope.

## Development Setup

```bash
# Clone and install in a virtual environment
git clone git@github.com:dcellison/kai.git
cd kai
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Verify everything works
make check    # ruff lint + format check
make test     # pytest
```

Requires **Python 3.13+**. See the
[Getting Started](https://github.com/dcellison/kai/wiki/Getting-Started)
wiki page for full setup instructions including `.env` configuration.

## Branch and PR Workflow

Direct pushes to `main` are blocked. All changes go through pull requests
with required CI checks.

1. **Create a branch** from `main` with a descriptive prefix:
   - `feature/` -- new functionality
   - `fix/` -- bug fixes
   - `docs/` -- documentation only
   - `refactor/` -- restructuring without behavior change
   - `chore/` -- dependency updates, CI changes, etc.
   - `test/` -- test additions or fixes

   Use kebab-case for the rest: `feature/file-exchange`, `fix/path-traversal`.

2. **Keep PRs focused.** One feature or fix per PR. Don't bundle unrelated
   changes -- if you notice something else worth fixing, open a separate PR.

3. **Review your own diff before submitting.** Every file in the diff should
   be there intentionally. Unrelated deletions, reformatting of untouched
   code, or tooling artifacts (`.idea/`, `.vscode/`, etc.) will get your
   PR sent back.

4. **CI must pass.** The pipeline runs ruff (lint + format) and pytest.
   Check locally first:
   ```bash
   make check && make test
   ```

## Code Style

### Python

- **Ruff** handles linting and formatting. The full rule configuration is in
  `pyproject.toml`. Run `make format` to auto-format before committing.
- **Line length:** 120 characters max.
- **Imports:** sorted by ruff's isort rules (stdlib, third-party, first-party).

### Comments and Docstrings

Kai's codebase is thoroughly commented. New code must match this standard:

- **Every function and class** gets a docstring. Single-line for simple
  helpers, multi-line with `Args:` / `Returns:` for anything non-trivial.
- **Module docstrings** explain the module's purpose and responsibilities.
- **Inline comments** explain *why*, not *what*. Focus on non-obvious logic,
  edge cases, and workarounds.
- **Section separators** divide logical groups within a module:
  ```python
  # -- Authorization --------------------------------------------------------
  ```

Example of the expected style:

```python
async def save_session(chat_id: int, session_id: str, model: str, cost_usd: float) -> None:
    """
    Save or update a Claude session for a chat.

    On conflict (existing chat_id), the session_id and model are updated,
    last_used_at is refreshed, and total_cost_usd is accumulated (not replaced).

    Args:
        chat_id: Telegram chat ID.
        session_id: Claude session identifier from the stream-json response.
        model: Model name used for this session (e.g., "sonnet").
        cost_usd: Cost of this particular interaction (added to running total).
    """
```

### Type Safety

The codebase passes Pyright in strict mode. New code should maintain this:

- Use type annotations on all function signatures.
- Use `assert` for narrowing `Optional` types from external libraries.
- Extract `@property` returns to local variables before narrowing (Pyright
  limitation).

## Security

Kai exposes a webhook server and API endpoints. If your change touches
networking, file I/O, or process execution:

- **Path confinement:** Use `Path.relative_to()` for directory containment
  checks, not string prefix matching (which is bypassable via symlinks).
- **Input validation:** Validate at system boundaries (user input, API
  payloads, external data). Don't trust anything from the network.
- **No new attack surface without discussion.** New endpoints, file
  operations, or shell commands should be discussed in the issue first.

## Tests

- Tests live in `tests/` and use **pytest** with **pytest-asyncio**.
- New features need tests. Bug fixes should include a regression test where
  practical.
- Test the function directly when possible (unit tests over integration
  tests). Mock external dependencies (Telegram API, filesystem for
  destructive operations).

Run the full suite with:
```bash
make test
```

## What Not to Include

- **Secrets or credentials** -- `.env` files, API keys, tokens. The
  `.gitignore` already covers these, but double-check your diff.
- **Generated files** -- IDE configs, OS metadata (`.DS_Store`), build
  artifacts, tool caches.
- **Unrelated changes** -- reformatting files you didn't modify, deleting
  files outside the scope of your feature, adding ignore rules for your
  personal tooling.

## License

By contributing, you agree that your contributions will be licensed under the
same [Apache 2.0 License](LICENSE) that covers the project.
