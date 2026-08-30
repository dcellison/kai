# Kai

Personal AI assistant accessed via Telegram. Python package at src/kai/.

## Development

- Run: `make run` or `python -m kai`
- Test: `make test` (pytest, ~1200 tests)
- Lint: `make check` (ruff check + ruff format)
- Config wizard: `make config`
- Install: `make install` (the Makefile invokes `sudo`; never prefix this command with `sudo`)

## Adding a New Env Var

Follow the existing pattern exactly:
1. Dataclass field in `config.py` (with default)
2. Validated parsing in `load_config()` (try/except ValueError, SystemExit on bad input)
3. Pass through `pool.py` to `claude.py` if it affects the inner Claude
4. Prompt in `install.py` `_cmd_config()` (with validation loop)
5. Document in `templates/.env`
6. Add to `_CONFIG_ENV_VARS` list in `tests/test_config.py`

## Key Patterns

- `config.py` anchors all path resolution via `PROJECT_ROOT` and `DATA_DIR`
- Inner Claude uses stream-json I/O, not interactive mode (auto-memory doesn't work)
- Inner Claude CLI flags: verify with `claude --help` before use; use `--settings` for settings.json options
- Pyright strict: use assert-narrowing for Optional, helper functions for @property returns

## Workshop UI

- Keep navigation controls consistent and visually quiet: no background or
  border at rest, with subtle hover and keyboard-focus feedback.
- Prefer a familiar icon over a text button when the icon communicates the
  same action. Preserve an accessible label and tooltip for every icon-only
  control.
- If the appropriate icon is ambiguous, ask Daniel before implementing it.

## Don'ts

- Don't manually re-register the Telegram webhook. Just restart the service.
- Don't add features or refactor beyond what was asked.
- Spec files go in `home/docs/specs/`, not the project root.
- Kai (the inner agent, on any backend) must NEVER modify source files in this repo. Read, review, and report only.
- Don't push follow-up commits until the current CI run has completed.
