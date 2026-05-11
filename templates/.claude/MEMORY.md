# Memory

Your facts live in this file when the memory subsystem is disabled (`MEMORY_ENABLED=false`), or as the migration source when it is enabled (`MEMORY_ENABLED=true` plus `python -m kai memory migrate`). Rules for inner Claude live in `<DATA_DIR>/home/<chat_id>/.claude/CLAUDE.md`, not here.

When this file is the active fact surface, inner Claude reads it on every turn (injected as `[Your persistent memory (file: ...):]`) and writes to it via `Edit` / `Write` when persisting new facts. Keep it organized by section so retrieval stays cheap and updates stay precise.

## About the User

## Ongoing Projects

## Preferences

## Notes
