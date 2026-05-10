# Preferences

This file is your always-on personal rule layer. Inner Claude reads it on every turn (injected as `[Your personal preferences (file: ...):]`) but does NOT auto-edit it. Treat it like `CLAUDE.md`: a curated set of always-on rules, edited deliberately, never silently appended.

Write to PREFERENCES.md ONLY when the operator explicitly instructs ("save this as a preference," "add this to PREFERENCES," "make this always-on"). Even then, surface the proposed wording and confirm before persisting. Proactive saves of inferred preferences route to `MEMORY.md` (or Qdrant in enabled mode), NOT here.

Keep PREFERENCES.md tight. Anything that should fire every turn (writing style, formatting directives, behavioral rules tied to the operator's specific working patterns) belongs here. Anything fact-shaped (project state, decisions, dated incidents, contextual references) belongs in `MEMORY.md`, where retrieval surfaces it on similarity rather than on every turn.

## Style

## Working Discipline

## Behavioral
