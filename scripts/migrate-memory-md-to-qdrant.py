#!/usr/bin/env python3
"""
Migrate operator-curated MEMORY.md content into Qdrant.

Phase 4 of #396 (issue #406). One-shot script that reads the operator's
deployed MEMORY.md, chunks it by section header with H3-to-H2 rollup
for short bodies, runs a per-chunk similarity check against existing
Qdrant content to skip likely duplicates, and writes survivors via
`memory.add_structured` with `source="migration"` metadata.

Headers above H3 (i.e. H4+) are flattened into their parent H3's body
text. The script's chunking granularity is H2 + H3 only, matching the
spec; deeper sub-headers contribute body text but do not produce
separate chunks.

Includes dry-run with score distribution summary, tunable similarity
threshold, idempotency guard via `memory.count_by_source`, and a
rollback path via `memory.delete_by_source` (wrapped in `asyncio.run`).

Usage:
    # Dry run (recommended first step; inspect the plan and score
    # distribution before committing to writes):
    python scripts/migrate-memory-md-to-qdrant.py --user-id 123 --dry-run

    # Real migration:
    python scripts/migrate-memory-md-to-qdrant.py --user-id 123

    # Rollback (deletes all migration-source rows for the user):
    python scripts/migrate-memory-md-to-qdrant.py --user-id 123 --rollback
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Exit codes (documented in the spec):
#   0 - success
#   1 - unexpected error
#   2 - idempotency guard fired (existing migration rows present)
#   3 - memory subsystem disabled (config.memory_enabled = False)
#   4 - MEMORY.md path does not exist
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_GUARD = 2
EXIT_MEMORY_DISABLED = 3
EXIT_FILE_MISSING = 4


# ── Chunking ───────────────────────────────────────────────────────


@dataclass
class Chunk:
    """One Qdrant-bound chunk extracted from MEMORY.md.

    Attributes:
        title: The header text (without the `#` markers). Empty for
            untitled H1 preamble.
        level: Heading level (1 = H1 preamble, 2 = H2 standalone or H2
            with rolled-up H3s, 3 = H3 standalone).
        body: The chunk's full body text WITHOUT the leading header
            line. For level=2 chunks with rolled-up H3 children, body
            includes the rolled-up content under the spec-mandated
            `\\n\\n### <title>\\n<body>` separator.
        parent_h2: For level=3 chunks, the H2 title above this chunk.
            None for level=1 and level=2.
        tag_slug: Slug of the title for level=3 chunks. Empty for
            level=1 and level=2 (which have no per-chunk topical tag
            beyond the universal `migration` tag).
    """

    title: str
    level: int
    body: str
    parent_h2: str | None
    tag_slug: str

    @property
    def text(self) -> str:
        """The full chunk text written to Qdrant (header + body).

        Format depends on level:
          - level=1: just the body (no `# Memory` line, since H1 is
            file-level metadata, not topical).
          - level=2: `## <title>\\n\\n<body>` (body may include rolled-
            up H3 sections).
          - level=3: `### <title>\\n\\n<body>` (body may include
            flattened H4+ sub-headers as part of the prose).
        """
        if self.level == 1:
            return self.body.strip()
        prefix = "#" * self.level
        return f"{prefix} {self.title}\n\n{self.body}".strip()


# Slug algorithm pinned by the spec (D6 worked-examples table). Lower-
# case, then collapse every run of non-alphanumeric characters into a
# single `-`, then strip leading/trailing `-`. Stable across all H3
# titles in the deployed MEMORY.md (parens, slashes, colons, backticks,
# version-string dots all handled uniformly).
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """Generate the metadata.tags H3-title slug (spec §D6)."""
    return _SLUG_RE.sub("-", title.lower()).strip("-")


# Header detection: lines starting with one or more `#` followed by a
# space and at least one non-space character. The count of leading `#`
# is the heading level. Captures only the title text after the `# `.
_HEADING_RE = re.compile(r"^(#+)\s+(.+?)\s*$")


def _chunk_memory_md(text: str, rollup_tokens: int) -> list[Chunk]:
    """Parse MEMORY.md text into Qdrant-bound chunks (spec §D1).

    Walks lines top-to-bottom maintaining (current_h2, current_h3)
    context. On each new heading, the pending chunk is emitted (if it
    has body content) and a fresh accumulator starts. Rollup happens
    in a second pass: H3 chunks below the body-token threshold get
    folded into their parent H2's chunk text under the
    `\\n\\n### <title>\\n<body>` separator from spec D1.

    Args:
        text: Full MEMORY.md file contents.
        rollup_tokens: H3 chunks with `_estimate_tokens(body)` strictly
            less than this value get rolled up into their parent H2.
            At-or-above stand alone. The default is 50 tokens, which
            corresponds to ~200 characters under the
            `len(text) // 4` heuristic.

    Returns:
        List of `Chunk` objects in document order. Headers with empty
        body and no children are dropped (they would produce one-line
        embeddings with no useful signal).
    """
    # Lazy import: keeps the module load light when only parsing is
    # exercised in tests (memory.py imports torch transitively).
    from kai.memory import _estimate_tokens

    # First pass: walk lines, accumulate raw segments.
    @dataclass
    class _Segment:
        title: str  # Empty string for the H1 preamble segment.
        level: int  # 1, 2, 3, or 4+ (4+ stays in body, no chunk emitted).
        body_lines: list[str] = field(default_factory=list)

    segments: list[_Segment] = []
    current = _Segment(title="", level=1)
    segments.append(current)

    for raw_line in text.splitlines():
        m = _HEADING_RE.match(raw_line)
        if m is None:
            current.body_lines.append(raw_line)
            continue
        level = len(m.group(1))
        title = m.group(2)
        if level >= 4:
            # H4+ stays inside the parent's body. Preserves the sub-
            # header line as text so retrieval can match on it. The
            # current segment continues; we do NOT start a new one.
            current.body_lines.append(raw_line)
            continue
        # H1, H2, or H3: start a new segment. Drop the H1 preamble
        # segment if its body is empty AND it has no title (no real
        # H1 line was seen, which is the case when the file starts
        # with a heading other than `# `).
        current = _Segment(title=title, level=level)
        segments.append(current)

    # Trim each segment's body. A trailing blank line before the next
    # heading would otherwise leak a `\n` into the chunk text.
    for seg in segments:
        # Drop leading and trailing blank lines but preserve internal
        # spacing (paragraph breaks).
        while seg.body_lines and not seg.body_lines[0].strip():
            seg.body_lines.pop(0)
        while seg.body_lines and not seg.body_lines[-1].strip():
            seg.body_lines.pop()

    # Build initial chunks from non-empty segments.
    chunks: list[Chunk] = []
    current_h2: str | None = None
    for seg in segments:
        body_text = "\n".join(seg.body_lines).strip()
        if seg.level == 1:
            if not body_text:
                continue  # No preamble; skip the placeholder H1 segment.
            chunks.append(
                Chunk(
                    title=seg.title or "Memory",
                    level=1,
                    body=body_text,
                    parent_h2=None,
                    tag_slug="",
                )
            )
        elif seg.level == 2:
            current_h2 = seg.title
            # Always emit H2 chunks even when their immediate body is
            # empty: rollup may fold short H3 children into them in
            # the second pass, at which point the chunk has content
            # to write. The empty-body case is filtered below if no
            # H3 rolled up.
            chunks.append(
                Chunk(
                    title=seg.title,
                    level=2,
                    body=body_text,
                    parent_h2=None,
                    tag_slug="",
                )
            )
        elif seg.level == 3:
            if not body_text:
                continue  # Empty H3 with no children: drop.
            chunks.append(
                Chunk(
                    title=seg.title,
                    level=3,
                    body=body_text,
                    parent_h2=current_h2,
                    tag_slug=slugify(seg.title),
                )
            )

    # Second pass: roll up small H3 chunks into their parent H2.
    # Walk in document order; when we encounter an H3 below threshold
    # whose parent_h2 matches the most recent H2 chunk, append its
    # `\n\n### <title>\n<body>` to that H2's body and drop the H3.
    rolled_up: list[Chunk] = []
    last_h2_idx: int | None = None
    for chunk in chunks:
        if chunk.level == 2:
            rolled_up.append(chunk)
            last_h2_idx = len(rolled_up) - 1
            continue
        if (
            chunk.level == 3
            and last_h2_idx is not None
            and chunk.parent_h2 == rolled_up[last_h2_idx].title
            and _estimate_tokens(chunk.body) < rollup_tokens
        ):
            # Roll up: append separator + title + body to parent H2.
            parent = rolled_up[last_h2_idx]
            separator = "\n\n" if parent.body else ""
            parent.body = f"{parent.body}{separator}### {chunk.title}\n{chunk.body}"
            continue
        rolled_up.append(chunk)

    # Final filter: drop H2 chunks that ended up with empty body (no
    # H2-level prose AND no H3 rolled up). Such a chunk would produce
    # a one-line `## Title` embedding with no retrieval value.
    return [c for c in rolled_up if c.body.strip()]


# ── Score distribution summary ─────────────────────────────────────


def _score_summary(scores: list[float], threshold: float) -> str:
    """Format the dry-run end-of-run score distribution (spec §D8)."""
    if not scores:
        return "Similarity score distribution: no chunks emitted (file empty after parsing).\n"

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    def pct(p: float) -> float:
        # Linear interpolation between adjacent ranks; matches numpy's
        # default percentile method without the import. n=1 returns
        # the single value regardless of percentile.
        if n == 1:
            return sorted_scores[0]
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return sorted_scores[lo] + frac * (sorted_scores[hi] - sorted_scores[lo])

    skipped = sum(1 for s in scores if s >= threshold)
    p90 = pct(0.90)
    near_misses = sum(1 for s in scores if p90 <= s < threshold)

    lines = [
        "",
        "Similarity score distribution (top-1 score per chunk):",
        f"  min:    {sorted_scores[0]:.3f}",
        f"  p25:    {pct(0.25):.3f}",
        f"  p50:    {pct(0.50):.3f}",
        f"  p75:    {pct(0.75):.3f}",
        f"  p90:    {p90:.3f}",
        f"  max:    {sorted_scores[-1]:.3f}",
        f"  Skipped at threshold {threshold:.2f}: {skipped}/{n} chunks",
        f"  Above p90 but below threshold: {near_misses} chunks (manual review recommended)",
    ]
    return "\n".join(lines) + "\n"


# ── Forward migration ──────────────────────────────────────────────


def _do_forward_migration(args: argparse.Namespace, chunks: list[Chunk]) -> int:
    """Execute the forward migration path (or dry-run preview).

    Returns the script exit code.
    """
    from kai.memory import add_structured, build_migration_metadata, search

    user_id_str = str(args.user_id)
    scores: list[float] = []
    added = 0
    skipped = 0
    errors = 0

    for idx, chunk in enumerate(chunks, start=1):
        results = search(chunk.text, user_id=user_id_str, limit=1)
        top_score = float(results[0].score) if results else 0.0
        scores.append(top_score)

        action = "SKIP" if top_score >= args.threshold else "ADD"
        title_label = chunk.title or f"(level-{chunk.level} chunk)"
        if args.verbose:
            # Print per-chunk plan inline. Earlier shape kept a
            # plan_lines list with no consumer past the loop body;
            # collapsed to a local since nothing outside the loop
            # reads accumulated plan output.
            print(f"CHUNK {idx:2d} {action} score={top_score:.3f} title={title_label!r}")

        if args.dry_run:
            continue
        if action == "SKIP":
            skipped += 1
            continue

        # Tags: ["migration", <h3-slug-or-empty>]. Empty slug is dropped
        # from the tag list to avoid storing a literal "" tag, which
        # would inflate /memory tag counts with a meaningless bucket.
        tags = ["migration"]
        if chunk.tag_slug:
            tags.append(chunk.tag_slug)
        section = chunk.parent_h2 or (chunk.title if chunk.level == 2 else "")
        subsection = chunk.title if chunk.level == 3 else ""
        try:
            mid = add_structured(
                chunk.text,
                user_id=user_id_str,
                memory_type="fact",
                tags=tags,
                # build_migration_metadata centralizes the migration-
                # row metadata shape so the writer here and the tests
                # in tests/test_memory.py drive the same code path.
                # The helper layers in `speaker` / `confidence` (the
                # migration constants) on top of section / subsection.
                metadata=build_migration_metadata(section=section, subsection=subsection),
            )
            if mid is None:
                # add_structured returns None on store failure or
                # disabled-memory; surface to stderr so the operator
                # can correlate the error count with specific chunks
                # (silent counter increment was the original bug).
                print(
                    f"ERROR on chunk {idx} ({title_label}): add_structured returned None",
                    file=sys.stderr,
                )
                errors += 1
            else:
                added += 1
        except Exception as exc:
            print(f"ERROR on chunk {idx} ({title_label}): {exc}", file=sys.stderr)
            errors += 1

    # Summary block. Always print the totals; print the score
    # distribution under --dry-run unless --no-summary is set.
    print()
    if args.dry_run:
        print(f"Dry run complete. {len(chunks)} chunks parsed.")
        if not args.no_summary:
            print(_score_summary(scores, args.threshold))
    else:
        print(f"Migration complete. Added: {added}, skipped: {skipped}, errors: {errors}.")

    return EXIT_OK if errors == 0 else EXIT_ERROR


# ── Rollback ───────────────────────────────────────────────────────


def _do_rollback(args: argparse.Namespace) -> int:
    """Execute --rollback (real or dry-run).

    Real mode wraps `delete_by_source` in `asyncio.run`. Dry-run mode
    calls the sync `count_by_source` and prints a preview without
    deleting anything.
    """
    from kai.memory import count_by_source, delete_by_source

    user_id_str = str(args.user_id)
    if args.dry_run:
        n = count_by_source(user_id_str, "migration")
        print(f"Dry-run rollback: would delete {n} migration entries for user_id {args.user_id}.")
        return EXIT_OK

    # Pre-check the count regardless of --yes so the zero-row case
    # short-circuits cleanly in both interactive and scripted paths.
    # Without this, --yes would call delete_by_source on a guaranteed-
    # empty result set, printing a confusing "Deleted 0 migration
    # entries" line for what was meant as a no-op verification.
    n = count_by_source(user_id_str, "migration")
    if n == 0:
        print(f"No migration entries to delete for user_id {args.user_id}. Exiting.")
        return EXIT_OK

    # Confirmation prompt unless --yes. Reads from stdin so the prompt
    # is interactive in a terminal but the operator can pass --yes in
    # scripted contexts.
    if not args.yes:
        prompt = f"Delete {n} migration entries for user_id {args.user_id}? [y/N]: "
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted.")
            return EXIT_OK

    deleted = asyncio.run(delete_by_source(user_id_str, "migration"))
    print(f"Deleted {deleted} migration entries for user_id {args.user_id}.")
    return EXIT_OK


# ── Main ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Migrate MEMORY.md content into Qdrant under source='migration'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--user-id", type=int, required=True, help="Telegram chat_id (required).")
    p.add_argument(
        "--memory-md",
        type=Path,
        default=None,
        help="Path to MEMORY.md (default: <DATA_DIR>/memory/<user_id>/MEMORY.md).",
    )
    p.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold for skip (default: 0.85).")
    p.add_argument(
        "--rollup-tokens",
        type=int,
        default=50,
        help="H3-to-H2 rollup body-token threshold; ~200 chars (default: 50).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan and summary, no writes.")
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable the score distribution summary printed under --dry-run.",
    )
    p.add_argument("--force", action="store_true", help="Bypass the idempotency guard.")
    p.add_argument(
        "--rollback",
        action="store_true",
        help="Delete all migration-source entries for --user-id; mutually exclusive with --force.",
    )
    p.add_argument("--yes", action="store_true", help="Skip the rollback confirmation prompt.")
    p.add_argument("-v", "--verbose", action="store_true", help="Per-chunk score logging.")
    return p


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject mutually-exclusive flag combinations early."""
    if args.rollback and args.force:
        parser.error("--rollback is mutually exclusive with --force")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    # Lazy imports for everything that pulls in torch/Mem0 so a `--help`
    # or argparse error exits without paying the import cost.
    from kai.config import DATA_DIR, load_config
    from kai.memory import count_by_source, init_memory

    config = load_config()
    if not config.memory_enabled:
        print(
            "ERROR: memory subsystem is disabled (config.memory_enabled = False). "
            "Set MEMORY_ENABLED=true and restart the service before running migration.",
            file=sys.stderr,
        )
        return EXIT_MEMORY_DISABLED

    init_memory(config)

    if args.rollback:
        return _do_rollback(args)

    # Forward migration. Resolve the MEMORY.md path.
    memory_md = args.memory_md or (DATA_DIR / "memory" / str(args.user_id) / "MEMORY.md")
    if not memory_md.exists():
        print(f"ERROR: MEMORY.md not found at {memory_md}.", file=sys.stderr)
        return EXIT_FILE_MISSING

    # Idempotency guard. Skip in --force mode (with a warning) or
    # under --dry-run (the operator is exploring; no risk of layered
    # writes since we do not write).
    user_id_str = str(args.user_id)
    if not args.dry_run:
        existing = count_by_source(user_id_str, "migration")
        if existing > 0 and not args.force:
            print(
                f"Migration source entries already exist for user_id {args.user_id} ({existing} entries).\n"
                "Refusing to run a second migration without --force; this would create\n"
                "duplicate Qdrant rows that the similarity check cannot reliably catch\n"
                "across edited chunks.\n"
                "\n"
                "To re-run cleanly:\n"
                f"  python scripts/migrate-memory-md-to-qdrant.py --user-id {args.user_id} --rollback\n"
                f"  python scripts/migrate-memory-md-to-qdrant.py --user-id {args.user_id}\n"
                "\n"
                "Or pass --force to layer this run on top of the prior one (NOT recommended).",
                file=sys.stderr,
            )
            return EXIT_GUARD
        if existing > 0 and args.force:
            print(
                f"WARNING: --force with {existing} existing migration rows. "
                f"Similarity check will skip exact duplicates but may miss near-duplicates.",
                file=sys.stderr,
            )

    chunks = _chunk_memory_md(memory_md.read_text(encoding="utf-8"), rollup_tokens=args.rollup_tokens)
    print(f"Parsed {len(chunks)} chunks from {memory_md}.")

    return _do_forward_migration(args, chunks)


if __name__ == "__main__":
    sys.exit(main())
