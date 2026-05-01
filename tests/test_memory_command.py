"""
Unit tests for `src/kai/memory_command.py`.

Covers:
- Subcommand parsing dispatch (every subcommand routes to the right
  send-helper, including the empty-query fall-through to help).
- Callback encoding/decoding round-trips, including a 64-byte ceiling
  check on the longest realistic callback.
- Pure builder rendering: dashboard, fact view, forget confirmation,
  search, stats, episode list view.
- Per-chat cache: insert, single-entry overwrite, lazy TTL expiry.
- Empty-state branches: zero-fact dashboard, zero-result search.

No real Mem0 instance is involved. The handler tests that drive
`_send_*` swap `kai.memory.get_stats` / `get_by_id` / `delete_by_id`
/ `search` / `get_all_episodes` for fakes via monkeypatch, and swap
`kai.memory.is_enabled` to return True.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai import memory_command
from kai.memory import MemoryResult, MemoryStats

# ── Fixture builders ────────────────────────────────────────────────


def _fact(
    fact_id: str,
    text: str,
    tags: list[str],
    confidence: float = 0.85,
    *,
    confirmation_quote: str = "",
    session_id: str = "session_test",
    prompt_version: str = "v3",
    created_at: str = "2026-04-17T10:00:00",
    updated_at: str | None = None,
    score: float = 0.0,
) -> MemoryResult:
    """Construct a MemoryResult shaped like an extracted fact.

    Defaults match what `memory_extraction.py` writes into Mem0
    metadata. Tests override only the fields that matter to them.
    """
    metadata: dict[str, Any] = {
        "source": "extracted",
        "tags": tags,
        "confidence": confidence,
        "session_id": session_id,
        "prompt_version": prompt_version,
        "type": "fact",
    }
    if confirmation_quote:
        metadata["confirmation_quote"] = confirmation_quote
    return MemoryResult(
        id=fact_id,
        text=text,
        score=score,
        memory_type="fact",
        metadata=metadata,
        created_at=created_at,
        updated_at=updated_at if updated_at is not None else created_at,
    )


def _stats(
    *,
    extracted_count: int = 0,
    episode_count: int = 0,
    migration_count: int = 0,
    by_tag: dict[str, int] | None = None,
    confidence_min: float | None = None,
    confidence_median: float | None = None,
    confidence_max: float | None = None,
    confidence_below_0_7: int = 0,
    confidence_below_0_6: int = 0,
    confirmation_quote_count: int = 0,
    by_prompt_version: dict[str, int] | None = None,
) -> MemoryStats:
    """Construct a MemoryStats with extracted-only aggregates.

    `episode_count` and `migration_count` (issue #407) default to zero
    so existing tests stay unchanged. Tests that exercise the
    multi-source dashboard / stats / fact-view branches override them
    explicitly.
    """
    return MemoryStats(
        total_count=extracted_count + episode_count + migration_count,
        by_type={"fact": extracted_count} if extracted_count else {},
        extracted_count=extracted_count,
        episode_count=episode_count,
        migration_count=migration_count,
        by_tag=by_tag or {},
        confidence_min=confidence_min,
        confidence_median=confidence_median,
        confidence_max=confidence_max,
        confidence_below_0_7=confidence_below_0_7,
        confidence_below_0_6=confidence_below_0_6,
        confirmation_quote_count=confirmation_quote_count,
        by_prompt_version=by_prompt_version or {},
    )


# ── Cache helpers (test-local) ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_cache():
    """Wipe the module-level screen cache between tests.

    The cache is module-global so a leak from one test would leak
    into the next. The autouse fixture guarantees test order is
    irrelevant to assertions about cache state.
    """
    memory_command._screen_cache.clear()
    yield
    memory_command._screen_cache.clear()


# ── Callback encode/decode ─────────────────────────────────────────


class TestCallbackCodec:
    """Round-trip every callback verb the dispatcher handles."""

    def test_encode_dashboard_verb_only(self):
        # Verb-only callbacks have no args section.
        assert memory_command._encode_callback("dash") == "mem:dash"

    def test_encode_with_args(self):
        encoded = memory_command._encode_callback("tag", "preference", "2")
        assert encoded == "mem:tag:preference:2"

    def test_decode_verb_only(self):
        action = memory_command._decode_callback("mem:dash")
        assert action is not None
        assert action.verb == "dash"
        assert action.args == []

    def test_decode_with_args(self):
        action = memory_command._decode_callback("mem:tag:preference:2")
        assert action is not None
        assert action.verb == "tag"
        assert action.args == ["preference", "2"]

    def test_decode_unknown_prefix_returns_none(self):
        # Other CallbackQueryHandlers register `ws:`, `voice:`, etc.
        # We must return None so the wrong dispatcher does not match.
        assert memory_command._decode_callback("ws:home") is None

    def test_decode_empty_body_returns_none(self):
        assert memory_command._decode_callback("mem:") is None

    def test_longest_realistic_callback_under_64_bytes(self):
        # The longest legitimate post-dismantle callback is the
        # episode-list page verb with a multi-digit page index:
        #   mem:eps:999  (12 bytes)
        # Verifying the constructor's assertion does not fire and the
        # encoded length is well under the 64-byte limit.
        encoded = memory_command._encode_callback("eps", "999")
        assert len(encoded.encode("utf-8")) <= 64
        assert encoded == "mem:eps:999"

    def test_overlong_callback_raises(self):
        # The 64-byte ceiling is enforced via `if/raise`, not assert,
        # so the check survives `python -O` (which strips assertions).
        # If it ever fires in real use it indicates a bug in the
        # caller passing an unexpectedly long arg, not a runtime
        # condition; raising loudly beats silent Telegram truncation.
        with pytest.raises(ValueError, match="callback_data too long"):
            memory_command._encode_callback("x", "a" * 100)


# ── Constants ───────────────────────────────────────────────────────


# ── Display helpers ─────────────────────────────────────────────────


class TestTruncate:
    def test_short_text_unchanged(self):
        assert memory_command._truncate("short", limit=80) == "short"

    def test_long_text_gets_ellipsis(self):
        text = "x" * 100
        result = memory_command._truncate(text, limit=20)
        assert len(result) == 20
        assert result.endswith("\u2026")

    def test_newlines_collapsed_to_spaces(self):
        # Embedded newlines would break the single-line list layout.
        assert memory_command._truncate("a\nb\rc") == "a b c"


class TestFormatDate:
    def test_date_only_slice(self):
        assert memory_command._format_date("2026-04-17T10:30:00") == "2026-04-17"

    def test_date_with_time(self):
        assert memory_command._format_date("2026-04-17T10:30:00", with_time=True) == "2026-04-17 10:30"

    def test_empty_returns_empty(self):
        assert memory_command._format_date("") == ""

    def test_short_iso_falls_back(self):
        # Mock fixtures sometimes pass dates without time. The function
        # must not blow up.
        assert memory_command._format_date("2026-04-17", with_time=True) == "2026-04-17"


# ── Builder: dashboard ─────────────────────────────────────────────


class TestBuildDashboard:
    def test_empty_state(self):
        text, kb = memory_command._build_dashboard(_stats(extracted_count=0))
        assert "No memories yet" in text
        assert kb is None


# ── Builder: pagination ────────────────────────────────────────────


class TestPaginate:
    def test_empty_returns_one_page(self):
        # Even with zero facts, total_pages must read as 1 so the
        # "page X of Y" footer does not show "page 1 of 0".
        window, page, total = memory_command._paginate([], 0)
        assert window == []
        assert page == 0
        assert total == 1

    def test_single_page(self):
        facts = [_fact(f"id{i}", f"t{i}", ["preference"]) for i in range(3)]
        window, page, total = memory_command._paginate(facts, 0)
        assert len(window) == 3
        assert page == 0
        assert total == 1

    def test_exact_page_boundary(self):
        # 5 facts at page size 5 must produce exactly 1 page (not 2).
        facts = [_fact(f"id{i}", f"t{i}", ["preference"]) for i in range(5)]
        _, _, total = memory_command._paginate(facts, 0)
        assert total == 1

    def test_last_page_partial(self):
        # 7 facts at page size 5: first page has 5, second has 2.
        facts = [_fact(f"id{i}", f"t{i}", ["preference"]) for i in range(7)]
        window, _, total = memory_command._paginate(facts, 1)
        assert len(window) == 2
        assert total == 2

    def test_page_clamped_when_too_high(self):
        # An out-of-range page falls back to the last valid page
        # rather than rendering an empty screen.
        facts = [_fact(f"id{i}", f"t{i}", ["preference"]) for i in range(7)]
        _, page, total = memory_command._paginate(facts, 99)
        assert page == total - 1

    def test_negative_page_clamped(self):
        facts = [_fact(f"id{i}", f"t{i}", ["preference"]) for i in range(3)]
        _, page, _ = memory_command._paginate(facts, -1)
        assert page == 0


# ── Builder: fact view ─────────────────────────────────────────────


class TestBuildFactView:
    def test_renders_all_metadata_lines(self):
        f = _fact(
            "id1",
            "Never use em dashes.",
            ["preference", "constraint"],
            confidence=0.92,
            session_id="session_abc123",
            prompt_version="v3",
            created_at="2026-04-17T14:32:00",
        )
        text, kb = memory_command._build_fact_view(f, return_to=("tag", ["preference", "0"]))
        assert '"Never use em dashes."' in text
        assert "preference, constraint" in text
        assert "Confidence:       0.92" in text
        assert "2026-04-17 14:32" in text
        assert "session_abc123" in text
        assert "v3" in text
        # No confirmation quote on a non-confirmed_action fact.
        assert "n/a" in text
        # Two buttons in one row: back + forget.
        row = kb.inline_keyboard[0]
        assert [btn.text for btn in row] == ["back", "forget"]

    def test_back_button_routes_to_return_to(self):
        f = _fact("id1", "x", ["preference"])
        _, kb = memory_command._build_fact_view(f, return_to=("tag", ["preference", "2"]))
        back_btn = kb.inline_keyboard[0][0]
        assert back_btn.callback_data == "mem:tag:preference:2"

    def test_no_return_to_falls_back_to_dashboard(self):
        f = _fact("id1", "x", ["preference"])
        _, kb = memory_command._build_fact_view(f, return_to=None)
        back_btn = kb.inline_keyboard[0][0]
        assert back_btn.callback_data == "mem:dash"

    def test_confirmed_action_renders_quote(self):
        f = _fact(
            "id1",
            "Will deploy on Friday.",
            ["confirmed_action"],
            confirmation_quote="Yes, deploy on Friday at 5pm please",
        )
        text, _ = memory_command._build_fact_view(f, return_to=None)
        assert "Yes, deploy on Friday" in text


# ── Builder: forget confirmations ──────────────────────────────────


class TestBuildForgetFactConfirm:
    def test_text_includes_fact_quote(self):
        f = _fact("id1", "Do not use em dashes.", ["preference"])
        text, kb = memory_command._build_forget_fact_confirm(f)
        assert "Forget this fact?" in text
        assert '"Do not use em dashes."' in text
        assert "cannot be undone" in text
        # Confirm + cancel buttons.
        labels = [btn.text for btn in kb.inline_keyboard[0]]
        assert labels == ["confirm forget", "cancel"]


# ── Builder: search results ────────────────────────────────────────


class TestBuildSearchResults:
    def test_empty_results(self):
        text, kb, ids = memory_command._build_search_results("anything", [], 0.3)
        assert "No matching memories found" in text
        assert ids == []
        # Single back button.
        assert kb.inline_keyboard[0][0].text == "back"

    def test_renders_score_not_confidence(self):
        # Spec §6.5: the bracketed number on the search screen is the
        # Mem0 similarity score, not the Haiku confidence.
        f = _fact("id1", "preference text", ["preference"], confidence=0.92, score=0.84)
        text, _, ids = memory_command._build_search_results("q", [f], 0.3)
        assert "[0.84]" in text  # score
        assert "[0.92]" not in text  # not confidence
        assert ids == ["id1"]

    def test_floor_appears_in_footer(self):
        f = _fact("id1", "x", ["preference"], score=0.84)
        text, _, _ = memory_command._build_search_results("q", [f], 0.42)
        assert "(0.4)" in text  # floor formatted to one decimal


# ── Builder: stats ─────────────────────────────────────────────────


class TestBuildStats:
    def test_empty_state(self):
        text, kb = memory_command._build_stats(_stats(extracted_count=0))
        # Empty-state wording mirrors the dashboard's two-bucket
        # surface (facts + episodes); migration folds into "facts"
        # so the phrasing has no third item.
        assert "No facts or episodes yet" in text
        assert kb.inline_keyboard[0][0].text == "back"

    def test_renders_full_aggregates(self):
        stats = _stats(
            extracted_count=142,
            by_tag={
                "preference": 38,
                "fact": 27,
                "decision": 22,
                "project": 18,
                "constraint": 14,
                "confirmed_action": 10,
                "schedule": 8,
                "relationship": 5,
                # location intentionally omitted to test zero-display
            },
            confidence_min=0.52,
            confidence_median=0.87,
            confidence_max=0.99,
            confidence_below_0_7=11,
            confidence_below_0_6=3,
            confirmation_quote_count=10,
            by_prompt_version={"v3": 128, "v2": 14},
        )
        text, _ = memory_command._build_stats(stats)
        # Per-bucket totals: an extracted-only operator sees just the
        # "Total facts:" line. The Total imported: row is gone post-
        # fold; migration counts roll into the same Total facts:
        # number. Total episodes: is still suppressed when zero.
        assert "Total facts:      142" in text
        assert "Total episodes:" not in text
        assert "Total imported:" not in text
        # No "By tag:" section in the stats surface (issue #416
        # dropped the per-tag count table along with the rest of
        # the tag UI).
        assert "By tag:" not in text
        # Confidence block
        assert "min               0.52" in text
        assert "median            0.87" in text
        assert "max               0.99" in text
        assert "below 0.7" in text
        # Percentages
        assert "(7.7%)" in text
        # Confirmed actions
        assert "10 with confirmation_quote" in text
        # Prompt versions sorted desc by count
        assert text.index("v3") < text.index("v2")

    def test_renders_n_a_for_missing_confidence(self):
        # If the memory.py invariant ever breaks (extracted_count > 0
        # but min/median/max are None), the stats screen must say
        # "n/a" rather than "0.00". A real 0.00 confidence reading
        # would be indistinguishable from a missing one otherwise.
        # Below-threshold counts get the same n/a treatment in this
        # state - rendering "0 (0.0%)" would read as "all facts
        # scored above the threshold" rather than "no data".
        stats = _stats(
            extracted_count=5,
            confidence_min=None,
            confidence_median=None,
            confidence_max=None,
        )
        text, _ = memory_command._build_stats(stats)
        assert "min               n/a" in text
        assert "median            n/a" in text
        assert "max               n/a" in text
        # Below-threshold rows fall back to (n/a) too.
        assert "below 0.7           0  (n/a)" in text
        assert "below 0.6           0  (n/a)" in text
        # And no spurious 0.00 / (0.0%) leaking in from the
        # confidence block.
        assert "0.00" not in text
        assert "(0.0%)" not in text

    def test_renders_percentages_when_confidence_present(self):
        # Inverse of the n/a test: when confidence data is present,
        # the below-threshold rows must still render as percentages
        # (regression guard against the n/a fallback over-firing).
        stats = _stats(
            extracted_count=10,
            confidence_min=0.5,
            confidence_median=0.8,
            confidence_max=0.95,
            confidence_below_0_7=2,
            confidence_below_0_6=1,
        )
        text, _ = memory_command._build_stats(stats)
        assert "below 0.7           2  (20.0%)" in text
        assert "below 0.6           1  (10.0%)" in text
        assert "(n/a)" not in text

    def test_renders_int_keyed_prompt_version_bucket(self):
        # The renderer must tolerate int keys in by_prompt_version.
        # A MemoryStats handed straight to the renderer (bypassing
        # the aggregation cast in memory.py) can surface non-string
        # keys; tests construct MemoryStats directly, and any caller
        # that does the same bypasses the cast. Without the str()
        # wraps in _build_stats, len(int) raises TypeError and the
        # stats view becomes unreachable.
        stats = _stats(
            extracted_count=1,
            # Helper annotation is dict[str, int]; the int key here
            # is the load-bearing fixture - len("1") works on the
            # un-fixed renderer, len(1) raises.
            by_prompt_version={1: 1},  # type: ignore[dict-item]
        )
        text, _kb = memory_command._build_stats(stats)
        assert "Prompt versions:" in text
        # Stringified int appears as the version label.
        assert "1" in text

    def test_renders_mixed_int_str_keyed_prompt_versions(self):
        # The sort comparator must tolerate dicts with mixed int/str
        # keys - Python 3 raises TypeError on int<->str comparison,
        # so the tiebreaker key must wrap item[0] in str() *before*
        # the sort runs. A single-key fixture cannot exercise this
        # path because sort never compares; need at least two keys
        # of different types and equal counts to force the
        # tiebreaker to fire.
        stats = _stats(
            extracted_count=2,
            # Equal counts force the int<->str tiebreaker comparison.
            by_prompt_version={1: 1, "2": 1},  # type: ignore[dict-item]
        )
        text, _kb = memory_command._build_stats(stats)
        assert "Prompt versions:" in text
        # Both buckets render without raising; presence is enough.
        assert "1" in text
        assert "2" in text


# ── Issue #407: multi-source UI surface ────────────────────────────
#
# Episode and migration rows joined extracted as user-visible sources
# in `/memory` (memory.py `USER_VISIBLE_SOURCES`). The four classes
# below exercise the dashboard, stats, fact-view, and forget-confirm
# branches added by the spec.


def _episode_fact(
    fact_id: str = "ep-1",
    text: str = "Set up CI for the kai repo. Outcome: succeeded.",
    *,
    tags: list[str] | None = None,
    outcome_quality: str | None = "good",
    approach: str | None = "Configured GitHub Actions step-by-step.",
    outcome: str | None = "CI ran on every PR within an hour.",
    actors: list[str] | None = None,
    lessons: str | None = None,
    created_at: str = "2026-04-28T15:00:00",
) -> MemoryResult:
    """Construct a MemoryResult shaped like an episode row.

    Episode rows do not carry confidence / session_id / prompt_version
    / confirmation_quote in production; tests deliberately omit those
    fields so the renderer's "omit extractor-only rows" branch is
    actually exercised. `outcome_quality` is None when the test wants
    to drive the no-quality branch of the header.

    `approach`, `outcome`, `actors`, and `lessons` are issue #412
    metadata. Defaults populate `approach`, `outcome`, and `actors`
    (the schema-required content-bearing fields, so existing tests
    that pre-date #412 render with a realistic body shape rather than
    empty `Approach:  ` / `Outcome:  ` rows). `lessons` defaults to
    None (the generator's "no lesson this time" sentinel), so tests
    must opt in to render the Lessons row.

    Presence-check (`is not None`) on `actors` matters: a caller that
    passes `actors=[]` to test the empty-list defensive-fallback path
    must see an empty list survive helper transit. `actors or
    ["operator"]` would silently collapse the empty list to the
    default and the test's intent would never reach the renderer.
    """
    metadata: dict[str, Any] = {
        "source": "episode",
        "tags": tags if tags is not None else [],
        "type": "fact",
    }
    if outcome_quality is not None:
        metadata["outcome_quality"] = outcome_quality
    if approach is not None:
        metadata["approach"] = approach
    if outcome is not None:
        metadata["outcome"] = outcome
    metadata["actors"] = actors if actors is not None else ["operator"]
    if lessons is not None:
        metadata["lessons"] = lessons
    return MemoryResult(
        id=fact_id,
        text=text,
        score=0.0,
        memory_type="fact",
        metadata=metadata,
        created_at=created_at,
        updated_at=created_at,
    )


def _migration_fact(
    fact_id: str = "mig-1",
    text: str = "### /backend\nRole-based backend assignment.",
    *,
    tags: list[str] | None = None,
    created_at: str = "2026-04-29T16:30:00",
) -> MemoryResult:
    """Construct a MemoryResult shaped like a migration row (#408).

    Tags default to ["migration"] which is what the migration script
    writes for chunks without an H3 slug; tests exercising tag rendering
    should pass an explicit list including the H3 slug too.
    """
    metadata: dict[str, Any] = {
        "source": "migration",
        "tags": tags if tags is not None else ["migration"],
        "type": "fact",
    }
    return MemoryResult(
        id=fact_id,
        text=text,
        score=0.0,
        memory_type="fact",
        metadata=metadata,
        created_at=created_at,
        updated_at=created_at,
    )


class TestDashboardMultiSource:
    """`/memory` dashboard surfacing of episode and migration rows."""

    def test_dashboard_headline_sums_extracted_and_migration(self):
        """Headline sums extracted+migration into a single "facts"
        count and lists episodes separately. Migration is internal
        plumbing; the operator does not see a separate "imported"
        count anywhere."""
        stats = _stats(
            extracted_count=12,
            episode_count=3,
            migration_count=25,
            by_tag={"preference": 5},
            confidence_median=0.85,
            confidence_min=0.6,
        )
        text, _ = memory_command._build_dashboard(stats)
        assert "37 facts" in text
        assert "3 episodes" in text
        assert "imported" not in text

    def test_dashboard_omits_zero_source_counts(self):
        """A fresh extracted-only operator sees a headline naming
        only their non-zero bucket. Zero-valued buckets are omitted
        from the comma list rather than rendered as "0 episodes"."""
        stats = _stats(
            extracted_count=10,
            by_tag={"preference": 5},
            confidence_median=0.9,
            confidence_min=0.8,
        )
        text, _ = memory_command._build_dashboard(stats)
        assert "Memories: 10 facts." in text
        assert "episodes" not in text
        assert "imported" not in text
        # Issue #416 dropped the "across N tags" suffix from the
        # headline when the tag UI was retired.
        assert "across" not in text

    def test_dashboard_episode_only_user_renders(self):
        """An episode-only operator (`episode_count > 0`,
        `extracted_count == 0`, `migration_count == 0`) sees the
        dashboard rendered (NOT _MSG_NO_FACTS). Footer points at
        the Episodes/Stats button row.

        Pins the empty-state guard fix from issue #407: the
        pre-#407 dashboard returned _MSG_NO_FACTS for this input
        via its extracted-only guard."""
        stats = _stats(episode_count=4)
        text, kb = memory_command._build_dashboard(stats)
        # Not the empty state.
        assert "No memories yet" not in text
        assert kb is not None
        # Headline shows the episode count.
        assert "4 episodes" in text
        # Footer two-sentence form names Episodes and Stats only.
        assert "Tap Episodes to browse. Tap Stats for details." in text
        # Keyboard has only the utility row, which includes the
        # Episodes button alongside Stats. No Search button.
        assert len(kb.inline_keyboard) == 1
        utility_row = kb.inline_keyboard[-1]
        assert [btn.text for btn in utility_row] == ["Episodes (4)", "Stats"]

    def test_dashboard_migration_only_user_renders(self):
        """Migration-only operator: migration_count > 0,
        extracted_count == 0, episode_count == 0. The empty-state
        guard does not fire because migration_count is positive.
        The Facts button renders (extracted + migration > 0) and
        the headline reads "25 facts"; Episodes does not render
        and the operator sees no "imported" wording anywhere."""
        stats = _stats(migration_count=25)
        text, kb = memory_command._build_dashboard(stats)
        assert "No memories yet" not in text
        assert kb is not None
        assert "25 facts" in text
        assert "imported" not in text
        # Migration-only is the facts-only footer branch.
        assert "Tap Facts to browse. Tap Stats for details." in text
        assert len(kb.inline_keyboard) == 1
        utility_row = kb.inline_keyboard[-1]
        assert [btn.text for btn in utility_row] == ["Facts (25)", "Stats"]

    def test_dashboard_all_zero_returns_empty_state(self):
        """The empty state fires only when every user-visible source
        is zero. Pins the combined-guard contract."""
        stats = _stats()  # all three counts default to 0
        text, kb = memory_command._build_dashboard(stats)
        assert text == memory_command._MSG_NO_FACTS
        assert kb is None


class TestStatsMultiSource:
    """`/memory stats` surfacing of episode and migration totals."""

    def test_stats_view_total_facts_sums_extracted_and_migration(self):
        """The headline sums extracted and migration into a single
        Total facts: line; episodes get their own line. There is no
        Total imported: row anywhere in the rendered output."""
        stats = _stats(
            extracted_count=12,
            episode_count=3,
            migration_count=25,
            by_tag={"preference": 12},
            confidence_min=0.6,
            confidence_median=0.85,
            confidence_max=0.95,
        )
        text, _ = memory_command._build_stats(stats)
        assert "Total facts:      37" in text
        assert "Total episodes:   3" in text
        assert "Total imported:" not in text
        assert "imported" not in text

    def test_stats_view_migration_only_renders_facts_total(self):
        """A migration-only operator sees a single Total facts: line
        with the migration count, no Total imported: row, and no
        "imported" wording anywhere."""
        stats = _stats(migration_count=25)
        text, _ = memory_command._build_stats(stats)
        assert "Total facts:      25" in text
        assert "imported" not in text

    def test_stats_view_all_zero_says_no_memories_yet(self):
        """All-zero empty-state mirrors the dashboard's two-bucket
        surface; migration folds into facts so the phrasing has no
        third item."""
        stats = _stats()
        text, kb = memory_command._build_stats(stats)
        assert text == "Memory stats\n\nNo facts or episodes yet."
        # Back button is still rendered so the user can return to the
        # dashboard rather than being stuck on the stats screen.
        assert kb.inline_keyboard[0][0].text == "back"

    def test_stats_view_episode_only_omits_extracted_sections(self):
        """An episode-only operator's stats output contains only the
        per-source headline. The three extracted-shaped sections
        (confidence block, confirmed-actions line, prompt-version
        table) are omitted because they read extractor-only metadata.

        Pins the omit-on-extracted-zero contract. The negative
        `"By tag:" not in text` assertion stays valid even though
        the per-tag section was retired across the board; keeping
        the pin protects against a regression that revives any
        per-tag rendering on this branch.
        """
        stats = _stats(episode_count=5)
        text, _ = memory_command._build_stats(stats)
        assert "Total episodes:   5" in text
        # None of the extracted-shaped sections render. The
        # per-tag table is retired entirely (issue #416), but the
        # negative assertion stays valid as a regression pin.
        assert "By tag:" not in text
        assert "Confidence:" not in text
        assert "Confirmed actions:" not in text
        assert "Prompt versions:" not in text


class TestFactViewMultiSource:
    """`_build_fact_view` per-source render branches."""

    def test_fact_view_renders_episode_with_outcome_quality(self):
        """Episode header includes outcome_quality as a parenthetical
        and the body shows Tags + Date only - none of the four
        extractor-only rows (Confidence/Session/Prompt version/
        Confirmation) appear."""
        fact = _episode_fact(outcome_quality="good", tags=["sophia/topic"])
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        # Header reads "Episode (good)".
        assert "Episode (good)" in text
        # Tags line renders.
        assert "Tags:" in text
        assert "sophia/topic" in text
        # Date line renders with HH:MM precision.
        assert "Date:" in text
        # Extractor-only rows are absent. Each label is checked
        # individually so a regression that adds back ANY one of them
        # trips the assertion.
        assert "Confidence:" not in text
        assert "Session:" not in text
        assert "Prompt version:" not in text
        assert "Confirmation:" not in text

    def test_fact_view_renders_episode_without_outcome_quality(self):
        """When `outcome_quality` is missing from metadata the header
        falls back to a bare "Episode" with no parenthetical."""
        fact = _episode_fact(outcome_quality=None)
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        # Header reads "Episode" without parenthetical. Use a regex-
        # style assertion: line begins with "Episode\n" (no "(").
        first_line = text.split("\n", 1)[0]
        assert first_line == "Episode"

    def test_fact_view_renders_migration_with_fact_header(self):
        """Migration row renders the `Fact` header (matching extracted)
        and the same minimal Tags + Date body it used to. The header
        no longer calls out the extracted/migration distinction; the
        operator-facing UI treats them as one bucket. The body shape
        stays minimal because migration rows do not carry the four
        extractor-only fields."""
        fact = _migration_fact(tags=["migration", "backend"])
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        # Header matches extracted; the literal "Imported" string
        # must not appear anywhere in the rendered text.
        first_line = text.split("\n", 1)[0]
        assert first_line == "Fact"
        assert "Imported" not in text
        # Tags include the migration H3 slug.
        assert "migration" in text
        assert "backend" in text
        # Date renders.
        assert "Date:" in text
        # No extractor-only rows.
        assert "Confidence:" not in text
        assert "Session:" not in text
        assert "Prompt version:" not in text
        assert "Confirmation:" not in text

    def test_fact_view_renders_extracted_unchanged(self):
        """Pins the unchanged-extracted-shape contract: extracted
        rows still render the original six-row block (Tags,
        Confidence, Date, Session, Prompt version, Confirmation).
        A regression that accidentally routes extracted rows
        through the episode / migration shape would drop four of
        these labels."""
        fact = _fact("id1", "Prefers tea.", ["preference"], confidence=0.85, prompt_version="v3")
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        first_line = text.split("\n", 1)[0]
        assert first_line == "Fact"
        assert "Tags:" in text
        assert "Confidence:" in text
        assert "Date:" in text
        assert "Session:" in text
        assert "Prompt version:" in text
        assert "Confirmation:" in text

    # ── Issue #412: episode detail-fields ──────────────────────────
    #
    # Adds `Approach`, `Outcome`, `Lessons` (when present), `Actors`
    # rows to the episode branch of `_build_fact_view`, between the
    # body quote and the existing `Tags` / `Date` footer. The
    # extracted and migration branches stay unchanged.

    def test_fact_view_episode_renders_approach_outcome(self):
        """approach and outcome are schema-required Sophia fields;
        always render with two-space-after-colon formatting."""
        fact = _episode_fact(approach="A1", outcome="O1")
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        assert "Approach:  A1" in text
        assert "Outcome:  O1" in text

    def test_fact_view_episode_renders_lessons_when_present(self):
        """lessons is optional; when the metadata key is set, the
        Lessons row renders alongside the always-on rows."""
        fact = _episode_fact(lessons="L1")
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        assert "Lessons:  L1" in text

    def test_fact_view_episode_omits_lessons_row_when_absent(self):
        """Pins the lessons-absence design intent: the metadata key
        is absent (not empty-string) when the generator chose not
        to record a lesson. The row is dropped entirely rather than
        rendered as `Lessons:  (none)`, which would misrepresent
        the generator's intent as "considered and rejected" rather
        than "no lesson this time"."""
        fact = _episode_fact(lessons=None)
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        assert "Lessons:" not in text

    def test_fact_view_episode_renders_actors_comma_joined(self):
        fact = _episode_fact(actors=["alice", "bob"])
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        assert "Actors:  alice, bob" in text

    def test_fact_view_episode_actors_empty_list_renders_none(self):
        """Exercises the defensive `(none)` fallback for an empty
        actors list. Production data cannot produce this state
        because the schema enforces `minItems=1`, so the test pins
        a code path that production cannot reach. Documented as an
        unreachable-state pin so a future reader does not delete
        it as testing an impossible state."""
        fact = _episode_fact(actors=[])
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        assert "Actors:  (none)" in text

    def test_fact_view_episode_field_order(self):
        """Pins the four-row vertical ordering against future edits:
        Approach -> Outcome -> Lessons (when present) -> Actors,
        before the existing Tags / Date footer."""
        # Custom body text avoids the labels appearing inside the
        # body quote (the default helper text contains "Outcome:"
        # in narrative form). text.index(label) on the rendered
        # string would otherwise pick up the body occurrence.
        fact = _episode_fact(
            text="A short body without label collisions.",
            approach="aaa",
            outcome="bbb",
            lessons="ccc",
            actors=["alice"],
            tags=["sophia/topic"],
        )
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        positions = {
            label: text.index(label) for label in ("Approach:", "Outcome:", "Lessons:", "Actors:", "Tags:", "Date:")
        }
        assert positions["Approach:"] < positions["Outcome:"]
        assert positions["Outcome:"] < positions["Lessons:"]
        assert positions["Lessons:"] < positions["Actors:"]
        assert positions["Actors:"] < positions["Tags:"]
        assert positions["Tags:"] < positions["Date:"]

    def test_fact_view_episode_label_uses_two_spaces_after_colon(self):
        """Pins the two-space-after-colon formatting in code so the
        convention does not drift away from prose comments. Each of
        the four new labels uses exactly two spaces after the colon
        - not one (cramped) and not pad-aligned, which is reserved
        for the wider extracted-row label set ("Confidence:",
        "Prompt version:", "Confirmation:") where pad-alignment
        improves scanability. Asserts on every label individually
        so a copy-paste error on any one row trips the test."""
        fact = _episode_fact(approach="A", outcome="O", lessons="L", actors=["X"])
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        # Positive: every label rendered with two-space spacing.
        assert "Approach:  A" in text
        assert "Outcome:  O" in text
        assert "Lessons:  L" in text
        assert "Actors:  X" in text
        # Negative regression for each label: pad-aligned form
        # (3+ spaces) and single-space form must NOT appear.
        for label in ("Approach", "Outcome", "Lessons", "Actors"):
            assert f"{label}:   " not in text
        assert "Approach: A" not in text
        assert "Outcome: O" not in text
        assert "Lessons: L" not in text
        assert "Actors: X" not in text

    def test_fact_view_episode_absent_approach_outcome_render_empty_fallback(self):
        """`approach` and `outcome` are schema-required content-bearing
        fields. The renderer's `or ""` defensive fallback for an
        absent metadata key surfaces the corruption as an empty
        labeled row rather than crashing. Production data cannot
        produce this state because the schema requires both fields,
        so the test pins a code path that production cannot reach -
        analogous to the actors-empty-list defensive-fallback pin.
        Documented as
        unreachable-state coverage so a future reader does not
        delete it."""
        fact = _episode_fact(approach=None, outcome=None)
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        # Both rows render with empty values after the two-space
        # separator. The trailing-whitespace match would be ambiguous
        # so anchor on a label-followed-by-newline pattern.
        assert "Approach:  \n" in text
        assert "Outcome:  \n" in text

    def test_fact_view_episode_unchanged_extracted_branch(self):
        """Confirms the extracted six-row block stays unchanged: a
        regression that accidentally routed extracted rows through
        the episode shape would drop the extractor-only labels and
        could pick up the new labels. Both directions are pinned
        here."""
        fact = _fact("id1", "Prefers tea.", ["preference"])
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        for label in ("Approach:", "Outcome:", "Lessons:", "Actors:"):
            assert label not in text

    def test_fact_view_episode_unchanged_migration_branch(self):
        """Required as a separate test even though the assertion is
        identical to the extracted-branch test, because the two cases
        exercise different code paths in `_build_fact_view`:
        extracted goes through the standalone extracted-branch
        `else:` arm (which never touches `detail_lines`), while
        migration goes through the combined episode-or-migration
        branch with `detail_lines = []` initialized and the inner
        `if source == "episode":` guard skipped, leaving the splice
        empty. A regression mis-indenting the
        `detail_lines.append(...)` calls so they fire for migration
        too would only be caught by this test."""
        fact = _migration_fact(tags=["migration", "backend"])
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        for label in ("Approach:", "Outcome:", "Lessons:", "Actors:"):
            assert label not in text

    def test_episode_detail_has_blank_lines_between_sections(self):
        """Each labelled section in the episode detail (Approach,
        Outcome, Lessons, Actors, Tags) must have a blank line
        before it. Pins the Actors-to-Tags separator added when the
        detail view gained inter-section spacing; a regression that
        drops any of the four blanks would let two adjacent labelled
        rows fuse together visually."""
        fact = _episode_fact(
            tags=["sophia/topic"],
            approach="Approach body that may be long.",
            outcome="Outcome body that resolves the conversation.",
            lessons="Lesson body the generator chose to record.",
            actors=["alice", "bob"],
        )
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        lines = text.split("\n")
        for label in ("Outcome:", "Lessons:", "Actors:", "Tags:"):
            # Find the line containing the label; assert the line
            # immediately above it is blank. The Approach line is
            # preceded by the body's trailing blank from the lines
            # list construction, so it does not need an extra check.
            label_idx = next(i for i, line in enumerate(lines) if line.startswith(label))
            assert lines[label_idx - 1] == "", f"missing blank line before {label}"

    def test_episode_detail_lessons_absent_collapses_blank_lines(self):
        """When `lessons` is absent the entire Lessons block (label
        line plus its trailing blank) is skipped, leaving exactly
        one blank line between Outcome and Actors. Pins against a
        regression that emits a double blank or no blank when the
        conditional drops out."""
        fact = _episode_fact(
            tags=["sophia/topic"],
            approach="Approach body.",
            outcome="Outcome body.",
            lessons=None,
            actors=["alice"],
        )
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        lines = text.split("\n")
        outcome_idx = next(i for i, line in enumerate(lines) if line.startswith("Outcome:"))
        actors_idx = next(i for i, line in enumerate(lines) if line.startswith("Actors:"))
        # Exactly one blank line between Outcome and Actors. That
        # means actors_idx - outcome_idx == 2 (Outcome line, blank,
        # Actors line). The literal Lessons row must be absent.
        assert actors_idx - outcome_idx == 2
        assert "Lessons:" not in text


class TestDashboardEpisodesButton:
    """Issue #410: Episodes (N) button on the dashboard utility row.

    Conditional on `stats.episode_count > 0`. Post-Search-removal the
    utility row is one of `[Stats]`, `[Facts (N), Stats]`,
    `[Episodes (N), Stats]`, or `[Facts (N), Episodes (N), Stats]`.
    Tests in this class pin the Episodes-button half of the matrix
    (visibility, placement, footer wording when Episodes is the only
    browse button)."""

    def test_dashboard_renders_episodes_button_when_episode_count_nonzero(self):
        # Mixed-source fixture: extracted_count=10 makes the Facts
        # button visible too, so the row is the both-buttons shape.
        # Episodes sits at index 1 (Facts at 0).
        stats = _stats(extracted_count=10, episode_count=4, by_tag={"preference": 5})
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        utility_row = kb.inline_keyboard[-1]
        labels = [btn.text for btn in utility_row]
        assert labels == ["Facts (10)", "Episodes (4)", "Stats"]
        # Callback shape: mem:eps:<page>; initial page is 0.
        # Episodes is at index 1 because Facts now occupies index 0.
        episodes_btn = utility_row[1]
        assert episodes_btn.callback_data == "mem:eps:0"

    def test_dashboard_omits_episodes_button_when_episode_count_zero(self):
        # extracted_count=10 makes Facts visible, so the row is
        # [Facts (10), Stats]. Episodes is correctly absent.
        # The test name still describes the omit-when-zero intent;
        # the row contents also confirm the Facts presence.
        stats = _stats(extracted_count=10, episode_count=0, by_tag={"preference": 5})
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        utility_row = kb.inline_keyboard[-1]
        labels = [btn.text for btn in utility_row]
        assert labels == ["Facts (10)", "Stats"]

    def test_dashboard_episode_only_user_renders_episodes_button(self):
        # Episode-only operator: utility row is the only keyboard
        # row, with Episodes alongside Stats. Pins the cross-section
        # of the episode-only surface and the Episodes button
        # placement. Facts is hidden because extracted_count and
        # migration_count both default to zero in _stats().
        stats = _stats(episode_count=4)
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        # Only one row total: the utility row.
        assert len(kb.inline_keyboard) == 1
        utility_row = kb.inline_keyboard[0]
        labels = [btn.text for btn in utility_row]
        assert labels == ["Episodes (4)", "Stats"]

    def test_dashboard_with_episodes_footer_wording(self):
        # Footer when Episodes is rendered (and Facts is not):
        # two-sentence form names Episodes and Stats. Episode-only
        # fixture; extracted and migration both default to zero so
        # Facts stays hidden.
        stats = _stats(episode_count=4)
        text, _ = memory_command._build_dashboard(stats)
        assert "Tap Episodes to browse. Tap Stats for details." in text

    def test_dashboard_facts_only_footer_wording(self):
        # Migration-only operator hits the facts-only footer branch
        # (extracted=0, episode=0, migration=25), so Facts is the
        # only browse button visible. Negative regression on
        # "Tap Episodes" stays valid: Episodes really is hidden here.
        stats = _stats(migration_count=25)
        text, _ = memory_command._build_dashboard(stats)
        assert "Tap Facts to browse. Tap Stats for details." in text
        # Negative regression: the Episodes branch wording must NOT
        # appear when no Episodes button is rendered.
        assert "Tap Episodes to browse" not in text


class TestDashboardFactsButton:
    """Facts (N) button on the dashboard utility row.

    Conditional on `stats.extracted_count + stats.migration_count > 0`.
    The label sums extracted and migration; the headline now sums
    them too (a single "<X> facts" headline part rather than
    extracted + imported split). Order on the utility row, when both
    Facts and Episodes are visible, is Facts first (left), then
    Episodes, then Stats.

    The empty-state guard at the top of `_build_dashboard` returns
    early when all three source counts are zero, so the
    no-browse-buttons footer wording is unreachable production code
    and is intentionally not pinned by a test."""

    def test_dashboard_renders_facts_button_when_extracted_or_migration_nonzero(self):
        # Facts label sums extracted (10) and migration (2). Episode
        # count is zero so this is the facts-only row shape.
        stats = _stats(extracted_count=10, migration_count=2)
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        utility_row = kb.inline_keyboard[-1]
        labels = [btn.text for btn in utility_row]
        assert labels == ["Facts (12)", "Stats"]
        # Callback shape: mem:facts:<page>; initial page is 0.
        facts_btn = utility_row[0]
        assert facts_btn.callback_data == "mem:facts:0"

    def test_dashboard_facts_button_label_sums_extracted_and_migration(self):
        # Explicit sum check on a fixture where both contributions
        # are non-zero. Pins the sum semantics so a regression that
        # uses only one of the two counts trips this assertion.
        stats = _stats(extracted_count=7, migration_count=18)
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        utility_row = kb.inline_keyboard[-1]
        labels = [btn.text for btn in utility_row]
        assert "Facts (25)" in labels

    def test_dashboard_omits_facts_button_when_extracted_and_migration_both_zero(self):
        # Episode-only operator: facts_visible is false, so the row
        # should be [Episodes, Stats] with no Facts button. Pins the
        # negative case so a future regression that always emits
        # Facts trips here.
        stats = _stats(episode_count=4)
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        utility_row = kb.inline_keyboard[-1]
        labels = [btn.text for btn in utility_row]
        assert labels == ["Episodes (4)", "Stats"]
        assert not any(label.startswith("Facts ") for label in labels)

    def test_dashboard_facts_only_user_renders_facts_button(self):
        # Migration-only operator: the row is exactly [Facts, Stats]
        # (no Episodes). Tests the migration-only case and the
        # facts-only row shape together.
        stats = _stats(migration_count=25)
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        utility_row = kb.inline_keyboard[-1]
        labels = [btn.text for btn in utility_row]
        assert labels == ["Facts (25)", "Stats"]

    def test_dashboard_both_browse_buttons_renders_facts_first(self):
        # Both browse buttons visible: Facts must come first
        # (left-most), then Episodes, then Stats. Pins the placement
        # so a regression that swaps the two browse buttons trips
        # here.
        stats = _stats(extracted_count=10, migration_count=2, episode_count=4)
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        utility_row = kb.inline_keyboard[-1]
        labels = [btn.text for btn in utility_row]
        assert labels == ["Facts (12)", "Episodes (4)", "Stats"]

    def test_dashboard_footer_both_browse_wording(self):
        # When both Facts and Episodes render, the footer enumerates
        # both browse buttons in the two-sentence form. Pins the
        # "Tap Facts or Episodes to browse." phrasing so a regression
        # to either single-button branch trips this assertion.
        stats = _stats(extracted_count=10, episode_count=4)
        text, _ = memory_command._build_dashboard(stats)
        assert "Tap Facts or Episodes to browse. Tap Stats for details." in text

    def test_dashboard_footer_facts_only_wording(self):
        # When Facts renders but Episodes does not, footer says
        # "Tap Facts to browse." (without Episodes). The cross-test
        # of the facts-only footer also lives in
        # TestDashboardEpisodesButton's
        # `test_dashboard_facts_only_footer_wording`; both pin the
        # same branch from different angles.
        stats = _stats(extracted_count=15)
        text, _ = memory_command._build_dashboard(stats)
        assert "Tap Facts to browse. Tap Stats for details." in text
        # Negative regression: the both-buttons phrasing must NOT
        # leak in when only Facts is visible.
        assert "Tap Facts or Episodes" not in text

    def test_dashboard_footer_episodes_only_wording(self):
        # When Episodes renders but Facts does not, footer reads the
        # episodes-only branch of the matrix. Pins the branch so a
        # sloppy refactor cannot silently rewrite this message.
        stats = _stats(episode_count=4)
        text, _ = memory_command._build_dashboard(stats)
        assert "Tap Episodes to browse. Tap Stats for details." in text
        # Negative regression: Facts-related wording must not leak
        # into the episodes-only branch.
        assert "Tap Facts" not in text


class TestBuildEpisodeListView:
    """Issue #410: paginated list view of episode rows."""

    def _episode(
        self,
        fact_id: str,
        text: str,
        outcome_quality: str | None = "success",
        *,
        created_at: str = "2026-04-29T10:00:00",
        updated_at: str | None = None,
    ) -> MemoryResult:
        metadata: dict[str, Any] = {"source": "episode", "type": "fact"}
        if outcome_quality is not None:
            metadata["outcome_quality"] = outcome_quality
        return MemoryResult(
            id=fact_id,
            text=text,
            score=0.0,
            memory_type="fact",
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at if updated_at is not None else created_at,
        )

    def test_outcome_quality_brackets_no_longer_render(self):
        """Negative regression: episode list rows no longer carry the
        per-row [<outcome_quality>] bracket. The outcome_quality
        field stays load-bearing in the episode detail view (header
        parenthetical) but is not surfaced in the list view; per-row
        quality bracketing was visual noise that did not help triage
        at the row level. A future revival of the bracket should be
        a deliberate spec change, not a silent edit; this assertion
        catches the silent-edit case."""
        eps = [
            self._episode("e1", "First episode", outcome_quality="success"),
            self._episode("e2", "Second episode", outcome_quality="partial"),
            self._episode("e3", "Third episode", outcome_quality="failure"),
            self._episode("e4", "Fourth episode", outcome_quality=None),
            self._episode("e5", "Fifth episode", outcome_quality="off-enum-value"),
        ]
        text, _, ids, _, _ = memory_command._build_episode_list_view(eps, 0)
        # All four enumerated bracket forms must be absent.
        assert "[success]" not in text
        assert "[partial]" not in text
        assert "[failure]" not in text
        assert "[----]" not in text
        # The fact texts still render in row order.
        assert "First episode" in text
        assert "Fifth episode" in text
        assert ids == ["e1", "e2", "e3", "e4", "e5"]

    def test_paginates(self):
        # 12 episodes at page-size 5 -> page 1: 5, page 2: 5, page 3: 2.
        eps = [self._episode(f"e{i}", f"Episode {i}") for i in range(12)]
        # Page 1 (index 0).
        _, kb, ids, page, total = memory_command._build_episode_list_view(eps, 0)
        assert len(ids) == 5
        assert page == 0
        assert total == 3
        nav_labels = [btn.text for btn in kb.inline_keyboard[-1]]
        assert nav_labels == ["back", "next >"]
        # Page 2 (middle).
        _, kb, ids, page, total = memory_command._build_episode_list_view(eps, 1)
        assert len(ids) == 5
        assert page == 1
        nav_labels = [btn.text for btn in kb.inline_keyboard[-1]]
        assert nav_labels == ["< prev", "back", "next >"]
        # Page 3 (partial last).
        _, kb, ids, page, total = memory_command._build_episode_list_view(eps, 2)
        assert len(ids) == 2
        assert page == 2
        nav_labels = [btn.text for btn in kb.inline_keyboard[-1]]
        assert nav_labels == ["< prev", "back"]

    def test_empty_state(self):
        text, kb, ids, page, total = memory_command._build_episode_list_view([], 0)
        assert text == "Episodes\n\nNo episodes yet."
        assert ids == []
        assert page == 0
        assert total == 1
        # Single back button on the keyboard.
        assert len(kb.inline_keyboard) == 1
        assert kb.inline_keyboard[0][0].text == "back"

    def test_truncates_long_text(self):
        long_text = "x" * 200
        eps = [self._episode("e1", long_text)]
        text, _, _, _, _ = memory_command._build_episode_list_view(eps, 0)
        assert long_text not in text
        assert "…" in text

    def test_header_uses_two_space_separator(self):
        # Header label and parenthetical are separated by exactly
        # two spaces. Pin so a single-space regression trips this
        # test.
        eps = [self._episode("e1", "Just one")]
        text, _, _, _, _ = memory_command._build_episode_list_view(eps, 0)
        assert text.startswith("Episodes  (page 1 of 1)")

    def test_episode_list_date_indent_four_spaces(self):
        # Date line on a non-empty row is indented exactly four
        # spaces, lining up with the start of the truncated fact
        # text on the line above (after the "N.  " prefix). Same
        # convention the facts list uses; pin the indent so a
        # regression to twelve spaces (the pre-bracket-removal
        # indent) trips this test.
        eps = [self._episode("e1", "Some episode", created_at="2026-04-30T10:00:00")]
        text, _, _, _, _ = memory_command._build_episode_list_view(eps, 0)
        date_line = next(line for line in text.split("\n") if line.lstrip().startswith("2026-04-30"))
        assert date_line == "    2026-04-30"


class TestBuildFactsListView:
    """Paginated list view of fact-bucket rows (extracted + migration).

    The list surface is intentionally source-agnostic: rows from both
    sources render with the same shape (`N. <text>` + indented date),
    no bracket field, no source label. The per-source distinction
    surfaces only on the fact-view detail screen, which inherits the
    issue #407 per-source rendering."""

    def _row(
        self,
        fact_id: str,
        text: str,
        source: str = "extracted",
        *,
        created_at: str = "2026-04-29T10:00:00",
        updated_at: str | None = None,
    ) -> MemoryResult:
        return MemoryResult(
            id=fact_id,
            text=text,
            score=0.0,
            memory_type="fact",
            metadata={"source": source, "type": "fact"},
            created_at=created_at,
            updated_at=updated_at if updated_at is not None else created_at,
        )

    def test_renders_extracted_and_migration_rows_indistinguishably(self):
        # Mixed list: one extracted row, one migration row. The list
        # view must not surface any source distinction in the rendered
        # text or in any per-row affordance. The fact-view detail
        # screen handles the per-source rendering separately.
        rows = [
            self._row("e1", "Extracted one", source="extracted"),
            self._row("m1", "Migration one", source="migration"),
        ]
        text, _kb, ids, _, _ = memory_command._build_facts_list_view(rows, 0)
        # No source labels, no bracket fields.
        assert "[extracted]" not in text
        assert "[migration]" not in text
        assert "Imported" not in text
        assert "Fact:" not in text
        # Both rows present, in input order (caller pre-sorts).
        assert "1.  Extracted one" in text
        assert "2.  Migration one" in text
        assert ids == ["e1", "m1"]

    def test_paginates(self):
        # 12 rows at page-size 5 -> page 1: 5, page 2: 5, page 3: 2.
        rows = [self._row(f"f{i}", f"Fact {i}") for i in range(12)]
        # Page 1 (index 0).
        _, kb, ids, page, total = memory_command._build_facts_list_view(rows, 0)
        assert len(ids) == 5
        assert page == 0
        assert total == 3
        nav_labels = [btn.text for btn in kb.inline_keyboard[-1]]
        assert nav_labels == ["back", "next >"]
        # Page 2 (middle).
        _, kb, ids, page, total = memory_command._build_facts_list_view(rows, 1)
        assert len(ids) == 5
        assert page == 1
        nav_labels = [btn.text for btn in kb.inline_keyboard[-1]]
        assert nav_labels == ["< prev", "back", "next >"]
        # Page 3 (partial last).
        _, kb, ids, page, total = memory_command._build_facts_list_view(rows, 2)
        assert len(ids) == 2
        assert page == 2
        nav_labels = [btn.text for btn in kb.inline_keyboard[-1]]
        assert nav_labels == ["< prev", "back"]

    def test_empty_state(self):
        text, kb, ids, page, total = memory_command._build_facts_list_view([], 0)
        assert text == "Facts\n\nNo facts yet."
        assert ids == []
        assert page == 0
        assert total == 1
        # Single back button on the keyboard.
        assert len(kb.inline_keyboard) == 1
        assert kb.inline_keyboard[0][0].text == "back"
        # Back goes to dashboard, not to a stale facts page.
        assert kb.inline_keyboard[0][0].callback_data == "mem:dash"

    def test_truncates_long_text(self):
        long_text = "x" * 200
        rows = [self._row("f1", long_text)]
        text, _, _, _, _ = memory_command._build_facts_list_view(rows, 0)
        assert long_text not in text
        assert "…" in text

    def test_header_uses_two_space_separator(self):
        # Header label and parenthetical are separated by exactly
        # two spaces, matching the episode header convention. Pin
        # so a single-space regression trips this test.
        rows = [self._row("f1", "Just one")]
        text, _, _, _, _ = memory_command._build_facts_list_view(rows, 0)
        assert text.startswith("Facts  (page 1 of 1)")

    def test_date_indent_four_spaces(self):
        # Date line is indented exactly four spaces, lining up under
        # the start of the fact text after the "N.  " prefix. Pin
        # the indent so a regression to twelve spaces (the episode
        # list's indent) trips this test.
        rows = [self._row("f1", "Some fact", created_at="2026-04-30T10:00:00")]
        text, _, _, _, _ = memory_command._build_facts_list_view(rows, 0)
        # The date line is the second non-blank content line.
        date_line = next(line for line in text.split("\n") if line.lstrip().startswith("2026-04-30"))
        assert date_line == "    2026-04-30"

    def test_number_buttons_use_fact_verb(self):
        # Number buttons share the `fact` verb with the episode list.
        # The fact-detail open is source-agnostic (cache.memory_ids
        # at integer index); the list-screen sentinel determines the
        # back-target, not the number-button verb.
        rows = [self._row(f"f{i}", f"Fact {i}") for i in range(3)]
        _, kb, _, _, _ = memory_command._build_facts_list_view(rows, 0)
        number_row = kb.inline_keyboard[0]
        assert [btn.text for btn in number_row] == ["1", "2", "3"]
        for i, btn in enumerate(number_row):
            assert btn.callback_data == f"mem:fact:{i}"

    def test_nav_buttons_use_facts_verb(self):
        # Prev and next buttons must encode `mem:facts:<page>`, not
        # `mem:eps:<page>`. Pin the verb so a copy-paste regression
        # from the episode list trips here.
        rows = [self._row(f"f{i}", f"Fact {i}") for i in range(12)]
        # Middle page so both prev and next render.
        _, kb, _, _, _ = memory_command._build_facts_list_view(rows, 1)
        nav_row = kb.inline_keyboard[-1]
        prev_btn = next(btn for btn in nav_row if btn.text == "< prev")
        next_btn = next(btn for btn in nav_row if btn.text == "next >")
        assert prev_btn.callback_data == "mem:facts:0"
        assert next_btn.callback_data == "mem:facts:2"


class TestEpsCallbackDispatch:
    """Issue #410: handle_memory_callback `eps` verb dispatch."""

    @pytest.mark.asyncio
    async def test_eps_verb_dispatches_to_send_episode_list(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        captured: dict[str, Any] = {}

        async def fake_send(update, context, chat_id, page, edit=False):
            captured["chat_id"] = chat_id
            captured["page"] = page
            captured["edit"] = edit

        monkeypatch.setattr(memory_command, "_send_episode_list", fake_send)
        upd = update_factory(callback_data="mem:eps:0")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        assert captured["chat_id"] == 100
        assert captured["page"] == 0
        assert captured["edit"] is True

    @pytest.mark.asyncio
    async def test_eps_verb_invalid_page_falls_back_to_zero(self, monkeypatch, update_factory, context_factory):
        # Stale or crafted callback with a non-integer page must
        # not crash; the dispatch falls back to page 0.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        captured: dict[str, Any] = {}

        async def fake_send(update, context, chat_id, page, edit=False):
            captured["page"] = page

        monkeypatch.setattr(memory_command, "_send_episode_list", fake_send)
        upd = update_factory(callback_data="mem:eps:notaninteger")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        assert captured["page"] == 0


class TestFactViewEpisodeReturn:
    """Issue #410: back-navigation from a fact opened from the
    episode list returns to the same page of that list."""

    @pytest.mark.asyncio
    async def test_fact_view_back_returns_to_episode_list(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)

        # Prime the cache as if the user is on page 2 of the
        # episode list.
        memory_command._set_cache(
            100,
            memory_command._ScreenCache(
                screen="episodes",
                page=2,
                memory_ids=["e1"],
            ),
        )

        # Stub get_by_id to return an episode-shaped row so the
        # fact view actually renders.
        episode = MemoryResult(
            id="e1",
            text="Some episode",
            score=0.0,
            memory_type="fact",
            metadata={"source": "episode", "outcome_quality": "success"},
            created_at="2026-04-29T10:00:00",
            updated_at="2026-04-29T10:00:00",
        )
        monkeypatch.setattr(memory_command.memory, "get_by_id", lambda *, user_id, memory_id: episode)

        upd = update_factory(callback_data="mem:fact:0")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)

        # The fact view's edit_message_text was called with a
        # keyboard whose back button targets the episode list at
        # page 2 (mem:eps:2).
        edit_call = upd.callback_query.edit_message_text.call_args
        assert edit_call is not None
        kb = edit_call.kwargs["reply_markup"]
        back_btn = kb.inline_keyboard[0][0]
        assert back_btn.text == "back"
        assert back_btn.callback_data == "mem:eps:2"


class TestFactsCallbackDispatch:
    """handle_memory_callback `facts` verb dispatch.

    Mirrors `TestEpsCallbackDispatch`: the `facts` verb is the
    initial entry point for the facts list browser, with a single
    integer page argument."""

    @pytest.mark.asyncio
    async def test_facts_verb_dispatches_to_send_facts_list(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        captured: dict[str, Any] = {}

        async def fake_send(update, context, chat_id, page, edit=False):
            captured["chat_id"] = chat_id
            captured["page"] = page
            captured["edit"] = edit

        monkeypatch.setattr(memory_command, "_send_facts_list", fake_send)
        upd = update_factory(callback_data="mem:facts:0")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        assert captured["chat_id"] == 100
        assert captured["page"] == 0
        assert captured["edit"] is True

    @pytest.mark.asyncio
    async def test_facts_verb_invalid_page_falls_back_to_zero(self, monkeypatch, update_factory, context_factory):
        # Stale or crafted callback with a non-integer page must
        # not crash; the dispatch falls back to page 0. Same
        # contract as `eps`.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        captured: dict[str, Any] = {}

        async def fake_send(update, context, chat_id, page, edit=False):
            captured["page"] = page

        monkeypatch.setattr(memory_command, "_send_facts_list", fake_send)
        upd = update_factory(callback_data="mem:facts:notaninteger")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        assert captured["page"] == 0


class TestFactViewFactsReturn:
    """Back-navigation from a fact opened from the facts list returns
    to the same page of that list. Mirrors `TestFactViewEpisodeReturn`
    using the new `"facts"` cache sentinel."""

    @pytest.mark.asyncio
    async def test_fact_view_back_returns_to_facts_list(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)

        # Prime the cache as if the user is on page 2 of the facts
        # list.
        memory_command._set_cache(
            100,
            memory_command._ScreenCache(
                screen="facts",
                page=2,
                memory_ids=["f1"],
            ),
        )

        # Stub get_by_id to return an extracted-shaped row so the
        # fact view renders. The source value here is irrelevant to
        # the back-target wiring; the cache sentinel is what
        # determines the back-target.
        fact = MemoryResult(
            id="f1",
            text="Some fact",
            score=0.0,
            memory_type="fact",
            metadata={"source": "extracted", "tags": ["preference"]},
            created_at="2026-04-29T10:00:00",
            updated_at="2026-04-29T10:00:00",
        )
        monkeypatch.setattr(memory_command.memory, "get_by_id", lambda *, user_id, memory_id: fact)

        upd = update_factory(callback_data="mem:fact:0")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)

        # The fact view's edit_message_text was called with a
        # keyboard whose back button targets the facts list at page
        # 2 (mem:facts:2). Distinct from the existing eps target.
        edit_call = upd.callback_query.edit_message_text.call_args
        assert edit_call is not None
        kb = edit_call.kwargs["reply_markup"]
        back_btn = kb.inline_keyboard[0][0]
        assert back_btn.text == "back"
        assert back_btn.callback_data == "mem:facts:2"


class TestForgetFactReturnFacts:
    """Post-delete return-to wiring for facts opened from the facts
    list. Mirrors the existing eps-branch behavior on `ffd`."""

    @pytest.mark.asyncio
    async def test_forget_fact_from_facts_list_returns_to_facts_list(
        self, monkeypatch, update_factory, context_factory
    ):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        # Mock the delete to succeed so the post-delete branch fires.
        monkeypatch.setattr(memory_command.memory, "delete_by_id", lambda *, user_id, memory_id: True)

        # Capture the post-delete dispatch by stubbing
        # _send_facts_list. The branch should call this rather than
        # _send_dashboard.
        captured: dict[str, Any] = {}

        async def fake_send_facts(update, context, chat_id, page, edit=False):
            captured["facts_called"] = True
            captured["page"] = page

        async def fake_send_dashboard(update, context, chat_id, edit=False):
            captured["dashboard_called"] = True

        monkeypatch.setattr(memory_command, "_send_facts_list", fake_send_facts)
        monkeypatch.setattr(memory_command, "_send_dashboard", fake_send_dashboard)

        # Prime the cache as if the user is on the fact-detail view
        # for a fact opened from page 1 of the facts list.
        memory_command._set_cache(
            100,
            memory_command._ScreenCache(
                screen="fact",
                memory_ids=["f1"],
                return_to=("facts", ["1"]),
            ),
        )

        upd = update_factory(callback_data="mem:ffd")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)

        # _send_facts_list must have been called with page 1, and
        # the dashboard fallback must NOT have been invoked.
        assert captured.get("facts_called") is True
        assert captured["page"] == 1
        assert captured.get("dashboard_called") is not True

    @pytest.mark.asyncio
    async def test_forget_fact_from_facts_list_invalid_page_falls_back_to_zero(
        self, monkeypatch, update_factory, context_factory
    ):
        # Defensive parsing: a malformed return_to page (e.g., from
        # a corrupted in-memory cache) must not crash; it falls back
        # to page 0. Same contract as the eps branch.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        monkeypatch.setattr(memory_command.memory, "delete_by_id", lambda *, user_id, memory_id: True)

        captured: dict[str, Any] = {}

        async def fake_send_facts(update, context, chat_id, page, edit=False):
            captured["page"] = page

        monkeypatch.setattr(memory_command, "_send_facts_list", fake_send_facts)

        memory_command._set_cache(
            100,
            memory_command._ScreenCache(
                screen="fact",
                memory_ids=["f1"],
                return_to=("facts", ["notaninteger"]),
            ),
        )

        upd = update_factory(callback_data="mem:ffd")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        assert captured["page"] == 0


class TestDashboardSearchButtonRemoved:
    """Pins the absence of the Search button on the dashboard.

    Telegram inline keyboards cannot accept text input, so the
    Search button could only ever surface a help alert with the
    slash-command syntax. The slash-command path `/memory search
    <q>` is the actual search entry point; the button was a
    redundant affordance that occupied a slot on the utility row
    without doing search-shaped work."""

    def test_dashboard_utility_row_has_no_search_button(self):
        # Mixed-source fixture so both browse buttons render. The
        # utility row has Facts, Episodes, Stats. No Search button
        # by label and no button with the help-callback data.
        stats = _stats(extracted_count=10, episode_count=4)
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        utility_row = kb.inline_keyboard[-1]
        labels = [btn.text for btn in utility_row]
        assert "Search" not in labels
        callbacks = [btn.callback_data for btn in utility_row]
        assert "mem:help" not in callbacks

    def test_dashboard_footer_no_search_callout(self):
        # Footer wording across all reachable branches no longer
        # contains the Search call-out. Negative regression on the
        # old "Search to find specific memories" phrasing.
        stats = _stats(extracted_count=10, episode_count=4)
        text, _ = memory_command._build_dashboard(stats)
        assert "Search" not in text
        assert "to find specific memories" not in text

    @pytest.mark.asyncio
    async def test_help_callback_falls_through_to_unknown_verb_dismiss(
        self, monkeypatch, update_factory, context_factory
    ):
        # Stale `mem:help` callbacks fired from chat history pre-
        # deploy must degrade gracefully via the unknown-verb
        # dismiss (`await query.answer()` with no args), the same
        # forward-compat path the retired `tag` verb relied on.
        # Stub `is_enabled` so the dispatch reaches the verb
        # branches; verify no send helper was invoked and the
        # answer call carries no toast or alert content.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)

        called: dict[str, bool] = {}

        async def fail_send(*args, **kwargs):
            called["sent"] = True
            raise AssertionError("no send helper should fire for the orphaned mem:help verb")

        # Cover the full set of send helpers so any accidental
        # routing trips the assertion.
        for name in ("_send_dashboard", "_send_stats", "_send_episode_list", "_send_facts_list", "_send_fact_view"):
            monkeypatch.setattr(memory_command, name, fail_send)

        upd = update_factory(callback_data="mem:help")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)

        # The dispatch's trailing `await query.answer()` is the
        # unknown-verb dismiss; it carries no positional arguments
        # and no `text=` kwarg.
        upd.callback_query.answer.assert_awaited_once_with()
        assert "sent" not in called


class TestHelpTextSummaryLine:
    """Issue #410 D7: `_HELP_TEXT` first line wording change."""

    def test_help_text_says_browse_memories(self):
        assert "/memory - Browse memories" in memory_command._HELP_TEXT
        # Pre-#410 wording must be gone so a future revert trips
        # this assertion.
        assert "Browse by tag" not in memory_command._HELP_TEXT


class TestForgetConfirmMultiSource:
    """`_build_forget_fact_confirm` per-source label injection."""

    def test_delete_confirm_extracted_keeps_verb_and_warning(self):
        """Pins the verb + warning regression-test contract per D4:
        the verb stays `Forget`, the noun for extracted rows stays
        `fact`, and the irreversibility cue (`This cannot be undone.`)
        remains. A future change that swaps wording should be a
        deliberate spec change, not a silent edit."""
        fact = _fact("id1", "Prefers tea.", ["preference"])
        text, kb = memory_command._build_forget_fact_confirm(fact)
        assert text.startswith("Forget")
        assert "fact" in text
        assert "This cannot be undone." in text
        # Inline-button flow unchanged.
        labels = [btn.text for btn in kb.inline_keyboard[0]]
        assert labels == ["confirm forget", "cancel"]

    def test_delete_confirm_episode_uses_episode_label(self):
        """Episode rows render the prompt with the `episode` noun.
        Verb and warning sentence are unchanged from the extracted
        case so the operator sees one consistent surface."""
        fact = _episode_fact()
        text, _ = memory_command._build_forget_fact_confirm(fact)
        assert text.startswith("Forget")
        assert "episode" in text
        assert "This cannot be undone." in text

    def test_delete_confirm_migration_uses_fact_label(self):
        """Migration rows render the prompt with the `fact` noun
        (matching extracted). The operator-facing UI no longer
        surfaces the extracted/migration distinction; both source
        rows share the prompt wording. Negative regression on the
        old `imported memory` label trips a future revert."""
        fact = _migration_fact()
        text, _ = memory_command._build_forget_fact_confirm(fact)
        assert text.startswith("Forget")
        assert "fact" in text
        assert "imported memory" not in text
        assert "This cannot be undone." in text


# ── Subcommand parsing dispatch ────────────────────────────────────


@pytest.fixture
def auth_config():
    """A Config-shaped object that authorizes user_id 999."""
    cfg = MagicMock()
    cfg.allowed_user_ids = {999}
    cfg.memory_search_floor = 0.3
    return cfg


@pytest.fixture
def update_factory():
    """Build an Update-shaped mock with chat_id and user_id."""

    def make(text: str = "", *, chat_id: int = 100, user_id: int = 999, callback_data: str | None = None):
        upd = MagicMock()
        upd.effective_chat = MagicMock(id=chat_id)
        upd.effective_user = MagicMock(id=user_id)
        if callback_data is None:
            upd.message = MagicMock()
            upd.message.reply_text = AsyncMock()
            upd.message.text = text
            upd.callback_query = None
        else:
            # Callback path - no text message, just a query object.
            upd.message = None
            upd.callback_query = MagicMock()
            upd.callback_query.data = callback_data
            upd.callback_query.answer = AsyncMock()
            upd.callback_query.edit_message_text = AsyncMock()
        # send_message used by _send_or_edit fresh-send branch.
        upd.effective_chat.send_message = AsyncMock(return_value=MagicMock(message_id=42))
        return upd

    return make


@pytest.fixture
def context_factory(auth_config):
    """Build a ContextTypes.DEFAULT_TYPE-shaped mock."""

    def make(args: list[str] | None = None):
        ctx = MagicMock()
        ctx.bot_data = {"config": auth_config}
        ctx.args = args or []
        return ctx

    return make


class TestCommandDispatch:
    """Verify each subcommand routes to the correct send-helper."""

    @pytest.mark.asyncio
    async def test_no_args_renders_dashboard(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        monkeypatch.setattr(
            memory_command.memory,
            "get_stats",
            lambda *, user_id: _stats(extracted_count=0),
        )
        upd = update_factory()
        ctx = context_factory(args=[])
        await memory_command.handle_memory_command(upd, ctx)
        upd.effective_chat.send_message.assert_awaited_once()
        sent_text = upd.effective_chat.send_message.call_args.kwargs["text"]
        assert "No memories yet" in sent_text

    @pytest.mark.asyncio
    async def test_memory_disabled_short_circuits(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: False)
        upd = update_factory()
        ctx = context_factory(args=[])
        await memory_command.handle_memory_command(upd, ctx)
        upd.message.reply_text.assert_awaited_once_with(memory_command._MSG_DISABLED)

    @pytest.mark.asyncio
    async def test_help_subcommand(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        upd = update_factory()
        ctx = context_factory(args=["help"])
        await memory_command.handle_memory_command(upd, ctx)
        upd.message.reply_text.assert_awaited_once_with(memory_command._HELP_TEXT)

    @pytest.mark.asyncio
    async def test_unknown_subcommand_falls_through_to_help(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        upd = update_factory()
        ctx = context_factory(args=["zzznotacommand"])
        await memory_command.handle_memory_command(upd, ctx)
        upd.message.reply_text.assert_awaited_once_with(memory_command._HELP_TEXT)

    @pytest.mark.asyncio
    async def test_search_with_empty_query_falls_through_to_help(self, monkeypatch, update_factory, context_factory):
        # Spec §5: "An empty query (`/memory search` with no text)
        # falls through to `/memory help`."
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        upd = update_factory()
        ctx = context_factory(args=["search"])
        await memory_command.handle_memory_command(upd, ctx)
        upd.message.reply_text.assert_awaited_once_with(memory_command._HELP_TEXT)

    @pytest.mark.asyncio
    async def test_search_with_query_calls_search(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        called: dict[str, Any] = {}

        def fake_search(query, *, user_id, limit):
            called["query"] = query
            called["user_id"] = user_id
            return []

        monkeypatch.setattr(memory_command.memory, "search", fake_search)
        upd = update_factory()
        ctx = context_factory(args=["search", "what", "did", "I", "say"])
        await memory_command.handle_memory_command(upd, ctx)
        assert called["query"] == "what did I say"
        assert called["user_id"] == "100"  # chat_id stringified

    @pytest.mark.asyncio
    async def test_search_filters_out_non_user_visible_rows(self, monkeypatch, update_factory, context_factory):
        # Search results are post-filtered to `USER_VISIBLE_SOURCES`
        # (`extracted` + `episode` + `migration`); rows from any
        # other source - most importantly legacy ""-source rows -
        # are dropped before reaching the operator. This site exists
        # because `memory.search` is a Mem0 vector lookup spanning
        # every source, while the data-layer reads (`get_by_id`,
        # `get_by_tag`) already gate at fetch time. Issue #410
        # widened this from extracted-only.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)

        # Build one user-visible hit (kept) and one non-user-visible
        # hit (dropped). Both clear the score floor so only source
        # decides. The non-user-visible row uses `source="track1"`,
        # exercising the negative-space contract for any value
        # outside the admit list.
        extracted = _fact("e1", "kept", ["preference"], score=0.9)
        non_user_visible = MemoryResult(
            id="t1",
            text="dropped",
            score=0.95,
            memory_type="fact",
            metadata={"source": "track1"},
            created_at="2026-04-17T10:00:00",
            updated_at="2026-04-17T10:00:00",
        )

        def fake_search(query, *, user_id, limit):
            return [extracted, non_user_visible]

        monkeypatch.setattr(memory_command.memory, "search", fake_search)
        upd = update_factory()
        ctx = context_factory(args=["search", "anything"])
        await memory_command.handle_memory_command(upd, ctx)

        # The cache memory_ids list is populated from the filtered
        # results - if the non-user-visible row leaked through, "t1"
        # would appear here. It must not.
        cache = memory_command._get_cache(100)
        assert cache is not None
        assert cache.memory_ids == ["e1"]

    @pytest.mark.asyncio
    async def test_search_post_filter_admits_user_visible_sources_drops_legacy(
        self, monkeypatch, update_factory, context_factory
    ):
        # Issue #410 D4: `_send_search` post-filter widens from
        # extracted-only to `USER_VISIBLE_SOURCES`. This test drives
        # the comprehensive contract: extracted + episode + migration
        # all surface; legacy ""-source rows are dropped. Replaces
        # v1's redundant 14/15/16 with a single multi-source fixture.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)

        extracted = MemoryResult(
            id="ext-1",
            text="extracted fact",
            score=0.9,
            memory_type="fact",
            metadata={"source": "extracted"},
            created_at="2026-04-29T10:00:00",
            updated_at="2026-04-29T10:00:00",
        )
        episode = MemoryResult(
            id="ep-1",
            text="episode summary",
            score=0.85,
            memory_type="fact",
            metadata={"source": "episode", "outcome_quality": "success"},
            created_at="2026-04-29T10:00:00",
            updated_at="2026-04-29T10:00:00",
        )
        migration = MemoryResult(
            id="mig-1",
            text="imported memory",
            score=0.8,
            memory_type="fact",
            metadata={"source": "migration", "tags": ["migration"]},
            created_at="2026-04-29T10:00:00",
            updated_at="2026-04-29T10:00:00",
        )
        legacy = MemoryResult(
            id="legacy-1",
            text="legacy row",
            score=0.95,  # high score still gets dropped on source filter
            memory_type="fact",
            metadata={"source": ""},
            created_at="2026-04-29T10:00:00",
            updated_at="2026-04-29T10:00:00",
        )

        def fake_search(query, *, user_id, limit):
            return [extracted, episode, migration, legacy]

        monkeypatch.setattr(memory_command.memory, "search", fake_search)
        upd = update_factory()
        ctx = context_factory(args=["search", "anything"])
        await memory_command.handle_memory_command(upd, ctx)

        # All three user-visible sources surfaced; legacy was dropped.
        cache = memory_command._get_cache(100)
        assert cache is not None
        assert set(cache.memory_ids) == {"ext-1", "ep-1", "mig-1"}
        assert "legacy-1" not in cache.memory_ids

    @pytest.mark.asyncio
    async def test_unauthorized_silently_dropped(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        upd = update_factory(user_id=12345)  # not in allowed_user_ids
        ctx = context_factory(args=[])
        await memory_command.handle_memory_command(upd, ctx)
        # Neither send_message nor reply_text should be called -
        # unauthorized users get silence, matching the rest of the bot.
        upd.message.reply_text.assert_not_called()
        upd.effective_chat.send_message.assert_not_called()


# ── Cache TTL ──────────────────────────────────────────────────────


class TestScreenCache:
    def test_set_and_get_round_trip(self):
        memory_command._set_cache(100, memory_command._ScreenCache(screen="dashboard"))
        entry = memory_command._get_cache(100)
        assert entry is not None
        assert entry.screen == "dashboard"

    def test_missing_chat_returns_none(self):
        assert memory_command._get_cache(999) is None

    def test_overwrite_replaces_prior(self):
        # Spec §7.4: "single-entry-per-chat behavior". A second
        # /memory invocation in the same chat overwrites the first.
        memory_command._set_cache(100, memory_command._ScreenCache(screen="dashboard"))
        memory_command._set_cache(100, memory_command._ScreenCache(screen="stats"))
        entry = memory_command._get_cache(100)
        assert entry is not None
        assert entry.screen == "stats"

    def test_expired_entry_dropped_on_access(self, monkeypatch):
        # Install a cache entry, then advance the monotonic clock past
        # the TTL. _get_cache must return None and remove the entry.
        memory_command._set_cache(100, memory_command._ScreenCache(screen="dashboard"))
        # Backdate created_at by more than the TTL.
        memory_command._screen_cache[100].created_at = time.monotonic() - memory_command._CACHE_TTL_S - 60
        assert memory_command._get_cache(100) is None
        assert 100 not in memory_command._screen_cache


# ── _send_or_edit contract ─────────────────────────────────────────


class TestSendOrEditContract:
    """Spec 310 §7.3 dispatcher contract: edit=True requires a callback.

    All `edit=True` callers today live in `handle_memory_callback`
    where `update.callback_query` is guaranteed. The check exists to
    surface contract violations from any future caller wiring up the
    `_send_*` helpers from outside the callback path - silent no-op
    is the worst failure mode here. `raise` rather than `assert` so
    the gate survives `python -O`."""

    @pytest.mark.asyncio
    async def test_edit_without_callback_query_raises(self, update_factory):
        # Build an update with no callback_query (text-message path).
        upd = update_factory("hi")
        with pytest.raises(ValueError, match="edit=True requires"):
            await memory_command._send_or_edit(upd, "hello", None, edit=True)

    @pytest.mark.asyncio
    async def test_edit_with_callback_query_dispatches(self, update_factory):
        # Sanity check that the contract gate is not too aggressive:
        # the legitimate edit path still works when callback_query
        # is set.
        upd = update_factory(callback_data="mem:dash")
        await memory_command._send_or_edit(upd, "hello", None, edit=True)
        upd.callback_query.edit_message_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_path_does_not_require_callback_query(self, update_factory):
        # edit=False is the fresh-send branch; callback_query may be
        # absent. Must not trip the contract check.
        upd = update_factory("hi")
        await memory_command._send_or_edit(upd, "hello", None, edit=False)
        upd.effective_chat.send_message.assert_awaited_once()

    def test_chat_id_helper_raises_when_effective_chat_is_none(self):
        # Round-6 #1: `_chat_id` was an `assert`; converted to
        # `if/raise` for `-O` safety. PTB routing guarantees a chat
        # in normal handlers, but a future caller wiring this from a
        # non-routed path would silently AttributeError under -O.
        upd = MagicMock()
        upd.effective_chat = None
        with pytest.raises(ValueError, match="effective_chat is None"):
            memory_command._chat_id(upd)

    def test_user_id_helper_raises_when_effective_user_is_none(self):
        # Round-6 #1: same conversion as `_chat_id`.
        upd = MagicMock()
        upd.effective_user = None
        with pytest.raises(ValueError, match="effective_user is None"):
            memory_command._user_id(upd)

    @pytest.mark.asyncio
    async def test_handle_memory_command_raises_without_message(self, monkeypatch):
        # Round-6 #1: top-of-handler `assert update.message is not None`
        # converted to `if/raise`. PTB CommandHandler guarantees a
        # message in normal use, but the contract should hold under
        # `python -O` too.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        upd = MagicMock()
        upd.message = None
        upd.effective_chat = MagicMock(id=100)
        upd.effective_user = MagicMock(id=999)
        ctx = MagicMock()
        with pytest.raises(ValueError, match=r"update\.message is None"):
            await memory_command.handle_memory_command(upd, ctx)

    @pytest.mark.asyncio
    async def test_handle_memory_callback_raises_without_callback_query(self):
        # Audit-all-sites: while addressing the three sites the round-6
        # review named, the same `assert` pattern was found at the top
        # of `handle_memory_callback` (callback_query and query.data).
        # Converted in the same sweep so a future round doesn't have
        # to re-flag the same anti-pattern.
        upd = MagicMock()
        upd.callback_query = None
        ctx = MagicMock()
        with pytest.raises(ValueError, match="callback_query is None"):
            await memory_command.handle_memory_callback(upd, ctx)

    @pytest.mark.asyncio
    async def test_send_path_raises_when_effective_chat_is_none(self):
        # The fresh-send branch's None-guard on `effective_chat` is
        # `if/raise` rather than `assert` so the contract holds under
        # `python -O` (which strips assertions). Mirrors the same
        # hardening used by `_encode_callback` and the `edit=True`
        # guard above.
        upd = MagicMock()
        upd.effective_chat = None
        upd.callback_query = None
        upd.message = None
        with pytest.raises(ValueError, match="effective_chat is None"):
            await memory_command._send_or_edit(upd, "hello", None, edit=False)

    @pytest.mark.asyncio
    async def test_fallback_send_clears_stale_keyboard(self, update_factory):
        # When edit_message_text raises a non-"not modified"
        # BadRequest (e.g., the original message was deleted), the
        # function falls back to a fresh send. The stale message in
        # chat would otherwise keep its inline keyboard, and a tap
        # on those buttons would trigger ghost callbacks against
        # state we have moved past. Strip the keyboard first.
        from telegram.error import BadRequest

        upd = update_factory(callback_data="mem:dash")
        upd.callback_query.edit_message_text = AsyncMock(side_effect=BadRequest("Message deleted"))
        upd.callback_query.edit_message_reply_markup = AsyncMock()
        await memory_command._send_or_edit(upd, "hello", None, edit=True)
        # Keyboard cleared on the original message before the
        # replacement send, and a fresh message went out.
        upd.callback_query.edit_message_reply_markup.assert_awaited_once()
        upd.effective_chat.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fallback_keyboard_clear_failure_is_swallowed(self, update_factory):
        # The keyboard-clear is best-effort: the original message may
        # be gone or immutable. If it raises BadRequest, swallow it
        # and continue with the replacement send. Already on an
        # error path; surfacing a second exception would mask the
        # original.
        from telegram.error import BadRequest

        upd = update_factory(callback_data="mem:dash")
        upd.callback_query.edit_message_text = AsyncMock(side_effect=BadRequest("Message deleted"))
        upd.callback_query.edit_message_reply_markup = AsyncMock(side_effect=BadRequest("Message to edit not found"))
        await memory_command._send_or_edit(upd, "hello", None, edit=True)
        # Send still happened despite the keyboard-clear failure.
        upd.effective_chat.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fallback_keyboard_clear_swallows_network_error(self, update_factory):
        # Round-7 #2: the keyboard-clear catch was BadRequest-only.
        # A transient NetworkError / TimedOut from PTB would escape
        # and short-circuit the fresh send below, surfacing as
        # "Memory query failed." for what should have been a successful
        # re-render. Broadened to `except Exception` so the best-effort
        # contract holds: cleanup failure does not abort the fallback
        # send under any error class.
        from telegram.error import BadRequest, NetworkError

        upd = update_factory(callback_data="mem:dash")
        upd.callback_query.edit_message_text = AsyncMock(side_effect=BadRequest("Message deleted"))
        upd.callback_query.edit_message_reply_markup = AsyncMock(side_effect=NetworkError("connection reset"))
        await memory_command._send_or_edit(upd, "hello", None, edit=True)
        # Fresh send happened despite the NetworkError on cleanup.
        upd.effective_chat.send_message.assert_awaited_once()


# ── Callback dispatch (smoke tests for the verb table) ─────────────


class TestCallbackDispatch:
    """Drive the callback handler with each verb to verify routing.

    These tests check that the right send-helper runs and the right
    edit/answer calls occur. The pure builders are exercised more
    thoroughly above; here we focus on dispatch wiring.
    """

    @pytest.mark.asyncio
    async def test_dash_verb_re_renders_dashboard(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        monkeypatch.setattr(
            memory_command.memory,
            "get_stats",
            lambda *, user_id: _stats(extracted_count=0),
        )
        upd = update_factory(callback_data="mem:dash")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        upd.callback_query.answer.assert_awaited()
        upd.callback_query.edit_message_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_callback_prefix_is_dismissed(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        upd = update_factory(callback_data="ws:home")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        # _decode_callback returned None, so we should silently ack.
        upd.callback_query.answer.assert_awaited()
        upd.callback_query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_fact_verb_with_no_cache_routes_to_dashboard(self, monkeypatch, update_factory, context_factory):
        # Cache miss (e.g. expired or restart) on a fact tap must
        # route gracefully back to the dashboard with the
        # "session expired" toast.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        monkeypatch.setattr(
            memory_command.memory,
            "get_stats",
            lambda *, user_id: _stats(extracted_count=0),
        )
        upd = update_factory(callback_data="mem:fact:0")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        toast = upd.callback_query.answer.call_args.args[0]
        assert toast == memory_command._MSG_SESSION_EXPIRED

    @pytest.mark.asyncio
    async def test_ffd_deletes_and_returns_to_episode_list(self, monkeypatch, update_factory, context_factory):
        # Forget single fact: confirm flow. Cache holds the fact id
        # and a return_to pointing at the episode list. After delete,
        # the handler should re-render the episode list at the same
        # page. Pins the eps return_to wiring through the ffd path.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        monkeypatch.setattr(
            memory_command.memory,
            "get_all_episodes",
            lambda *, user_id: [],  # episode list is now empty
        )
        deleted: list[str] = []

        def fake_delete(*, user_id, memory_id):
            deleted.append(memory_id)
            return True

        monkeypatch.setattr(memory_command.memory, "delete_by_id", fake_delete)
        memory_command._set_cache(
            100,
            memory_command._ScreenCache(
                screen="forget_fact_confirm",
                memory_ids=["mem-id-1"],
                return_to=("eps", ["0"]),
            ),
        )
        upd = update_factory(callback_data="mem:ffd")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        assert deleted == ["mem-id-1"]
        toast = upd.callback_query.answer.call_args.args[0]
        assert toast == "Forgotten."

    @pytest.mark.asyncio
    async def test_send_failure_yields_single_query_failed_answer(self, monkeypatch, update_factory, context_factory):
        # Regression for the double-answer bug. Round-3 code answered
        # the query BEFORE _send_*; if the send raised, the outer
        # except answered again with _MSG_QUERY_FAILED, which Telegram
        # rejected as BadRequest. The fix is to defer query.answer()
        # to AFTER the send, so the except path answers an
        # unanswered query exactly once.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)

        # Pick a verb whose Mem0 call is NOT wrapped in a per-helper
        # try/except, so the exception escapes to the outer dispatch
        # except. The `ffc` verb dispatches `_send_forget_fact_confirm`,
        # which calls `memory.get_by_id` directly without an inner
        # guard - if Mem0 is down, RuntimeError bubbles up to
        # handle_memory_callback's except. Verbs like `dash` won't
        # work here because `_send_dashboard` catches its own
        # `get_stats` failure and renders an in-screen error instead
        # of propagating.
        def boom(*, user_id, memory_id):
            raise RuntimeError("mem0 down")

        monkeypatch.setattr(memory_command.memory, "get_by_id", boom)
        # Prime the cache so the `ffc` handler reads a memory id
        # from it before invoking the now-blown-up `get_by_id`.
        memory_command._set_cache(
            100,
            memory_command._ScreenCache(
                screen="fact",
                memory_ids=["any-id"],
            ),
        )
        upd = update_factory(callback_data="mem:ffc")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        # query.answer must have been called exactly once, with the
        # _MSG_QUERY_FAILED toast - never a redundant pre-send answer.
        assert upd.callback_query.answer.await_count == 1
        toast = upd.callback_query.answer.call_args.args[0]
        assert toast == memory_command._MSG_QUERY_FAILED

    @pytest.mark.asyncio
    async def test_callback_disabled_short_circuits(self, monkeypatch, update_factory, context_factory):
        """Switch point 4 (#434): when memory is disabled, the
        callback handler's `if not memory.is_enabled():` early return
        fires and the user gets the _MSG_DISABLED toast WITHOUT any
        edit_message_text dispatch. Mirrors
        `TestCommandDispatch::test_memory_disabled_short_circuits`
        which covers the same contract on the slash-command side;
        without this test, a regression that removed the callback
        handler's early return would let the dashboard re-render fire
        under disabled mode and confuse a user who just ran a
        forget-flow on a now-disabled install.
        """
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: False)
        upd = update_factory(callback_data="mem:dash")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        # The disabled-mode toast is the contract: the operator
        # tapped a button on a /memory dashboard that is no longer
        # active, and the only acceptable response is the
        # disabled-mode reply via query.answer().
        upd.callback_query.answer.assert_awaited_once_with(memory_command._MSG_DISABLED)
        # No dashboard re-render: the early return must fire BEFORE
        # any verb dispatch reaches edit_message_text. A regression
        # that left the answer call in place but moved it AFTER
        # dispatch would slip past the answer assertion alone, so
        # the negative assertion here pins the early-return shape.
        upd.callback_query.edit_message_text.assert_not_called()


# ── Issue #416: tag browse-axis dismantle ──────────────────────────


class TestTagBrowseAxisRetired:
    """Regression pins for the issue #416 dismantle.

    Each test pins a specific dismantle assertion so a future edit
    that revives any of the deleted surfaces (per-tag dashboard
    buttons, the `across N tags` headline, the per-tag stats table,
    `/memory forget <tag>`, the `tag` callback verb) trips a clear
    test failure rather than silently re-introducing the surface.
    """

    def test_retired_tag_enum_constant_is_absent(self):
        # The closed-vocabulary tag-enum constant from the pre-
        # dismantle surface is fully retired. A future edit that
        # imports it from memory_extraction or re-declares it
        # locally would defeat the dismantle. The constant name is
        # constructed via string concatenation so this test does
        # not itself register as a stale reference under the dual
        # grep gate that checks for revival of the deleted symbols.
        constant_name = "_TAG" + "_ENUM"
        assert constant_name not in vars(memory_command), (
            f"constant {constant_name} was resurrected; the tag-dismantle defense is broken"
        )

    def test_dashboard_renders_no_per_tag_button_rows(self):
        # The dashboard keyboard is the utility row only post-dismantle;
        # any inline-button row containing a single button labelled
        # `<word> (<int>)` (the prior per-tag-row shape) is a regression.
        stats = _stats(
            extracted_count=10,
            by_tag={"preference": 5, "fact": 3},
            confidence_median=0.85,
            confidence_min=0.6,
        )
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        for row in kb.inline_keyboard:
            if len(row) == 1:
                label = row[0].text
                # The prior per-tag shape was `<tag> (<count>)`. The
                # surviving Episodes button (`Episodes (N)`) has the
                # same `<word> (<int>)` shape and lands in a single-
                # button row when it's the only utility-row item, so
                # exempt it by prefix; any other single-button row
                # ending in `)` is a regression.
                assert not label.endswith(")") or label.startswith("Episodes ("), (
                    f"unexpected single-button row: {label!r}"
                )

    def test_dashboard_headline_omits_tag_count(self):
        # The "across N tags" headline suffix is gone. Permissive
        # substring check on the leading word avoids over-pinning a
        # specific wording variant.
        stats = _stats(
            extracted_count=10,
            by_tag={"preference": 5, "fact": 3},
            confidence_median=0.85,
            confidence_min=0.6,
        )
        text, _ = memory_command._build_dashboard(stats)
        assert "across " not in text

    def test_stats_renders_no_by_tag_table_for_extracted_only(self):
        # Complementary to the existing episode-only assertion in
        # TestStatsMultiSource: an extracted-only operator's stats
        # output has no `By tag:` header. The pre-Sub-B dashboard
        # rendered it under the `extracted_count > 0` gate; the
        # post-dismantle dashboard does not.
        stats = _stats(extracted_count=5, by_tag={"preference": 3})
        text, _ = memory_command._build_stats(stats)
        assert "By tag:" not in text

    def test_help_text_omits_forget_tag_line(self):
        # The `/memory forget <tag>` line is dropped from the help
        # text. A future edit that revives the bulk-forget command
        # must also re-add the help line, so this pin guards against
        # half-revivals where the command works but help lies about
        # its absence.
        assert "/memory forget" not in memory_command._HELP_TEXT

    @pytest.mark.asyncio
    async def test_forget_subcommand_returns_help_text(self, monkeypatch, update_factory, context_factory):
        # The /memory forget text path is retired; the unknown-
        # subcommand fallthrough sends the help text. Both arg-less
        # and arg-bearing variants must land on help.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        for args in [["forget"], ["forget", "preference"]]:
            upd = update_factory()
            ctx = context_factory(args=args)
            await memory_command.handle_memory_command(upd, ctx)
            msg = upd.message.reply_text.call_args.args[0]
            assert msg == memory_command._HELP_TEXT, f"expected help text for /memory {' '.join(args)}, got {msg!r}"

    @pytest.mark.asyncio
    async def test_unknown_callback_verb_silently_dismisses(self, monkeypatch, update_factory, context_factory):
        # A stale `mem:tag:<name>:0` callback fired from chat history
        # after deploy lands on the unknown-verb branch. The handler
        # must dismiss with `query.answer()` (no arguments) and not
        # invoke any send-helper. Pins the degraded-but-graceful
        # behavior that makes the deletion safe for users with stale
        # dashboards.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        # If a send-helper fired, its data-layer call would surface
        # as a missing monkeypatch (KeyError or unexpected call).
        # Mock get_stats defensively in case the dispatcher routed
        # through dashboard for any reason; the assertion that
        # edit_message_text was NOT called is the load-bearing check.
        monkeypatch.setattr(
            memory_command.memory,
            "get_stats",
            lambda *, user_id: _stats(extracted_count=0),
        )
        upd = update_factory(callback_data="mem:tag:preference:0")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        # query.answer was called exactly once with no arguments.
        # Both positional args and kwargs must be empty: a stray
        # positional toast text or a `show_alert=True` kwarg would
        # both cross the "silent dismiss" line.
        assert upd.callback_query.answer.await_count == 1
        call = upd.callback_query.answer.call_args
        assert call.args == () and call.kwargs == {}
        # No edit_message_text invocation - the dashboard did not
        # re-render for this stale verb.
        upd.callback_query.edit_message_text.assert_not_called()

    def test_fact_view_renders_tags_row_for_extracted(self):
        # The fact-view detail surface still shows a Tags: row on
        # extracted rows post-Sub-B. The padding is wider than on
        # episode/migration rows because extracted rows align with
        # Confidence/Date/Session/Prompt-version/Confirmation, but
        # the row presence is the load-bearing pin.
        fact = _fact("a", "Sample preference", ["preference", "constraint"], confidence=0.9)
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        assert "Tags:" in text
        # Both tags appear in the rendered row.
        assert "preference" in text
        assert "constraint" in text

    def test_fact_view_renders_tags_row_for_episode(self):
        # Episode rows render Tags: with the tighter padding shape
        # (only Tags + Date in that block). The row is required;
        # the padding asymmetry vs extracted is intentional.
        fact = _episode_fact(tags=["sophia/topic", "browser"])
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        assert "Tags:" in text
        assert "sophia/topic" in text

    def test_fact_view_renders_tags_row_for_migration(self):
        # Migration rows render Tags: with the same tighter padding
        # as episode rows. Migration metadata defaults to the
        # ["migration"] tag in the fixture; an explicit tag list
        # exercises the multi-tag rendering.
        fact = _migration_fact(tags=["migration", "/backend"])
        text, _ = memory_command._build_fact_view(fact, return_to=None)
        assert "Tags:" in text
        assert "migration" in text
        assert "/backend" in text
