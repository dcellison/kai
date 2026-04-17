"""
Unit tests for `src/kai/memory_command.py` (spec 310).

Covers:
- Subcommand parsing dispatch (every subcommand routes to the right
  send-helper, including the empty-query and unknown-tag fall-throughs).
- Callback encoding/decoding round-trips, including a 64-byte ceiling
  check on the longest realistic callback.
- Pure builder rendering: dashboard, tag view (with pagination edge
  cases), fact view, forget confirmations, search, stats.
- Per-chat cache: insert, single-entry overwrite, lazy TTL expiry.
- Empty-state branches: zero-fact dashboard, zero-result search,
  off-enum tag rejection.

Per spec §10.3, no real Mem0 instance is involved. The handler tests
that drive `_send_*` swap `kai.memory.get_stats` / `get_by_tag` /
`get_by_id` / `delete_by_id` / `search` for fakes via monkeypatch,
and swap `kai.memory.is_enabled` to return True.
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
    by_tag: dict[str, int] | None = None,
    confidence_min: float | None = None,
    confidence_median: float | None = None,
    confidence_max: float | None = None,
    confidence_below_0_7: int = 0,
    confidence_below_0_6: int = 0,
    confirmation_quote_count: int = 0,
    by_prompt_version: dict[str, int] | None = None,
) -> MemoryStats:
    """Construct a MemoryStats with extracted-only aggregates."""
    return MemoryStats(
        total_count=extracted_count,
        by_type={"fact": extracted_count} if extracted_count else {},
        extracted_count=extracted_count,
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
        # The longest legitimate callback is the forget-by-tag verb
        # with the longest tag name (`confirmed_action`, 16 chars):
        #   mem:ftd:confirmed_action  (24 bytes)
        # Verifying the constructor's assertion does not fire and the
        # encoded length is well under the 64-byte limit.
        encoded = memory_command._encode_callback("ftd", "confirmed_action")
        assert len(encoded.encode("utf-8")) <= 64
        assert encoded == "mem:ftd:confirmed_action"

    def test_overlong_callback_raises(self):
        # The 64-byte ceiling is enforced via `if/raise`, not assert,
        # so the check survives `python -O` (which strips assertions).
        # If it ever fires in real use it indicates a bug in the
        # caller passing an unexpectedly long arg, not a runtime
        # condition; raising loudly beats silent Telegram truncation.
        with pytest.raises(ValueError, match="callback_data too long"):
            memory_command._encode_callback("x", "a" * 100)


# ── Constants ───────────────────────────────────────────────────────


class TestHelpTextLength:
    """Round-7 #1 regression: `_HELP_TEXT` is used both in
    `update.message.reply_text` (no length cap) AND in
    `query.answer(_HELP_TEXT, show_alert=True)` from the dashboard's
    Search button. The latter goes through Telegram's
    `answerCallbackQuery` API, whose `text` field is capped at 200
    chars. Exceeding the cap causes a 400 BadRequest, which the
    outer except handler converts to "Memory query failed." -
    confusing UX for what should be the help screen.
    """

    def test_help_text_under_telegram_callback_alert_limit(self):
        # 200 is the Telegram-documented limit for
        # answerCallbackQuery.text. If someone extends `_HELP_TEXT`
        # and trips this assertion, either trim wording or split
        # the dashboard help flow into its own shorter toast string.
        assert len(memory_command._HELP_TEXT) <= 200


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


class TestBar:
    def test_zero_count_empty_bar(self):
        assert memory_command._bar(0, 10) == ""

    def test_max_count_full_width(self):
        bar = memory_command._bar(10, 10, width=8)
        assert bar == "\u2593" * 8

    def test_nonzero_min_one_block(self):
        # A single-fact tag should not render as an empty bar; users
        # would lose the visual signal that the tag exists.
        assert memory_command._bar(1, 100, width=8) == "\u2593"


# ── Builder: dashboard ─────────────────────────────────────────────


class TestBuildDashboard:
    def test_empty_state(self):
        text, kb = memory_command._build_dashboard(_stats(extracted_count=0))
        assert "No memories yet" in text
        assert kb is None

    def test_renders_summary_and_tags(self):
        stats = _stats(
            extracted_count=10,
            by_tag={"preference": 5, "fact": 3, "decision": 2},
            confidence_median=0.86,
            confidence_min=0.52,
        )
        text, kb = memory_command._build_dashboard(stats)
        assert "10 facts across 3 tags" in text
        # Sort order: descending count. Use the leading-space prefix
        # so the bare tag rows match without colliding with the
        # "X facts across" wording in the summary header.
        assert text.index("  preference") < text.index("  fact") < text.index("  decision")
        assert "median 0.86, min 0.52" in text
        assert kb is not None
        # 3 tag rows + 1 footer row of (Search, Stats).
        assert len(kb.inline_keyboard) == 4
        # Footer row holds the two utility buttons.
        footer = kb.inline_keyboard[-1]
        assert [btn.text for btn in footer] == ["Search", "Stats"]

    def test_zero_count_tags_hidden(self):
        # Spec §6.1: dashboard hides zero-count tags. Stats screen
        # shows them. This test enforces the dashboard half.
        stats = _stats(
            extracted_count=5,
            by_tag={"preference": 5, "location": 0},
            confidence_median=0.9,
            confidence_min=0.9,
        )
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        # 1 nonzero tag row + 1 footer row.
        assert len(kb.inline_keyboard) == 2
        assert "preference" in kb.inline_keyboard[0][0].text
        assert "location" not in kb.inline_keyboard[0][0].text

    def test_callback_data_for_tag_button(self):
        stats = _stats(extracted_count=3, by_tag={"preference": 3})
        _, kb = memory_command._build_dashboard(stats)
        assert kb is not None
        button = kb.inline_keyboard[0][0]
        assert button.callback_data == "mem:tag:preference:0"


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


# ── Builder: tag view ──────────────────────────────────────────────


class TestBuildTagView:
    def test_renders_facts_with_confidence_and_date(self):
        facts = [
            _fact("a", "First fact", ["preference"], confidence=0.92, updated_at="2026-04-17"),
            _fact("b", "Second fact", ["preference"], confidence=0.75, updated_at="2026-02-14"),
        ]
        text, _, ids, page, total = memory_command._build_tag_view("preference", facts, 0)
        assert "preference  (page 1 of 1)" in text
        assert "[0.92]  First fact" in text
        assert "[0.75]  Second fact" in text
        assert "2026-04-17" in text
        assert ids == ["a", "b"]
        assert page == 0
        assert total == 1

    def test_truncates_long_text(self):
        long_text = "x" * 200
        facts = [_fact("a", long_text, ["preference"], confidence=0.9)]
        text, _, _, _, _ = memory_command._build_tag_view("preference", facts, 0)
        # Truncated text appears with ellipsis. The full 200-char
        # string must NOT appear in the rendered output.
        assert long_text not in text
        assert "\u2026" in text

    def test_pagination_buttons(self):
        facts = [_fact(f"id{i}", f"t{i}", ["preference"]) for i in range(12)]
        # Middle page should have prev, back, next.
        _, kb, _, _, _ = memory_command._build_tag_view("preference", facts, 1)
        nav_row = kb.inline_keyboard[-1]
        labels = [btn.text for btn in nav_row]
        assert labels == ["< prev", "back", "next >"]

    def test_first_page_no_prev_button(self):
        facts = [_fact(f"id{i}", f"t{i}", ["preference"]) for i in range(12)]
        _, kb, _, _, _ = memory_command._build_tag_view("preference", facts, 0)
        nav_row = kb.inline_keyboard[-1]
        assert "< prev" not in [btn.text for btn in nav_row]
        assert "next >" in [btn.text for btn in nav_row]

    def test_last_page_no_next_button(self):
        facts = [_fact(f"id{i}", f"t{i}", ["preference"]) for i in range(12)]
        _, kb, _, _, _ = memory_command._build_tag_view("preference", facts, 2)
        nav_row = kb.inline_keyboard[-1]
        assert "next >" not in [btn.text for btn in nav_row]
        assert "< prev" in [btn.text for btn in nav_row]

    def test_empty_tag_renders_back_button(self):
        text, kb, ids, _, _ = memory_command._build_tag_view("preference", [], 0)
        assert "No memories with this tag" in text
        assert ids == []
        assert kb.inline_keyboard[0][0].text == "back"

    def test_missing_confidence_renders_placeholder(self):
        # Legacy or pre-#335 rows may lack confidence; the screen
        # must not KeyError. Verify the placeholder appears.
        bad = MemoryResult(
            id="bad",
            text="legacy fact",
            score=0.0,
            memory_type="fact",
            metadata={"source": "extracted", "tags": ["preference"]},
            created_at="2026-04-17",
            updated_at="2026-04-17",
        )
        text, _, _, _, _ = memory_command._build_tag_view("preference", [bad], 0)
        assert "[----]" in text


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


class TestBuildForgetTagConfirm:
    def test_count_appears_in_button_label(self):
        text, kb = memory_command._build_forget_tag_confirm("preference", 38)
        assert "Forget all 38 facts" in text
        assert "Tags are independent" in text
        confirm_btn = kb.inline_keyboard[0][0]
        # Spec §6.7: "confirm forget 38 facts" - the count is in the
        # button label as a concreteness cue.
        assert confirm_btn.text == "confirm forget 38 facts"
        assert confirm_btn.callback_data == "mem:ftd:preference"

    def test_singular_when_count_is_one(self):
        # Round-5 review #3: count==1 used to render "1 facts" /
        # "1 memories" / "confirm forget 1 facts". A tag with one
        # surviving member is a real case (deletes whittle the
        # count). Verify the singular form across all three sites
        # (header, body, button) so a future edit can't regress
        # any one of them in isolation.
        text, kb = memory_command._build_forget_tag_confirm("preference", 1)
        assert "Forget all 1 fact " in text
        assert "1 memory." in text
        assert "1 facts" not in text
        assert "1 memories" not in text
        confirm_btn = kb.inline_keyboard[0][0]
        assert confirm_btn.text == "confirm forget 1 fact"

    def test_plural_when_count_is_two(self):
        # Boundary on the other side: count==2 must use the plural
        # forms. Otherwise the singular branch would silently catch
        # everything.
        text, kb = memory_command._build_forget_tag_confirm("preference", 2)
        assert "Forget all 2 facts " in text
        assert "2 memories." in text
        confirm_btn = kb.inline_keyboard[0][0]
        assert confirm_btn.text == "confirm forget 2 facts"


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
        assert "No extracted facts yet" in text
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
        # Total
        assert "Total:            142 facts" in text
        # Spec §6.1 asymmetry: zero-count tags ARE shown in stats.
        assert "location" in text
        assert "  0" in text  # location's zero count
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
    async def test_search_filters_out_non_extracted_rows(self, monkeypatch, update_factory, context_factory):
        # Spec §7.5 / round-5 review #1: every read path must scope to
        # source=="extracted". `memory.search()` is a Mem0 vector
        # lookup that spans all sources (Track 1 exchanges, legacy
        # rows). Without a post-filter, a non-extracted row could
        # surface in results, then fail get_by_id's source check on
        # tap and render "no longer exists." for a row the user just
        # saw. The filter lives in `_send_search` alongside the score
        # floor.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)

        # Build one extracted hit (kept) and one non-extracted hit
        # (dropped). Both clear the score floor so only source
        # decides. The legacy/Track-1 row uses the same MemoryResult
        # shape Mem0 returns; the only difference is metadata.source.
        extracted = _fact("e1", "kept", ["preference"], score=0.9)
        non_extracted = MemoryResult(
            id="t1",
            text="dropped",
            score=0.95,
            memory_type="fact",
            metadata={"source": "track1"},
            created_at="2026-04-17T10:00:00",
            updated_at="2026-04-17T10:00:00",
        )

        def fake_search(query, *, user_id, limit):
            return [extracted, non_extracted]

        monkeypatch.setattr(memory_command.memory, "search", fake_search)
        upd = update_factory()
        ctx = context_factory(args=["search", "anything"])
        await memory_command.handle_memory_command(upd, ctx)

        # The cache memory_ids list is populated from the filtered
        # results - if the non-extracted row leaked through, "t1"
        # would appear here. It must not.
        cache = memory_command._get_cache(100)
        assert cache is not None
        assert cache.memory_ids == ["e1"]

    @pytest.mark.asyncio
    async def test_forget_unknown_tag_rejected(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        upd = update_factory()
        ctx = context_factory(args=["forget", "notatag"])
        await memory_command.handle_memory_command(upd, ctx)
        msg = upd.message.reply_text.call_args.args[0]
        assert "Unknown tag" in msg

    @pytest.mark.asyncio
    async def test_forget_no_tag_shows_usage(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        upd = update_factory()
        ctx = context_factory(args=["forget"])
        await memory_command.handle_memory_command(upd, ctx)
        msg = upd.message.reply_text.call_args.args[0]
        assert "Usage:" in msg

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
        # Round-5 review #2: the fresh-send branch's None-guard on
        # `effective_chat` was an `assert`, which `python -O` strips.
        # Replaced with `if/raise` so the contract holds in optimized
        # bytecode. This mirrors the same pattern used by
        # `_encode_callback`, the `edit=True` guard above, and
        # `_send_forget_tag_confirm`.
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

    @pytest.mark.asyncio
    async def test_send_forget_tag_confirm_raises_without_message(self, monkeypatch):
        # `_send_forget_tag_confirm` is only called from the text
        # command path where update.message is guaranteed. The check
        # is `if/raise` (not `assert`) so it survives `python -O`,
        # consistent with the other contract gates in this module.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        monkeypatch.setattr(
            memory_command.memory,
            "get_by_tag",
            lambda *, user_id, tag: [_fact("a", "x", ["preference"])],
        )
        upd = MagicMock()
        upd.message = None
        ctx = MagicMock()
        with pytest.raises(ValueError, match=r"requires update\.message"):
            await memory_command._send_forget_tag_confirm(upd, ctx, 100, "preference")


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
    async def test_ffd_deletes_and_returns_to_tag(self, monkeypatch, update_factory, context_factory):
        # Forget single fact: confirm flow. Cache holds the fact id
        # and a return_to pointing at a tag view. After delete, the
        # handler should re-render the tag view.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        monkeypatch.setattr(
            memory_command.memory,
            "get_by_tag",
            lambda *, user_id, tag: [],  # tag view is now empty
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
                return_to=("tag", ["preference", "0"]),
            ),
        )
        upd = update_factory(callback_data="mem:ffd")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        assert deleted == ["mem-id-1"]
        toast = upd.callback_query.answer.call_args.args[0]
        assert toast == "Forgotten."

    @pytest.mark.asyncio
    async def test_ftd_rejects_unknown_tag(self, monkeypatch, update_factory, context_factory):
        # A stale or crafted callback could carry an arbitrary tag
        # string. The handler must validate against `_TAG_ENUM`
        # before invoking get_by_tag/delete_by_id - mirrors the same
        # gate as the /memory forget text path.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        monkeypatch.setattr(
            memory_command.memory,
            "get_stats",
            lambda *, user_id: _stats(extracted_count=0),
        )

        # If validation fails to fire, get_by_tag would be invoked
        # with the bogus tag - blow up loudly to surface the bug.
        def boom(*, user_id, tag):
            raise AssertionError(f"get_by_tag called with bogus tag {tag!r}")

        monkeypatch.setattr(memory_command.memory, "get_by_tag", boom)
        monkeypatch.setattr(
            memory_command.memory,
            "delete_by_id",
            lambda *, user_id, memory_id: True,
        )
        upd = update_factory(callback_data="mem:ftd:not_a_real_tag")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        toast = upd.callback_query.answer.call_args.args[0]
        assert toast == memory_command._MSG_SESSION_EXPIRED

    @pytest.mark.asyncio
    async def test_tag_verb_rejects_unknown_tag(self, monkeypatch, update_factory, context_factory):
        # Symmetric to test_ftd_rejects_unknown_tag: the read-only
        # `tag` verb must also enforce _TAG_ENUM. A crafted callback
        # `mem:tag:not_a_real_tag:0` would otherwise reach get_by_tag
        # with an off-enum string.
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        monkeypatch.setattr(
            memory_command.memory,
            "get_stats",
            lambda *, user_id: _stats(extracted_count=0),
        )

        def boom(*, user_id, tag):
            raise AssertionError(f"get_by_tag called with bogus tag {tag!r}")

        monkeypatch.setattr(memory_command.memory, "get_by_tag", boom)
        upd = update_factory(callback_data="mem:tag:not_a_real_tag:0")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        toast = upd.callback_query.answer.call_args.args[0]
        assert toast == memory_command._MSG_SESSION_EXPIRED

    @pytest.mark.asyncio
    async def test_ftd_bulk_deletes_by_tag(self, monkeypatch, update_factory, context_factory):
        monkeypatch.setattr(memory_command.memory, "is_enabled", lambda: True)
        # Two facts on the tag, both delete cleanly.
        monkeypatch.setattr(
            memory_command.memory,
            "get_by_tag",
            lambda *, user_id, tag: [
                _fact("a", "x", ["preference"]),
                _fact("b", "y", ["preference"]),
            ],
        )
        # After delete, dashboard re-render queries get_stats.
        monkeypatch.setattr(
            memory_command.memory,
            "get_stats",
            lambda *, user_id: _stats(extracted_count=0),
        )
        deleted: list[str] = []

        def fake_delete(*, user_id, memory_id):
            deleted.append(memory_id)
            return True

        monkeypatch.setattr(memory_command.memory, "delete_by_id", fake_delete)
        upd = update_factory(callback_data="mem:ftd:preference")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        assert sorted(deleted) == ["a", "b"]
        toast = upd.callback_query.answer.call_args.args[0]
        assert toast == "Forgot 2 facts."

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
        # except. The `ftd` (forget-by-tag) verb calls get_by_tag
        # directly inside the dispatcher, with no inner guard - if
        # Mem0 is down, RuntimeError bubbles up to handle_memory_callback's
        # except. Verbs like `dash` won't work here because
        # _send_dashboard catches its own get_stats failure and renders
        # an in-screen error instead of propagating.
        def boom(*, user_id, tag):
            raise RuntimeError("mem0 down")

        monkeypatch.setattr(memory_command.memory, "get_by_tag", boom)
        upd = update_factory(callback_data="mem:ftd:preference")
        ctx = context_factory()
        await memory_command.handle_memory_callback(upd, ctx)
        # query.answer must have been called exactly once, with the
        # _MSG_QUERY_FAILED toast - never a redundant pre-send answer.
        assert upd.callback_query.answer.await_count == 1
        toast = upd.callback_query.answer.call_args.args[0]
        assert toast == memory_command._MSG_QUERY_FAILED


# ── Tag enum drift guard ───────────────────────────────────────────


def test_tag_enum_matches_extraction_schema():
    """The `_TAG_ENUM` mirror must stay in sync with the schema source.

    Drift would silently exclude valid tags from `/memory stats`
    (a tag with rows but no enum entry would never appear). Both
    lists are short and unlikely to change often, but a unit test
    catches the next time they do.
    """
    from kai.memory_extraction import _FACT_SCHEMA

    schema_enum = tuple(_FACT_SCHEMA["properties"]["facts"]["items"]["properties"]["tags"]["items"]["enum"])
    assert schema_enum == memory_command._TAG_ENUM
