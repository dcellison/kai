"""
Bot-level TOTP gate tests.

These tests call handle_message() directly with mock Update/Context objects
and patch the kai.totp functions at the bot module level (kai.bot.*) to
control gate behavior without touching real files or subprocess calls.

All downstream machinery (Claude, locks, session logging) is also mocked
so tests that reach past the gate complete cleanly without starting processes.

Every test also patches _is_authorized to return True. handle_message is
wrapped by @_require_auth which silently drops updates from unauthorized users
(returning early with no action). Since TOTP tests are about the gate, not
access control, we bypass the auth check in all cases.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from kai.bot import handle_document, handle_message, handle_photo, handle_voice

# ── Test helpers ──────────────────────────────────────────────────────────


def _make_update(text: str = "hello") -> MagicMock:
    """
    Create a minimal mock Update sufficient for handle_message().

    Sets up update.message, update.message.text, and the async methods
    the TOTP gate calls (reply_text, delete, effective_chat.send_message).
    """
    update = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.delete = AsyncMock()
    update.effective_chat.id = 12345
    update.effective_chat.send_message = AsyncMock()
    return update


def _make_context(user_data: dict | None = None) -> MagicMock:
    """Create a mock PTB context with controllable user_data."""
    ctx = MagicMock()
    ctx.user_data = user_data if user_data is not None else {}
    # Set TOTP config attributes on the mock so the gate can read them
    # directly (matching how real Config provides typed defaults).
    cfg = ctx.bot_data["config"]
    cfg.totp_session_minutes = 30
    cfg.totp_challenge_seconds = 120
    cfg.totp_lockout_attempts = 3
    cfg.totp_lockout_minutes = 15
    return ctx


def _fake_lock(*_args, **_kwargs):
    """Return a real asyncio.Lock to stand in for the per-chat lock.

    Uses a real Lock instead of a bare async context manager so that both
    async-with and .locked() work (the latter is needed by _notify_if_queued).
    """
    return asyncio.Lock()


def _downstream_patches() -> dict:
    """
    Return a dict of bare attribute names -> mocks for bot machinery that runs
    after the gate. Keys are attribute names within kai.bot (no module prefix),
    suitable for use with patch.multiple("kai.bot", **_downstream_patches()).

    Applied when a test expects the gate to pass and execution to continue
    into normal Claude handling. Prevents actual subprocess spawning.
    """
    return {
        "_is_authorized": MagicMock(return_value=True),
        "_handle_response": AsyncMock(),
        "_get_pool": MagicMock(return_value=MagicMock(get_model=MagicMock(return_value="opus"))),
        "log_message": MagicMock(),
        "_set_responding": MagicMock(),
        "_clear_responding": MagicMock(),
        "get_lock": MagicMock(return_value=_fake_lock()),
    }


# ── Gate-blocking tests ───────────────────────────────────────────────────


async def test_gate_prompts_when_auth_expired():
    """Gate sends the 'Session expired' prompt when TOTP is configured and auth window has lapsed."""
    update = _make_update("hello")
    ctx = _make_context()  # no totp_authenticated_at -> defaults to 0 -> always expired

    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
    ):
        await handle_message(update, ctx)

    update.message.reply_text.assert_called_once_with("Session expired. Enter code from authenticator.")
    # Gate must have set the pending challenge state.
    assert "totp_pending" in ctx.user_data


async def test_gate_skipped_when_auth_active():
    """Gate is transparent when totp_authenticated_at is within the session window."""
    update = _make_update("hello")
    # Set auth time to 1 second ago - well within the 30-minute default.
    ctx = _make_context({"totp_authenticated_at": time.time() - 1})

    with patch.multiple("kai.bot", is_totp_configured=MagicMock(return_value=True), **_downstream_patches()):
        await handle_message(update, ctx)

    # The challenge prompt must NOT have been sent.
    update.message.reply_text.assert_not_called()


async def test_gate_skipped_when_totp_not_configured():
    """Gate is fully transparent when TOTP is not configured (secret file absent)."""
    update = _make_update("hello")
    ctx = _make_context()  # no auth state at all

    with patch.multiple("kai.bot", is_totp_configured=MagicMock(return_value=False), **_downstream_patches()):
        await handle_message(update, ctx)

    update.message.reply_text.assert_not_called()


async def test_code_message_deleted_after_verification():
    """The code message is deleted from chat regardless of whether verification succeeds or fails."""
    pending = {"expires_at": time.time() + 120}
    update = _make_update("123456")
    ctx = _make_context({"totp_pending": pending})

    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
        patch("kai.bot.get_lockout_remaining", return_value=0),
        patch("kai.bot.verify_code", return_value=False),
        patch("kai.bot.get_failure_count", return_value=1),
    ):
        await handle_message(update, ctx)

    update.message.delete.assert_called_once()


async def test_challenge_expires_after_two_minutes():
    """A pending challenge that is past its expires_at is rejected with an expiry message."""
    # expires_at in the past
    pending = {"expires_at": time.time() - 1}
    update = _make_update("123456")
    ctx = _make_context({"totp_pending": pending})

    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
    ):
        await handle_message(update, ctx)

    update.message.reply_text.assert_called_once_with("TOTP challenge expired. Send another message to try again.")
    # Pending state must be cleared so the next message re-issues the challenge.
    assert "totp_pending" not in ctx.user_data


async def test_successful_auth_sets_timestamp():
    """Successful code verification records totp_authenticated_at in context.user_data."""
    pending = {"expires_at": time.time() + 120}
    update = _make_update("123456")
    ctx = _make_context({"totp_pending": pending})

    before = time.time()
    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
        patch("kai.bot.get_lockout_remaining", return_value=0),
        patch("kai.bot.verify_code", return_value=True),
    ):
        await handle_message(update, ctx)
    after = time.time()

    assert "totp_authenticated_at" in ctx.user_data
    assert before <= ctx.user_data["totp_authenticated_at"] <= after
    # Pending state must be cleared after successful auth.
    assert "totp_pending" not in ctx.user_data


async def test_lockout_message_shown_when_rate_limited():
    """When the global lockout is active, the user sees a lockout message (not a 'remaining' message)."""
    pending = {"expires_at": time.time() + 120}
    update = _make_update("123456")
    ctx = _make_context({"totp_pending": pending})

    # 600 seconds = 10 minutes remaining in lockout.
    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
        patch("kai.bot.get_lockout_remaining", return_value=600),
    ):
        await handle_message(update, ctx)

    sent = update.effective_chat.send_message.call_args[0][0]
    assert "Locked out" in sent
    assert "10" in sent  # 600 // 60 = 10 minutes


async def test_invalid_code_shows_remaining_attempts():
    """An invalid code that doesn't trigger lockout shows the remaining attempt count."""
    pending = {"expires_at": time.time() + 120}
    update = _make_update("000000")
    ctx = _make_context({"totp_pending": pending})

    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
        # Not locked out before the attempt.
        patch("kai.bot.get_lockout_remaining", return_value=0),
        patch("kai.bot.verify_code", return_value=False),
        # After a failed attempt with default lockout_attempts=3, failures=1 -> 2 remaining.
        patch("kai.bot.get_failure_count", return_value=1),
    ):
        await handle_message(update, ctx)

    sent = update.effective_chat.send_message.call_args[0][0]
    assert "Invalid code" in sent
    assert "2" in sent  # lockout_attempts(3) - failures(1) = 2 remaining


# ── Media handler TOTP tests ────────────────────────────────────────────
# These verify that photo, document, and voice handlers check TOTP
# before reaching Claude - the bypass that this PR fixes.


def _make_photo_update() -> MagicMock:
    """Create a mock Update with a photo attachment."""
    update = MagicMock()
    update.message.photo = [MagicMock()]  # non-empty = has photo
    update.message.caption = "What is this?"
    update.message.reply_text = AsyncMock()
    update.message.delete = AsyncMock()
    update.effective_chat.id = 12345
    update.effective_chat.send_message = AsyncMock()
    return update


def _make_document_update() -> MagicMock:
    """Create a mock Update with a document attachment."""
    update = MagicMock()
    update.message.document = MagicMock()
    update.message.document.file_name = "test.txt"
    update.message.document.file_size = 100
    update.message.caption = None
    update.message.reply_text = AsyncMock()
    update.message.delete = AsyncMock()
    update.effective_chat.id = 12345
    update.effective_chat.send_message = AsyncMock()
    return update


def _make_voice_update() -> MagicMock:
    """Create a mock Update with a voice message."""
    update = MagicMock()
    update.message.voice = MagicMock()
    update.message.reply_text = AsyncMock()
    update.message.delete = AsyncMock()
    update.effective_chat.id = 12345
    update.effective_chat.send_message = AsyncMock()
    return update


async def test_photo_requires_totp():
    """Sending a photo with expired TOTP session triggers challenge, not Claude."""
    update = _make_photo_update()
    ctx = _make_context()  # no totp_authenticated_at -> expired

    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
    ):
        await handle_photo(update, ctx)

    # Challenge sent, Claude NOT invoked
    update.message.reply_text.assert_called_once_with("Session expired. Enter code from authenticator.")


async def test_document_requires_totp():
    """Sending a document with expired TOTP session triggers challenge, not Claude."""
    update = _make_document_update()
    ctx = _make_context()

    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
    ):
        await handle_document(update, ctx)

    update.message.reply_text.assert_called_once_with("Session expired. Enter code from authenticator.")


async def test_voice_requires_totp():
    """Sending a voice message with expired TOTP session triggers challenge, not Claude."""
    update = _make_voice_update()
    ctx = _make_context()

    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
    ):
        await handle_voice(update, ctx)

    update.message.reply_text.assert_called_once_with("Session expired. Enter code from authenticator.")


async def test_photo_passes_with_valid_totp():
    """Photo with valid TOTP session proceeds past the gate (no challenge sent)."""
    update = _make_photo_update()
    ctx = _make_context(user_data={"totp_authenticated_at": time.time()})
    ctx.bot.get_file = AsyncMock(return_value=MagicMock(download_as_bytearray=AsyncMock(return_value=b"img")))

    with (
        patch("kai.bot._is_authorized", return_value=True),
        patch("kai.bot.is_totp_configured", return_value=True),
        # Patch downstream to prevent actual processing past the gate.
        # log_message MUST be patched to avoid writing test data (chat_id
        # 12345, MagicMock paths) to the real production history files.
        patch("kai.bot._get_pool"),
        patch("kai.bot.log_message"),
        patch("kai.bot._notify_if_queued", new_callable=AsyncMock, return_value=False),
        patch("kai.bot._acquire_lock_or_kill", new_callable=AsyncMock, return_value=None),
    ):
        await handle_photo(update, ctx)

    # Challenge NOT sent (gate passed, processing continued)
    for call in update.message.reply_text.call_args_list:
        assert "Session expired" not in call[0][0]
