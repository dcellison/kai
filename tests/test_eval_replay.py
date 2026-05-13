"""Tests for `kai.eval.replay`.

The replay module re-extracts an older history window into a sandbox
`user_id` so a spec-implementation cycle can compare v8-prompt and
v9-prompt sandboxes without disturbing production. The unit tests pin
the safety guards (sandbox-prefix enforcement), the file-selection
logic (date-range filter), the pairing semantics (which records become
which pairs), the rolling prior buffer (PRIOR CONTEXT window of N
pairs), tolerance for partial-corruption (malformed JSONL lines), the
two side-effect-free dry-run / would-be-modify contracts, and the
reset path's call to `memory.delete_all`. PRIOR CONTEXT truncation is
deferred to `_build_extraction_payload`; the replay only feeds raw
pair strings, so we assert the call shape (replay does not roll its
own truncation).

PII posture: tests use synthetic JSONL fixtures generated inside
`tmp_path`; no real operator history files are touched.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.eval import replay

# A canonical chat_id used across fixtures; aligns with the production
# operator id so tests exercise the same shape the real bot writes.
_CHAT_ID = 12345


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one record per line; matches the bot's `log_message` shape."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _rec(direction: str, text: str, *, chat_id: int = _CHAT_ID, ts: str = "2026-05-12T10:00:00") -> dict:
    """Build a record dict with the fields production writes."""
    return {"ts": ts, "dir": direction, "chat_id": chat_id, "text": text}


# ── Sandbox-prefix guard ─────────────────────────────────────────────


class TestSandboxPrefixGuard:
    """The `--user-id` prefix check is the structural defense against
    accidentally writing replay output to a real user_id. The guard
    raises before any extraction or storage call, which means tests
    can exercise it without standing up a Mem0 store."""

    def test_real_chat_id_raises(self):
        with pytest.raises(ValueError, match="must start with"):
            replay._validate_user_id("2114582497")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must start with"):
            replay._validate_user_id("")

    def test_almost_prefix_raises(self):
        # Casing matters; an unprefixed value with the right characters
        # in a different order still fails. The literal prefix is the
        # check, not a fuzzy match.
        with pytest.raises(ValueError, match="must start with"):
            replay._validate_user_id("Sandbox-464")

    def test_correct_prefix_accepted(self):
        # No exception means the guard passed.
        replay._validate_user_id("sandbox-464")
        replay._validate_user_id("sandbox-")
        replay._validate_user_id("sandbox-anything-else")


# ── Date-range filter ────────────────────────────────────────────────


class TestDateRangeFilter:
    """The replay reads JSONL files named `YYYY-MM-DD.jsonl` from a
    chat-history directory. The date-range filter selects which files
    enter the replay; non-date-named files are skipped silently so an
    operator backup or rotated artifact does not abort the run."""

    def test_three_files_pick_middle(self, tmp_path: Path):
        history = tmp_path / "history"
        history.mkdir()
        for d in ("2026-05-10", "2026-05-11", "2026-05-12"):
            (history / f"{d}.jsonl").write_text("")
        picked = replay._iter_history_files(history, date(2026, 5, 11), date(2026, 5, 11))
        assert [p.name for p in picked] == ["2026-05-11.jsonl"]

    def test_no_filter_returns_all_oldest_first(self, tmp_path: Path):
        history = tmp_path / "history"
        history.mkdir()
        for d in ("2026-05-12", "2026-05-10", "2026-05-11"):
            (history / f"{d}.jsonl").write_text("")
        picked = replay._iter_history_files(history, None, None)
        # Oldest first; alphabetical sort happens to match for ISO dates,
        # but the function sorts by parsed date so the order is correct
        # even for date-strings that would lex-sort differently.
        assert [p.name for p in picked] == [
            "2026-05-10.jsonl",
            "2026-05-11.jsonl",
            "2026-05-12.jsonl",
        ]

    def test_non_date_named_files_skipped(self, tmp_path: Path):
        history = tmp_path / "history"
        history.mkdir()
        (history / "2026-05-12.jsonl").write_text("")
        (history / "backup.jsonl").write_text("")
        (history / "rotated-2026-05-12.jsonl").write_text("")
        picked = replay._iter_history_files(history, None, None)
        assert [p.name for p in picked] == ["2026-05-12.jsonl"]

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        picked = replay._iter_history_files(tmp_path / "does-not-exist", None, None)
        assert picked == []


# ── Pairing semantics ───────────────────────────────────────────────


class TestPairing:
    """The replay reuses `history._pair_records_chronologically` so the
    pair stream fed to `extract_and_store` matches what production
    builds from the same records. These tests pin the integration: the
    replay's filter pass drops the right records and hands the right
    list to the pairing helper."""

    def test_clean_alternating_records(self, tmp_path: Path):
        history = tmp_path / "history"
        history.mkdir()
        _write_jsonl(
            history / "2026-05-12.jsonl",
            [
                _rec("user", "Hi"),
                _rec("assistant", "Hello"),
                _rec("user", "What's up?"),
                _rec("assistant", "Working on the eval."),
            ],
        )
        files = replay._iter_history_files(history, None, None)
        records = replay._load_records(files, _CHAT_ID)
        pairs = replay._build_pairs(records)
        assert pairs == [("Hi", "Hello"), ("What's up?", "Working on the eval.")]

    def test_orphan_assistant_dropped(self, tmp_path: Path):
        # An assistant record with no pending user is the orphan path
        # described in history._pair_records_chronologically: drop it.
        history = tmp_path / "history"
        history.mkdir()
        _write_jsonl(
            history / "2026-05-12.jsonl",
            [
                _rec("assistant", "Mystery reply"),
                _rec("user", "Hi"),
                _rec("assistant", "Hello"),
            ],
        )
        files = replay._iter_history_files(history, None, None)
        records = replay._load_records(files, _CHAT_ID)
        pairs = replay._build_pairs(records)
        assert pairs == [("Hi", "Hello")]

    def test_other_chat_id_filtered_out(self, tmp_path: Path):
        history = tmp_path / "history"
        history.mkdir()
        _write_jsonl(
            history / "2026-05-12.jsonl",
            [
                _rec("user", "From _CHAT_ID"),
                _rec("assistant", "Reply", chat_id=99999),
            ],
        )
        files = replay._iter_history_files(history, None, None)
        records = replay._load_records(files, _CHAT_ID)
        # Assistant record dropped (wrong chat_id), so the user record
        # has no partner and the pair list is empty.
        pairs = replay._build_pairs(records)
        assert pairs == []

    def test_missing_chat_id_field_dropped(self, tmp_path: Path):
        # Legacy pre-Phase-2 records have no chat_id field. The replay
        # is strict (matches history.get_recent_pairs's strictness on
        # provenance) and drops them.
        history = tmp_path / "history"
        history.mkdir()
        _write_jsonl(
            history / "2026-05-12.jsonl",
            [
                {"ts": "2026-05-12T10:00:00", "dir": "user", "text": "Legacy"},
                _rec("user", "Modern"),
                _rec("assistant", "Reply"),
            ],
        )
        records = replay._load_records(replay._iter_history_files(history, None, None), _CHAT_ID)
        pairs = replay._build_pairs(records)
        assert pairs == [("Modern", "Reply")]


# ── Rolling prior buffer ─────────────────────────────────────────────


class TestRollingPriorBuffer:
    """`_run_replay` maintains a rolling buffer of the previous
    `context_turns` pairs and passes it as `prior_pairs` to
    `extract_and_store`. The first pair sees an empty buffer; later
    pairs see at most `context_turns` predecessors."""

    @pytest.mark.asyncio
    async def test_buffer_grows_then_caps(self):
        pairs = [(f"u{i}", f"a{i}") for i in range(5)]
        captured: list[list[tuple[str, str]]] = []

        async def fake_store(user_text, assistant_text, *, user_id, config, prior_pairs):
            captured.append(list(prior_pairs))
            return 0

        with patch("kai.memory_extraction.extract_and_store", side_effect=fake_store):
            await replay._run_replay(
                pairs,
                user_id="sandbox-test",
                context_turns=2,
                config=MagicMock(),
                dry_run=False,
            )

        # First pair: empty prior. Second pair: one prior. Third pair:
        # two priors (the cap). Fourth and fifth pairs: still two
        # priors (the oldest gets dropped each iteration).
        assert captured[0] == []
        assert captured[1] == [("u0", "a0")]
        assert captured[2] == [("u0", "a0"), ("u1", "a1")]
        assert captured[3] == [("u1", "a1"), ("u2", "a2")]
        assert captured[4] == [("u2", "a2"), ("u3", "a3")]

    @pytest.mark.asyncio
    async def test_zero_context_turns_passes_empty_buffer(self):
        # context_turns=0 disables the prior-context window entirely.
        # Used by callers that want the pre-#392 single-turn behavior.
        captured: list[list[tuple[str, str]]] = []

        async def fake_store(user_text, assistant_text, *, user_id, config, prior_pairs):
            captured.append(list(prior_pairs))
            return 0

        with patch("kai.memory_extraction.extract_and_store", side_effect=fake_store):
            await replay._run_replay(
                [("u0", "a0"), ("u1", "a1")],
                user_id="sandbox-test",
                context_turns=0,
                config=MagicMock(),
                dry_run=False,
            )

        # Both calls see empty prior; the buffer collects entries but
        # gets immediately trimmed to zero by the `if len(prior) >
        # context_turns: prior.pop(0)` branch.
        assert captured == [[], []]


# ── Malformed-line tolerance ────────────────────────────────────────


class TestMalformedTolerance:
    """The replay tolerates partial-corruption mirroring
    `history.get_recent_pairs`'s behavior: skip individual bad lines,
    keep the rest. Catches an interrupted-write or operator-edit
    scenario without aborting an entire replay run."""

    def test_mixed_malformed_lines_skipped(self, tmp_path: Path):
        history = tmp_path / "history"
        history.mkdir()
        # Build a file with: a valid record, an empty line, a non-JSON
        # line, a truncated JSON line, and another valid record.
        raw = "\n".join(
            [
                json.dumps(_rec("user", "first")),
                "",
                "garbage not json",
                '{"dir": "assistant", "chat_id": ',  # truncated
                json.dumps(_rec("assistant", "second")),
            ]
        )
        (history / "2026-05-12.jsonl").write_text(raw, encoding="utf-8")

        files = replay._iter_history_files(history, None, None)
        records = replay._load_records(files, _CHAT_ID)
        pairs = replay._build_pairs(records)

        assert pairs == [("first", "second")]


# ── Dry-run flag ────────────────────────────────────────────────────


class TestDryRun:
    """Dry-run reports the pair count without calling
    `extract_and_store` or writing to the store. Operators use it to
    inspect what a real run would do before spending wall-clock on
    actual extraction."""

    @pytest.mark.asyncio
    async def test_dry_run_skips_extract_and_store(self):
        pairs = [("u0", "a0"), ("u1", "a1"), ("u2", "a2")]
        store_mock = AsyncMock(return_value=0)

        with patch("kai.memory_extraction.extract_and_store", store_mock):
            counters, samples = await replay._run_replay(
                pairs,
                user_id="sandbox-test",
                context_turns=3,
                config=MagicMock(),
                dry_run=True,
            )

        # extract_and_store was NEVER called; only the pair walk ran.
        store_mock.assert_not_called()
        # Pair count still updates so the operator sees what the real
        # run would have processed.
        assert counters["pairs_processed"] == 3
        assert counters["facts_stored"] == 0
        # Dry-run captures payload-shape samples for the first
        # `_DRY_RUN_SAMPLE_LIMIT` pairs (3 here, with all 3 pairs
        # sampled). Verbatim text is not captured; only structural
        # shape (prior depth + char counts) lands in the sample.
        assert len(samples) == 3
        assert samples[0] == {
            "index": 1,
            "prior_count": 0,
            "user_chars": 2,
            "assistant_chars": 2,
        }
        assert samples[1]["prior_count"] == 1
        assert samples[2]["prior_count"] == 2


# ── Reset flag ──────────────────────────────────────────────────────


class TestReset:
    """`--reset` calls `memory.delete_all(user_id=...)` before the
    replay loop starts. The unit test mocks the deletion path so a
    test run does not touch a real Qdrant store; the assertion is that
    the helper was invoked with the sandbox user_id."""

    @pytest.mark.asyncio
    async def test_reset_calls_delete_all(self):
        delete_mock = MagicMock()
        with patch("kai.memory.delete_all", delete_mock):
            await replay._reset_sandbox("sandbox-test")
        delete_mock.assert_called_once_with(user_id="sandbox-test")


# ── Memory store initialization ─────────────────────────────────────


class TestInitMemory:
    """`_async_main` MUST initialize the memory store before invoking the
    replay loop. Without this, the module-level `_memory` handle in
    `kai.memory` stays None, every storage primitive
    (`search`, `add_structured`, `delete_all`) short-circuits to a
    silent no-op, and the replay extracts facts via `claude --print`
    only to discard them. The failure mode is invisible at the unit
    level - `extract_and_store` returns 0 stored per pair, the summary
    line reports `facts_stored=0`, and no exception is raised - so the
    only place the regression can be caught structurally is here, by
    asserting the init call happens. This test exists because that bug
    actually shipped pre-init-fix: a 93-pair sandbox replay produced
    zero stored facts because `_memory` was None, and the symptom was
    only diagnosable by reading the `consolidate.intent` log lines for
    `outcome=dropped_backend`."""

    @pytest.mark.asyncio
    async def test_async_main_initializes_memory(self, tmp_path):
        # Create a history file with one valid pair so the replay walks
        # but does not block on missing files.
        history_dir = tmp_path / "history" / str(_CHAT_ID)
        history_dir.mkdir(parents=True)
        _write_jsonl(
            history_dir / "2026-05-12.jsonl",
            [_rec("user", "hi"), _rec("assistant", "hello")],
        )

        init_mock = MagicMock()
        config_mock = MagicMock()
        config_mock.episode_classifier_context_turns = 3
        # Patch every dependency reached after init_memory so the unit
        # test cannot reach a real config / store / subprocess. The
        # only assertion is that init_memory was invoked with the
        # config object - the call shape protecting against the silent
        # storage no-op.
        with (
            patch("kai.config.load_config", return_value=config_mock),
            patch("kai.memory.init_memory", init_mock),
            patch("kai.memory.delete_all"),
            patch("kai.memory.get_all_facts", return_value=[]),
            patch("kai.memory_extraction.extract_and_store", AsyncMock(return_value=0)),
        ):
            rc = await replay._async_main(
                [
                    "--chat-id",
                    str(_CHAT_ID),
                    "--user-id",
                    "sandbox-init-test",
                    "--history-dir",
                    str(history_dir),
                ]
            )

        assert rc == 0
        init_mock.assert_called_once_with(config_mock)


# ── --log-file structured-log capture ───────────────────────────────


class TestLogFileCapture:
    """`--log-file` MUST attach an INFO-level FileHandler to the root
    logger so structured Kai log lines (`memory.consolidate.intent`,
    `memory.extract:`, etc.) land in a file instead of being dropped
    by Python's WARNING-only last-resort handler.

    This test exists because that bug actually shipped: the #465
    five-T sweep ran ~6 hours of compute and produced zero
    gate-fire-count data because the replay module never configured
    logging. The deterministic gate-fire signal the spec named as
    PR-body evidence was silently dropped at the logger boundary. The
    fix is the `--log-file` flag; this test pins the contract."""

    def _make_history(self, tmp_path: Path) -> Path:
        """One-pair synthetic history so the replay walks but the
        extractor and store are stubbed before any subprocess fires."""
        history_dir = tmp_path / "history" / str(_CHAT_ID)
        history_dir.mkdir(parents=True)
        _write_jsonl(
            history_dir / "2026-05-12.jsonl",
            [_rec("user", "hi"), _rec("assistant", "hello")],
        )
        return history_dir

    @pytest.mark.asyncio
    async def test_log_file_flag_captures_intent_log_lines(self, tmp_path):
        """End-to-end: an INFO-level `memory.consolidate.intent` line
        emitted via the production `_emit_intent_log` helper must
        appear in the file the operator named. The replay loop is
        short-circuited (no real extraction) and we directly emit a
        synthetic intent log from inside the patched extract path so
        the test is fast and deterministic."""
        history_dir = self._make_history(tmp_path)
        log_path = tmp_path / "replay-with-logs.log"

        # Drive a synthetic intent log line from inside the patched
        # extract path. _emit_intent_log is the production helper that
        # `_store_facts` calls on every branch; emitting through it
        # verifies the same call shape production uses, not a stub.
        async def _fake_extract(*args, **kwargs):
            from kai.memory_extraction import _emit_intent_log

            _emit_intent_log(
                user_id=kwargs["user_id"],
                intent="new",
                original_intent=None,
                new_id="test-id",
                replaced_id=None,
                outcome="stored",
            )
            return 1

        config_mock = MagicMock()
        config_mock.episode_classifier_context_turns = 3

        with (
            patch("kai.config.load_config", return_value=config_mock),
            patch("kai.memory.init_memory"),
            patch("kai.memory.delete_all"),
            patch("kai.memory.get_all_facts", return_value=[]),
            patch("kai.memory_extraction.extract_and_store", side_effect=_fake_extract),
        ):
            rc = await replay._async_main(
                [
                    "--chat-id",
                    str(_CHAT_ID),
                    "--user-id",
                    "sandbox-log-test",
                    "--history-dir",
                    str(history_dir),
                    "--log-file",
                    str(log_path),
                ]
            )

        assert rc == 0
        assert log_path.exists(), "log file was not created"
        captured = log_path.read_text(encoding="utf-8")
        # The intent log line carries the literal marker the spec
        # documents as the wire format. Asserting the substring (not
        # the exact line) keeps the test resilient to formatter
        # adjustments while pinning the load-bearing tag.
        assert "memory.consolidate.intent" in captured
        # The JSON payload's outcome field MUST be preserved verbatim
        # so a downstream `jq '.outcome'` parse over a real sweep log
        # produces usable counts. This is the assertion that prevents
        # a future "summarize before write" refactor from breaking
        # gate-fire analysis.
        assert '"outcome":"stored"' in captured

    def _root_handler_count(self) -> int:
        """Snapshot of the root logger's handler count so the
        no-flag test can assert no permanent global mutation."""
        import logging

        return len(logging.getLogger().handlers)

    @pytest.mark.asyncio
    async def test_log_file_default_attaches_no_handler(self, tmp_path):
        """Regression guard: without `--log-file` the replay must NOT
        attach a handler to the root logger. The previous behavior
        (no logging) was a bug, but the FIX must be explicit (the
        flag), not silent. A future refactor that auto-enables logging
        could leak handlers across test runs or surprise an operator
        whose terminal session was previously quiet."""
        history_dir = self._make_history(tmp_path)
        before = self._root_handler_count()

        config_mock = MagicMock()
        config_mock.episode_classifier_context_turns = 3

        with (
            patch("kai.config.load_config", return_value=config_mock),
            patch("kai.memory.init_memory"),
            patch("kai.memory.delete_all"),
            patch("kai.memory.get_all_facts", return_value=[]),
            patch("kai.memory_extraction.extract_and_store", AsyncMock(return_value=0)),
        ):
            rc = await replay._async_main(
                [
                    "--chat-id",
                    str(_CHAT_ID),
                    "--user-id",
                    "sandbox-no-log-test",
                    "--history-dir",
                    str(history_dir),
                ]
            )

        assert rc == 0
        # Handler count unchanged: the replay did not silently attach
        # logging machinery. Operators who omit --log-file get the
        # same (broken-but-documented) behavior as before the fix.
        assert self._root_handler_count() == before


# ── PRIOR CONTEXT truncation deferred to extract_and_store ──────────


class TestPriorContextTruncationDeferred:
    """The replay does NOT roll its own truncation for prior turns. It
    passes raw pair strings to `extract_and_store`, which internally
    invokes `_build_extraction_payload` and applies the production
    `_PRIOR_USER_CHARS` / `_PRIOR_ASSISTANT_CHARS` caps. The assertion
    here is on the call shape: the prior_pairs the replay passes are
    the unmodified raw strings; trusting `_build_extraction_payload`
    to do the truncation is the contract.

    A regression in which the replay rolled its own (likely diverging)
    truncation logic would land in this test as an inequality on the
    captured `prior_pairs`."""

    @pytest.mark.asyncio
    async def test_replay_passes_untruncated_pairs(self):
        # Build a pair with a deliberately oversized user-side prior
        # turn (5000 chars). Production's _PRIOR_USER_CHARS cap (800)
        # would truncate, but only inside _build_extraction_payload.
        # The replay must pass the full string through; the truncation
        # is the downstream's job.
        long_user = "X" * 5000
        long_assistant = "Y" * 5000
        pairs = [
            (long_user, long_assistant),  # The prior pair.
            ("current_u", "current_a"),
        ]
        captured_prior: list[list[tuple[str, str]]] = []

        async def fake_store(user_text, assistant_text, *, user_id, config, prior_pairs):
            captured_prior.append(list(prior_pairs))
            return 0

        with patch("kai.memory_extraction.extract_and_store", side_effect=fake_store):
            await replay._run_replay(
                pairs,
                user_id="sandbox-test",
                context_turns=1,
                config=MagicMock(),
                dry_run=False,
            )

        # The second call's prior_pairs is the first pair, untruncated.
        # If the replay had rolled its own truncation, the captured
        # strings would be shorter than 5000 chars here.
        assert len(captured_prior) == 2
        assert captured_prior[0] == []
        assert captured_prior[1] == [(long_user, long_assistant)]


# ── Summary breakdowns (spec §4.2 step 6) ───────────────────────────


class TestFormatBreakdowns:
    """`_format_breakdowns` produces the per-tag / per-speaker /
    per-prompt-version aggregation of a sandbox user's post-replay
    fact set. Spec §4.2 step 6 names these three groupings as the
    summary the operator inspects. The unit test pins the grouping
    rules (tags fan out, speaker and prompt_version are single-valued)
    and the deterministic sort order (count desc, name asc) so two
    runs with the same fact multiset render byte-identical output."""

    def _fact(self, *, tags: list[str], speaker: str, prompt_version: str):
        # Lightweight stub mirroring `MemoryResult.metadata` access.
        # The format helper only reads `metadata`, not the other
        # MemoryResult fields, so a SimpleNamespace suffices here.
        from types import SimpleNamespace

        return SimpleNamespace(metadata={"tags": tags, "speaker": speaker, "prompt_version": prompt_version})

    def test_groups_by_tag_speaker_and_prompt_version(self):
        facts = [
            self._fact(tags=["preference"], speaker="user", prompt_version="9"),
            self._fact(tags=["preference", "location"], speaker="user", prompt_version="9"),
            self._fact(tags=["fact"], speaker="assistant", prompt_version="9"),
        ]
        out = replay._format_breakdowns(facts)
        # Tag fan-out: "preference" appears in 2 facts, contributes
        # twice; "location" appears in 1 fact, once; "fact" in 1, once.
        assert "preference: 2" in out
        assert "location: 1" in out
        assert "fact: 1" in out
        # Speaker single-valued: 2 user + 1 assistant.
        assert "user: 2" in out
        assert "assistant: 1" in out
        # Prompt-version single-valued: all 3 on v9.
        assert "9: 3" in out

    def test_empty_facts_renders_none_buckets(self):
        # A zero-fact run is the normal output of an extraction-
        # disabled or all-suppressed window. The summary section must
        # still render (parseable empty block) rather than crash or
        # skip.
        out = replay._format_breakdowns([])
        assert "by tag (0 distinct):" in out
        assert "by speaker (0 distinct):" in out
        assert "by prompt_version (0 distinct):" in out
        assert "(none)" in out

    def test_missing_metadata_fields_default_to_unknown(self):
        # Older fact rows predating the speaker / prompt_version
        # fields land in the store with those keys absent. The format
        # helper should bucket them under "unknown" rather than KeyError.
        from types import SimpleNamespace

        facts = [SimpleNamespace(metadata={"tags": ["fact"]})]
        out = replay._format_breakdowns(facts)
        assert "unknown: 1" in out

    def test_sort_order_is_count_desc_then_name_asc(self):
        facts = [
            self._fact(tags=["alpha"], speaker="user", prompt_version="9"),
            self._fact(tags=["beta"], speaker="user", prompt_version="9"),
            self._fact(tags=["beta"], speaker="user", prompt_version="9"),
            self._fact(tags=["gamma"], speaker="user", prompt_version="9"),
        ]
        out = replay._format_breakdowns(facts)
        # beta (count=2) precedes alpha (count=1, comes before gamma
        # alphabetically). gamma trails alpha.
        beta_pos = out.index("beta:")
        alpha_pos = out.index("alpha:")
        gamma_pos = out.index("gamma:")
        assert beta_pos < alpha_pos < gamma_pos


# ── Dry-run payload-shape sample (spec §4.2) ────────────────────────


class TestFormatDryRunSamples:
    """`_format_dry_run_samples` answers the spec §4.2 `--dry-run`
    contract: "report the would-be pair count AND a sample of payload
    shapes." The pair count is in the top-level summary; this helper
    renders the per-pair structural sample. Verbatim text is NOT in
    the output (operator-personal history could land in stdout
    otherwise); only structural shape: prior depth, user/assistant
    character counts."""

    def test_renders_one_pair(self):
        samples = [{"index": 1, "prior_count": 0, "user_chars": 12, "assistant_chars": 200}]
        out = replay._format_dry_run_samples(samples)
        assert "first 1 pair(s)" in out
        assert "pair 1: prior_pairs=0 user_chars=12 assistant_chars=200" in out

    def test_renders_three_pairs(self):
        samples = [
            {"index": 1, "prior_count": 0, "user_chars": 10, "assistant_chars": 20},
            {"index": 2, "prior_count": 1, "user_chars": 30, "assistant_chars": 40},
            {"index": 3, "prior_count": 2, "user_chars": 50, "assistant_chars": 60},
        ]
        out = replay._format_dry_run_samples(samples)
        assert "first 3 pair(s)" in out
        assert out.count("pair ") == 3

    def test_no_verbatim_text_in_output(self):
        # The sample dict deliberately carries no verbatim text;
        # confirm `_format_dry_run_samples` does not introduce any.
        # The "no operator-personal stdout leak" property is what
        # the helper's docstring promises.
        samples = [{"index": 1, "prior_count": 0, "user_chars": 5, "assistant_chars": 5}]
        out = replay._format_dry_run_samples(samples)
        for forbidden in ("user_text", "assistant_text", "content="):
            assert forbidden not in out

    def test_empty_samples_does_not_crash(self):
        out = replay._format_dry_run_samples([])
        assert "no pairs to sample" in out
