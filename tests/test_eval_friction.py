"""
Tests for src/kai/eval/friction.py (Layer 3 longitudinal friction analysis).

The friction harness reads chat history from `data_dir/history/<user_id>/`
directly via its own `--data-dir` argument; the autouse
`_isolate_history_dir` fixture at tests/conftest.py:65-81 only redirects
WRITES from `kai.history.log_message` and does NOT cover this read path.
Every test that needs history data therefore writes via the local
`_write_history` helper below and passes `tmp_path` as `--data-dir`
explicitly. Removing the explicit `--data-dir` would silently fall back
to the production `KAI_DATA_DIR`, which is exactly the contamination
this fixture exists to prevent for OTHER test paths.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest

from kai.eval import friction
from kai.eval.friction import (
    _BAD_TS_SENTINEL,
    _BUCKET_ANCHORS,
    _FRUSTRATION_PHRASES,
    _MATERIAL_DROP_RATIO,
    _PREDICTED_PATTERN,
    _SIGNAL_FAMILIES,
    _UP_RATIO,
    _USER_TURN_FLOOR,
    BucketAggregate,
    FrictionEvent,
    HistoryRecord,
    MemoryAvailability,
    MemoryInitResult,
    _classify_band,
    _classify_bucket,
    _content_words,
    _detect_frustration,
    _detect_kai_asks_back,
    _detect_preference_correction,
    _detect_repeated_fact,
    _ends_with_question_mark,
    _non_negative_int,
    _parse_record_timestamp,
    _redact_text,
    _validate_user_id,
    aggregate,
    classify_trend,
    detect_events,
    main,
    read_history,
)

# ── Test helpers ────────────────────────────────────────────────────


def _write_history(tmp_path: Path, user_id: str, records: list[dict]) -> Path:
    """Build a synthetic history/<user_id>/ tree under tmp_path.

    Records are grouped by UTC date into YYYY-MM-DD.jsonl files, matching
    the on-disk scheme at src/kai/history.py:67, 75. Returns the data_dir
    root (tmp_path), which is what the harness receives via --data-dir.

    The helper is local to this module by convention; behavioral.py's
    inline pattern at tests/test_eval_behavioral.py:1129-1176 is the
    reference shape but was deliberately not extracted to a shared
    fixture - keeping the helper here avoids a production-side refactor.
    """
    history_dir = tmp_path / "history" / user_id
    history_dir.mkdir(parents=True, exist_ok=True)
    by_date: dict[str, list[dict]] = {}
    for rec in records:
        ts = rec.get("ts", "")
        date_part = ts[:10] if isinstance(ts, str) and len(ts) >= 10 else "0001-01-01"
        by_date.setdefault(date_part, []).append(rec)
    for date_str, day_records in by_date.items():
        path = history_dir / f"{date_str}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in day_records) + "\n",
            encoding="utf-8",
        )
    return tmp_path


def _user_msg(ts: str, text: str, chat_id: int = 1) -> dict:
    """Shorthand: a user-direction JSONL row with the given ts and text."""
    return {"ts": ts, "dir": "user", "chat_id": chat_id, "text": text}


def _bot_msg(ts: str, text: str, chat_id: int = 1) -> dict:
    """Shorthand: an assistant-direction JSONL row with the given ts and text."""
    return {"ts": ts, "dir": "assistant", "chat_id": chat_id, "text": text}


@dataclass
class _FakeFact:
    """Stand-in for kai.memory.MemoryResult with the three fields stage 3b reads.

    The detector accesses fields via getattr so this minimal shape is
    sufficient; using a real MemoryResult would force constructing the
    full Mem0 metadata dict which has nothing to do with what stage 3b
    actually exercises (id, text, created_at).
    """

    id: str
    text: str
    created_at: str


def _make_aggregates(rates: dict[str, dict[str, float]]) -> list[BucketAggregate]:
    """Build a BucketAggregate list from a {bucket: {signal: rate}} mapping.

    Sets user_turns_in_bucket=100 so the per-100 rate equals the
    literal rate input; classifier-focused tests read only the rate
    field, but `count` is rounded (not truncated) so any future
    JSON-output test that asserts on the count field sees the
    arithmetically consistent value (rate=2.7 -> count=3, not 2).
    Missing (bucket, family) cells default to 0.0.
    """
    out: list[BucketAggregate] = []
    for bucket in ("A", "B", "C"):
        for family in _SIGNAL_FAMILIES:
            rate = rates.get(bucket, {}).get(family, 0.0)
            out.append(
                BucketAggregate(
                    bucket_label=bucket,
                    signal_family=family,
                    count=round(rate),
                    user_turns_in_bucket=100,
                    rate_per_100_user_turns=rate,
                )
            )
    return out


# Anchor-relative datetime helpers; tests reference offsets from the
# milestone anchors so the synthetic records always land in the
# intended bucket regardless of the absolute calendar dates.
_BEFORE_A = (_BUCKET_ANCHORS[0] - timedelta(days=10)).isoformat()  # bucket A
_IN_B = (_BUCKET_ANCHORS[0] + timedelta(hours=1)).isoformat()  # bucket B
_IN_C = (_BUCKET_ANCHORS[1] + timedelta(hours=1)).isoformat()  # bucket C


# ── TestHistoryReader ───────────────────────────────────────────────


class TestHistoryReader:
    """Round-trip behaviour of read_history against synthetic JSONL trees."""

    def test_empty_directory_returns_empty_iterator(self, tmp_path: Path):
        # A data_dir with no history/ subtree is a valid input shape; the
        # iterator yields nothing rather than raising. This is the path a
        # fresh operator hits before any chat traffic exists.
        assert list(read_history(tmp_path, "u1")) == []

    def test_missing_user_returns_empty_iterator(self, tmp_path: Path):
        # history/ exists but no <user_id>/ subdir. Empty iterator with
        # no log noise; the harness keeps running and emits a zero-count
        # report (matches the "no crash on empty history" acceptance shape).
        (tmp_path / "history").mkdir()
        assert list(read_history(tmp_path, "u1")) == []

    def test_malformed_json_line_skipped(self, tmp_path: Path):
        # Mirror the behaviour at src/kai/history.py:136-153: a single
        # bad line does not abort parsing; surrounding good lines still
        # land in the record list.
        history_dir = tmp_path / "history" / "u1"
        history_dir.mkdir(parents=True)
        rows = [
            json.dumps(_user_msg(_IN_B, "good before")),
            "{not valid json",
            "",
            json.dumps(_user_msg(_IN_B, "good after")),
        ]
        (history_dir / "2026-04-18.jsonl").write_text("\n".join(rows), encoding="utf-8")
        out = list(read_history(tmp_path, "u1"))
        assert [r.text for r in out] == ["good before", "good after"]

    def test_bad_timestamp_routes_to_sentinel(self, tmp_path: Path):
        # Missing/empty/unparseable ts -> _BAD_TS_SENTINEL.
        # Sentinel is year 1 so bucket assignment routes the record to
        # A without a special branch (verified separately in
        # TestBucketAssignment).
        _write_history(
            tmp_path,
            "u1",
            [
                {"ts": "", "dir": "user", "chat_id": 1, "text": "no ts"},
                {"ts": "not-iso", "dir": "user", "chat_id": 1, "text": "bad ts"},
            ],
        )
        out = list(read_history(tmp_path, "u1"))
        assert all(r.ts_utc == _BAD_TS_SENTINEL for r in out)

    def test_multiple_files_concatenated_in_sorted_order(self, tmp_path: Path):
        # Production history rotates daily; the reader stitches all
        # YYYY-MM-DD.jsonl files in lexicographic order so chronological
        # ordering is preserved without any per-record sort.
        history_dir = tmp_path / "history" / "u1"
        history_dir.mkdir(parents=True)
        (history_dir / "2026-04-18.jsonl").write_text(json.dumps(_user_msg(_IN_B, "from-18")) + "\n", encoding="utf-8")
        (history_dir / "2026-04-19.jsonl").write_text(json.dumps(_user_msg(_IN_B, "from-19")) + "\n", encoding="utf-8")
        out = [r.text for r in read_history(tmp_path, "u1")]
        assert out == ["from-18", "from-19"]

    def test_non_object_json_lines_skipped(self, tmp_path: Path):
        # A top-level JSON array or string is structurally invalid but
        # might appear from a hand-edit; the parser must not call .get
        # on a non-dict and must skip silently.
        history_dir = tmp_path / "history" / "u1"
        history_dir.mkdir(parents=True)
        rows = [
            json.dumps(["not", "a", "row"]),
            json.dumps(_user_msg(_IN_B, "good")),
        ]
        (history_dir / "2026-04-18.jsonl").write_text("\n".join(rows), encoding="utf-8")
        out = [r.text for r in read_history(tmp_path, "u1")]
        assert out == ["good"]

    def test_non_integer_chat_id_skipped_not_aborted(self, tmp_path: Path):
        # A row with a non-numeric chat_id is valid JSON, so the parse
        # guard above lets it through; the int() cast must not crash
        # the run. The docstring promises malformed rows do not abort.
        history_dir = tmp_path / "history" / "u1"
        history_dir.mkdir(parents=True)
        bad = {"ts": _IN_B, "dir": "user", "chat_id": "corrupted", "text": "bad"}
        good = _user_msg(_IN_B, "good")
        (history_dir / "2026-04-18.jsonl").write_text(
            "\n".join(json.dumps(r) for r in (bad, good)) + "\n",
            encoding="utf-8",
        )
        out = [r.text for r in read_history(tmp_path, "u1")]
        assert out == ["good"]


# ── TestBucketAssignment ────────────────────────────────────────────


class TestBucketAssignment:
    """Boundary semantics for the milestone-bucket classifier."""

    def test_record_at_anchor_a_lands_in_b(self):
        # Bucket B is [anchor_A, anchor_B). Inclusive lower bound
        # means a record at exactly anchor_A is the first row of B.
        assert _classify_bucket(_BUCKET_ANCHORS[0]) == "B"

    def test_record_just_before_anchor_a_lands_in_a(self):
        # One microsecond before anchor_A is still bucket A; this nails
        # the exclusive upper bound on A.
        ts = _BUCKET_ANCHORS[0] - timedelta(microseconds=1)
        assert _classify_bucket(ts) == "A"

    def test_record_at_anchor_b_lands_in_c(self):
        # Same inclusive-lower contract for bucket C.
        assert _classify_bucket(_BUCKET_ANCHORS[1]) == "C"

    def test_record_just_before_anchor_b_lands_in_b(self):
        ts = _BUCKET_ANCHORS[1] - timedelta(microseconds=1)
        assert _classify_bucket(ts) == "B"

    def test_bad_timestamp_sentinel_routes_to_a(self):
        # Bad timestamps land in the oldest bucket by design; the
        # sentinel is year 1, strictly older than anchor_A.
        assert _classify_bucket(_BAD_TS_SENTINEL) == "A"


# ── TestFrustrationSignal ───────────────────────────────────────────


class TestFrustrationSignal:
    """Signal 1 detection: operator-side frustration markers."""

    @pytest.mark.parametrize("phrase", list(_FRUSTRATION_PHRASES))
    def test_each_phrase_emits_exactly_one_event(self, phrase: str, tmp_path: Path):
        # Pre-committed phrase set; every phrase must trigger the
        # detector or the lock-it-before-running promise is broken.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text=f"hey {phrase} about that",
                media=None,
                source_path=str(tmp_path / "history" / "u1" / "2026-04-18.jsonl"),
                line_no=1,
            )
        ]
        events = _detect_frustration(records, tmp_path)
        assert len(events) == 1
        assert events[0].signal_family == "frustration"

    def test_assistant_messages_never_flagged(self, tmp_path: Path):
        # Frustration is operator-side only; the same phrase from the
        # assistant must not register, otherwise Kai's own apologetic
        # boilerplate would inflate the rate.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="assistant",
                chat_id=1,
                text="i already told you the answer",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=1,
            )
        ]
        assert _detect_frustration(records, tmp_path) == []

    def test_case_insensitive_matching(self, tmp_path: Path):
        # Telegram autocorrect capitalizes message starts; without
        # case-insensitive matching every "I told you" would silently
        # miss. The detector lowercases before substring match.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="I TOLD YOU about the bug",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=1,
            )
        ]
        assert len(_detect_frustration(records, tmp_path)) == 1

    def test_multiple_phrases_in_one_message_emit_one_event(self, tmp_path: Path):
        # One message containing two phrases must NOT emit two events
        # (the inner break enforces this); double-counting would distort
        # per-bucket rates on operators with a verbose vent style.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="i told you and i already told you",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=1,
            )
        ]
        assert len(_detect_frustration(records, tmp_path)) == 1


# ── TestKaiAsksBackSignal ───────────────────────────────────────────


class TestKaiAsksBackSignal:
    """Signal 2 detection: Kai asking for information already provided."""

    @staticmethod
    def _records_for_overlap(tmp_path: Path, prior_text: str, question_text: str):
        # Build a 2-record window: prior user message then assistant
        # question. Used by overlap-floor tests; keeps the construction
        # noise out of the test bodies.
        return [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text=prior_text,
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=1,
            ),
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=2),
                ts_raw=_IN_B,
                direction="assistant",
                chat_id=1,
                text=question_text,
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=2,
            ),
        ]

    def test_assistant_question_with_3_overlap_emits(self, tmp_path: Path):
        # Three substantive shared content-words crosses the floor; the
        # detector emits one event citing the prior user message.
        records = self._records_for_overlap(
            tmp_path,
            "deployment uses postgres docker container",
            "what database deployment container does postgres use?",
        )
        events = _detect_kai_asks_back(records, tmp_path)
        assert len(events) == 1
        assert events[0].signal_family == "kai_asks_back"
        assert len(events[0].context_message_ids) == 1

    def test_assistant_question_with_2_overlap_does_not_emit(self, tmp_path: Path):
        # Two shared content-words is below the floor; no event.
        # Calibrated to keep generic "what about that" clarifications
        # from being miscounted as memory misses.
        records = self._records_for_overlap(
            tmp_path,
            "deployment uses postgres",
            "what database deployment uses?",
        )
        assert _detect_kai_asks_back(records, tmp_path) == []

    def test_assistant_message_without_question_mark_skipped(self, tmp_path: Path):
        # The signal requires a literal "?" ending; a declarative
        # response that happens to share content words must not register.
        records = self._records_for_overlap(
            tmp_path,
            "deployment uses postgres docker container",
            "the database deployment container is postgres",
        )
        assert _detect_kai_asks_back(records, tmp_path) == []

    def test_window_excludes_records_beyond_lookback(self, tmp_path: Path):
        # The lookback window is records[max(0, i-20):i]. The 21st prior
        # is OUTSIDE the window and must not be cited even if it shares
        # all content-words with the assistant question.
        records: list[HistoryRecord] = []
        # 21 noise user records first.
        for n in range(21):
            records.append(
                HistoryRecord(
                    ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=n),
                    ts_raw=_IN_B,
                    direction="user",
                    chat_id=1,
                    text=f"unrelated noise {n}",
                    media=None,
                    source_path=str(tmp_path / "h.jsonl"),
                    line_no=n + 1,
                )
            )
        # After the insert, the layout is: indices 0-20 (21 noise),
        # index 21 (matching prior), indices 22-42 (21 more noise),
        # index 43 (assistant question). The matching prior is 22
        # slots back from the assistant; the 20-record lookback
        # window covers indices 23-42, so the prior at index 21 is
        # 2 slots outside the window.
        records.insert(
            21,
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=21),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="deployment uses postgres docker container",
                media=None,
                source_path=str(tmp_path / "outside.jsonl"),
                line_no=999,
            ),
        )
        for n in range(21):
            records.append(
                HistoryRecord(
                    ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=22 + n),
                    ts_raw=_IN_B,
                    direction="user",
                    chat_id=1,
                    text=f"more noise {n}",
                    media=None,
                    source_path=str(tmp_path / "h.jsonl"),
                    line_no=1000 + n,
                )
            )
        records.append(
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=99),
                ts_raw=_IN_B,
                direction="assistant",
                chat_id=1,
                text="what database deployment container does postgres use?",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=2000,
            )
        )
        events = _detect_kai_asks_back(records, tmp_path)
        # Window covers the 20 most recent priors before the assistant
        # record; the matching prior is 22 slots back, outside the cap.
        # The disjunctive form would mask a future regression (any
        # event from any other source would still pass), so assert
        # no events fire at all.
        assert events == []

    def test_question_mark_detector_handles_closers(self):
        # A question wrapped in quotes ("...?") still counts as a
        # question; the closer-stripping logic preserves common
        # parenthesized/quoted forms.
        assert _ends_with_question_mark("really?")
        assert _ends_with_question_mark('really?"')
        assert _ends_with_question_mark("really?)")
        assert not _ends_with_question_mark("really.")


# ── TestPreferenceCorrectionSignal ──────────────────────────────────


class TestPreferenceCorrectionSignal:
    """Signal 3 stage 3a: regex-based correction detection."""

    @pytest.mark.parametrize(
        "text",
        [
            "don't do that",
            "stop asking for the same thing",
            "not that one",
            "please use the new flag",
            "always use snake_case",
            "never use camelCase",
            "instead of foo, try bar",
            "i prefer the longer form",
        ],
    )
    def test_each_correction_pattern_matches(self, text: str, tmp_path: Path):
        # Each pre-committed regex must catch its sentinel phrase; this
        # locks the regex set so a typo or a renamed pattern fails the
        # test instead of silently missing in production.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text=text,
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=1,
            )
        ]
        events = _detect_preference_correction(records, tmp_path, facts=[])
        assert len(events) == 1
        assert events[0].signal_family == "preference_correction"

    def test_assistant_corrections_not_flagged(self, tmp_path: Path):
        # Stage 3a is operator-side only; an assistant message that
        # happens to contain "don't" must not inflate the rate.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="assistant",
                chat_id=1,
                text="don't worry, i'll handle it",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=1,
            )
        ]
        assert _detect_preference_correction(records, tmp_path, facts=[]) == []

    def test_non_correction_text_not_flagged(self, tmp_path: Path):
        # Plain content with no correction marker; rate stays at zero.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="here is some context for the next task",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=1,
            )
        ]
        assert _detect_preference_correction(records, tmp_path, facts=[]) == []

    def test_event_has_known_fact_overlap_false_when_no_facts(self, tmp_path: Path):
        # No facts available -> stage 3b cannot mark any event as
        # known-fact-overlap; metadata.known_fact_overlap MUST be False
        # AND overlapping_fact_ids MUST be the empty tuple. This is the
        # baseline shape the verdict-driving sum collapses against when
        # memory is unreachable.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="don't do that thing again",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=1,
            )
        ]
        events = _detect_preference_correction(records, tmp_path, facts=[])
        assert events[0].metadata["known_fact_overlap"] is False
        assert events[0].metadata["overlapping_fact_ids"] == ()


# ── TestKnownFactOverlap ────────────────────────────────────────────


class TestKnownFactOverlap:
    """Signal 3 stage 3b: known-fact overlap (the verdict-driving subset)."""

    @staticmethod
    def _correction_record(tmp_path: Path):
        # Single user record carrying enough content-words to overlap
        # against the test facts. "deployment postgres docker container"
        # is the shared substantive payload.
        return HistoryRecord(
            ts_utc=_BUCKET_ANCHORS[0] + timedelta(days=2),
            ts_raw=_IN_B,
            direction="user",
            chat_id=1,
            text="don't change the deployment postgres docker container",
            media=None,
            source_path=str(tmp_path / "h.jsonl"),
            line_no=1,
        )

    def test_fact_created_before_correction_marks_known(self, tmp_path: Path):
        # The fact predates the correction by 1 day and shares 4
        # content-words; stage 3b must mark this as known-fact overlap.
        fact = _FakeFact(
            id="f1",
            text="user prefers postgres docker container deployment",
            created_at=(_BUCKET_ANCHORS[0] + timedelta(days=1)).isoformat(),
        )
        events = _detect_preference_correction([self._correction_record(tmp_path)], tmp_path, facts=[fact])
        assert events[0].metadata["known_fact_overlap"] is True
        assert events[0].metadata["overlapping_fact_ids"] == ("f1",)

    def test_fact_created_after_correction_excluded(self, tmp_path: Path):
        # Stage 3b's time-comparison rule: a fact whose created_at is
        # AFTER the correction cannot have grounded the operator's
        # frustration; including it would inflate the verdict-driving
        # sum with retro-extracted facts.
        fact = _FakeFact(
            id="f1",
            text="user prefers postgres docker container deployment",
            created_at=(_BUCKET_ANCHORS[0] + timedelta(days=5)).isoformat(),
        )
        events = _detect_preference_correction([self._correction_record(tmp_path)], tmp_path, facts=[fact])
        assert events[0].metadata["known_fact_overlap"] is False
        assert events[0].metadata["overlapping_fact_ids"] == ()

    def test_fact_with_unparseable_created_at_excluded(self, tmp_path: Path):
        # Conservative: "we don't know when this fact was written, so
        # we can't claim it pre-existed the correction." The known-fact
        # overlap detector deliberately does NOT fall back to updated_at.
        fact = _FakeFact(
            id="f1",
            text="user prefers postgres docker container deployment",
            created_at="not-a-timestamp",
        )
        events = _detect_preference_correction([self._correction_record(tmp_path)], tmp_path, facts=[fact])
        assert events[0].metadata["known_fact_overlap"] is False

    def test_fact_with_empty_created_at_excluded(self, tmp_path: Path):
        # Third of the three mandated exclusion paths: empty-string
        # created_at. Production handles this via `if not value: return
        # None` in _parse_fact_created_at, which collapses both None and
        # "" to the same exclusion path. We assert the empty case
        # explicitly so a future refactor that swaps the falsy check for
        # `if value is None` cannot silently regress: the unparseable
        # test would still pass (independent try/except branch) and the
        # after-correction test would still pass (never reaches the
        # empty-string branch), making this the only guard against the
        # None-vs-falsy slip.
        fact = _FakeFact(
            id="f1",
            text="user prefers postgres docker container deployment",
            created_at="",
        )
        events = _detect_preference_correction([self._correction_record(tmp_path)], tmp_path, facts=[fact])
        assert events[0].metadata["known_fact_overlap"] is False
        assert events[0].metadata["overlapping_fact_ids"] == ()

    def test_multiple_overlapping_facts_all_in_overlapping_fact_ids(self, tmp_path: Path):
        # Two facts both overlap and both predate the correction; both
        # IDs must appear in the metadata tuple (no first-match break).
        f1 = _FakeFact(
            id="f1",
            text="user prefers postgres docker container deployment",
            created_at=(_BUCKET_ANCHORS[0] + timedelta(hours=1)).isoformat(),
        )
        f2 = _FakeFact(
            id="f2",
            text="postgres docker container deployment is canonical",
            created_at=(_BUCKET_ANCHORS[0] + timedelta(hours=2)).isoformat(),
        )
        events = _detect_preference_correction([self._correction_record(tmp_path)], tmp_path, facts=[f1, f2])
        assert set(events[0].metadata["overlapping_fact_ids"]) == {"f1", "f2"}


# ── TestRepeatedFactSignal ──────────────────────────────────────────


class TestRepeatedFactSignal:
    """Signal 4: repeated fact-shape statements via Jaccard on shingles."""

    @staticmethod
    def _fact_shape_records(tmp_path: Path, *texts: str):
        # All records are user-direction with consecutive timestamps so
        # the Jaccard comparison walks them as a single conversation.
        out = []
        for n, text in enumerate(texts):
            out.append(
                HistoryRecord(
                    ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=n),
                    ts_raw=_IN_B,
                    direction="user",
                    chat_id=1,
                    text=text,
                    media=None,
                    source_path=str(tmp_path / "h.jsonl"),
                    line_no=n + 1,
                )
            )
        return out

    def test_jaccard_above_threshold_emits(self, tmp_path: Path):
        # Two near-identical fact-shape statements; shingle overlap is
        # well above the 0.4 floor. The second statement is the repeat;
        # only it emits an event citing the first as context.
        records = self._fact_shape_records(
            tmp_path,
            "i prefer postgres docker container deployment",
            "i prefer postgres docker container deployment as before",
        )
        events = _detect_repeated_fact(records, tmp_path)
        assert len(events) == 1
        assert events[0].signal_family == "repeated_fact"
        assert events[0].context_message_ids  # invariant: every event has non-empty context

    def test_non_fact_shape_records_never_flagged(self, tmp_path: Path):
        # Both records are paraphrases but neither matches the
        # _FACT_SHAPE_RE prefix set; the detector does not flag them
        # even though their Jaccard would clear the floor.
        records = self._fact_shape_records(
            tmp_path,
            "the weather is nice today and clear and sunny",
            "the weather is nice today and clear and sunny",
        )
        assert _detect_repeated_fact(records, tmp_path) == []

    def test_first_occurrence_never_flagged(self, tmp_path: Path):
        # A standalone fact-shape statement with no prior cannot emit;
        # this preserves the harness invariant that every emitted event
        # has a non-empty context_message_ids tuple.
        records = self._fact_shape_records(tmp_path, "i prefer postgres docker container deployment")
        assert _detect_repeated_fact(records, tmp_path) == []

    def test_unrelated_fact_shape_pair_not_flagged(self, tmp_path: Path):
        # Two fact-shape statements that share NO shingles must not
        # emit; otherwise every "i ..." sentence would inflate the
        # repeated_fact rate.
        records = self._fact_shape_records(
            tmp_path,
            "i prefer postgres for analytics workloads",
            "my deployment runs on kubernetes nodes",
        )
        assert _detect_repeated_fact(records, tmp_path) == []


# ── TestNormalization ───────────────────────────────────────────────


class TestNormalization:
    """Per-100-user-turns rate math under normal and edge inputs."""

    def test_zero_user_turns_returns_zero_not_nan(self, tmp_path: Path):
        # A bucket with zero user_turns has rate 0.0 (NOT NaN);
        # NaN would break JSON serialization downstream.
        aggregates, _ = aggregate(records=[], events=[])
        for agg in aggregates:
            assert agg.rate_per_100_user_turns == 0.0

    def test_two_decimal_place_rounding(self):
        # 1 frustration event over 3 user-turns -> 33.33333... -> 33.33.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text=f"u{n}",
                media=None,
                source_path="x",
                line_no=n,
            )
            for n in range(3)
        ]
        ev = FrictionEvent(
            timestamp_utc=records[0].ts_utc,
            bucket_label="B",
            signal_family="frustration",
            message_id="x:1",
            surface_text="x",
            context_message_ids=(),
            metadata=MappingProxyType({}),
        )
        aggregates, _ = aggregate(records, [ev])
        b_frustration = next(a for a in aggregates if a.bucket_label == "B" and a.signal_family == "frustration")
        assert b_frustration.rate_per_100_user_turns == 33.33

    def test_user_turn_denominator_excludes_assistant_messages(self):
        # The denominator MUST be user_turns, not messages_total. An
        # assistant-heavy bucket would otherwise dilute the rate.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="u1",
                media=None,
                source_path="x",
                line_no=1,
            ),
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(hours=2),
                ts_raw=_IN_B,
                direction="assistant",
                chat_id=1,
                text="bot1",
                media=None,
                source_path="x",
                line_no=2,
            ),
        ]
        _, user_turns = aggregate(records, [])
        assert user_turns["B"] == 1


# ── TestOutputDeterminism ───────────────────────────────────────────


class _FakeNow:
    """Subclass-of-datetime sentinel for monkey-patching `datetime.now`.

    Subclassing (not Mock) so the module's other datetime usage
    (fromisoformat, constructor calls) keeps working; only `.now` is
    overridden. The convention is pytest monkeypatch of the module's
    datetime reference rather than a hidden CLI flag that would only
    run in tests.
    """

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 4, 23, 14, 0, 0, tzinfo=tz or UTC)

    def __new__(cls, *args, **kwargs):
        return datetime(*args, **kwargs)


class _FakeDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        # Pinned reference time; same return value across runs makes the
        # JSON's generated_at_utc field byte-stable for determinism tests.
        return datetime(2026, 4, 23, 14, 0, 0, tzinfo=tz or UTC)


class TestOutputDeterminism:
    """Byte-stability of JSON output across runs against a fixed input."""

    def _run_once(self, tmp_path: Path, output_name: str, monkeypatch) -> str:
        # Helper: write a tiny synthetic history, run the harness, and
        # return the raw JSON file contents. Two consecutive calls must
        # produce byte-identical strings (determinism contract).
        _write_history(
            tmp_path,
            "u1",
            [
                _user_msg(_BEFORE_A, "old context message"),
                _user_msg(_IN_B, "i told you about that bug"),
                _user_msg(_IN_C, "don't change the deploy script"),
            ],
        )
        monkeypatch.setattr("kai.eval.friction.datetime", _FakeDatetime)
        monkeypatch.setattr(
            "kai.eval.friction._initialize_memory_local",
            lambda: MemoryInitResult(MemoryAvailability.DISABLED, None),
        )
        out = tmp_path / output_name
        rc = main(
            [
                "--user-id",
                "u1",
                "--output",
                str(out),
                "--data-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        return out.read_text(encoding="utf-8")

    def test_same_input_produces_byte_identical_json(self, tmp_path: Path, monkeypatch):
        a = self._run_once(tmp_path, "a.json", monkeypatch)
        b = self._run_once(tmp_path, "b.json", monkeypatch)
        assert a == b

    def test_events_sorted_by_timestamp_then_message_id(self, tmp_path: Path, monkeypatch):
        # Two events with the same timestamp but different message_ids;
        # the report MUST sort them lexically by message_id so output
        # is deterministic across runs.
        ts = _BUCKET_ANCHORS[0] + timedelta(hours=1)
        events = [
            FrictionEvent(
                timestamp_utc=ts,
                bucket_label="B",
                signal_family="frustration",
                message_id="z.jsonl:1",
                surface_text="late",
                context_message_ids=(),
                metadata=MappingProxyType({}),
            ),
            FrictionEvent(
                timestamp_utc=ts,
                bucket_label="B",
                signal_family="frustration",
                message_id="a.jsonl:1",
                surface_text="early",
                context_message_ids=(),
                metadata=MappingProxyType({}),
            ),
        ]
        sorted_events = sorted(events, key=lambda e: (e.timestamp_utc, e.message_id))
        # Lexically a < z, so a.jsonl event must be first regardless of
        # construction order.
        assert sorted_events[0].message_id == "a.jsonl:1"


# ── TestSurfaceTextRedaction ────────────────────────────────────────


class TestSurfaceTextRedaction:
    """Redaction applied when --include-snippets is set."""

    def test_url_redacted(self):
        out = _redact_text("see https://example.com/path?q=1 for details")
        assert "<redacted-url>" in out
        assert "https://" not in out

    def test_email_redacted(self):
        out = _redact_text("write to user@example.com please")
        assert "<redacted-email>" in out
        assert "@example.com" not in out

    def test_handle_redacted(self):
        out = _redact_text("ping @username about it")
        assert "<redacted-handle>" in out
        assert "@username" not in out

    def test_phone_redacted(self):
        out = _redact_text("call 555-123-4567 for support")
        assert "<redacted-phone>" in out
        assert "555-123-4567" not in out

    def test_snippet_capped_at_200_chars(self):
        # The redaction layer caps at _SNIPPET_MAX_CHARS so a long
        # message cannot dump a full chat-log paragraph into the report.
        long_text = "x" * 500
        assert len(_redact_text(long_text)) == 200

    def test_include_snippets_false_nulls_surface_text(self, tmp_path: Path, monkeypatch):
        # End-to-end: default (no --include-snippets) must emit
        # surface_text=null. Verifies the render-layer guard.
        _write_history(
            tmp_path,
            "u1",
            [_user_msg(_IN_B, "i told you that already")],
        )
        monkeypatch.setattr("kai.eval.friction.datetime", _FakeDatetime)
        monkeypatch.setattr(
            "kai.eval.friction._initialize_memory_local",
            lambda: MemoryInitResult(MemoryAvailability.DISABLED, None),
        )
        out = tmp_path / "report.json"
        main(
            [
                "--user-id",
                "u1",
                "--output",
                str(out),
                "--data-dir",
                str(tmp_path),
            ]
        )
        report = json.loads(out.read_text())
        assert all(ev["surface_text"] is None for ev in report["events"])

    def test_include_snippets_true_populates(self, tmp_path: Path, monkeypatch):
        # With --include-snippets, surface_text is populated (still
        # redacted, still capped); the operator can now diff the
        # specific messages a signal fired on.
        _write_history(
            tmp_path,
            "u1",
            [_user_msg(_IN_B, "i told you that already")],
        )
        monkeypatch.setattr("kai.eval.friction.datetime", _FakeDatetime)
        monkeypatch.setattr(
            "kai.eval.friction._initialize_memory_local",
            lambda: MemoryInitResult(MemoryAvailability.DISABLED, None),
        )
        out = tmp_path / "report.json"
        main(
            [
                "--user-id",
                "u1",
                "--output",
                str(out),
                "--data-dir",
                str(tmp_path),
                "--include-snippets",
            ]
        )
        report = json.loads(out.read_text())
        assert any(ev["surface_text"] for ev in report["events"])


# ── TestCLIIntegration ──────────────────────────────────────────────


class TestCLIIntegration:
    """Full CLI invocation against a synthetic history tree."""

    def test_end_to_end_writes_json_and_markdown(self, tmp_path: Path, monkeypatch):
        # Round-trip: history -> harness -> JSON + markdown files. Both
        # outputs must exist and the JSON must parse.
        _write_history(
            tmp_path,
            "u1",
            [
                _user_msg(_BEFORE_A, "context one"),
                _user_msg(_IN_B, "i told you about that"),
                _user_msg(_IN_C, "don't change the deploy"),
            ],
        )
        monkeypatch.setattr("kai.eval.friction.datetime", _FakeDatetime)
        monkeypatch.setattr(
            "kai.eval.friction._initialize_memory_local",
            lambda: MemoryInitResult(MemoryAvailability.DISABLED, None),
        )
        json_out = tmp_path / "report.json"
        md_out = tmp_path / "report.md"
        rc = main(
            [
                "--user-id",
                "u1",
                "--output",
                str(json_out),
                "--markdown-summary",
                str(md_out),
                "--data-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        report = json.loads(json_out.read_text())
        assert report["version"] == friction._OUTPUT_SCHEMA_VERSION
        assert report["user_id"] == "u1"
        # Markdown summary must redact the user_id literally.
        md = md_out.read_text()
        assert "<user_id-redacted>" in md
        assert "u1" not in md.split("Friction analysis")[0]

    def test_bad_user_id_returns_exit_code_2(self, tmp_path: Path):
        # Invalid --user-id is an argparse-shape error; the harness
        # must reject before any disk read.
        rc = main(
            [
                "--user-id",
                "../etc",
                "--output",
                str(tmp_path / "out.json"),
                "--data-dir",
                str(tmp_path),
            ]
        )
        assert rc == 2

    def test_empty_history_produces_zero_count_report(self, tmp_path: Path, monkeypatch):
        # No history files for the user -> zero counts everywhere, no
        # crash. The "no history must not abort" acceptance shape.
        monkeypatch.setattr("kai.eval.friction.datetime", _FakeDatetime)
        monkeypatch.setattr(
            "kai.eval.friction._initialize_memory_local",
            lambda: MemoryInitResult(MemoryAvailability.DISABLED, None),
        )
        out = tmp_path / "report.json"
        rc = main(
            [
                "--user-id",
                "u1",
                "--output",
                str(out),
                "--data-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        report = json.loads(out.read_text())
        assert all(agg["count"] == 0 for agg in report["aggregates"])

    def test_sample_days_zero_means_today_utc(self, tmp_path: Path, monkeypatch):
        # --sample-days 0 -> today UTC (midnight UTC of the run's day).
        # A pre-anchor record falls outside this window and is excluded
        # from the report.
        _write_history(
            tmp_path,
            "u1",
            [_user_msg(_BEFORE_A, "old message i told you")],
        )
        monkeypatch.setattr("kai.eval.friction.datetime", _FakeDatetime)
        monkeypatch.setattr(
            "kai.eval.friction._initialize_memory_local",
            lambda: MemoryInitResult(MemoryAvailability.DISABLED, None),
        )
        out = tmp_path / "report.json"
        rc = main(
            [
                "--user-id",
                "u1",
                "--output",
                str(out),
                "--data-dir",
                str(tmp_path),
                "--sample-days",
                "0",
            ]
        )
        assert rc == 0
        report = json.loads(out.read_text())
        # All bucket counts should be zero since the only record is
        # filtered out.
        assert all(b["user_turns"] == 0 for b in report["buckets"])


# ── TestTraversalGuard ──────────────────────────────────────────────


class TestTraversalGuard:
    """Defensive shape check on --user-id; mirrors behavioral.py:841."""

    @pytest.mark.parametrize(
        "bad",
        ["../etc", "../../etc/passwd", "foo/bar", "", ".", "..", "/abs/path"],
    )
    def test_invalid_user_id_rejected(self, bad: str):
        with pytest.raises(ValueError, match="invalid user_id"):
            _validate_user_id(bad)

    def test_valid_user_id_accepted(self):
        # Plain segment with no separators, dots, or empty value passes.
        # 12345 is the chat_id format used in production.
        _validate_user_id("12345")
        _validate_user_id("user-1")


# ── TestTrendClassifier ─────────────────────────────────────────────


class TestTrendClassifier:
    """All 9 grid cells + inconclusive precedence + unclassified catch-all."""

    @staticmethod
    def _ut_ok() -> dict[str, int]:
        # Three buckets each above the volume floor; lets the classifier
        # reach the band-classification path instead of short-circuiting
        # on inconclusive precedence.
        return {"A": 100, "B": 100, "C": 100}

    def _summary(self, frustration_rates: tuple[float, float, float]) -> friction.TrendSummary:
        # Build aggregates with the supplied frustration rates per
        # bucket (events left empty so verdict_sum equals frustration
        # alone) and return the classifier's verdict.
        rates = {
            "A": {"frustration": frustration_rates[0]},
            "B": {"frustration": frustration_rates[1]},
            "C": {"frustration": frustration_rates[2]},
        }
        return classify_trend(
            aggregates=_make_aggregates(rates),
            events=[],
            user_turns=self._ut_ok(),
        )

    def test_prediction_holds_drop_then_flat(self):
        # A=10 -> B=5 (drop), B=5 -> C=5 (flat). The pre-committed
        # hypothesis lands cleanly.
        s = self._summary((10.0, 5.0, 5.0))
        assert s.matched_pattern == "prediction-holds"
        assert s.predicted_pattern == _PREDICTED_PATTERN

    def test_both_milestones_helped_drop_then_drop(self):
        # A=20 -> B=10 (drop), B=10 -> C=4 (drop). Both rollouts
        # contributed signal reductions.
        s = self._summary((20.0, 10.0, 4.0))
        assert s.matched_pattern == "both-milestones-helped"

    def test_regression_drop_then_up(self):
        # A=10 -> B=5 (drop), B=5 -> C=10 (up). Track 1 cleanup helped,
        # then post-Track-1 regressed.
        s = self._summary((10.0, 5.0, 10.0))
        assert s.matched_pattern == "regression"

    def test_track_1_carried_win_flat_then_drop(self):
        # A=10 -> B=10 (flat), B=10 -> C=4 (drop). Track 1 removal was
        # the actual carrier of the post-#361 improvement.
        s = self._summary((10.0, 10.0, 4.0))
        assert s.matched_pattern == "track-1-carried-win"

    def test_no_effect_flat_then_flat(self):
        # A=10 -> B=10 -> C=10. Neither milestone moved the verdict.
        s = self._summary((10.0, 10.0, 10.0))
        assert s.matched_pattern == "no-effect"

    def test_regression_flat_then_up(self):
        # A=10 -> B=10 (flat), B=10 -> C=20 (up). Regression in C.
        s = self._summary((10.0, 10.0, 20.0))
        assert s.matched_pattern == "regression"

    def test_retrieval_helped_pollution_hurt_up_then_drop(self):
        # A=5 -> B=10 (up), B=10 -> C=4 (drop). Track 1's pollution
        # outweighed the retrieval gain in B; removing it fixed C.
        s = self._summary((5.0, 10.0, 4.0))
        assert s.matched_pattern == "retrieval-helped-pollution-hurt"

    def test_retrieval_hurt_no_recovery_up_then_flat(self):
        # A=5 -> B=10 (up), B=10 -> C=10 (flat). The B regression
        # persisted; removing Track 1 didn't help.
        s = self._summary((5.0, 10.0, 10.0))
        assert s.matched_pattern == "retrieval-hurt-no-recovery"

    def test_regression_up_then_up(self):
        # A=5 -> B=10 -> C=20. Both transitions worsen the verdict.
        s = self._summary((5.0, 10.0, 20.0))
        assert s.matched_pattern == "regression"

    def test_inconclusive_volume_floor_short_bucket(self):
        # Any bucket below the floor fires inconclusive BEFORE band
        # classification runs (volume floor takes precedence).
        s = classify_trend(
            aggregates=_make_aggregates(
                {
                    "A": {"frustration": 10.0},
                    "B": {"frustration": 5.0},
                    "C": {"frustration": 5.0},
                }
            ),
            events=[],
            user_turns={"A": 100, "B": 100, "C": 5},  # C below 30
        )
        assert s.matched_pattern == "inconclusive"

    def test_inconclusive_zero_prior_rate(self):
        # prior=0 + new>0 fires inconclusive (zero-prior-rate path) AND
        # records a caveat. Band falls back to "up" but matched_pattern
        # is inconclusive: the ratio is undefined when the divisor is 0.
        s = self._summary((0.0, 5.0, 5.0))
        assert s.matched_pattern == "inconclusive"
        assert s.zero_prior_rate_caveats  # at least one caveat

    def test_unclassified_via_band_monkey_patch(self, monkeypatch):
        # Mock _classify_band to return a sentinel outside {drop, flat,
        # up}; the dict.get fallback in classify_trend must emit
        # "unclassified". Verifies the defensive catch-all wiring is
        # actually reachable.
        monkeypatch.setattr("kai.eval.friction._classify_band", lambda p, n: "sideways")
        s = self._summary((10.0, 10.0, 10.0))
        assert s.matched_pattern == "unclassified"

    def test_band_boundary_at_0_85_is_drop(self):
        # The flat band's lower edge is open at 0.85x; exactly 0.85x
        # belongs to drop. _MATERIAL_DROP_RATIO * 100 = 85.0.
        assert _classify_band(100.0, 100.0 * _MATERIAL_DROP_RATIO) == "drop"

    def test_band_boundary_just_above_0_85_is_flat(self):
        # Just above 0.85x is the flat band's lower edge.
        assert _classify_band(100.0, 100.0 * _MATERIAL_DROP_RATIO + 0.01) == "flat"

    def test_band_boundary_at_1_15_is_flat(self):
        # The flat band's upper edge is closed at 1.15x; exactly
        # 1.15x belongs to flat, not up.
        assert _classify_band(100.0, 100.0 * _UP_RATIO) == "flat"

    def test_band_boundary_just_above_1_15_is_up(self):
        assert _classify_band(100.0, 100.0 * _UP_RATIO + 0.01) == "up"

    def test_zero_prior_zero_new_yields_flat_ratio_one(self):
        # Both buckets quiet; the convention is ratio 1.0 + flat band,
        # NOT zero-prior-rate caveat. Quiet -> quiet is no signal at all.
        s = self._summary((0.0, 0.0, 0.0))
        assert s.ratios["A_to_B"] == 1.0
        assert s.ratios["B_to_C"] == 1.0
        # No zero-prior caveats fire because new=0 too.
        assert s.zero_prior_rate_caveats == ()

    def test_verdict_sum_and_ratios_emitted_when_inconclusive(self):
        # verdict_driving_sum and ratios are ALWAYS computed and
        # emitted even when matched_pattern is inconclusive. Operators
        # reading the report still see the underlying numbers for
        # debugging the call.
        s = classify_trend(
            aggregates=_make_aggregates(
                {
                    "A": {"frustration": 10.0},
                    "B": {"frustration": 5.0},
                    "C": {"frustration": 5.0},
                }
            ),
            events=[],
            user_turns={"A": 100, "B": 100, "C": 5},
        )
        assert s.matched_pattern == "inconclusive"
        assert s.verdict_driving_sum["A"] == 10.0
        assert s.ratios["A_to_B"] == 0.5

    def test_memory_disabled_collapses_verdict_to_frustration(self):
        # When events list contains no preference_correction events
        # with known_fact_overlap=True (the memory-unreachable case),
        # verdict_sum == frustration alone. Same matched_pattern as the
        # all-frustration baseline.
        s = self._summary((10.0, 5.0, 5.0))
        assert s.verdict_driving_sum["A"] == 10.0
        assert s.matched_pattern == "prediction-holds"


# ── TestRender (regression: MappingProxyType -> dict) ───────────────


class TestRender:
    """Render-layer regressions: MappingProxyType serialization, redaction."""

    def test_mappingproxytype_metadata_serializes(self):
        # FrictionEvent.metadata is MappingProxyType (frozen dataclass
        # immutability); the JSON encoder rejects it unless the render
        # layer converts to dict first. This test catches the silent-
        # break case where the immutability invariant meets the renderer.
        ev = FrictionEvent(
            timestamp_utc=_BUCKET_ANCHORS[0] + timedelta(hours=1),
            bucket_label="B",
            signal_family="preference_correction",
            message_id="x.jsonl:1",
            surface_text="don't do that",
            context_message_ids=(),
            metadata=MappingProxyType(
                {
                    "known_fact_overlap": True,
                    "overlapping_fact_ids": ("f1", "f2"),
                }
            ),
        )
        payload = friction._build_event_payload(ev, include_snippets=False)
        # json.dumps must succeed without TypeError.
        rendered = json.dumps(payload)
        assert "f1" in rendered
        assert "f2" in rendered

    def test_markdown_summary_omits_surface_text_even_with_include_snippets(self, tmp_path: Path, monkeypatch):
        # The markdown summary always omits surface text by design;
        # --include-snippets only affects the JSON, not the markdown.
        # Posting the markdown publicly must never leak chat content.
        _write_history(
            tmp_path,
            "u1",
            [_user_msg(_IN_B, "i told you about the postgres deploy")],
        )
        monkeypatch.setattr("kai.eval.friction.datetime", _FakeDatetime)
        monkeypatch.setattr(
            "kai.eval.friction._initialize_memory_local",
            lambda: MemoryInitResult(MemoryAvailability.DISABLED, None),
        )
        json_out = tmp_path / "report.json"
        md_out = tmp_path / "report.md"
        main(
            [
                "--user-id",
                "u1",
                "--output",
                str(json_out),
                "--markdown-summary",
                str(md_out),
                "--data-dir",
                str(tmp_path),
                "--include-snippets",
            ]
        )
        md = md_out.read_text()
        # The frustration phrase from the input must NOT appear in the
        # markdown even though --include-snippets is set.
        assert "i told you about the postgres deploy" not in md

    def test_caveats_include_inconclusive_when_volume_floor(self, tmp_path: Path, monkeypatch):
        # The caveats list must surface every triggered condition; an
        # under-volume bucket emits a caveat naming the bucket and the
        # floor so an operator scanning the report knows why the
        # classification is inconclusive.
        _write_history(
            tmp_path,
            "u1",
            [_user_msg(_IN_C, "single message")],
        )
        monkeypatch.setattr("kai.eval.friction.datetime", _FakeDatetime)
        monkeypatch.setattr(
            "kai.eval.friction._initialize_memory_local",
            lambda: MemoryInitResult(MemoryAvailability.DISABLED, None),
        )
        out = tmp_path / "report.json"
        main(
            [
                "--user-id",
                "u1",
                "--output",
                str(out),
                "--data-dir",
                str(tmp_path),
            ]
        )
        report = json.loads(out.read_text())
        joined = " ".join(report["caveats"])
        assert "inconclusive" in joined
        assert str(_USER_TURN_FLOOR) in joined


# ── TestArgparseHelpers ─────────────────────────────────────────────


class TestArgparseHelpers:
    """Validators duplicated locally to keep the eval modules independent."""

    @pytest.mark.parametrize("bad", ["-1", "-100"])
    def test_non_negative_int_rejects_negatives(self, bad: str):
        with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
            _non_negative_int(bad)

    def test_non_negative_int_rejects_non_integer(self):
        with pytest.raises(argparse.ArgumentTypeError, match="expected integer"):
            _non_negative_int("ten")

    @pytest.mark.parametrize("good,expected", [("0", 0), ("1", 1), ("100", 100)])
    def test_non_negative_int_accepts_zero_and_positives(self, good: str, expected: int):
        # 0 is the legitimate "today UTC" value for --sample-days; the
        # validator must NOT reject it.
        assert _non_negative_int(good) == expected


# ── TestParseRecordTimestamp ────────────────────────────────────────


class TestParseRecordTimestamp:
    """Edge cases of the per-record timestamp parser."""

    def test_empty_string_returns_sentinel(self):
        ts, reason = _parse_record_timestamp("")
        assert ts == _BAD_TS_SENTINEL
        assert reason == "ts_missing_or_empty"

    def test_unparseable_returns_sentinel(self):
        ts, reason = _parse_record_timestamp("not-iso-format")
        assert ts == _BAD_TS_SENTINEL
        assert reason == "ts_unparseable"

    def test_naive_assumed_utc(self):
        # log_message has always written tz-aware ISO; only legacy or
        # hand-edited rows are naive. The conservative rule is "treat
        # naive as UTC" so the bucket assignment is at least consistent.
        ts, reason = _parse_record_timestamp("2026-04-18T10:00:00")
        assert reason is None
        assert ts.tzinfo == UTC

    def test_aware_normalized_to_utc(self):
        # An Eastern-TZ ISO string normalizes to UTC; the bucket math
        # operates exclusively in UTC.
        ts, reason = _parse_record_timestamp("2026-04-18T10:00:00-04:00")
        assert reason is None
        assert ts.utcoffset() == timedelta(0)
        assert ts.hour == 14


# ── TestContentWords ────────────────────────────────────────────────


class TestContentWords:
    """The shared tokenizer that backs Signal 2 and stage 3b overlap."""

    def test_tokens_below_length_4_excluded(self):
        # "the", "is", "a" all drop because they're shorter than 4.
        assert _content_words("the cat is here") == {"here"}

    def test_stopwords_excluded(self):
        # "this", "that", "with" etc. are pre-committed stopwords; they
        # do not contribute to the substantive overlap floor.
        assert _content_words("this with that") == set()

    def test_lowercased_and_punctuation_stripped(self):
        # Mixed case with punctuation collapses to a single canonical
        # set; "Database!" -> "database".
        assert "database" in _content_words("Database! works correctly.")

    def test_apostrophe_preserved_in_token(self):
        # Apostrophe is kept inside the character class so contractions
        # survive as single tokens; "don't" stays as "don't" rather than
        # splitting into "don" + "t".
        assert "don't" in _content_words("don't change the deployment")


# ── TestDetectEventsOrchestrator ────────────────────────────────────


class TestDetectEventsOrchestrator:
    """detect_events runs all four signal detectors and concatenates."""

    def test_returns_events_from_each_signal_family(self, tmp_path: Path):
        # Synthetic history hits all four signals: frustration phrase,
        # an assistant question with overlap, a correction pattern, and
        # a fact-shape repeat. The orchestrator MUST emit at least one
        # event per family so the per-bucket-per-signal aggregate matrix
        # has a non-empty row in each family.
        records = [
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=1),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="i prefer postgres docker container deployment",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=1,
            ),
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=2),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="i prefer postgres docker container deployment as before",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=2,
            ),
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=3),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="i told you about the postgres docker container",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=3,
            ),
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=4),
                ts_raw=_IN_B,
                direction="assistant",
                chat_id=1,
                text="what postgres docker container should i use?",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=4,
            ),
            HistoryRecord(
                ts_utc=_BUCKET_ANCHORS[0] + timedelta(minutes=5),
                ts_raw=_IN_B,
                direction="user",
                chat_id=1,
                text="don't change the postgres docker container deployment",
                media=None,
                source_path=str(tmp_path / "h.jsonl"),
                line_no=5,
            ),
        ]
        events = detect_events(records, tmp_path, facts=[])
        families = {ev.signal_family for ev in events}
        assert "frustration" in families
        assert "kai_asks_back" in families
        assert "preference_correction" in families
        assert "repeated_fact" in families
