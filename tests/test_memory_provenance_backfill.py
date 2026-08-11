"""Tests for the transcript-provenance backfill harness.

Covers the spec v2 test plan:

1. TestSelection - source filter, present-provenance exclusion.
2. TestOverlapMath - identical, disjoint, lifted-phrase cases.
3. TestGatingBuckets - STRONG_MATCH / AMBIGUOUS_OVERLAP / NO_CANDIDATE.
4. TestDelayedExtractionSafety - the v1 failure mode: true source is
   hours before created_at, an unrelated near-in-time turn exists.
   Overlap-based matching must pick the true source (or skip), NOT
   stamp the near-in-time turn.
5. TestApplyDriftGates - SKIP_ROW_GONE, SKIP_DESELECTED,
   SKIP_METADATA_DRIFT, SKIP_TRANSCRIPT_DRIFT.
6. TestApplyWritesFourFields - successful apply calls update_metadata
   with `data=row.text` (not `text=`), merges the four source_* keys,
   read_transcript_provenance(merged).present is True.
7. TestPreImageShape - dump line carries applied_source_block with
   the four required keys.
8. TestRollback - skip on operator correction; restore when block
   matches.
9. TestHeaderValidation - wrong-user file fails before any read.
10. TestCLIDispatch - --apply without --yes exits 2; scoring flags
    rejected in mutating modes.
11. TestHistoryUnreadable - whole row bucketed as HISTORY_UNREADABLE
    when any intersecting JSONL exists but cannot be read.
12. TestReportShape - report renders without IO; carries the right
    sections in stable order.
13. TestOptionalVerification - non-ok fetch_transcript_context after
    apply logs a warning but does not roll back.

Mock-only at the pipeline edges (memory.get_all, memory.get_by_id,
memory.update_metadata) and at the JSONL boundary (tmp_path + the
private `_LOG_DIR` redirect, same trick the reclassify tests use).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kai.config import Config
from kai.eval import retrieval_scoped  # noqa: F401  (force-import sibling so memory module loads with same constants)
from kai.memory import (
    SOURCE_CHAT_ID_KEY,
    SOURCE_DATE_KEY,
    SOURCE_USER_TEXT_SHA256_KEY,
    SOURCE_USER_TS_KEY,
    MemoryResult,
    read_transcript_provenance,
)
from kai.memory_provenance_backfill import (
    SKIP_AMBIGUOUS_OVERLAP,
    SKIP_NO_CANDIDATE,
    Candidate,
    PreImage,
    Proposal,
    _build_applied_source_block,
    _date_files_in_window,
    _gate_match,
    _metadata_fingerprint,
    _normalize_text,
    _overlap_score,
    _records_for_row_window,
    _score_candidates,
    _shingles,
    _verify_jsonl_assistant_text,
    _verify_jsonl_user_text,
    parse_preimages,
    parse_proposals,
    render_preimages,
    render_proposals,
    render_report,
    run_apply,
    run_dry_run,
    run_rollback,
    select_rows,
    validate_header,
)

# ── Shared fixtures ────────────────────────────────────────────────


_BASE_CONFIG = Config(
    telegram_bot_token="test-token",
    allowed_user_ids={12345},
    memory_enabled=True,
)


@pytest.fixture
def history_dir(tmp_path, monkeypatch):
    """Redirect the backfill module's `_LOG_DIR` to a tmp directory.

    The module imports `_LOG_DIR` from kai.history at import time, so
    the local binding is what tests must patch. Patching kai.history._LOG_DIR
    alone would not redirect the backfill module's reads.
    """
    from kai import memory_provenance_backfill as mod

    history_root = tmp_path / "history"
    monkeypatch.setattr(mod, "_LOG_DIR", history_root)
    return history_root


def _write_jsonl(history_root: Path, chat_id: int, date: str, records: list[dict]) -> Path:
    """Write a JSONL fixture file at the standard layout path."""
    user_dir = history_root / str(chat_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{date}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _user_rec(ts: str, text: str, chat_id: int = 1) -> dict:
    """One user-direction JSONL record."""
    return {"ts": ts, "dir": "user", "chat_id": chat_id, "text": text}


def _assistant_rec(ts: str, text: str, chat_id: int = 1) -> dict:
    """One assistant-direction JSONL record (Pass 2 fixtures)."""
    return {"ts": ts, "dir": "assistant", "chat_id": chat_id, "text": text}


def _alive_row(
    *,
    row_id: str,
    text: str,
    created_at: str,
    source: str = "extracted",
    metadata_extra: dict | None = None,
) -> MemoryResult:
    """Build a MemoryResult with the minimal metadata shape backfill reads."""
    metadata = {"source": source}
    if metadata_extra:
        metadata.update(metadata_extra)
    return MemoryResult(
        id=row_id,
        text=text,
        score=0.0,
        memory_type="fact",
        metadata=metadata,
        created_at=created_at,
    )


# ── Test 1: Selection ──────────────────────────────────────────────


class TestSelection:
    def test_eligible_source_and_not_present_provenance_passes(self):
        rows = [
            _alive_row(row_id="a", text="x", created_at="2026-06-13T00:00:00+00:00", source="extracted"),
            _alive_row(row_id="b", text="x", created_at="2026-06-13T00:00:00+00:00", source="episode"),
        ]
        assert [r.id for r in select_rows(rows)] == ["a", "b"]

    def test_migration_source_excluded(self):
        rows = [
            _alive_row(row_id="m", text="x", created_at="2026-06-13T00:00:00+00:00", source="migration"),
        ]
        assert select_rows(rows) == []

    def test_user_raw_source_excluded(self):
        rows = [
            _alive_row(row_id="u", text="x", created_at="2026-06-13T00:00:00+00:00", source="user_raw"),
        ]
        assert select_rows(rows) == []

    def test_present_provenance_excluded(self):
        rows = [
            _alive_row(
                row_id="p",
                text="x",
                created_at="2026-06-13T00:00:00+00:00",
                metadata_extra={
                    SOURCE_CHAT_ID_KEY: 1,
                    SOURCE_DATE_KEY: "2026-06-12",
                    SOURCE_USER_TS_KEY: "2026-06-12T10:00:00+00:00",
                    SOURCE_USER_TEXT_SHA256_KEY: "abc",
                },
            ),
        ]
        assert select_rows(rows) == []


# ── Test 2: Overlap math ───────────────────────────────────────────


class TestOverlapMath:
    def test_identical_texts_score_one(self):
        toks = _normalize_text("the quick brown fox jumps over the lazy dog")
        sh = _shingles(toks, 4)
        assert _overlap_score(sh, sh) == pytest.approx(1.0)

    def test_disjoint_texts_score_zero(self):
        a = _shingles(_normalize_text("alpha beta gamma delta epsilon"), 4)
        b = _shingles(_normalize_text("rho sigma tau upsilon phi"), 4)
        assert _overlap_score(a, b) == 0.0

    def test_lifted_phrase_scores_above_zero(self):
        # Row text quotes a 5-word phrase from a longer candidate. The
        # 4-token shingles of the row are entirely contained in the
        # candidate's shingles, so the overlap is 1.0 (intersection
        # over min, not Jaccard).
        row = _shingles(_normalize_text("we deploy on monday at noon"), 4)
        cand = _shingles(
            _normalize_text("I just want to confirm we deploy on monday at noon before the freeze starts on tuesday."),
            4,
        )
        assert _overlap_score(row, cand) == pytest.approx(1.0)

    def test_empty_text_scores_zero(self):
        a = _shingles(_normalize_text("anything goes here"), 4)
        b: set = set()
        assert _overlap_score(a, b) == 0.0
        assert _overlap_score(b, a) == 0.0


# ── Test 3: Gating buckets ─────────────────────────────────────────


class TestGatingBuckets:
    def _cand(self, ts: str, score: float) -> Candidate:
        return Candidate(ts=ts, text=f"text-{ts}", sha256="x", overlap_score=score, gap_seconds=10.0)

    def test_strong_match_with_no_runner_up(self):
        winner, bucket, runner_up = _gate_match(
            [self._cand("t1", 0.6)],
            min_overlap=0.3,
            strong_overlap_ratio=2.0,
        )
        assert bucket == "STRONG_MATCH"
        assert winner is not None and winner.ts == "t1"
        assert runner_up == 0.0

    def test_strong_match_with_dominant_runner_up(self):
        winner, bucket, runner_up = _gate_match(
            [self._cand("t1", 0.8), self._cand("t2", 0.3)],
            min_overlap=0.3,
            strong_overlap_ratio=2.0,
        )
        assert bucket == "STRONG_MATCH"
        assert winner is not None and winner.ts == "t1"
        assert runner_up == pytest.approx(0.3)

    def test_ambiguous_overlap_when_runner_up_close(self):
        winner, bucket, runner_up = _gate_match(
            [self._cand("t1", 0.6), self._cand("t2", 0.5)],
            min_overlap=0.3,
            strong_overlap_ratio=2.0,
        )
        assert bucket == SKIP_AMBIGUOUS_OVERLAP
        assert winner is None
        assert runner_up == pytest.approx(0.5)

    def test_no_candidate_when_max_below_min_overlap(self):
        winner, bucket, _runner_up = _gate_match(
            [self._cand("t1", 0.2), self._cand("t2", 0.1)],
            min_overlap=0.3,
            strong_overlap_ratio=2.0,
        )
        assert bucket == SKIP_NO_CANDIDATE
        assert winner is None

    def test_no_candidate_when_empty_list(self):
        winner, bucket, _ = _gate_match([], min_overlap=0.3, strong_overlap_ratio=2.0)
        assert bucket == SKIP_NO_CANDIDATE
        assert winner is None


# ── Test 4: Delayed-extraction safety ──────────────────────────────


class TestDelayedExtractionSafety:
    """The exact v1 failure mode the v2 design exists to prevent."""

    def test_true_source_picked_over_near_in_time_unrelated_turn(self, history_dir):
        # Memory row created at 10:08. True source was 8 hours earlier
        # at 02:00, lifted the phrase "deploy on monday at noon" from
        # the user. An unrelated user turn arrived at 10:07 (one
        # minute before created_at) talking about lunch. With timing-
        # based matching, the near-in-time turn wins. With overlap-
        # based matching, the true source wins because its text
        # actually shares phrasing with the row.
        chat_id = 1
        true_source_ts = "2026-06-13T02:00:00+00:00"
        near_in_time_ts = "2026-06-13T10:07:00+00:00"
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-13",
            [
                _user_rec(true_source_ts, "let us deploy on monday at noon if the build is green"),
                _user_rec(near_in_time_ts, "where should we go for lunch"),
            ],
        )

        row_text = "deploy on monday at noon"
        created_at_dt = datetime(2026, 6, 13, 10, 8, 0, tzinfo=UTC)
        records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is False
        scored = _score_candidates(row_text, records, shingle_n=4, created_at_dt=created_at_dt)
        winner, bucket, _ = _gate_match(scored, min_overlap=0.3, strong_overlap_ratio=2.0)
        assert bucket == "STRONG_MATCH"
        assert winner is not None
        assert winner.ts == true_source_ts

    def test_no_overlap_rows_skip_rather_than_match_near_in_time(self, history_dir):
        # Row text shares NO meaningful overlap with anything in the
        # window. The near-in-time turn is NOT a STRONG_MATCH; it goes
        # to NO_CANDIDATE.
        chat_id = 1
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-13",
            [
                _user_rec("2026-06-13T10:07:00+00:00", "where should we go for lunch"),
            ],
        )
        created_at_dt = datetime(2026, 6, 13, 10, 8, 0, tzinfo=UTC)
        records, _ = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        scored = _score_candidates(
            "the operator prefers strong type checking",
            records,
            shingle_n=4,
            created_at_dt=created_at_dt,
        )
        winner, bucket, _ = _gate_match(scored, min_overlap=0.3, strong_overlap_ratio=2.0)
        assert bucket == SKIP_NO_CANDIDATE
        assert winner is None


# ── Test 5: Apply-time drift gates ─────────────────────────────────


class TestApplyDriftGates:
    def _make_proposals_file(self, tmp_path: Path, proposals: list[Proposal]) -> Path:
        header = {"run_id": "bp-test", "user_id": "1", "generated_at": "2026-06-13T00:00:00+00:00"}
        path = tmp_path / "proposals.jsonl"
        path.write_text(render_proposals(header, proposals), encoding="utf-8")
        return path

    def _proposal(
        self,
        *,
        row_id: str = "m1",
        chat_id: int = 1,
        date: str = "2026-06-12",
        user_ts: str = "2026-06-12T10:00:00+00:00",
        user_text_sha: str = "abc",
        prior_fp: str = "fp-prior",
    ) -> Proposal:
        return Proposal(
            memory_id=row_id,
            chat_id=chat_id,
            date=date,
            user_ts=user_ts,
            user_text_sha256=user_text_sha,
            overlap_score=0.9,
            runner_up_overlap_score=0.1,
            candidate_count=2,
            gap_seconds=60.0,
            prior_metadata_sha256=prior_fp,
            row_text_snippet="row",
            candidate_text_snippet="cand",
        )

    def test_apply_rejects_proposal_with_mismatched_chat_id(self, tmp_path, history_dir):
        # Proposal carries chat_id=2 while the CLI runs for user_id="1".
        # The artifact-level user_id check passes (header user_id="1"),
        # but the per-proposal chat-id guard catches the mismatch and
        # refuses to apply BEFORE any store access.
        proposals_path = self._make_proposals_file(tmp_path, [self._proposal(chat_id=2)])
        get_calls: list = []
        update_calls: list = []
        with (
            patch(
                "kai.memory.get_by_id",
                side_effect=lambda **kwargs: get_calls.append(kwargs) or None,
            ),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: update_calls.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        assert code == 1
        # Neither read nor write reached Mem0.
        assert get_calls == []
        assert update_calls == []

    def test_skip_row_gone(self, tmp_path, history_dir):
        proposals_path = self._make_proposals_file(tmp_path, [self._proposal()])
        # get_by_id returns None -> SKIP_ROW_GONE; no writes.
        update_calls: list = []
        with (
            patch("kai.memory.get_by_id", return_value=None),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: update_calls.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        assert code == 0
        assert update_calls == []

    def test_skip_deselected_when_present_provenance(self, tmp_path, history_dir):
        proposals_path = self._make_proposals_file(tmp_path, [self._proposal()])
        row = _alive_row(
            row_id="m1",
            text="x",
            created_at="2026-06-13T00:00:00+00:00",
            metadata_extra={
                SOURCE_CHAT_ID_KEY: 1,
                SOURCE_DATE_KEY: "2026-06-12",
                SOURCE_USER_TS_KEY: "2026-06-12T10:00:00+00:00",
                SOURCE_USER_TEXT_SHA256_KEY: "abc",
            },
        )
        update_calls: list = []
        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: update_calls.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        assert code == 0
        assert update_calls == []

    def test_skip_metadata_drift(self, tmp_path, history_dir):
        # Build a row whose fingerprint will NOT match the proposal's.
        row = _alive_row(
            row_id="m1",
            text="row body",
            created_at="2026-06-13T00:00:00+00:00",
            metadata_extra={"session_id": "s2"},
        )
        proposals_path = self._make_proposals_file(tmp_path, [self._proposal(prior_fp="different-fingerprint")])
        update_calls: list = []
        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: update_calls.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        assert code == 0
        assert update_calls == []

    def test_skip_transcript_drift(self, tmp_path, history_dir):
        # Proposal's user_text_sha256 does not match the JSONL line.
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [_user_rec(user_ts, "current text on disk")],
        )
        row = _alive_row(row_id="m1", text="row body", created_at="2026-06-13T00:00:00+00:00")
        prior_fp = _metadata_fingerprint(row.metadata)
        proposals_path = self._make_proposals_file(
            tmp_path,
            [
                self._proposal(
                    user_text_sha="sha-from-an-earlier-snapshot",
                    prior_fp=prior_fp,
                )
            ],
        )
        update_calls: list = []
        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: update_calls.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        assert code == 0
        assert update_calls == []


# ── Test 6: Apply writes the four required fields ────────────────


class TestApplyWritesFourFields:
    def test_successful_apply_calls_update_metadata_with_data_keyword(self, tmp_path, history_dir):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        user_text = "the live user line text"
        _write_jsonl(history_dir, chat_id, "2026-06-12", [_user_rec(user_ts, user_text)])
        user_text_sha = hashlib.sha256(user_text.encode("utf-8")).hexdigest()

        row = _alive_row(row_id="m1", text="row body", created_at="2026-06-13T00:00:00+00:00")
        prior_fp = _metadata_fingerprint(row.metadata)
        header = {"run_id": "bp-test", "user_id": "1", "generated_at": "2026-06-13T00:00:00+00:00"}
        proposal = Proposal(
            memory_id="m1",
            chat_id=chat_id,
            date="2026-06-12",
            user_ts=user_ts,
            user_text_sha256=user_text_sha,
            overlap_score=0.9,
            runner_up_overlap_score=0.0,
            candidate_count=1,
            gap_seconds=60.0,
            prior_metadata_sha256=prior_fp,
            row_text_snippet="row",
            candidate_text_snippet="cand",
        )
        proposals_path = tmp_path / "proposals.jsonl"
        proposals_path.write_text(render_proposals(header, [proposal]), encoding="utf-8")

        captured: list[dict] = []

        def fake_update(**kwargs):
            captured.append(kwargs)
            return True

        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch("kai.memory.update_metadata", side_effect=fake_update),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        assert code == 0
        assert len(captured) == 1
        call = captured[0]
        # The v1 typo was `text=row.text`; v2 must use `data=`.
        assert "data" in call
        assert call["data"] == row.text
        assert "text" not in call
        merged = call["metadata"]
        # The four required source_* keys all land in the merged dict
        # and resolve to present provenance.
        assert merged[SOURCE_CHAT_ID_KEY] == chat_id
        assert merged[SOURCE_DATE_KEY] == "2026-06-12"
        assert merged[SOURCE_USER_TS_KEY] == user_ts
        assert merged[SOURCE_USER_TEXT_SHA256_KEY] == user_text_sha
        assert read_transcript_provenance(merged).present is True


# ── Test 7: PreImage shape ─────────────────────────────────────────


class TestPreImageShape:
    def test_dump_carries_applied_source_block(self, tmp_path, history_dir):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        user_text = "the user turn that produced the row"
        _write_jsonl(history_dir, chat_id, "2026-06-12", [_user_rec(user_ts, user_text)])
        user_text_sha = hashlib.sha256(user_text.encode("utf-8")).hexdigest()

        row = _alive_row(row_id="m1", text="row body", created_at="2026-06-13T00:00:00+00:00")
        prior_fp = _metadata_fingerprint(row.metadata)
        header = {"run_id": "bp-test", "user_id": "1", "generated_at": "2026-06-13T00:00:00+00:00"}
        proposal = Proposal(
            memory_id="m1",
            chat_id=chat_id,
            date="2026-06-12",
            user_ts=user_ts,
            user_text_sha256=user_text_sha,
            overlap_score=0.9,
            runner_up_overlap_score=0.0,
            candidate_count=1,
            gap_seconds=60.0,
            prior_metadata_sha256=prior_fp,
            row_text_snippet="row",
            candidate_text_snippet="cand",
        )
        proposals_path = tmp_path / "proposals.jsonl"
        proposals_path.write_text(render_proposals(header, [proposal]), encoding="utf-8")
        out_dir = tmp_path / "out"

        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch("kai.memory.update_metadata", return_value=True),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=out_dir))
        assert code == 0
        preimage_path = out_dir / "backfill-bp-test-preimages.jsonl"
        text = preimage_path.read_text(encoding="utf-8")
        header_line, *row_lines = [ln for ln in text.splitlines() if ln.strip()]
        assert json.loads(header_line)["run_id"] == "bp-test"
        assert len(row_lines) == 1
        row_obj = json.loads(row_lines[0])
        assert row_obj["type"] == "preimage"
        assert row_obj["memory_id"] == "m1"
        # The four required keys must all appear in applied_source_block.
        applied = row_obj["applied_source_block"]
        assert applied[SOURCE_CHAT_ID_KEY] == chat_id
        assert applied[SOURCE_DATE_KEY] == "2026-06-12"
        assert applied[SOURCE_USER_TS_KEY] == user_ts
        assert applied[SOURCE_USER_TEXT_SHA256_KEY] == user_text_sha

    def test_exclusive_creation_refuses_to_overwrite(self, tmp_path, history_dir):
        # Run apply once, then run it again with the same run_id; the
        # second run must abort without truncating the first run's
        # pre-image file.
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        user_text = "the user turn"
        _write_jsonl(history_dir, chat_id, "2026-06-12", [_user_rec(user_ts, user_text)])
        user_text_sha = hashlib.sha256(user_text.encode("utf-8")).hexdigest()
        row = _alive_row(row_id="m1", text="row body", created_at="2026-06-13T00:00:00+00:00")
        prior_fp = _metadata_fingerprint(row.metadata)
        header = {"run_id": "bp-test", "user_id": "1", "generated_at": "2026-06-13T00:00:00+00:00"}
        proposal = Proposal(
            memory_id="m1",
            chat_id=chat_id,
            date="2026-06-12",
            user_ts=user_ts,
            user_text_sha256=user_text_sha,
            overlap_score=0.9,
            runner_up_overlap_score=0.0,
            candidate_count=1,
            gap_seconds=60.0,
            prior_metadata_sha256=prior_fp,
            row_text_snippet="row",
            candidate_text_snippet="cand",
        )
        proposals_path = tmp_path / "proposals.jsonl"
        proposals_path.write_text(render_proposals(header, [proposal]), encoding="utf-8")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Pre-create the pre-image file with sentinel content.
        preimage_path = out_dir / "backfill-bp-test-preimages.jsonl"
        preimage_path.write_text("sentinel content", encoding="utf-8")

        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch("kai.memory.update_metadata", return_value=True) as upd,
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=out_dir))
        assert code == 1
        # No update_metadata call must have happened.
        assert upd.call_count == 0
        # The sentinel content must still be there - exclusive creation
        # refused to truncate the existing file.
        assert preimage_path.read_text(encoding="utf-8") == "sentinel content"


# ── Test 8: Rollback ───────────────────────────────────────────────


class TestRollback:
    def _make_preimage_file(self, tmp_path: Path, preimages: list[PreImage]) -> Path:
        header = {"run_id": "bp-test", "user_id": "1", "generated_at": "2026-06-13T00:00:00+00:00"}
        path = tmp_path / "preimages.jsonl"
        path.write_text(render_preimages(header, preimages), encoding="utf-8")
        return path

    def test_skip_operator_correction_when_source_block_changed(self, tmp_path):
        applied_block = {
            SOURCE_CHAT_ID_KEY: 1,
            SOURCE_DATE_KEY: "2026-06-12",
            SOURCE_USER_TS_KEY: "2026-06-12T10:00:00+00:00",
            SOURCE_USER_TEXT_SHA256_KEY: "abc",
        }
        preimage = PreImage(
            memory_id="m1",
            text="row body",
            metadata_before={"source": "extracted"},
            applied_source_block=applied_block,
        )
        path = self._make_preimage_file(tmp_path, [preimage])
        # Current row's source_* fields have been changed by another
        # writer (the operator manually corrected the user_ts).
        row = _alive_row(
            row_id="m1",
            text="row body",
            created_at="2026-06-13T00:00:00+00:00",
            metadata_extra={
                SOURCE_CHAT_ID_KEY: 1,
                SOURCE_DATE_KEY: "2026-06-12",
                SOURCE_USER_TS_KEY: "2026-06-12T11:00:00+00:00",  # changed!
                SOURCE_USER_TEXT_SHA256_KEY: "abc",
            },
        )
        update_calls: list = []
        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: update_calls.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_rollback(_BASE_CONFIG, "1", preimages_path=path))
        # Skipped, no writes.
        assert code == 0
        assert update_calls == []

    def test_restores_when_source_block_matches(self, tmp_path):
        applied_block = {
            SOURCE_CHAT_ID_KEY: 1,
            SOURCE_DATE_KEY: "2026-06-12",
            SOURCE_USER_TS_KEY: "2026-06-12T10:00:00+00:00",
            SOURCE_USER_TEXT_SHA256_KEY: "abc",
        }
        metadata_before = {"source": "extracted", "session_id": "s1"}
        preimage = PreImage(
            memory_id="m1",
            text="row body",
            metadata_before=metadata_before,
            applied_source_block=applied_block,
        )
        path = self._make_preimage_file(tmp_path, [preimage])
        # Current row has the applied block intact: rollback proceeds.
        current_md = dict(metadata_before)
        current_md.update(applied_block)
        row = _alive_row(
            row_id="m1",
            text="row body",
            created_at="2026-06-13T00:00:00+00:00",
            metadata_extra={**current_md, "source": "extracted"},
        )
        # Override source so the metadata exactly matches what apply would have produced.
        row = MemoryResult(
            id="m1",
            text="row body",
            score=0.0,
            memory_type="fact",
            metadata={**metadata_before, **applied_block},
            created_at="2026-06-13T00:00:00+00:00",
        )
        captured: list[dict] = []
        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: captured.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_rollback(_BASE_CONFIG, "1", preimages_path=path))
        assert code == 0
        assert len(captured) == 1
        # update_metadata uses data= keyword and metadata=metadata_before.
        assert captured[0]["data"] == "row body"
        assert captured[0]["metadata"] == metadata_before

    def test_rollback_exit_one_when_only_attempted_write_failed(self, tmp_path):
        # Two preimages: one drifts (skipped as operator_correction),
        # one matches and is attempted but update_metadata returns
        # False. With the old "exit 1 only when no skips" rule, the
        # run would have exited 0; the corrected rule (attempted =
        # restored + failed; exit 1 when attempted and restored == 0)
        # exits 1.
        applied_block = {
            SOURCE_CHAT_ID_KEY: 1,
            SOURCE_DATE_KEY: "2026-06-12",
            SOURCE_USER_TS_KEY: "2026-06-12T10:00:00+00:00",
            SOURCE_USER_TEXT_SHA256_KEY: "abc",
        }
        metadata_before = {"source": "extracted"}
        preimages = [
            PreImage(
                memory_id="drifted",
                text="row body",
                metadata_before=metadata_before,
                applied_source_block=applied_block,
            ),
            PreImage(
                memory_id="failed",
                text="row body",
                metadata_before=metadata_before,
                applied_source_block=applied_block,
            ),
        ]
        path = self._make_preimage_file(tmp_path, preimages)

        def fake_get(*, user_id: str, memory_id: str):
            if memory_id == "drifted":
                # current source block differs -> SKIP_OPERATOR_CORRECTION
                drifted_block = dict(applied_block)
                drifted_block[SOURCE_USER_TS_KEY] = "2026-06-12T11:00:00+00:00"
                return MemoryResult(
                    id="drifted",
                    text="row body",
                    score=0.0,
                    memory_type="fact",
                    metadata={**metadata_before, **drifted_block},
                    created_at="2026-06-13T00:00:00+00:00",
                )
            return MemoryResult(
                id="failed",
                text="row body",
                score=0.0,
                memory_type="fact",
                metadata={**metadata_before, **applied_block},
                created_at="2026-06-13T00:00:00+00:00",
            )

        with (
            patch("kai.memory.get_by_id", side_effect=fake_get),
            patch("kai.memory.update_metadata", return_value=False),
        ):
            code = asyncio.run(run_rollback(_BASE_CONFIG, "1", preimages_path=path))
        assert code == 1

    def test_rollback_rejects_preimage_with_mismatched_chat_id(self, tmp_path):
        # PreImage's applied_source_block carries chat_id=2 while the
        # CLI runs for user_id="1". The pre-image guard rejects the
        # whole rollback before any store access.
        applied_block = {
            SOURCE_CHAT_ID_KEY: 2,  # mismatched
            SOURCE_DATE_KEY: "2026-06-12",
            SOURCE_USER_TS_KEY: "2026-06-12T10:00:00+00:00",
            SOURCE_USER_TEXT_SHA256_KEY: "abc",
        }
        preimage = PreImage(
            memory_id="m1",
            text="row body",
            metadata_before={"source": "extracted"},
            applied_source_block=applied_block,
        )
        path = self._make_preimage_file(tmp_path, [preimage])
        update_calls: list = []
        with (
            patch("kai.memory.get_by_id", return_value=None),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: update_calls.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_rollback(_BASE_CONFIG, "1", preimages_path=path))
        assert code == 1
        assert update_calls == []


# ── Test 9: Header validation ──────────────────────────────────────


class TestHeaderValidation:
    def test_validate_header_rejects_wrong_user(self):
        assert validate_header({"run_id": "bp-x", "user_id": "2"}, user_id="1") is not None

    def test_validate_header_accepts_matching_user(self):
        assert validate_header({"run_id": "bp-x", "user_id": "1"}, user_id="1") is None

    def test_validate_header_rejects_missing_run_id(self):
        assert validate_header({"user_id": "1"}, user_id="1") is not None


# ── Test 10: CLI dispatch ──────────────────────────────────────────


class TestCLIDispatch:
    def test_scoring_flags_rejected_in_mutating_mode(self, tmp_path, capsys):
        from kai.memory_admin import _cmd_backfill_provenance

        ns = SimpleNamespace(
            user_id="1",
            window_seconds=3600,  # explicit scoring flag in mutating mode
            min_overlap=None,
            strong_overlap_ratio=None,
            overlap_shingle_n=None,
            no_assistant_pass=None,
            assistant_max_user_gap_seconds=None,
            sample=None,
            out_dir=None,
            apply=str(tmp_path / "p.jsonl"),
            rollback=None,
            yes=False,
        )
        code = _cmd_backfill_provenance(ns)
        assert code == 2
        err = capsys.readouterr().err
        assert "--window-seconds" in err

    def test_apply_without_yes_exits_two(self, tmp_path, history_dir, capsys):
        # The plan-only path: --apply with a valid proposals file but
        # no --yes prints the planned change count and exits 2.
        proposal = Proposal(
            memory_id="m1",
            chat_id=1,
            date="2026-06-12",
            user_ts="2026-06-12T10:00:00+00:00",
            user_text_sha256="abc",
            overlap_score=0.9,
            runner_up_overlap_score=0.0,
            candidate_count=1,
            gap_seconds=60.0,
            prior_metadata_sha256="fp",
            row_text_snippet="row",
            candidate_text_snippet="cand",
        )
        proposals_path = tmp_path / "proposals.jsonl"
        header = {"run_id": "bp-test", "user_id": "1", "generated_at": "2026-06-13T00:00:00+00:00"}
        proposals_path.write_text(render_proposals(header, [proposal]), encoding="utf-8")

        from kai.memory_admin import _cmd_backfill_provenance

        ns = SimpleNamespace(
            user_id="1",
            window_seconds=None,
            min_overlap=None,
            strong_overlap_ratio=None,
            overlap_shingle_n=None,
            no_assistant_pass=None,
            assistant_max_user_gap_seconds=None,
            sample=None,
            out_dir=str(tmp_path / "out"),
            apply=str(proposals_path),
            rollback=None,
            yes=False,
        )
        # _initialize_memory needs memory to look enabled.
        with (
            patch("kai.memory_admin._initialize_memory", return_value=_BASE_CONFIG),
        ):
            code = _cmd_backfill_provenance(ns)
        assert code == 2
        out = capsys.readouterr().out
        assert "would apply 1 row" in out


# ── Test 11: HISTORY_UNREADABLE ────────────────────────────────────


class TestHistoryUnreadable:
    def test_unreadable_file_buckets_whole_row(self, history_dir, monkeypatch):
        # Create a file that exists but raises on read.
        chat_id = 1
        path = _write_jsonl(history_dir, chat_id, "2026-06-13", [_user_rec("2026-06-13T01:00:00+00:00", "x")])

        # Patch _read_jsonl_records to fail for this specific file.
        from kai import memory_provenance_backfill as mod

        original = mod._read_jsonl_records

        def fail_for(p: Path):
            if p == path:
                return None
            return original(p)

        monkeypatch.setattr(mod, "_read_jsonl_records", fail_for)

        created_at_dt = datetime(2026, 6, 13, 2, 0, 0, tzinfo=UTC)
        _records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is True
        # records may carry entries from other files in the window
        # (a multi-day window over partially-readable history), but
        # the any_unreadable flag is the one that triggers the whole-
        # row HISTORY_UNREADABLE bucket in the driver.

    def test_malformed_jsonl_line_collapses_file_to_unreadable(self, history_dir):
        # An intersecting file exists but contains a malformed JSON
        # line. Silently skipping the bad line would let scoring run
        # against partial history; the contract is fail-closed at the
        # file level. The row buckets as HISTORY_UNREADABLE.
        chat_id = 1
        user_dir = history_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / "2026-06-13.jsonl"
        # One valid line, one malformed line. The valid line alone
        # would NOT be a STRONG_MATCH (no overlap), but the point is
        # that the file is treated as unreadable regardless of which
        # candidates remain after skipping the bad line.
        path.write_text(
            json.dumps({"ts": "2026-06-13T01:00:00+00:00", "dir": "user", "chat_id": 1, "text": "x"})
            + "\n"
            + "this line is not valid json\n",
            encoding="utf-8",
        )
        created_at_dt = datetime(2026, 6, 13, 2, 0, 0, tzinfo=UTC)
        _records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is True

    def test_malformed_user_ts_collapses_file_to_unreadable(self, history_dir):
        # Same partial-history hazard, one layer deeper: a user record
        # whose ts field is missing or unparseable would otherwise be
        # silently dropped, leaving an unrelated valid candidate free
        # to win the overlap. _records_for_row_window must set
        # any_unreadable=True so the row buckets HISTORY_UNREADABLE.
        chat_id = 1
        user_dir = history_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / "2026-06-13.jsonl"
        path.write_text(
            # Valid user record we don't want to silently lean on.
            json.dumps({"ts": "2026-06-13T01:00:00+00:00", "dir": "user", "chat_id": 1, "text": "alpha"})
            + "\n"
            # User record with missing ts.
            + json.dumps({"dir": "user", "chat_id": 1, "text": "beta"})
            + "\n",
            encoding="utf-8",
        )
        created_at_dt = datetime(2026, 6, 13, 2, 0, 0, tzinfo=UTC)
        _records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is True

    def test_unparseable_user_ts_collapses_file_to_unreadable(self, history_dir):
        chat_id = 1
        user_dir = history_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / "2026-06-13.jsonl"
        path.write_text(
            # User record with a non-ISO ts string.
            json.dumps({"ts": "this is not a timestamp", "dir": "user", "chat_id": 1, "text": "alpha"}) + "\n",
            encoding="utf-8",
        )
        created_at_dt = datetime(2026, 6, 13, 2, 0, 0, tzinfo=UTC)
        _records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is True

    def test_dry_run_buckets_history_unreadable_when_true_source_has_bad_ts(self, tmp_path, history_dir):
        # End-to-end: the true source user record has a malformed ts,
        # and a tempting valid candidate sits next to it with text that
        # would otherwise share strong overlap with the row. Dry-run
        # must bucket the row HISTORY_UNREADABLE, NOT STRONG_MATCH
        # against the tempting candidate.
        from kai.memory_provenance_backfill import run_dry_run

        chat_id = 1
        user_dir = history_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / "2026-06-13.jsonl"
        # Bad-ts record carries text that mirrors the row content.
        # Tempting valid candidate shares strong overlap with the row.
        path.write_text(
            json.dumps({"ts": "not a real timestamp", "dir": "user", "chat_id": 1, "text": "true source content"})
            + "\n"
            + json.dumps(
                {
                    "ts": "2026-06-13T01:30:00+00:00",
                    "dir": "user",
                    "chat_id": 1,
                    "text": "deploy the new build on monday at noon and watch the staging logs",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        row = _alive_row(
            row_id="m1",
            text="deploy the new build on monday at noon",
            created_at="2026-06-13T02:00:00+00:00",
        )
        out_dir = tmp_path / "out"
        with patch("kai.memory.get_all", return_value=[row]):
            code = asyncio.run(
                run_dry_run(
                    _BASE_CONFIG,
                    str(chat_id),
                    window_seconds=86400,
                    min_overlap=0.3,
                    strong_overlap_ratio=2.0,
                    shingle_n=4,
                    sample=10,
                    out_dir=out_dir,
                )
            )
        assert code == 0
        # The proposals file must be empty (header only). The tempting
        # candidate's strong overlap with the row is NOT what we want
        # to act on; the corrupt file makes scoring untrustworthy.
        proposals_path = next(out_dir.glob("backfill-*-proposals.jsonl"))
        lines = [ln for ln in proposals_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        # Header only, no proposal lines.
        assert len(lines) == 1
        assert json.loads(lines[0])["type"] == "header"

    def test_date_files_iterate_every_day_in_window(self, tmp_path):
        chat_id = 1
        start = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 6, 13, 23, 59, 0, tzinfo=UTC)
        # The driver patches _LOG_DIR; here we just check the helper's
        # math against a sentinel root.
        from kai import memory_provenance_backfill as mod

        original = mod._LOG_DIR
        try:
            mod._LOG_DIR = tmp_path / "history"
            files = _date_files_in_window(chat_id, start, end)
        finally:
            mod._LOG_DIR = original
        assert [f.name for f in files] == [
            "2026-06-10.jsonl",
            "2026-06-11.jsonl",
            "2026-06-12.jsonl",
            "2026-06-13.jsonl",
        ]


# ── Test 12: Report shape ──────────────────────────────────────────


class TestReportShape:
    def test_report_renders_sections_in_stable_order(self):
        from kai.memory_provenance_backfill import RowMatch

        row = _alive_row(row_id="m1", text="row body", created_at="2026-06-13T00:00:00+00:00")
        proposal = Proposal(
            memory_id="m1",
            chat_id=1,
            date="2026-06-12",
            user_ts="2026-06-12T10:00:00+00:00",
            user_text_sha256="abc",
            overlap_score=0.9,
            runner_up_overlap_score=0.0,
            candidate_count=1,
            gap_seconds=60.0,
            prior_metadata_sha256="fp",
            row_text_snippet="row body",
            candidate_text_snippet="cand text",
        )
        matches = [
            RowMatch(
                row=row,
                bucket="STRONG_MATCH",
                top_candidates=[
                    Candidate(
                        ts="2026-06-12T10:00:00+00:00",
                        text="cand text",
                        sha256="abc",
                        overlap_score=0.9,
                        gap_seconds=60.0,
                    )
                ],
                proposal=proposal,
            ),
        ]
        report = render_report(
            run_id="bp-test",
            user_id="1",
            window_seconds=86400,
            min_overlap=0.3,
            strong_overlap_ratio=2.0,
            shingle_n=4,
            assistant_pass_enabled=True,
            assistant_max_user_gap_seconds=600,
            scanned=10,
            selected=1,
            matches=matches,
            skips={"strong_match": ["m1"]},
            sample_size=10,
        )
        # Heading + parameters + counts + sample sections present.
        assert "# Backfill provenance dry-run report: bp-test" in report
        assert "User: 1" in report
        assert "scanned: 10" in report
        assert "proposals (STRONG_MATCH): 1" in report
        assert "## Sample of STRONG_MATCH proposals" in report
        assert "m1 (STRONG_MATCH)" in report


# ── Test 13: Optional verification on apply ────────────────────────


class TestOptionalVerification:
    def test_non_ok_lookup_logs_warning_not_rollback(self, tmp_path, history_dir, caplog):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        user_text = "lifted phrase about timing matters"
        _write_jsonl(history_dir, chat_id, "2026-06-12", [_user_rec(user_ts, user_text)])
        user_text_sha = hashlib.sha256(user_text.encode("utf-8")).hexdigest()

        row = _alive_row(row_id="m1", text="row body", created_at="2026-06-13T00:00:00+00:00")
        prior_fp = _metadata_fingerprint(row.metadata)
        header = {"run_id": "bp-test", "user_id": "1", "generated_at": "2026-06-13T00:00:00+00:00"}
        proposal = Proposal(
            memory_id="m1",
            chat_id=chat_id,
            date="2026-06-12",
            user_ts=user_ts,
            user_text_sha256=user_text_sha,
            overlap_score=0.9,
            runner_up_overlap_score=0.0,
            candidate_count=1,
            gap_seconds=60.0,
            prior_metadata_sha256=prior_fp,
            row_text_snippet="row",
            candidate_text_snippet="cand",
        )
        proposals_path = tmp_path / "proposals.jsonl"
        proposals_path.write_text(render_proposals(header, [proposal]), encoding="utf-8")

        captured: list[dict] = []

        def fake_update(**kwargs):
            captured.append(kwargs)
            return True

        # Force fetch_transcript_context to return a non-ok reason.
        fake_lookup = SimpleNamespace(reason="ts_not_found", context=None)
        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch("kai.memory.update_metadata", side_effect=fake_update),
            patch("kai.memory_provenance_backfill.fetch_transcript_context", return_value=fake_lookup),
            caplog.at_level("WARNING"),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        # The write still landed (no rollback on verification miss).
        assert code == 0
        assert len(captured) == 1
        # A warning was emitted naming the miss.
        assert any("verification miss" in rec.message for rec in caplog.records)


# ── Test 14: Bonus: proposal+preimage round-trip ───────────────────


class TestArtifactRoundTrip:
    """Strict parsing rejects hand-edits that miss required fields."""

    def test_proposal_missing_applied_field_rejected(self, tmp_path):
        # Build a proposals file by hand without overlap_score; parsing
        # must raise ValueError naming the bad line.
        header = {"type": "header", "run_id": "bp-x", "user_id": "1"}
        bad_proposal = {
            "type": "proposal",
            "memory_id": "m1",
            "chat_id": 1,
            "date": "2026-06-12",
            "user_ts": "2026-06-12T10:00:00+00:00",
            "user_text_sha256": "abc",
            # overlap_score missing
            "candidate_count": 1,
            "gap_seconds": 60.0,
            "prior_metadata_sha256": "fp",
        }
        text = json.dumps(header) + "\n" + json.dumps(bad_proposal) + "\n"
        with pytest.raises(ValueError, match="overlap_score"):
            parse_proposals(text)

    def test_preimage_missing_applied_source_block_key_rejected(self, tmp_path):
        # Build a preimage line whose applied_source_block lacks the
        # date key; parser must reject before any rollback dispatch.
        header = {"type": "header", "run_id": "bp-x", "user_id": "1"}
        bad_preimage = {
            "type": "preimage",
            "memory_id": "m1",
            "text": "row",
            "metadata_before": {"source": "extracted"},
            "applied_source_block": {
                SOURCE_CHAT_ID_KEY: 1,
                # SOURCE_DATE_KEY missing
                SOURCE_USER_TS_KEY: "2026-06-12T10:00:00+00:00",
                SOURCE_USER_TEXT_SHA256_KEY: "abc",
            },
        }
        text = json.dumps(header) + "\n" + json.dumps(bad_preimage) + "\n"
        with pytest.raises(ValueError, match=SOURCE_DATE_KEY):
            parse_preimages(text)


# ── Bonus: dry-run end-to-end ───────────────────────────────────────


class TestDryRunEndToEnd:
    def test_dry_run_emits_strong_match_proposal(self, tmp_path, history_dir):
        chat_id = 1
        user_ts = "2026-06-13T01:00:00+00:00"
        user_text = "deploy the new build on monday at noon if the staging looks green"
        _write_jsonl(history_dir, chat_id, "2026-06-13", [_user_rec(user_ts, user_text)])

        row = _alive_row(
            row_id="m1",
            text="deploy the new build on monday at noon",
            created_at="2026-06-13T01:05:00+00:00",
        )
        out_dir = tmp_path / "out"

        with patch("kai.memory.get_all", return_value=[row]):
            code = asyncio.run(
                run_dry_run(
                    _BASE_CONFIG,
                    str(chat_id),
                    window_seconds=86400,
                    min_overlap=0.3,
                    strong_overlap_ratio=2.0,
                    shingle_n=4,
                    sample=10,
                    out_dir=out_dir,
                )
            )
        assert code == 0
        proposals_files = list(out_dir.glob("backfill-*-proposals.jsonl"))
        assert len(proposals_files) == 1
        text = proposals_files[0].read_text(encoding="utf-8")
        header_line, *rows = [ln for ln in text.splitlines() if ln.strip()]
        assert json.loads(header_line)["user_id"] == "1"
        assert len(rows) == 1
        proposal = json.loads(rows[0])
        assert proposal["memory_id"] == "m1"
        assert proposal["chat_id"] == 1
        assert proposal["user_ts"] == user_ts
        # overlap_score is positive; runner-up is zero (single candidate).
        assert proposal["overlap_score"] > 0.0
        assert proposal["runner_up_overlap_score"] == 0.0


# ── Bonus: _build_applied_source_block ─────────────────────────────


class TestAppliedSourceBlock:
    def test_packs_four_keys_from_proposal(self):
        proposal = Proposal(
            memory_id="m1",
            chat_id=42,
            date="2026-06-12",
            user_ts="2026-06-12T10:00:00+00:00",
            user_text_sha256="abc",
            overlap_score=0.9,
            runner_up_overlap_score=0.0,
            candidate_count=1,
            gap_seconds=60.0,
            prior_metadata_sha256="fp",
            row_text_snippet="row",
            candidate_text_snippet="cand",
        )
        block = _build_applied_source_block(proposal)
        assert block == {
            SOURCE_CHAT_ID_KEY: 42,
            SOURCE_DATE_KEY: "2026-06-12",
            SOURCE_USER_TS_KEY: "2026-06-12T10:00:00+00:00",
            SOURCE_USER_TEXT_SHA256_KEY: "abc",
        }


# ── Pass 2 (assistant-turn matching) ──────────────────────────────
#
# The next ten test classes cover Pass 2 end-to-end: the full-window
# cache, the algorithm (overlap on assistant turns + conservative
# pairing rule), per-record validation on both directions, the
# apply-time assistant drift gate, parser gates on the new fields,
# CLI flags, the no-double-read property, the user-only-fields
# diff projection, and the report shape.


# Shared row/text constants for Pass 2 tests. The user text and
# assistant text are picked so the row text overlaps the ASSISTANT
# turn's phrasing (4-token shingles) but does NOT overlap the user
# turn. That's the failure mode Pass 2 exists to catch: a row whose
# only meaningful overlap in the window is with an assistant turn.
_PASS2_ROW_TEXT = "operator prefers strong type checking with pyright strict mode"
_PASS2_USER_TEXT = "what should we use for static analysis on this codebase"
_PASS2_ASSISTANT_TEXT = (
    "given your conventions the operator prefers strong type checking with pyright strict mode for the kai code"
)


def _make_pass2_proposal(
    *,
    row_id: str = "m1",
    chat_id: int = 1,
    user_date: str = "2026-06-12",
    user_ts: str = "2026-06-12T10:00:00+00:00",
    user_text_sha: str = "user-sha",
    assistant_ts: str = "2026-06-12T10:01:00+00:00",
    assistant_text_sha: str = "assistant-sha",
    prior_fp: str = "fp-prior",
) -> Proposal:
    """Build a fully-populated Pass 2 proposal for apply-time tests."""
    return Proposal(
        memory_id=row_id,
        chat_id=chat_id,
        date=user_date,
        user_ts=user_ts,
        user_text_sha256=user_text_sha,
        overlap_score=0.9,
        runner_up_overlap_score=0.0,
        candidate_count=1,
        gap_seconds=60.0,
        prior_metadata_sha256=prior_fp,
        row_text_snippet="row",
        candidate_text_snippet="user-text",
        match_pass="assistant",
        assistant_ts=assistant_ts,
        assistant_text_sha256=assistant_text_sha,
        assistant_text_snippet="assistant-text",
    )


def _make_proposals_file(tmp_path: Path, proposals: list[Proposal]) -> Path:
    """Render a proposals file for apply tests."""
    header = {"run_id": "bp-test", "user_id": "1", "generated_at": "2026-06-13T00:00:00+00:00"}
    path = tmp_path / "proposals.jsonl"
    path.write_text(render_proposals(header, proposals), encoding="utf-8")
    return path


# ── Test 1: Full-window cache includes assistant records ──────────


class TestFullWindowCache:
    """The window reader retains assistant records so Pass 1 and
    Pass 2 share a single disk read per JSONL file in the window."""

    def test_window_returns_both_directions_in_order(self, history_dir):
        chat_id = 1
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-13",
            [
                _user_rec("2026-06-13T01:00:00+00:00", "user one"),
                _assistant_rec("2026-06-13T01:01:00+00:00", "assistant one"),
                _user_rec("2026-06-13T01:30:00+00:00", "user two"),
                _assistant_rec("2026-06-13T01:31:00+00:00", "assistant two"),
            ],
        )
        created_at_dt = datetime(2026, 6, 13, 2, 0, 0, tzinfo=UTC)
        records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is False
        assert [r["dir"] for r in records] == ["user", "assistant", "user", "assistant"]
        assert [r["text"] for r in records] == ["user one", "assistant one", "user two", "assistant two"]

    def test_malformed_assistant_ts_collapses_window(self, history_dir):
        # An assistant record with a malformed ts must collapse the row
        # even when every user record is clean. The reader's
        # per-record validation contract covers both directions so a
        # malformed assistant turn cannot hide the boundary Pass 2's
        # pairing walk depends on.
        chat_id = 1
        user_dir = history_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / "2026-06-13.jsonl"
        path.write_text(
            json.dumps({"ts": "2026-06-13T01:00:00+00:00", "dir": "user", "chat_id": 1, "text": "u1"})
            + "\n"
            + json.dumps({"ts": "not-a-timestamp", "dir": "assistant", "chat_id": 1, "text": "a1"})
            + "\n",
            encoding="utf-8",
        )
        created_at_dt = datetime(2026, 6, 13, 2, 0, 0, tzinfo=UTC)
        _records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is True


# ── Tests 2-8: Pass 2 algorithm ───────────────────────────────────


class TestPass2Algorithm:
    """Pass 2 end-to-end through `run_dry_run`: overlap on assistant
    turns + conservative pairing rule + max-gap guard."""

    def _dry_run(self, history_dir: Path, tmp_path: Path, row: MemoryResult, **kwargs) -> Path:
        out_dir = tmp_path / "out"
        with patch("kai.memory.get_all", return_value=[row]):
            code = asyncio.run(
                run_dry_run(
                    _BASE_CONFIG,
                    "1",
                    window_seconds=kwargs.get("window_seconds", 86400),
                    min_overlap=kwargs.get("min_overlap", 0.3),
                    strong_overlap_ratio=kwargs.get("strong_overlap_ratio", 2.0),
                    shingle_n=kwargs.get("shingle_n", 4),
                    sample=kwargs.get("sample", 10),
                    out_dir=out_dir,
                    assistant_pass_enabled=kwargs.get("assistant_pass_enabled", True),
                    assistant_max_user_gap_seconds=kwargs.get("assistant_max_user_gap_seconds", 600),
                )
            )
        assert code == 0
        return out_dir

    def _proposals_in(self, out_dir: Path) -> list[dict]:
        proposals_path = next(out_dir.glob("backfill-*-proposals.jsonl"))
        text = proposals_path.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # First line is the header; everything after is one proposal.
        return [json.loads(ln) for ln in lines[1:]]

    def _report_in(self, out_dir: Path) -> str:
        report_path = next(out_dir.glob("backfill-*-report.md"))
        return report_path.read_text(encoding="utf-8")

    # Test 2.
    def test_pass2_finds_paired_user_via_assistant_overlap(self, history_dir, tmp_path):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        assistant_ts = "2026-06-12T10:01:00+00:00"
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec(user_ts, _PASS2_USER_TEXT),
                _assistant_rec(assistant_ts, _PASS2_ASSISTANT_TEXT),
            ],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")
        out_dir = self._dry_run(history_dir, tmp_path, row)
        proposals = self._proposals_in(out_dir)
        assert len(proposals) == 1
        p = proposals[0]
        assert p["match_pass"] == "assistant"
        assert p["user_ts"] == user_ts
        assert p["assistant_ts"] == assistant_ts
        assert p["user_text_sha256"]  # non-empty
        assert p["assistant_text_sha256"]  # non-empty
        # The proposal date is the USER ts's UTC date, not the assistant's.
        assert p["date"] == "2026-06-12"

    # Test 3.
    def test_pass2_walks_across_midnight_boundary(self, history_dir, tmp_path):
        chat_id = 1
        user_ts = "2026-06-12T23:55:00+00:00"
        assistant_ts = "2026-06-13T00:01:00+00:00"
        _write_jsonl(history_dir, chat_id, "2026-06-12", [_user_rec(user_ts, _PASS2_USER_TEXT)])
        _write_jsonl(history_dir, chat_id, "2026-06-13", [_assistant_rec(assistant_ts, _PASS2_ASSISTANT_TEXT)])
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-13T00:05:00+00:00")
        out_dir = self._dry_run(history_dir, tmp_path, row)
        proposals = self._proposals_in(out_dir)
        assert len(proposals) == 1
        p = proposals[0]
        assert p["match_pass"] == "assistant"
        assert p["user_ts"] == user_ts
        assert p["assistant_ts"] == assistant_ts
        # The date stamped onto the row is the USER's UTC date, even
        # though the assistant turn rolled into the next day.
        assert p["date"] == "2026-06-12"

    # Test 4.
    def test_pass2_skips_when_no_preceding_user(self, history_dir, tmp_path):
        chat_id = 1
        # Assistant turn alone in the window. No prior user means the
        # pairing rule has no source to bind the winner to.
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [_assistant_rec("2026-06-12T10:01:00+00:00", _PASS2_ASSISTANT_TEXT)],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")
        out_dir = self._dry_run(history_dir, tmp_path, row)
        proposals = self._proposals_in(out_dir)
        assert proposals == []
        report = self._report_in(out_dir)
        assert "skipped no_preceding_user: 1" in report

    # Test 5.
    def test_pass2_skips_when_preceding_user_exceeds_max_gap(self, history_dir, tmp_path):
        chat_id = 1
        # User at 09:00, assistant at 10:00 = 3600s gap. Default max-
        # gap is 600s. The user-to-assistant pairing is rejected as
        # untrustworthy even though the user turn is the only candidate.
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec("2026-06-12T09:00:00+00:00", _PASS2_USER_TEXT),
                _assistant_rec("2026-06-12T10:00:00+00:00", _PASS2_ASSISTANT_TEXT),
            ],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:30:00+00:00")
        out_dir = self._dry_run(history_dir, tmp_path, row)
        proposals = self._proposals_in(out_dir)
        assert proposals == []
        report = self._report_in(out_dir)
        # Max-gap collapses to the same bucket as "no preceding user
        # in window": the operator's tuning advice (widen the gap) is
        # the same in both cases.
        assert "skipped no_preceding_user: 1" in report

    def test_pass2_skips_when_user_timestamp_is_after_assistant(self, history_dir, tmp_path):
        chat_id = 1
        # The user line sits BEFORE the assistant in JSONL order but
        # AFTER it by timestamp. An assistant turn cannot be sourced
        # by a later user turn; the file's apparent "preceding"
        # relationship is corrupt. The pairing rule must refuse
        # rather than stamp a wrong-direction pointer.
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec("2026-06-12T10:02:00+00:00", _PASS2_USER_TEXT),
                _assistant_rec("2026-06-12T10:01:00+00:00", _PASS2_ASSISTANT_TEXT),
            ],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")
        out_dir = self._dry_run(history_dir, tmp_path, row)
        proposals = self._proposals_in(out_dir)
        assert proposals == []
        report = self._report_in(out_dir)
        assert "skipped no_preceding_user: 1" in report

    # Test 6.
    def test_pass2_skips_ambiguous_user_pairing(self, history_dir, tmp_path):
        chat_id = 1
        # Two user turns sit between window start (no prior assistant)
        # and the winning assistant. The pairing rule refuses rather
        # than picking the closer one heuristically.
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec("2026-06-12T09:55:00+00:00", "first unrelated user turn"),
                _user_rec("2026-06-12T09:58:00+00:00", "second unrelated user turn"),
                _assistant_rec("2026-06-12T10:00:00+00:00", _PASS2_ASSISTANT_TEXT),
            ],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")
        out_dir = self._dry_run(history_dir, tmp_path, row)
        proposals = self._proposals_in(out_dir)
        assert proposals == []
        report = self._report_in(out_dir)
        assert "skipped ambiguous_user_pairing: 1" in report

    # Test 7.
    def test_pass2_skips_ambiguous_assistant_overlap(self, history_dir, tmp_path):
        chat_id = 1
        # Two assistant turns with identical text both score 1.0
        # against the row; the dominance ratio (1.0) sits below the
        # 2.0 threshold and the row collapses to the assistant-
        # ambiguous bucket without ever reaching pairing.
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec("2026-06-12T09:55:00+00:00", _PASS2_USER_TEXT),
                _assistant_rec("2026-06-12T09:58:00+00:00", _PASS2_ASSISTANT_TEXT),
                _assistant_rec("2026-06-12T10:00:00+00:00", _PASS2_ASSISTANT_TEXT),
            ],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")
        out_dir = self._dry_run(history_dir, tmp_path, row)
        proposals = self._proposals_in(out_dir)
        assert proposals == []
        report = self._report_in(out_dir)
        assert "skipped assistant_ambiguous_overlap: 1" in report

    # Test 8.
    def test_pass2_runs_only_on_pass1_no_candidate(self, history_dir, tmp_path):
        chat_id = 1
        # Pass 1 STRONG_MATCH: the user turn shares strong overlap with
        # the row. Pass 2 must never see this row. We count calls to
        # the shared scorer and assert exactly one (Pass 1's).
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec("2026-06-12T10:00:00+00:00", _PASS2_ROW_TEXT + " plus some extra trailing words"),
                _assistant_rec("2026-06-12T10:01:00+00:00", "an unrelated assistant ack"),
            ],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")

        from kai import memory_provenance_backfill as mod

        calls: list[list[dict]] = []
        original = mod._score_candidates

        def counting(row_text, candidates, *, shingle_n, created_at_dt):
            calls.append(list(candidates))
            return original(row_text, candidates, shingle_n=shingle_n, created_at_dt=created_at_dt)

        with patch.object(mod, "_score_candidates", side_effect=counting):
            out_dir = self._dry_run(history_dir, tmp_path, row)

        # Only Pass 1 ran; Pass 2's assistant scoring never executed.
        assert len(calls) == 1
        # The single call's candidates were the user records, not
        # the assistant ones.
        assert all(rec["dir"] == "user" for rec in calls[0])
        proposals = self._proposals_in(out_dir)
        assert len(proposals) == 1
        assert proposals[0]["match_pass"] == "user"


# ── Tests 9-10: Per-record validation ─────────────────────────────


class TestPerRecordValidation:
    """Per-record chat_id and ts checks fire on both user and
    assistant records. Partial-history corruption on either
    direction collapses the row before scoring can lean on it."""

    def _expect_unreadable(self, history_dir: Path, records: list[dict]) -> None:
        chat_id = 1
        user_dir = history_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / "2026-06-13.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        created_at_dt = datetime(2026, 6, 13, 2, 0, 0, tzinfo=UTC)
        _records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is True

    # Test 9 part 1.
    def test_user_record_with_mismatched_chat_id_collapses_window(self, history_dir):
        self._expect_unreadable(
            history_dir,
            [{"ts": "2026-06-13T01:00:00+00:00", "dir": "user", "chat_id": 999, "text": "x"}],
        )

    # Test 9 part 2.
    def test_assistant_record_with_mismatched_chat_id_collapses_window(self, history_dir):
        self._expect_unreadable(
            history_dir,
            [{"ts": "2026-06-13T01:00:00+00:00", "dir": "assistant", "chat_id": 999, "text": "x"}],
        )

    # Test 9 part 3.
    def test_missing_chat_id_on_either_direction_collapses_window(self, history_dir):
        self._expect_unreadable(
            history_dir,
            [
                {"ts": "2026-06-13T01:00:00+00:00", "dir": "user", "text": "x"},
            ],
        )
        # Re-run with an assistant record missing chat_id.
        chat_id = 1
        path = history_dir / str(chat_id) / "2026-06-13.jsonl"
        path.write_text(
            json.dumps({"ts": "2026-06-13T01:00:00+00:00", "dir": "assistant", "text": "x"}) + "\n",
            encoding="utf-8",
        )
        created_at_dt = datetime(2026, 6, 13, 2, 0, 0, tzinfo=UTC)
        _records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is True

    # Test 10 part 1.
    def test_missing_ts_on_user_record_collapses_window(self, history_dir):
        self._expect_unreadable(
            history_dir,
            [{"dir": "user", "chat_id": 1, "text": "x"}],
        )

    # Test 10 part 2.
    def test_missing_ts_on_assistant_record_collapses_window(self, history_dir):
        self._expect_unreadable(
            history_dir,
            [{"dir": "assistant", "chat_id": 1, "text": "x"}],
        )

    # Test 10 part 3.
    def test_non_string_ts_on_assistant_record_collapses_window(self, history_dir):
        self._expect_unreadable(
            history_dir,
            [{"ts": 123456, "dir": "assistant", "chat_id": 1, "text": "x"}],
        )

    def test_out_of_window_corruption_does_not_collapse_clean_in_window_pair(self, history_dir, tmp_path):
        # A row at 10:05 with a 24h window scores only against
        # records up to 10:05 on the same day. A later record at
        # 15:00 with a mismatched chat_id or non-string text cannot
        # affect Pass 1 scoring, Pass 2 assistant matching, or
        # boundary pairing for this row; the reader must skip it
        # silently rather than flag the whole window unreadable.
        # Same-day files routinely carry post-created_at records
        # (the assistant's reply to a later user turn), so an over-
        # eager gate would over-reject otherwise-clean rows.
        chat_id = 1
        user_dir = history_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / "2026-06-12.jsonl"
        path.write_text(
            json.dumps({"ts": "2026-06-12T10:00:00+00:00", "dir": "user", "chat_id": 1, "text": "live user line"})
            + "\n"
            + json.dumps(
                {"ts": "2026-06-12T10:01:00+00:00", "dir": "assistant", "chat_id": 1, "text": "live assistant line"}
            )
            + "\n"
            + json.dumps(
                {"ts": "2026-06-12T15:00:00+00:00", "dir": "user", "chat_id": 999, "text": "later mismatched record"}
            )
            + "\n"
            + json.dumps({"ts": "2026-06-12T15:30:00+00:00", "dir": "user", "chat_id": 1, "text": 42})
            + "\n",
            encoding="utf-8",
        )
        created_at_dt = datetime(2026, 6, 12, 10, 5, 0, tzinfo=UTC)
        records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is False
        # Only the two in-window records made it into the cache;
        # the corrupt later records were silently skipped.
        assert [r["text"] for r in records] == ["live user line", "live assistant line"]

    def test_out_of_window_malformed_ts_still_collapses(self, history_dir):
        # The asymmetry that makes the in-window scoping safe: a
        # malformed `ts` ANYWHERE in the file collapses the row,
        # because without a parseable timestamp the reader cannot
        # prove the record falls outside the window. Fail-closed is
        # the only safe call.
        chat_id = 1
        user_dir = history_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / "2026-06-12.jsonl"
        path.write_text(
            json.dumps({"ts": "2026-06-12T10:00:00+00:00", "dir": "user", "chat_id": 1, "text": "live user"})
            + "\n"
            + json.dumps({"ts": "this is not a timestamp", "dir": "user", "chat_id": 1, "text": "x"})
            + "\n",
            encoding="utf-8",
        )
        created_at_dt = datetime(2026, 6, 12, 10, 5, 0, tzinfo=UTC)
        _records, any_unreadable = _records_for_row_window(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is True


# ── Tests 11-15: Apply-time gates on Pass 2 proposals ─────────────


class TestPass2ApplyGates:
    """Apply re-verifies the assistant evidence and the chat_id on
    both JSONL lines; the four source_* keys written to the row's
    metadata are user-side only on both passes."""

    def _row_with_fp(self, row_id: str = "m1") -> tuple[MemoryResult, str]:
        row = _alive_row(row_id=row_id, text="row body", created_at="2026-06-13T00:00:00+00:00")
        return row, _metadata_fingerprint(row.metadata)

    # Test 11.
    def test_assistant_drift_text_changed(self, tmp_path, history_dir):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        assistant_ts = "2026-06-12T10:01:00+00:00"
        user_text = "live user line"
        # The JSONL assistant line has changed text since dry-run: the
        # proposal's assistant_text_sha256 will not match.
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec(user_ts, user_text),
                _assistant_rec(assistant_ts, "edited assistant text after the fact"),
            ],
        )
        row, fp = self._row_with_fp()
        proposal = _make_pass2_proposal(
            chat_id=chat_id,
            user_ts=user_ts,
            user_text_sha=hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
            assistant_ts=assistant_ts,
            assistant_text_sha=hashlib.sha256(b"original assistant text").hexdigest(),
            prior_fp=fp,
        )
        proposals_path = _make_proposals_file(tmp_path, [proposal])
        update_calls: list = []
        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: update_calls.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        # No survivors -> exit 0, no writes.
        assert code == 0
        assert update_calls == []

    # Test 12.
    def test_user_drift_still_fires_on_pass2_proposal(self, tmp_path, history_dir):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        assistant_ts = "2026-06-12T10:01:00+00:00"
        assistant_text = "the actual live assistant text"
        # User line text drifted; the assistant line is intact.
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec(user_ts, "edited user text after the fact"),
                _assistant_rec(assistant_ts, assistant_text),
            ],
        )
        row, fp = self._row_with_fp()
        proposal = _make_pass2_proposal(
            chat_id=chat_id,
            user_ts=user_ts,
            user_text_sha=hashlib.sha256(b"original user text").hexdigest(),
            assistant_ts=assistant_ts,
            assistant_text_sha=hashlib.sha256(assistant_text.encode("utf-8")).hexdigest(),
            prior_fp=fp,
        )
        proposals_path = _make_proposals_file(tmp_path, [proposal])
        update_calls: list = []
        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch(
                "kai.memory.update_metadata",
                side_effect=lambda **kwargs: update_calls.append(kwargs) or True,
            ),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        assert code == 0
        assert update_calls == []

    # Test 13.
    def test_apply_chat_id_check_at_user_line(self, tmp_path, history_dir):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        assistant_ts = "2026-06-12T10:01:00+00:00"
        user_text = "live user line"
        assistant_text = "live assistant line"
        # Same ts, hash, and text on the user record, but the JSONL's
        # chat_id is wrong (a polluted restore). The ownership gate
        # must reject the user line.
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec(user_ts, user_text, chat_id=999),
                _assistant_rec(assistant_ts, assistant_text),
            ],
        )
        # The full-window reader's per-record chat_id check would
        # collapse this whole window, but apply-time verification uses
        # `_verify_jsonl_user_text` against the JSONL directly. Test
        # that helper in isolation: it must refuse the polluted record.
        ok = _verify_jsonl_user_text(
            chat_id=chat_id,
            date="2026-06-12",
            user_ts=user_ts,
            expected_sha256=hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
        )
        assert ok is False

    # Test 14.
    def test_apply_chat_id_check_at_assistant_line(self, tmp_path, history_dir):
        chat_id = 1
        assistant_ts = "2026-06-12T10:01:00+00:00"
        assistant_text = "live assistant line"
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [_assistant_rec(assistant_ts, assistant_text, chat_id=999)],
        )
        ok = _verify_jsonl_assistant_text(
            chat_id=chat_id,
            assistant_ts=assistant_ts,
            expected_sha256=hashlib.sha256(assistant_text.encode("utf-8")).hexdigest(),
        )
        assert ok is False

    # Test 15.
    def test_apply_writes_only_four_user_side_source_keys_for_pass2(self, tmp_path, history_dir):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        assistant_ts = "2026-06-12T10:01:00+00:00"
        user_text = "live user line"
        assistant_text = "live assistant line"
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec(user_ts, user_text),
                _assistant_rec(assistant_ts, assistant_text),
            ],
        )
        row = _alive_row(row_id="m1", text="row body", created_at="2026-06-13T00:00:00+00:00")
        fp = _metadata_fingerprint(row.metadata)
        proposal = _make_pass2_proposal(
            chat_id=chat_id,
            user_ts=user_ts,
            user_text_sha=hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
            assistant_ts=assistant_ts,
            assistant_text_sha=hashlib.sha256(assistant_text.encode("utf-8")).hexdigest(),
            prior_fp=fp,
        )
        proposals_path = _make_proposals_file(tmp_path, [proposal])
        captured: list[dict] = []

        def fake_update(**kwargs):
            captured.append(kwargs)
            return True

        with (
            patch("kai.memory.get_by_id", return_value=row),
            patch("kai.memory.update_metadata", side_effect=fake_update),
        ):
            code = asyncio.run(run_apply(_BASE_CONFIG, "1", proposals_path=proposals_path, out_dir=tmp_path / "out"))
        assert code == 0
        assert len(captured) == 1
        merged = captured[0]["metadata"]
        # The four user-side keys land in the merged dict.
        assert merged[SOURCE_CHAT_ID_KEY] == chat_id
        assert merged[SOURCE_DATE_KEY] == "2026-06-12"
        assert merged[SOURCE_USER_TS_KEY] == user_ts
        assert merged[SOURCE_USER_TEXT_SHA256_KEY] == hashlib.sha256(user_text.encode("utf-8")).hexdigest()
        # No assistant-side stamp leaks into the persisted metadata.
        assert "source_assistant_ts" not in merged
        assert "source_assistant_text_sha256" not in merged


# ── --no-assistant-pass disables Pass 2 ───────────────────────────


class TestNoAssistantPass:
    def test_no_assistant_pass_drops_pass2_match(self, history_dir, tmp_path):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        assistant_ts = "2026-06-12T10:01:00+00:00"
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec(user_ts, _PASS2_USER_TEXT),
                _assistant_rec(assistant_ts, _PASS2_ASSISTANT_TEXT),
            ],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")
        out_dir = tmp_path / "out"
        with patch("kai.memory.get_all", return_value=[row]):
            code = asyncio.run(
                run_dry_run(
                    _BASE_CONFIG,
                    "1",
                    window_seconds=86400,
                    min_overlap=0.3,
                    strong_overlap_ratio=2.0,
                    shingle_n=4,
                    sample=10,
                    out_dir=out_dir,
                    assistant_pass_enabled=False,
                    assistant_max_user_gap_seconds=600,
                )
            )
        assert code == 0
        proposals_path = next(out_dir.glob("backfill-*-proposals.jsonl"))
        lines = [ln for ln in proposals_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        # Header only; row falls into NO_CANDIDATE without Pass 2.
        assert len(lines) == 1
        report = (next(out_dir.glob("backfill-*-report.md"))).read_text(encoding="utf-8")
        assert "skipped no_candidate: 1" in report
        # No Pass 2 buckets fired.
        assert "skipped assistant_ambiguous_overlap: 0" in report
        assert "skipped no_preceding_user: 0" in report
        assert "skipped ambiguous_user_pairing: 0" in report

    def test_no_assistant_pass_pass1_proposals_still_uniform(self, history_dir, tmp_path):
        # A Pass 1 STRONG_MATCH emitted under --no-assistant-pass MUST
        # still carry the four new schema fields as empty strings;
        # otherwise downstream tools would have to branch on the flag.
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [_user_rec(user_ts, _PASS2_ROW_TEXT + " plus trailing extras")],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")
        out_dir = tmp_path / "out"
        with patch("kai.memory.get_all", return_value=[row]):
            code = asyncio.run(
                run_dry_run(
                    _BASE_CONFIG,
                    "1",
                    window_seconds=86400,
                    min_overlap=0.3,
                    strong_overlap_ratio=2.0,
                    shingle_n=4,
                    sample=10,
                    out_dir=out_dir,
                    assistant_pass_enabled=False,
                    assistant_max_user_gap_seconds=600,
                )
            )
        assert code == 0
        proposals_path = next(out_dir.glob("backfill-*-proposals.jsonl"))
        lines = [ln for ln in proposals_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2  # header + one Pass 1 proposal
        proposal = json.loads(lines[1])
        assert proposal["match_pass"] == "user"
        assert proposal["assistant_ts"] == ""
        assert proposal["assistant_text_sha256"] == ""
        assert proposal["assistant_text_snippet"] == ""


# ── Tests 17, 18, 25: Proposal parser schema gates ────────────────


class TestProposalSchemaGates:
    def _header_line(self) -> str:
        return json.dumps({"type": "header", "run_id": "bp-x", "user_id": "1"})

    def _legacy_proposal(self) -> dict:
        # Legacy proposal shape: no match_pass, no assistant_* fields.
        # This is the file shape on disk for proposal files written
        # before Pass 2 existed; the parser must still accept them.
        return {
            "type": "proposal",
            "memory_id": "m1",
            "chat_id": 1,
            "date": "2026-06-12",
            "user_ts": "2026-06-12T10:00:00+00:00",
            "user_text_sha256": "abc",
            "overlap_score": 0.9,
            "runner_up_overlap_score": 0.0,
            "candidate_count": 1,
            "gap_seconds": 60.0,
            "prior_metadata_sha256": "fp",
            "row_text_snippet": "row",
            "candidate_text_snippet": "cand",
        }

    # Test 17.
    def test_parser_accepts_legacy_proposal(self):
        text = self._header_line() + "\n" + json.dumps(self._legacy_proposal()) + "\n"
        _header, proposals = parse_proposals(text)
        assert len(proposals) == 1
        p = proposals[0]
        # Defaults applied: Pass 1, no assistant evidence.
        assert p.match_pass == "user"
        assert p.assistant_ts == ""
        assert p.assistant_text_sha256 == ""
        assert p.assistant_text_snippet == ""

    # Test 18 part 1.
    def test_parser_rejects_pass2_with_empty_assistant_ts(self):
        bad = dict(self._legacy_proposal())
        bad["match_pass"] = "assistant"
        bad["assistant_ts"] = ""
        bad["assistant_text_sha256"] = "ash"
        text = self._header_line() + "\n" + json.dumps(bad) + "\n"
        with pytest.raises(ValueError, match="assistant_ts"):
            parse_proposals(text)

    # Test 18 part 2.
    def test_parser_rejects_pass2_with_empty_assistant_sha(self):
        bad = dict(self._legacy_proposal())
        bad["match_pass"] = "assistant"
        bad["assistant_ts"] = "2026-06-12T10:01:00+00:00"
        bad["assistant_text_sha256"] = ""
        text = self._header_line() + "\n" + json.dumps(bad) + "\n"
        with pytest.raises(ValueError, match="assistant_text_sha256"):
            parse_proposals(text)

    # Test 25.
    def test_parser_rejects_pass2_with_malformed_assistant_ts(self):
        # Authorization-artifact gate: a hand-edited Pass 2 proposal
        # whose assistant_ts is non-empty but unparseable must be
        # rejected BEFORE the apply-time helper tries to derive the
        # JSONL date from it.
        bad = dict(self._legacy_proposal())
        bad["match_pass"] = "assistant"
        bad["assistant_ts"] = "not-a-date"
        bad["assistant_text_sha256"] = "ash"
        text = self._header_line() + "\n" + json.dumps(bad) + "\n"
        with pytest.raises(ValueError, match="assistant_ts"):
            parse_proposals(text)
        # And the other unparseable shape: a syntactically date-shaped
        # string that fromisoformat still rejects.
        bad["assistant_ts"] = "2026-13-99T25:99:99"
        text = self._header_line() + "\n" + json.dumps(bad) + "\n"
        with pytest.raises(ValueError, match="assistant_ts"):
            parse_proposals(text)


# ── Tests 19, 20, 21: CLI flags for Pass 2 ────────────────────────


class TestPass2CLIFlags:
    def _ns(self, tmp_path: Path, **overrides) -> SimpleNamespace:
        base = dict(
            user_id="1",
            window_seconds=None,
            min_overlap=None,
            strong_overlap_ratio=None,
            overlap_shingle_n=None,
            no_assistant_pass=None,
            assistant_max_user_gap_seconds=None,
            sample=None,
            out_dir=None,
            apply=None,
            rollback=None,
            yes=False,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    # Test 19.
    def test_cli_rejects_no_assistant_pass_in_mutating_mode(self, tmp_path, capsys):
        from kai.memory_admin import _cmd_backfill_provenance

        ns = self._ns(
            tmp_path,
            no_assistant_pass=True,
            apply=str(tmp_path / "proposals.jsonl"),
        )
        code = _cmd_backfill_provenance(ns)
        assert code == 2
        err = capsys.readouterr().err
        assert "--no-assistant-pass" in err

    # Test 20.
    def test_cli_rejects_assistant_max_user_gap_in_mutating_mode(self, tmp_path, capsys):
        from kai.memory_admin import _cmd_backfill_provenance

        ns = self._ns(
            tmp_path,
            assistant_max_user_gap_seconds=900,
            apply=str(tmp_path / "proposals.jsonl"),
        )
        code = _cmd_backfill_provenance(ns)
        assert code == 2
        err = capsys.readouterr().err
        assert "--assistant-max-user-gap-seconds" in err

    # Test 21.
    def test_cli_validates_assistant_max_user_gap_positive(self, tmp_path, capsys):
        from kai.memory_admin import _cmd_backfill_provenance

        ns = self._ns(tmp_path, assistant_max_user_gap_seconds=0)
        with patch("kai.memory_admin._initialize_memory", return_value=_BASE_CONFIG):
            code = _cmd_backfill_provenance(ns)
        assert code == 2
        err = capsys.readouterr().err
        assert "--assistant-max-user-gap-seconds" in err
        # Negative also rejected.
        ns = self._ns(tmp_path, assistant_max_user_gap_seconds=-1)
        with patch("kai.memory_admin._initialize_memory", return_value=_BASE_CONFIG):
            code = _cmd_backfill_provenance(ns)
        assert code == 2


# ── Test 22: Cache prevents double JSONL read ─────────────────────


class TestPass2Caching:
    def test_one_disk_read_per_file_across_both_passes(self, history_dir, tmp_path):
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        assistant_ts = "2026-06-12T10:01:00+00:00"
        # Row text overlaps only the assistant turn, so Pass 1 returns
        # NO_CANDIDATE and Pass 2 runs. Without the shared cache, the
        # window file would be read twice (once per pass).
        _write_jsonl(
            history_dir,
            chat_id,
            "2026-06-12",
            [
                _user_rec(user_ts, _PASS2_USER_TEXT),
                _assistant_rec(assistant_ts, _PASS2_ASSISTANT_TEXT),
            ],
        )
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")

        from kai import memory_provenance_backfill as mod

        reads: list[Path] = []
        original = mod._read_jsonl_records

        def counting(path):
            reads.append(path)
            return original(path)

        out_dir = tmp_path / "out"
        with (
            patch("kai.memory.get_all", return_value=[row]),
            patch.object(mod, "_read_jsonl_records", side_effect=counting),
        ):
            code = asyncio.run(
                run_dry_run(
                    _BASE_CONFIG,
                    "1",
                    window_seconds=86400,
                    min_overlap=0.3,
                    strong_overlap_ratio=2.0,
                    shingle_n=4,
                    sample=10,
                    out_dir=out_dir,
                )
            )
        assert code == 0
        # The window straddles two UTC dates (24h window ending at
        # 10:05 on 2026-06-12 starts at 10:05 on 2026-06-11). Each
        # date file is read once, NOT twice (the cache covers both
        # passes for this row).
        counts: dict[str, int] = {}
        for p in reads:
            counts[p.name] = counts.get(p.name, 0) + 1
        for name, count in counts.items():
            assert count == 1, f"{name} was read {count} times across the two passes"


# ── Pass-1-only projection round-trip ─────────────────────────────


class TestPass1OnlyProjection:
    def test_pass1_proposal_projection_matches_user_only_shape(self, history_dir, tmp_path):
        # Backward-compatibility contract: a Pass 1 proposal emitted
        # by the current code, after dropping the four assistant-side
        # audit fields, must carry exactly the user-only field set.
        # Downstream readers can therefore round-trip a legacy
        # proposal file through the current parser without losing
        # information.
        chat_id = 1
        user_ts = "2026-06-12T10:00:00+00:00"
        user_text = _PASS2_ROW_TEXT + " plus some trailing words"
        _write_jsonl(history_dir, chat_id, "2026-06-12", [_user_rec(user_ts, user_text)])
        row = _alive_row(row_id="m1", text=_PASS2_ROW_TEXT, created_at="2026-06-12T10:05:00+00:00")
        out_dir = tmp_path / "out"
        with patch("kai.memory.get_all", return_value=[row]):
            code = asyncio.run(
                run_dry_run(
                    _BASE_CONFIG,
                    "1",
                    window_seconds=86400,
                    min_overlap=0.3,
                    strong_overlap_ratio=2.0,
                    shingle_n=4,
                    sample=10,
                    out_dir=out_dir,
                    assistant_pass_enabled=False,
                )
            )
        assert code == 0
        proposals_path = next(out_dir.glob("backfill-*-proposals.jsonl"))
        lines = [ln for ln in proposals_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2
        proposal_raw = json.loads(lines[1])
        # The four assistant-side audit fields are present and empty
        # on a Pass 1 proposal (schema is uniform across passes);
        # drop them to recover the user-only field set.
        assistant_fields = {"match_pass", "assistant_ts", "assistant_text_sha256", "assistant_text_snippet"}
        user_only_shape = {k: v for k, v in proposal_raw.items() if k not in assistant_fields}
        # The user-only projection must carry the "type" tag plus
        # every audit field whose value is derived from the user
        # turn or the row itself; the value-level assertions live in
        # the Pass 1 scoring tests (this one owns the SHAPE contract).
        assert set(user_only_shape.keys()) == {
            "type",
            "memory_id",
            "chat_id",
            "date",
            "user_ts",
            "user_text_sha256",
            "overlap_score",
            "runner_up_overlap_score",
            "candidate_count",
            "gap_seconds",
            "prior_metadata_sha256",
            "row_text_snippet",
            "candidate_text_snippet",
        }


# ── Test 24: Report shape covers Pass 2 ───────────────────────────


class TestPass2ReportShape:
    def test_report_renders_pass_distribution_and_buckets(self):
        from kai.memory_provenance_backfill import RowMatch

        row_user = _alive_row(row_id="u1", text="ru", created_at="2026-06-13T00:00:00+00:00")
        row_assistant = _alive_row(row_id="a1", text="ra", created_at="2026-06-13T00:00:00+00:00")
        pass1 = Proposal(
            memory_id="u1",
            chat_id=1,
            date="2026-06-12",
            user_ts="2026-06-12T10:00:00+00:00",
            user_text_sha256="us",
            overlap_score=0.9,
            runner_up_overlap_score=0.0,
            candidate_count=1,
            gap_seconds=60.0,
            prior_metadata_sha256="fp",
            row_text_snippet="row user",
            candidate_text_snippet="user cand text",
        )
        pass2 = Proposal(
            memory_id="a1",
            chat_id=1,
            date="2026-06-12",
            user_ts="2026-06-12T10:00:00+00:00",
            user_text_sha256="us",
            overlap_score=0.85,
            runner_up_overlap_score=0.0,
            candidate_count=1,
            gap_seconds=60.0,
            prior_metadata_sha256="fp",
            row_text_snippet="row assistant",
            candidate_text_snippet="paired user text",
            match_pass="assistant",
            assistant_ts="2026-06-12T10:01:00+00:00",
            assistant_text_sha256="as",
            assistant_text_snippet="discriminator assistant text",
        )
        matches = [
            RowMatch(row=row_user, bucket="STRONG_MATCH", top_candidates=[], proposal=pass1),
            RowMatch(row=row_assistant, bucket="STRONG_MATCH", top_candidates=[], proposal=pass2),
        ]
        report = render_report(
            run_id="bp-test",
            user_id="1",
            window_seconds=86400,
            min_overlap=0.3,
            strong_overlap_ratio=2.0,
            shingle_n=4,
            assistant_pass_enabled=True,
            assistant_max_user_gap_seconds=600,
            scanned=2,
            selected=2,
            matches=matches,
            skips={"strong_match": ["u1", "a1"]},
            sample_size=10,
        )
        # The pass distribution line carries user/assistant split.
        assert "proposals (STRONG_MATCH): 2  (user: 1, assistant: 1)" in report
        # Every new skip bucket renders in a stable order so future
        # readers can locate them by line without reading the whole
        # report.
        idx_assistant_ambiguous = report.index("skipped assistant_ambiguous_overlap:")
        idx_no_preceding = report.index("skipped no_preceding_user:")
        idx_ambiguous_pair = report.index("skipped ambiguous_user_pairing:")
        assert idx_assistant_ambiguous < idx_no_preceding < idx_ambiguous_pair
        # The Pass 2 sample carries both the assistant discriminator
        # and the paired source user snippet; the operator can audit
        # either side without opening the proposals JSONL.
        assert "STRONG_MATCH via assistant pass" in report
        assert "Assistant discriminator" in report
        assert "discriminator assistant text" in report
        assert "Paired source user" in report
        assert "paired user text" in report
