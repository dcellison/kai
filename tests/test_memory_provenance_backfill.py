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
    _candidates_for_row,
    _date_files_in_window,
    _gate_match,
    _metadata_fingerprint,
    _normalize_text,
    _overlap_score,
    _score_candidates,
    _shingles,
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
    webhook_secret="test-secret",
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
        records, any_unreadable = _candidates_for_row(chat_id, created_at_dt, window_seconds=86400)
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
        records, _ = _candidates_for_row(chat_id, created_at_dt, window_seconds=86400)
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
        _records, any_unreadable = _candidates_for_row(chat_id, created_at_dt, window_seconds=86400)
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
        _records, any_unreadable = _candidates_for_row(chat_id, created_at_dt, window_seconds=86400)
        assert any_unreadable is True

    def test_malformed_user_ts_collapses_file_to_unreadable(self, history_dir):
        # Same partial-history hazard, one layer deeper: a user record
        # whose ts field is missing or unparseable would otherwise be
        # silently dropped, leaving an unrelated valid candidate free
        # to win the overlap. _candidates_for_row must set
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
        _records, any_unreadable = _candidates_for_row(chat_id, created_at_dt, window_seconds=86400)
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
        _records, any_unreadable = _candidates_for_row(chat_id, created_at_dt, window_seconds=86400)
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
