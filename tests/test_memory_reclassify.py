"""
Unit tests for `src/kai/memory_reclassify.py`.

Covers the spec's test plan: selection, verdict gating, envelope
parsing, the dry-run (no store writes, artifacts, abort guard,
defaults resolution), apply (header guard, re-checks, pre-images,
write shape, audit lines), and rollback (header guard, exact restore,
operator shielding).

No real Mem0, no real subprocesses, no real timeouts. The reasoner is
a fake object whose `run` pops canned outcomes; memory primitives are
monkeypatched; artifacts land in `tmp_path`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import kai.memory_projects as mp_mod
from kai import memory, memory_reclassify
from kai.config import MemoryProjectConfig
from kai.memory import MemoryResult
from kai.oneshot import OneShotResult, OneShotTimeout

# ── Fixture builders ────────────────────────────────────────────────


def _row(
    rid: str,
    text: str = "some stored fact",
    *,
    source: str = "extracted",
    scope_md: dict[str, Any] | None = None,
) -> MemoryResult:
    """A MemoryResult with optional scope metadata overlaid."""
    metadata: dict[str, Any] = {"type": "fact", "source": source, "tags": ["t"], "confidence": 0.9}
    if scope_md is not None:
        metadata.update(scope_md)
    return MemoryResult(id=rid, text=text, score=0.0, memory_type="fact", metadata=metadata, created_at="")


def _legacy(rid: str, text: str = "some stored fact") -> MemoryResult:
    """A legacy row: no scope keys at all."""
    return _row(rid, text)


def _extraction_global(rid: str) -> MemoryResult:
    return _row(rid, scope_md={"scope": "global", "scope_source": "extraction_default"})


def _project_cfg(pid: str, *, enabled: bool = True) -> MemoryProjectConfig:
    return MemoryProjectConfig(
        project_id=pid,
        display_name=pid,
        workspace_roots=(Path("/work") / pid,),
        memory_enabled=enabled,
        default_scope_for_new_facts="project",
    )


_REGISTRY = {"kai": _project_cfg("kai"), "anvil": _project_cfg("anvil")}


def _verdict(scope: str, *, project_id: str | None = None, confidence: float = 0.9, reason: str = "r") -> dict:
    return {"scope": scope, "project_id": project_id, "confidence": confidence, "reason": reason}


def _envelope(verdict: dict) -> str:
    """Wrap a verdict the way the claude schema-mode envelope does."""
    return json.dumps({"is_error": False, "structured_output": verdict})


class FakeReasoner:
    """Pops one canned outcome per run() call.

    Outcomes are either exceptions (raised) or strings (returned as
    OneShotResult.text). Records every prompt for assertions.
    """

    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.prompts: list[str] = []

    async def run(self, *, prompt, system_prompt, model, timeout, purpose, json_schema):
        self.prompts.append(prompt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return OneShotResult(text=outcome, backend="claude", model=model)


def _config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.memory_projects = {}
    cfg.user_configs = {}
    cfg.default_backend = "claude"
    cfg.default_provider = ""
    cfg.memory_extraction_timeout_s = 60
    cfg.session_db_path = tmp_path / "kai.db"
    return cfg


@pytest.fixture
def dry_run_env(monkeypatch, tmp_path):
    """Common monkeypatching for run_dry_run tests.

    Returns a dict of mutable knobs the test can adjust before
    calling `run()`: rows, registry, reasoner outcomes.
    """
    env: dict[str, Any] = {
        "rows": [],
        "registry": dict(_REGISTRY),
        "outcomes": [],
        "config": _config(tmp_path),
        "out_dir": tmp_path / "out",
    }

    async def fake_registry(config):
        return env["registry"]

    monkeypatch.setattr(memory_reclassify, "load_project_registry", fake_registry)
    monkeypatch.setattr(memory_reclassify.memory, "get_all", lambda *, user_id, limit: env["rows"])
    monkeypatch.setattr(memory_reclassify, "resolve_user_model", lambda *a, **k: "model-x")
    env["reasoner"] = None

    def fake_build(backend, *, os_user=None, provider=""):
        env["reasoner"] = FakeReasoner(env["outcomes"])
        return env["reasoner"]

    monkeypatch.setattr(memory_reclassify, "_build_memory_reasoner", fake_build)
    update_mock = MagicMock(return_value=True)
    monkeypatch.setattr(memory_reclassify.memory, "update_metadata", update_mock)
    env["update_metadata"] = update_mock

    async def run(**overrides):
        kwargs = {
            "backend": "claude",
            "os_user": None,
            "provider": "",
            "threshold": 0.8,
            "sample": 10,
            "out_dir": env["out_dir"],
        }
        kwargs.update(overrides)
        return await memory_reclassify.run_dry_run(env["config"], "100", **kwargs)

    env["run"] = run
    return env


# ── Selection ───────────────────────────────────────────────────────


class TestSelectRows:
    def test_admits_legacy_and_extraction_default_global(self):
        rows = [_legacy("a"), _extraction_global("b")]
        selected = memory_reclassify.select_rows(rows)
        assert [r.id for r, _ in selected] == ["a", "b"]

    @pytest.mark.parametrize(
        "row",
        [
            # Operator and classifier provenance are shielded.
            _row("op", scope_md={"scope": "global", "scope_source": "operator"}),
            _row("cl", scope_md={"scope": "global", "scope_source": "classifier"}),
            # Invalid arm: corrupted scope value resolves global but
            # invalid_defaulted, and must not be masked.
            _row("inv", scope_md={"scope": "bogus"}),
            # Valid scope without provenance: also the invalid arm.
            _row("noprov", scope_md={"scope": "global"}),
            # Non-global rows are out of population.
            _row("proj", scope_md={"scope": "project", "project_id": "kai", "scope_source": "operator"}),
            _row("task", scope_md={"scope": "task", "scope_source": "extraction_default"}),
            # Non-user-visible sources never appear.
            _row("raw", source="user_raw"),
            _row("nosrc", source=""),
        ],
    )
    def test_excludes_everything_else(self, row):
        assert memory_reclassify.select_rows([row]) == []


# ── Verdict gating ──────────────────────────────────────────────────


class TestGateVerdict:
    def _gate(self, verdict, *, registry=None, threshold=0.8, row=None):
        row = row or _legacy("a")
        resolved = memory.resolve_memory_scope(row.metadata)
        return memory_reclassify.gate_verdict(
            verdict,
            row=row,
            resolved=resolved,
            registry=_REGISTRY if registry is None else registry,
            threshold=threshold,
        )

    def test_project_verdict_at_threshold_proposes(self):
        proposal, skip = self._gate(_verdict("project", project_id="kai", confidence=0.8))
        assert skip is None
        assert proposal is not None
        assert proposal.verdict == "project"
        assert proposal.project_id == "kai"
        assert proposal.prior_scope_source == "legacy_default"
        assert proposal.text_sha256

    def test_project_below_threshold_skips(self):
        proposal, skip = self._gate(_verdict("project", project_id="kai", confidence=0.79))
        assert proposal is None
        assert skip == memory_reclassify.SKIP_BELOW_THRESHOLD

    def test_unregistered_target_skips(self):
        proposal, skip = self._gate(_verdict("project", project_id="ghost"))
        assert proposal is None
        assert skip == memory_reclassify.SKIP_UNREGISTERED_TARGET

    def test_disabled_target_skips(self):
        registry = {"kai": _project_cfg("kai", enabled=False)}
        proposal, skip = self._gate(_verdict("project", project_id="kai"), registry=registry)
        assert proposal is None
        assert skip == memory_reclassify.SKIP_DISABLED_TARGET

    def test_global_verdict_proposes_restamp(self):
        proposal, skip = self._gate(_verdict("global"))
        assert skip is None
        assert proposal is not None
        assert proposal.verdict == "global"
        assert proposal.project_id is None

    def test_global_below_threshold_skips(self):
        proposal, skip = self._gate(_verdict("global", confidence=0.5))
        assert proposal is None
        assert skip == memory_reclassify.SKIP_BELOW_THRESHOLD

    def test_global_verdict_discards_stray_target(self):
        proposal, _ = self._gate(_verdict("global", project_id="kai"))
        assert proposal is not None
        assert proposal.project_id is None

    @pytest.mark.parametrize(
        "verdict",
        [
            None,
            "not a dict",
            {"scope": "task", "project_id": None, "confidence": 0.9, "reason": "r"},
            {"scope": "project", "project_id": "kai", "confidence": "high", "reason": "r"},
            {"scope": "project", "project_id": "kai", "confidence": 1.7, "reason": "r"},
        ],
    )
    def test_malformed_verdicts_skip(self, verdict):
        proposal, skip = self._gate(verdict)
        assert proposal is None
        assert skip == memory_reclassify.SKIP_MALFORMED_OUTPUT


# ── Envelope parsing ────────────────────────────────────────────────


class TestParseVerdictEnvelope:
    def test_structured_output_nesting_preferred(self):
        v = memory_reclassify.parse_verdict_envelope(_envelope(_verdict("global")))
        assert v is not None and v["scope"] == "global"

    def test_root_fallback(self):
        v = memory_reclassify.parse_verdict_envelope(json.dumps(_verdict("project", project_id="kai")))
        assert v is not None and v["project_id"] == "kai"

    @pytest.mark.parametrize(
        "raw",
        [
            "not json at all",
            json.dumps(["a", "list"]),
            json.dumps({"is_error": True, "structured_output": {"scope": "global"}}),
            json.dumps({"unrelated": 1}),
        ],
    )
    def test_rejections_return_none(self, raw):
        assert memory_reclassify.parse_verdict_envelope(raw) is None


# ── Artifact formats ────────────────────────────────────────────────


class TestArtifactFormats:
    def _proposal(self) -> memory_reclassify.Proposal:
        return memory_reclassify.Proposal(
            memory_id="m1",
            verdict="project",
            project_id="kai",
            confidence=0.9,
            reason="r",
            prior_scope_source="legacy_default",
            text_sha256="ab",
        )

    def test_proposals_round_trip(self):
        text = memory_reclassify.render_proposals({"run_id": "rs-1", "user_id": "100"}, [self._proposal()])
        header, rows = memory_reclassify.parse_artifact(text, row_type="proposal")
        assert header["run_id"] == "rs-1"
        assert rows[0]["memory_id"] == "m1"
        assert rows[0]["text_sha256"] == "ab"

    def test_preimages_round_trip(self):
        pre = memory_reclassify.PreImage(memory_id="m1", text="t", metadata={"source": "extracted"})
        text = memory_reclassify.render_preimages({"run_id": "rs-1", "user_id": "100"}, [pre])
        header, rows = memory_reclassify.parse_artifact(text, row_type="preimage")
        assert header["run_id"] == "rs-1"
        assert rows[0]["metadata"] == {"source": "extracted"}

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "not json\n",
            json.dumps({"type": "proposal"}) + "\n",
            json.dumps({"type": "header", "run_id": "rs-1"}) + "\n" + json.dumps({"type": "preimage"}),
        ],
    )
    def test_structural_problems_raise(self, text):
        with pytest.raises(ValueError):
            memory_reclassify.parse_artifact(text, row_type="proposal")

    def test_validate_header_user_mismatch(self):
        err = memory_reclassify.validate_header({"run_id": "rs-1", "user_id": "200"}, user_id="100")
        assert err is not None and "200" in err

    def test_validate_header_missing_run_id(self):
        err = memory_reclassify.validate_header({"user_id": "100"}, user_id="100")
        assert err is not None and "run_id" in err

    def test_validate_header_ok(self):
        assert memory_reclassify.validate_header({"run_id": "rs-1", "user_id": "100"}, user_id="100") is None


class TestParseProposals:
    """Strict row validation: a hand-edited proposals file must fail
    parsing as a whole, never crash apply midway through writes."""

    def _line(self, **overrides) -> str:
        row = {
            "type": "proposal",
            "memory_id": "m1",
            "verdict": "project",
            "project_id": "kai",
            "confidence": 0.9,
            "reason": "r",
            "prior_scope_source": "legacy_default",
            "text_sha256": "ab",
        }
        row.update(overrides)
        header = json.dumps({"type": "header", "run_id": "rs-1", "user_id": "100"})
        return header + "\n" + json.dumps(row)

    def test_valid_round_trip(self):
        header, proposals = memory_reclassify.parse_proposals(self._line())
        assert header["run_id"] == "rs-1"
        assert proposals[0].verdict == "project"
        assert proposals[0].confidence == 0.9

    def test_global_verdict_normalizes_stray_target(self):
        _, proposals = memory_reclassify.parse_proposals(self._line(verdict="global", project_id="kai"))
        assert proposals[0].project_id is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"memory_id": ""},
            {"memory_id": 7},
            {"verdict": "task"},
            {"verdict": "oops"},
            {"project_id": None},
            {"confidence": "high"},
            {"confidence": 1.7},
            {"text_sha256": None},
        ],
    )
    def test_invalid_rows_raise_with_line_number(self, overrides):
        with pytest.raises(ValueError, match="line 2"):
            memory_reclassify.parse_proposals(self._line(**overrides))


class TestParsePreimages:
    """Same strictness for the restore artifact: a malformed pre-image
    row could otherwise silently restore an empty row."""

    def _line(self, **overrides) -> str:
        row = {"type": "preimage", "memory_id": "m1", "text": "t", "metadata": {"source": "extracted"}}
        row.update(overrides)
        header = json.dumps({"type": "header", "run_id": "rs-1", "user_id": "100"})
        return header + "\n" + json.dumps(row)

    def test_valid_round_trip(self):
        _, preimages = memory_reclassify.parse_preimages(self._line())
        assert preimages[0].metadata == {"source": "extracted"}

    @pytest.mark.parametrize(
        "overrides",
        [
            {"memory_id": ""},
            {"text": ""},
            {"text": None},
            {"metadata": "not a dict"},
            {"metadata": None},
        ],
    )
    def test_invalid_rows_raise_with_line_number(self, overrides):
        with pytest.raises(ValueError, match="line 2"):
            memory_reclassify.parse_preimages(self._line(**overrides))


# ── Report rendering ────────────────────────────────────────────────


class TestRenderReport:
    def _render(self, run_id: str = "rs-test") -> str:
        proposals = [
            memory_reclassify.Proposal(
                memory_id=f"m{i}",
                verdict="project" if i % 2 else "global",
                project_id="kai" if i % 2 else None,
                confidence=0.9,
                reason=f"reason {i}",
                prior_scope_source="legacy_default",
                text_sha256="ab",
            )
            for i in range(6)
        ]
        return memory_reclassify.render_report(
            run_id=run_id,
            user_id="100",
            backend="claude",
            threshold=0.8,
            scanned=10,
            selected=8,
            proposals=proposals,
            skips={"below_threshold": ["m9"], "reasoner_failure": []},
            sample_size=3,
            texts={f"m{i}": f"text of row {i}" for i in range(6)},
        )

    def test_report_contains_counts_sample_and_proposals(self):
        report = self._render()
        assert "rows scanned: 10   selected: 8" in report
        assert "3 project moves, 3 global re-stamps" in report
        assert "1 below_threshold" in report
        assert "## Eyeball sample (3 of 6)" in report
        assert "text of row" in report
        assert "## All proposals" in report
        assert "- m9: below_threshold" in report

    def test_sample_is_reproducible_per_run_id(self):
        assert self._render("rs-a") == self._render("rs-a")
        # Different run ids draw different samples (statistically;
        # with 6 choose 3 the two seeds used here differ).
        assert self._render("rs-a") != self._render("rs-b")


# ── Defaults resolution ─────────────────────────────────────────────


class TestResolveClassificationSettings:
    def test_user_overrides_win_over_global(self, tmp_path):
        cfg = _config(tmp_path)
        user = MagicMock()
        user.backend = "codex"
        user.provider = "deepseek"
        user.os_user = "alice"
        cfg.user_configs = {100: user}
        settings = memory_reclassify.resolve_classification_settings(
            cfg, "100", backend=None, os_user=None, provider=None
        )
        assert settings == ("codex", "alice", "deepseek")

    def test_unsupported_effective_backend_errors(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.default_backend = "not-a-reasoner-backend"
        settings = memory_reclassify.resolve_classification_settings(
            cfg, "100", backend=None, os_user=None, provider=None
        )
        assert isinstance(settings, str)
        assert "--backend" in settings

    def test_explicit_flags_override_resolution(self, tmp_path):
        cfg = _config(tmp_path)
        user = MagicMock()
        user.backend = "codex"
        user.provider = "deepseek"
        user.os_user = "alice"
        cfg.user_configs = {100: user}
        settings = memory_reclassify.resolve_classification_settings(
            cfg, "100", backend="claude", os_user="bob", provider="p"
        )
        assert settings == ("claude", "bob", "p")


# ── Dry run ─────────────────────────────────────────────────────────


class TestDryRun:
    @pytest.mark.asyncio
    async def test_writes_artifacts_and_no_store_writes(self, dry_run_env):
        env = dry_run_env
        env["rows"] = [_legacy("a", "kai webhook internals"), _extraction_global("b")]
        env["outcomes"] = [
            _envelope(_verdict("project", project_id="kai", confidence=0.93)),
            _envelope(_verdict("global", confidence=0.85)),
        ]
        code = await env["run"]()
        assert code == 0
        env["update_metadata"].assert_not_called()
        files = sorted(p.name for p in env["out_dir"].iterdir())
        assert any(name.endswith("-proposals.jsonl") for name in files)
        assert any(name.endswith("-report.md") for name in files)
        proposals_text = next(p for p in env["out_dir"].iterdir() if p.name.endswith(".jsonl")).read_text()
        header, rows = memory_reclassify.parse_artifact(proposals_text, row_type="proposal")
        assert header["user_id"] == "100"
        assert header["backend"] == "claude"
        assert {r["memory_id"] for r in rows} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_prompt_carries_enabled_projects_only(self, dry_run_env):
        env = dry_run_env
        env["registry"] = {"kai": _project_cfg("kai"), "off": _project_cfg("off", enabled=False)}
        env["rows"] = [_legacy("a")]
        env["outcomes"] = [_envelope(_verdict("global"))]
        await env["run"]()
        prompt = env["reasoner"].prompts[0]
        assert "kai: kai" in prompt
        assert "off" not in prompt.split("Rules:")[0].split("Registered projects:")[1]

    @pytest.mark.asyncio
    async def test_abort_after_five_consecutive_failures(self, dry_run_env):
        env = dry_run_env
        env["rows"] = [_legacy(f"r{i}") for i in range(8)]
        env["outcomes"] = [OneShotTimeout() for _ in range(5)] + [_envelope(_verdict("global"))] * 3
        code = await env["run"]()
        assert code == 1
        # The sixth row was never classified: the guard fired first.
        assert len(env["reasoner"].prompts) == 5

    @pytest.mark.asyncio
    async def test_four_failures_then_success_continues(self, dry_run_env):
        env = dry_run_env
        env["rows"] = [_legacy(f"r{i}") for i in range(6)]
        env["outcomes"] = [OneShotTimeout() for _ in range(4)] + [_envelope(_verdict("global"))] * 2
        code = await env["run"]()
        assert code == 0
        assert len(env["reasoner"].prompts) == 6

    @pytest.mark.asyncio
    async def test_malformed_output_counts_toward_abort(self, dry_run_env):
        env = dry_run_env
        env["rows"] = [_legacy(f"r{i}") for i in range(6)]
        env["outcomes"] = ["garbage"] * 5 + [_envelope(_verdict("global"))]
        code = await env["run"]()
        assert code == 1

    @pytest.mark.asyncio
    async def test_unsupported_backend_exits_2(self, dry_run_env):
        env = dry_run_env
        env["config"].default_backend = "not-a-reasoner-backend"
        code = await env["run"](backend=None)
        assert code == 2


# ── Apply ───────────────────────────────────────────────────────────


def _proposals_file(
    tmp_path: Path, rows: list[MemoryResult], *, run_id: str = "rs-apply", user_id: str = "100"
) -> Path:
    """Write a proposals file moving every row to project kai."""
    proposals = [
        memory_reclassify.Proposal(
            memory_id=r.id,
            verdict="project",
            project_id="kai",
            confidence=0.9,
            reason="r",
            prior_scope_source="legacy_default",
            text_sha256=memory_reclassify._text_sha256(r.text),
        )
        for r in rows
    ]
    path = tmp_path / "proposals.jsonl"
    path.write_text(
        memory_reclassify.render_proposals({"run_id": run_id, "user_id": user_id}, proposals), encoding="utf-8"
    )
    return path


@pytest.fixture
def apply_env(monkeypatch, tmp_path):
    """Common monkeypatching for run_apply / run_rollback tests."""
    env: dict[str, Any] = {
        "store": {},
        "registry": dict(_REGISTRY),
        "config": _config(tmp_path),
        "out_dir": tmp_path / "out",
        "updates": [],
        "update_ok": True,
    }

    async def fake_registry(config):
        return env["registry"]

    monkeypatch.setattr(memory_reclassify, "load_project_registry", fake_registry)
    monkeypatch.setattr(
        memory_reclassify.memory,
        "get_by_id",
        lambda *, user_id, memory_id: env["store"].get(memory_id),
    )

    def fake_update(*, user_id, memory_id, data, metadata):
        env["updates"].append({"user_id": user_id, "memory_id": memory_id, "data": data, "metadata": metadata})
        return env["update_ok"]

    monkeypatch.setattr(memory_reclassify.memory, "update_metadata", fake_update)
    return env


class TestApply:
    @pytest.mark.asyncio
    async def test_happy_path_write_shape_and_preimages(self, apply_env, tmp_path, caplog):
        env = apply_env
        row = _legacy("a", "kai internals")
        env["store"] = {"a": row}
        path = _proposals_file(tmp_path, [row])
        with caplog.at_level(logging.INFO, logger="kai.memory_reclassify"):
            code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=path, out_dir=env["out_dir"])
        assert code == 0
        # Write shape: full prior metadata preserved, scope keys
        # overlaid, run id stamped, verdict confidence carried.
        update = env["updates"][0]
        assert update["user_id"] == "100"
        assert update["data"] == row.text
        md = update["metadata"]
        for key, value in row.metadata.items():
            if key not in {"scope", "project_id", "workspace_root", "scope_confidence", "scope_source"}:
                assert md[key] == value
        assert md["scope"] == "project"
        assert md["project_id"] == "kai"
        assert md["scope_source"] == "classifier"
        assert md["scope_confidence"] == 0.9
        assert md[memory.SCOPE_RUN_ID_KEY] == "rs-apply"
        # Pre-image dump: headed, complete, prior state exact.
        pre_path = env["out_dir"] / "reclassify-rs-apply-preimages.jsonl"
        header, rows = memory_reclassify.parse_artifact(pre_path.read_text(), row_type="preimage")
        assert header["run_id"] == "rs-apply"
        assert header["user_id"] == "100"
        assert rows[0]["metadata"] == row.metadata
        # One audit line with the run id.
        line = next(r.message for r in caplog.records if r.message.startswith(memory.SCOPE_CHANGE_EVENT))
        assert '"run_id":"rs-apply"' in line
        assert '"from_scope_source":"legacy_default"' in line
        assert '"to_project_id":"kai"' in line

    @pytest.mark.asyncio
    async def test_header_user_mismatch_aborts_before_store(self, apply_env, tmp_path, monkeypatch):
        env = apply_env
        row = _legacy("a")
        path = _proposals_file(tmp_path, [row], user_id="200")
        fetches: list[str] = []
        monkeypatch.setattr(
            memory_reclassify.memory,
            "get_by_id",
            lambda *, user_id, memory_id: fetches.append(memory_id),
        )
        code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=path, out_dir=env["out_dir"])
        assert code == 1
        assert fetches == []
        assert env["updates"] == []

    @pytest.mark.asyncio
    async def test_recheck_skips(self, apply_env, tmp_path):
        env = apply_env
        gone = _legacy("gone")
        moved = _row("moved", scope_md={"scope": "project", "project_id": "kai", "scope_source": "operator"})
        drifted_orig = _legacy("drift", "original text")
        drifted_now = _legacy("drift", "edited text")
        ok = _legacy("ok")
        env["store"] = {"moved": moved, "drift": drifted_now, "ok": ok}
        path = _proposals_file(tmp_path, [gone, _legacy("moved"), drifted_orig, ok])
        code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=path, out_dir=env["out_dir"])
        assert code == 0
        # Only the clean row was written; the operator-moved row was
        # deselected, the drifted row hash-mismatched, the gone row
        # vanished.
        assert [u["memory_id"] for u in env["updates"]] == ["ok"]
        # Pre-images contain only the surviving row.
        pre_path = env["out_dir"] / "reclassify-rs-apply-preimages.jsonl"
        _, rows = memory_reclassify.parse_artifact(pre_path.read_text(), row_type="preimage")
        assert [r["memory_id"] for r in rows] == ["ok"]

    @pytest.mark.asyncio
    async def test_session_db_only_project_applies(self, apply_env, tmp_path, monkeypatch):
        """Pins the mode-neutral registry bootstrap: a target that only
        the DB layer knows applies when the bootstrap loaded it, and
        skips as unregistered when the registry is empty."""
        env = apply_env
        row = _legacy("a")
        env["store"] = {"a": row}
        path = _proposals_file(tmp_path, [row])

        env["registry"] = {"kai": _project_cfg("kai")}
        code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=path, out_dir=env["out_dir"])
        assert code == 0
        assert len(env["updates"]) == 1

        env["updates"].clear()
        env["registry"] = {}
        code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=path, out_dir=env["out_dir"])
        assert code == 0
        assert env["updates"] == []

    @pytest.mark.asyncio
    async def test_load_project_registry_mirrors_daemon_startup(self, monkeypatch, tmp_path):
        """The real bootstrap loads session-DB rows through the same
        cache loader daemon startup uses and merges under YAML."""
        init_db = AsyncMock()
        rows = AsyncMock(
            return_value=[
                {
                    "project_id": "dbproj",
                    "display_name": "dbproj",
                    "workspace_root": str(tmp_path / "dbproj"),
                    "memory_enabled": True,
                    "default_scope_for_new_facts": "project",
                    "created_by": 100,
                }
            ]
        )
        monkeypatch.setattr(mp_mod.sessions, "init_db", init_db)
        monkeypatch.setattr(mp_mod.sessions, "get_memory_project_rows", rows)
        monkeypatch.setattr(mp_mod, "_db_registry", {})
        monkeypatch.setattr(mp_mod, "_db_creators", {})
        cfg = _config(tmp_path)
        registry = await memory_reclassify.load_project_registry(cfg)
        init_db.assert_awaited_once_with(cfg.session_db_path)
        assert "dbproj" in registry

    @pytest.mark.asyncio
    async def test_preimage_dump_failure_aborts_with_no_writes(self, apply_env, tmp_path):
        env = apply_env
        row = _legacy("a")
        env["store"] = {"a": row}
        path = _proposals_file(tmp_path, [row])
        # out_dir is an existing FILE: mkdir(exist_ok=True) raises,
        # which must abort before any store write.
        blocked = tmp_path / "blocked"
        blocked.write_text("")
        code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=path, out_dir=blocked)
        assert code == 1
        assert env["updates"] == []

    @pytest.mark.asyncio
    async def test_all_writes_failing_exits_1(self, apply_env, tmp_path):
        env = apply_env
        row = _legacy("a")
        env["store"] = {"a": row}
        env["update_ok"] = False
        path = _proposals_file(tmp_path, [row])
        code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=path, out_dir=env["out_dir"])
        assert code == 1

    @pytest.mark.asyncio
    async def test_existing_preimage_file_is_never_truncated(self, apply_env, tmp_path):
        """A re-run of the same apply must not overwrite the first
        run's rollback material; the path derives from the run id, so
        only exclusive creation protects it."""
        env = apply_env
        row = _legacy("a")
        env["store"] = {"a": row}
        path = _proposals_file(tmp_path, [row])
        env["out_dir"].mkdir(parents=True)
        existing = env["out_dir"] / "reclassify-rs-apply-preimages.jsonl"
        existing.write_text("rollback material from the first apply\n")
        code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=path, out_dir=env["out_dir"])
        assert code == 1
        assert env["updates"] == []
        assert existing.read_text() == "rollback material from the first apply\n"

    @pytest.mark.asyncio
    async def test_no_survivors_writes_no_preimage_file(self, apply_env, tmp_path):
        """The accidental-re-apply shape: every row was reclassified
        by the first run, so the re-check deselects all of them. The
        second run must not leave a header-only pre-image file where
        the first run's rollback material lives."""
        env = apply_env
        reclassified = _row("a", scope_md={"scope": "project", "project_id": "kai", "scope_source": "classifier"})
        env["store"] = {"a": reclassified}
        path = _proposals_file(tmp_path, [_legacy("a")])
        code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=path, out_dir=env["out_dir"])
        assert code == 0
        assert env["updates"] == []
        assert not (env["out_dir"] / "reclassify-rs-apply-preimages.jsonl").exists()

    @pytest.mark.asyncio
    async def test_malformed_proposal_row_aborts_before_store(self, apply_env, tmp_path, monkeypatch):
        """A hand-edited row with an invalid verdict must abort the
        whole apply during parsing, before any fetch or write."""
        env = apply_env
        row = _legacy("a")
        good = _proposals_file(tmp_path, [row]).read_text()
        corrupted = good.replace('"verdict":"project"', '"verdict":"task"')
        bad_path = tmp_path / "edited.jsonl"
        bad_path.write_text(corrupted)
        fetches: list[str] = []
        monkeypatch.setattr(
            memory_reclassify.memory,
            "get_by_id",
            lambda *, user_id, memory_id: fetches.append(memory_id),
        )
        code = await memory_reclassify.run_apply(env["config"], "100", proposals_path=bad_path, out_dir=env["out_dir"])
        assert code == 1
        assert fetches == []
        assert env["updates"] == []


# ── Rollback ────────────────────────────────────────────────────────


def _preimage_file(tmp_path: Path, preimages: list[memory_reclassify.PreImage], *, user_id: str = "100") -> Path:
    path = tmp_path / "preimages.jsonl"
    path.write_text(
        memory_reclassify.render_preimages({"run_id": "rs-orig", "user_id": user_id}, preimages), encoding="utf-8"
    )
    return path


class TestRollback:
    @pytest.mark.asyncio
    async def test_restores_text_and_metadata_exactly(self, apply_env, tmp_path, caplog):
        env = apply_env
        # Current state: classifier-moved row. Pre-image: the legacy
        # original.
        current = _row(
            "a", "kai internals", scope_md={"scope": "project", "project_id": "kai", "scope_source": "classifier"}
        )
        env["store"] = {"a": current}
        original = _legacy("a", "kai internals")
        path = _preimage_file(
            tmp_path, [memory_reclassify.PreImage(memory_id="a", text=original.text, metadata=original.metadata)]
        )
        with caplog.at_level(logging.INFO, logger="kai.memory_reclassify"):
            code = await memory_reclassify.run_rollback(env["config"], "100", preimages_path=path)
        assert code == 0
        update = env["updates"][0]
        assert update["data"] == original.text
        assert update["metadata"] == original.metadata
        line = next(r.message for r in caplog.records if r.message.startswith(memory.SCOPE_CHANGE_EVENT))
        assert '"run_id":"rs-orig"' in line
        assert '"rollback":true' in line

    @pytest.mark.asyncio
    async def test_operator_corrected_rows_are_skipped(self, apply_env, tmp_path):
        env = apply_env
        current = _row("a", scope_md={"scope": "project", "project_id": "anvil", "scope_source": "operator"})
        env["store"] = {"a": current}
        path = _preimage_file(tmp_path, [memory_reclassify.PreImage(memory_id="a", text="t", metadata={})])
        code = await memory_reclassify.run_rollback(env["config"], "100", preimages_path=path)
        assert code == 0
        assert env["updates"] == []

    @pytest.mark.asyncio
    async def test_header_user_mismatch_aborts(self, apply_env, tmp_path):
        env = apply_env
        path = _preimage_file(
            tmp_path, [memory_reclassify.PreImage(memory_id="a", text="t", metadata={})], user_id="200"
        )
        code = await memory_reclassify.run_rollback(env["config"], "100", preimages_path=path)
        assert code == 1
        assert env["updates"] == []

    @pytest.mark.asyncio
    async def test_missing_rows_skip_counted(self, apply_env, tmp_path):
        env = apply_env
        env["store"] = {}
        path = _preimage_file(tmp_path, [memory_reclassify.PreImage(memory_id="ghost", text="t", metadata={})])
        code = await memory_reclassify.run_rollback(env["config"], "100", preimages_path=path)
        assert code == 0
        assert env["updates"] == []

    @pytest.mark.asyncio
    async def test_malformed_preimage_row_aborts_before_store(self, apply_env, tmp_path, monkeypatch):
        """A pre-image row without text must abort the whole rollback
        during parsing; writing it back would wipe the row's content."""
        env = apply_env
        good = _preimage_file(tmp_path, [memory_reclassify.PreImage(memory_id="a", text="t", metadata={})]).read_text()
        corrupted = good.replace('"text":"t"', '"text":""')
        bad_path = tmp_path / "edited.jsonl"
        bad_path.write_text(corrupted)
        fetches: list[str] = []
        monkeypatch.setattr(
            memory_reclassify.memory,
            "get_by_id",
            lambda *, user_id, memory_id: fetches.append(memory_id),
        )
        code = await memory_reclassify.run_rollback(env["config"], "100", preimages_path=bad_path)
        assert code == 1
        assert fetches == []
        assert env["updates"] == []


# ── Transcript provenance in the report ─────────────────────────────


class TestReportProvenanceQuote:
    def _proposals(self):
        return [
            memory_reclassify.Proposal(
                memory_id=f"m{i}",
                verdict="project" if i % 2 else "global",
                project_id="kai" if i % 2 else None,
                confidence=0.9,
                reason=f"reason {i}",
                prior_scope_source="legacy_default",
                text_sha256="ab",
            )
            for i in range(3)
        ]

    def test_present_entries_render_said_line(self):
        proposals = self._proposals()
        report = memory_reclassify.render_report(
            run_id="rs-test",
            user_id="100",
            backend="claude",
            threshold=0.8,
            scanned=3,
            selected=3,
            proposals=proposals,
            skips={},
            sample_size=3,
            texts={p.memory_id: f"text {p.memory_id}" for p in proposals},
            provenance_user_texts={"m0": "what did the user actually say"},
        )
        assert 'said:   "what did the user actually say"' in report
        # The other two proposals (no entry) render unchanged.
        assert report.count("said:") == 1

    def test_no_provenance_argument_renders_identically(self):
        """The optional kwarg defaults to None; older callers see the
        original report shape."""
        proposals = self._proposals()
        kwargs = dict(
            run_id="rs-test",
            user_id="100",
            backend="claude",
            threshold=0.8,
            scanned=3,
            selected=3,
            proposals=proposals,
            skips={},
            sample_size=3,
            texts={p.memory_id: f"text {p.memory_id}" for p in proposals},
        )
        without = memory_reclassify.render_report(**kwargs)
        with_empty = memory_reclassify.render_report(**kwargs, provenance_user_texts={})
        assert without == with_empty
        assert "said:" not in without


class TestReportProvenanceQuoteOwnership:
    """A forged or restored row whose source_chat_id points at another
    chat must not contribute a quote to the dry-run report."""

    def test_cross_chat_pointer_skipped(self, monkeypatch, tmp_path):
        from kai import memory as memory_module

        # The proposal's source row carries provenance whose chat_id
        # is 2; the CLI is running for user_id="1". Even if a JSONL
        # entry happens to line up, the helper must refuse before any
        # filesystem read and the report must not quote it.
        path = tmp_path / "history" / "2" / "2026-06-13.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"ts": "2026-06-13T09:00:00+00:00", "dir": "user", "chat_id": 2, "text": "secret"}\n')

        # Redirect history dir so the helper looks at our fixture.
        from kai import history

        monkeypatch.setattr(history, "_LOG_DIR", tmp_path / "history")

        fact = MemoryResult(
            id="m1",
            text="row body",
            score=0.0,
            memory_type="fact",
            metadata={
                "source": "extracted",
                "tags": ["t"],
                "confidence": 0.9,
                memory_module.SOURCE_CHAT_ID_KEY: 2,
                memory_module.SOURCE_DATE_KEY: "2026-06-13",
                memory_module.SOURCE_USER_TS_KEY: "2026-06-13T09:00:00+00:00",
                memory_module.SOURCE_USER_TEXT_SHA256_KEY: __import__("hashlib").sha256(b"secret").hexdigest(),
                memory_module.SOURCE_ASSISTANT_TS_KEY: "2026-06-13T09:00:30+00:00",
            },
            created_at="",
        )
        from kai.memory import resolve_memory_scope

        resolved = resolve_memory_scope(fact.metadata)
        proposal = memory_reclassify.Proposal(
            memory_id="m1",
            verdict="global",
            project_id=None,
            confidence=0.9,
            reason="r",
            prior_scope_source="legacy_default",
            text_sha256="ab",
        )
        quotes = memory_reclassify._collect_provenance_quotes([(fact, resolved)], [proposal], user_id="1")
        assert quotes == {}
