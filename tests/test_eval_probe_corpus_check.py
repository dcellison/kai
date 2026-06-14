"""Tests for the scoped-evaluator probe-corpus coverage check.

Three layers:

1. TestClassify - pure-function classification of a single probe
   (collision / positive-only / both / neither).
2. TestCoverageReport - bucketing per project, minimums and shortfalls,
   the `__none__` vs `__unregistered__` distinction, the empty-registry
   degenerate case.
3. TestRenderReport - the rendered output names every section in
   stable order and surfaces shortfalls verbatim.
4. TestCLIExitCodes - end-to-end exit code for met-all / shortfalls /
   load-failure paths.

The check is a pure structural pass over (probes, registry); no Mem0,
no filesystem outside the CLI smoke. Coverage minimums in the tests
are tuned to small numbers so the fixtures stay readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from kai.config import MemoryProjectConfig
from kai.eval.probe_corpus_check import (
    CoverageMinimums,
    ProjectCoverage,
    _classify_probe,
    check_corpus_coverage,
    main,
    render_report,
)
from kai.eval.retrieval_scoped import ScopedProbe

# ── Shared fixtures ────────────────────────────────────────────────


def _probe(
    *,
    line: int = 1,
    expected_fact_id: str | None = "fact-a",
    expected_excluded_fact_ids: tuple[str, ...] = (),
    workspace: str | None = None,
) -> ScopedProbe:
    return ScopedProbe(
        question="q",
        expected_fact_id=expected_fact_id,
        expected_excluded_fact_ids=expected_excluded_fact_ids,
        workspace=workspace,
        line_number=line,
    )


def _registry(tmp_path: Path, *project_ids: str) -> dict[str, MemoryProjectConfig]:
    """Build a registry whose project roots live under tmp_path/<project_id>."""
    registry: dict[str, MemoryProjectConfig] = {}
    for pid in project_ids:
        root = tmp_path / pid
        root.mkdir(parents=True, exist_ok=True)
        registry[pid] = MemoryProjectConfig(
            project_id=pid,
            display_name=pid,
            workspace_roots=(root.resolve(),),
            memory_enabled=True,
            default_scope_for_new_facts="project",
        )
    return registry


_MINIMUMS = CoverageMinimums(
    min_collision_per_project=2,
    min_positive_per_project=1,
    min_non_project=1,
)


# ── Test 1: Classify ───────────────────────────────────────────────


class TestClassify:
    def test_positive_only_when_no_exclusions(self):
        is_collision, is_positive_only = _classify_probe(_probe(expected_fact_id="x", expected_excluded_fact_ids=()))
        assert is_collision is False
        assert is_positive_only is True

    def test_collision_only_when_no_positive(self):
        is_collision, is_positive_only = _classify_probe(
            _probe(expected_fact_id=None, expected_excluded_fact_ids=("bad",))
        )
        assert is_collision is True
        assert is_positive_only is False

    def test_both_when_positive_and_exclusion(self):
        is_collision, is_positive_only = _classify_probe(
            _probe(expected_fact_id="x", expected_excluded_fact_ids=("bad",))
        )
        # Both polarities present. positive_only is False because the
        # probe contributes to the safety signal too; it is NOT a
        # "positive only" probe.
        assert is_collision is True
        assert is_positive_only is False


# ── Test 2: Coverage report ────────────────────────────────────────


class TestCoverageReport:
    def test_per_project_bucketing(self, tmp_path):
        registry = _registry(tmp_path, "kai", "anvil")
        kai_root = str(registry["kai"].workspace_roots[0])
        anvil_root = str(registry["anvil"].workspace_roots[0])
        probes = [
            _probe(line=1, expected_fact_id="x", expected_excluded_fact_ids=("y",), workspace=kai_root),
            _probe(line=2, expected_fact_id=None, expected_excluded_fact_ids=("z",), workspace=kai_root),
            _probe(line=3, expected_fact_id="p", workspace=anvil_root),
        ]
        report = check_corpus_coverage(probes, registry, _MINIMUMS)
        assert report.per_project["kai"] == ProjectCoverage(
            project_id="kai",
            total_probes=2,
            collision_probes=2,
            # Both kai probes carry exclusions; neither is positive-only.
            positive_only_probes=0,
        )
        assert report.per_project["anvil"] == ProjectCoverage(
            project_id="anvil",
            total_probes=1,
            collision_probes=0,
            positive_only_probes=1,
        )

    def test_non_project_vs_unregistered(self, tmp_path):
        registry = _registry(tmp_path, "kai")
        probes = [
            # workspace=null is deliberate non-project.
            _probe(line=1, expected_fact_id="x", workspace=None),
            # workspace set but not registered: authoring mistake.
            _probe(line=2, expected_fact_id="y", workspace=str(tmp_path / "other")),
        ]
        report = check_corpus_coverage(probes, registry, _MINIMUMS)
        assert report.non_project_count == 1
        assert report.unregistered_workspace_count == 1
        # Neither contributed to kai's counts.
        assert report.per_project["kai"].total_probes == 0

    def test_empty_registry_is_a_shortfall_by_default(self):
        # The tool's primary job is per-project safety coverage; an
        # empty memory-enabled registry plus no per-project shortfalls
        # would have been a meaningless "all met" before. Now it
        # surfaces an explicit shortfall so the operator either fixes
        # the registry or opts in to --non-project-only.
        report = check_corpus_coverage([], {}, _MINIMUMS)
        assert report.per_project == {}
        assert any("registry has no memory-enabled projects" in s for s in report.shortfalls)
        # The non-project minimum still fires too because the corpus
        # is empty.
        assert any("non-project (workspace=null)" in s for s in report.shortfalls)

    def test_empty_registry_passes_under_non_project_only(self, tmp_path):
        # Corpus deliberately exercises only the global section; the
        # non-project minimum is the only check that applies.
        probes = [_probe(line=1, expected_fact_id="x", workspace=None)]
        report = check_corpus_coverage([*probes], {}, _MINIMUMS, non_project_only=True)
        assert report.shortfalls == []

    def test_disabled_project_not_required_for_coverage(self, tmp_path):
        # A YAML-pinned project with memory_enabled=False: production
        # scope filter returns global-only at admission, so per-project
        # collision probes against it would assert a property retrieval
        # does not enforce. The coverage check filters disabled
        # projects out of per-project minimums.
        registry = _registry(tmp_path, "kai")
        disabled_root = tmp_path / "disabled"
        disabled_root.mkdir(parents=True, exist_ok=True)
        registry["disabled"] = MemoryProjectConfig(
            project_id="disabled",
            display_name="disabled",
            workspace_roots=(disabled_root.resolve(),),
            memory_enabled=False,
            default_scope_for_new_facts="project",
        )
        # Corpus exercises kai correctly but has nothing for the
        # disabled project; this should pass without complaint about
        # the disabled project.
        kai_root = str(registry["kai"].workspace_roots[0])
        probes = [
            _probe(line=1, expected_fact_id=None, expected_excluded_fact_ids=("y",), workspace=kai_root),
            _probe(line=2, expected_fact_id=None, expected_excluded_fact_ids=("z",), workspace=kai_root),
            _probe(line=3, expected_fact_id="p", workspace=kai_root),
            _probe(line=4, expected_fact_id="q", workspace=None),
        ]
        report = check_corpus_coverage(probes, registry, _MINIMUMS)
        # `disabled` does not appear in per_project (filtered out).
        assert "disabled" not in report.per_project
        # No shortfall mentions the disabled project.
        assert all("disabled" not in s for s in report.shortfalls)
        # And the run is otherwise clean.
        assert report.shortfalls == []

    def test_probe_pinned_to_disabled_project_does_not_count_as_unregistered(self, tmp_path):
        # A probe whose workspace matches a disabled project's root
        # should NOT be bucketed as unregistered (the path IS
        # registered); it just does not contribute to coverage.
        registry = _registry(tmp_path, "kai")
        disabled_root = tmp_path / "disabled"
        disabled_root.mkdir(parents=True, exist_ok=True)
        registry["disabled"] = MemoryProjectConfig(
            project_id="disabled",
            display_name="disabled",
            workspace_roots=(disabled_root.resolve(),),
            memory_enabled=False,
            default_scope_for_new_facts="project",
        )
        probes = [_probe(line=1, expected_fact_id="x", workspace=str(disabled_root))]
        report = check_corpus_coverage(probes, registry, _MINIMUMS, non_project_only=False)
        assert report.unregistered_workspace_count == 0

    def test_unregistered_workspace_is_a_shortfall(self, tmp_path):
        # Probe with a workspace path that no registered project
        # matches: the operator forgot to pin a chat-registered project
        # or the path is stale. The check must surface this.
        registry = _registry(tmp_path, "kai")
        kai_root = str(registry["kai"].workspace_roots[0])
        probes = [
            # Coverage for kai is fine.
            _probe(line=1, expected_fact_id=None, expected_excluded_fact_ids=("y",), workspace=kai_root),
            _probe(line=2, expected_fact_id=None, expected_excluded_fact_ids=("z",), workspace=kai_root),
            _probe(line=3, expected_fact_id="p", workspace=kai_root),
            _probe(line=4, expected_fact_id="q", workspace=None),
            # One probe targeting an unregistered path.
            _probe(line=5, expected_fact_id="r", workspace=str(tmp_path / "stale")),
        ]
        report = check_corpus_coverage(probes, registry, _MINIMUMS)
        assert report.unregistered_workspace_count == 1
        assert any("unregistered workspace" in s for s in report.shortfalls)

    def test_shortfalls_name_every_gap(self, tmp_path):
        registry = _registry(tmp_path, "kai")
        kai_root = str(registry["kai"].workspace_roots[0])
        probes = [
            # One collision probe; need two.
            _probe(line=1, expected_fact_id=None, expected_excluded_fact_ids=("y",), workspace=kai_root),
            # Two non-project probes; need one. Satisfied.
            _probe(line=2, expected_fact_id="z", workspace=None),
        ]
        # Note: no positive-only probes for kai project.
        report = check_corpus_coverage(probes, registry, _MINIMUMS)
        assert "collision" in report.shortfalls[0]
        assert "positive-only" in report.shortfalls[1]
        # Non-project minimum is met (2 >= 1) so no shortfall there.
        assert all("non-project" not in s for s in report.shortfalls)

    def test_all_minimums_met_produces_empty_shortfalls(self, tmp_path):
        registry = _registry(tmp_path, "kai")
        kai_root = str(registry["kai"].workspace_roots[0])
        probes = [
            _probe(line=1, expected_fact_id=None, expected_excluded_fact_ids=("y",), workspace=kai_root),
            _probe(line=2, expected_fact_id=None, expected_excluded_fact_ids=("z",), workspace=kai_root),
            _probe(line=3, expected_fact_id="p", workspace=kai_root),
            _probe(line=4, expected_fact_id="q", workspace=None),
        ]
        report = check_corpus_coverage(probes, registry, _MINIMUMS)
        assert report.shortfalls == []


# ── Test 3: Render report ──────────────────────────────────────────


class TestRenderReport:
    def test_met_all_minimums_renders_clean_summary(self, tmp_path):
        registry = _registry(tmp_path, "kai")
        kai_root = str(registry["kai"].workspace_roots[0])
        probes = [
            _probe(line=1, expected_fact_id=None, expected_excluded_fact_ids=("y",), workspace=kai_root),
            _probe(line=2, expected_fact_id=None, expected_excluded_fact_ids=("z",), workspace=kai_root),
            _probe(line=3, expected_fact_id="p", workspace=kai_root),
            _probe(line=4, expected_fact_id="q", workspace=None),
        ]
        report = check_corpus_coverage(probes, registry, _MINIMUMS)
        out = render_report(report)
        assert "Probe corpus coverage report" in out
        assert "Total probes: 4" in out
        assert "Non-project (workspace=null): 1 (need 1)" in out
        assert "All minimums met." in out

    def test_shortfall_lines_render(self, tmp_path):
        registry = _registry(tmp_path, "kai")
        probes = []  # nothing.
        report = check_corpus_coverage(probes, registry, _MINIMUMS)
        out = render_report(report)
        assert "Shortfalls:" in out
        # Per-project AND non-project shortfalls both surface.
        assert "collision probes" in out
        assert "positive-only probes" in out
        assert "non-project" in out


# ── Test 4: CLI exit codes ─────────────────────────────────────────


def _write_probes(path: Path, probes: list[dict]) -> None:
    """Write a v2 probe file the scoped loader will accept."""
    with open(path, "w", encoding="utf-8") as f:
        for p in probes:
            f.write(json.dumps(p) + "\n")


class TestCLIExitCodes:
    def test_exit_zero_when_minimums_met(self, tmp_path, capsys):
        registry = _registry(tmp_path, "kai")
        kai_root = str(registry["kai"].workspace_roots[0])
        probes_path = tmp_path / "probes.jsonl"
        _write_probes(
            probes_path,
            [
                {"question": "q1", "expected_excluded_fact_ids": ["y"], "workspace": kai_root},
                {"question": "q2", "expected_excluded_fact_ids": ["z"], "workspace": kai_root},
                {"question": "q3", "expected_fact_id": "p", "workspace": kai_root},
                {"question": "q4", "expected_fact_id": "q"},
            ],
        )
        # Patch load_config to return a config whose memory_projects
        # is our test registry. The CLI uses the YAML registry, so we
        # do not need to touch the session DB or Mem0.
        fake_config = type(
            "FakeConfig",
            (),
            {"memory_projects": registry},
        )()
        with patch("kai.config.load_config", return_value=fake_config):
            code = main(
                [
                    str(probes_path),
                    "--min-collision-per-project",
                    "2",
                    "--min-positive-per-project",
                    "1",
                    "--min-non-project",
                    "1",
                ]
            )
        assert code == 0
        out = capsys.readouterr().out
        assert "All minimums met." in out

    def test_exit_two_when_shortfalls_present(self, tmp_path, capsys):
        registry = _registry(tmp_path, "kai")
        kai_root = str(registry["kai"].workspace_roots[0])
        probes_path = tmp_path / "probes.jsonl"
        _write_probes(
            probes_path,
            # Only one collision probe; needs two. No non-project probe.
            [{"question": "q1", "expected_excluded_fact_ids": ["y"], "workspace": kai_root}],
        )
        fake_config = type("FakeConfig", (), {"memory_projects": registry})()
        with patch("kai.config.load_config", return_value=fake_config):
            code = main(
                [
                    str(probes_path),
                    "--min-collision-per-project",
                    "2",
                    "--min-positive-per-project",
                    "1",
                    "--min-non-project",
                    "1",
                ]
            )
        assert code == 2
        out = capsys.readouterr().out
        assert "Shortfalls:" in out

    def test_exit_one_when_probes_load_fails(self, tmp_path, capsys):
        # File that exists but contains invalid JSONL: the scoped
        # loader raises ValueError; the CLI must exit 1 (IO/load
        # failure) and NOT exit 2 (which is "loaded but shortfalls").
        probes_path = tmp_path / "probes.jsonl"
        probes_path.write_text("this is not json at all\n", encoding="utf-8")
        code = main([str(probes_path)])
        assert code == 1
        err = capsys.readouterr().err
        assert "failed to load probes" in err

    def test_exit_one_when_probes_file_missing(self, tmp_path, capsys):
        # Nonexistent path: the loader raises OSError; CLI exits 1.
        code = main([str(tmp_path / "missing.jsonl")])
        assert code == 1
        err = capsys.readouterr().err
        assert "failed to load probes" in err

    def test_exit_two_when_registry_empty_without_non_project_only(self, tmp_path, capsys):
        # Empty YAML registry plus a corpus that satisfies the non-
        # project minimum: under the old behavior this exited 0 (false
        # green); under the new behavior the empty registry is itself
        # a shortfall and exit is 2.
        probes_path = tmp_path / "probes.jsonl"
        _write_probes(probes_path, [{"question": "q", "expected_fact_id": "x"}])
        fake_config = type("FakeConfig", (), {"memory_projects": {}})()
        with patch("kai.config.load_config", return_value=fake_config):
            code = main(
                [
                    str(probes_path),
                    "--min-collision-per-project",
                    "2",
                    "--min-positive-per-project",
                    "1",
                    "--min-non-project",
                    "1",
                ]
            )
        assert code == 2
        out = capsys.readouterr().out
        assert "registry has no memory-enabled projects" in out

    def test_exit_zero_under_non_project_only_with_empty_registry(self, tmp_path, capsys):
        # Same setup as above but the operator opts in to non-project-
        # only: empty registry is no longer a shortfall.
        probes_path = tmp_path / "probes.jsonl"
        _write_probes(probes_path, [{"question": "q", "expected_fact_id": "x"}])
        fake_config = type("FakeConfig", (), {"memory_projects": {}})()
        with patch("kai.config.load_config", return_value=fake_config):
            code = main(
                [
                    str(probes_path),
                    "--non-project-only",
                    "--min-non-project",
                    "1",
                ]
            )
        assert code == 0

    def test_exit_one_when_probes_file_empty(self, tmp_path, capsys):
        # Probe file with only comment lines; loader returns [].
        probes_path = tmp_path / "probes.jsonl"
        probes_path.write_text("# just a comment\n", encoding="utf-8")
        code = main([str(probes_path)])
        assert code == 1
        err = capsys.readouterr().err
        assert "no probes loaded" in err
