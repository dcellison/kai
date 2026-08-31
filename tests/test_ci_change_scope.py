from scripts.ci_change_scope import ChangeScope, classify_paths


def test_client_sources_and_generated_assets_use_only_client_lane() -> None:
    assert classify_paths(
        [
            "workshop-client/src/App.tsx",
            "workshop-client/src/styles.css",
            "src/kai/workshop/static/app.js",
            "src/kai/workshop/static/app.css",
        ]
    ) == ChangeScope(client=True, full=False, dependency=False)


def test_documentation_only_change_needs_no_runtime_lane() -> None:
    assert classify_paths(["README.md", "docs/design.md", "home/docs/specs/workshop.md"]) == ChangeScope(
        client=False,
        full=False,
        dependency=False,
    )


def test_unknown_and_mixed_paths_fail_closed_to_complete_validation() -> None:
    assert classify_paths(["workshop-client/src/App.tsx", "mystery/new-input.dat"]) == ChangeScope(
        client=True,
        full=True,
        dependency=False,
    )


def test_backend_and_test_changes_retain_client_and_full_validation() -> None:
    assert classify_paths(["src/kai/workshop/client_api.py", "tests/test_bot.py"]) == ChangeScope(
        client=True,
        full=True,
        dependency=False,
    )


def test_python_dependency_inputs_select_full_and_audit_lanes() -> None:
    for path in (
        "pyproject.toml",
        "requirements/constraints.txt",
        ".github/workflows/dependency-audit.yml",
        "scripts/ci_change_scope.py",
        "Makefile",
    ):
        assert classify_paths([path]) == ChangeScope(client=True, full=True, dependency=True)


def test_empty_or_invalid_input_fails_closed() -> None:
    assert classify_paths([]) == ChangeScope(client=True, full=True, dependency=False)
    assert classify_paths(["../outside"]) == ChangeScope(client=True, full=True, dependency=False)


def test_force_full_selects_every_lane() -> None:
    assert classify_paths(["README.md"], force_full=True) == ChangeScope(
        client=True,
        full=True,
        dependency=True,
    )
