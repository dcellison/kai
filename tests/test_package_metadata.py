"""Tests for release package metadata consistency."""

import tomllib
from pathlib import Path

import kai


def test_package_version_matches_runtime_version():
    """The build metadata and runtime version must identify the same release."""
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)

    assert project["project"]["version"] == kai.__version__
