"""Tests for scripts/module-sizes.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "module-sizes.py"


def _load_module_sizes():
    spec = importlib.util.spec_from_file_location("kai_module_sizes_script", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_iter_python_files_excludes_cache_trees(tmp_path):
    module_sizes = _load_module_sizes()
    root = tmp_path / "src" / "kai"
    root.mkdir(parents=True)
    kept = root / "bot.py"
    kept.write_text("print('ok')\n")
    cache_dir = root / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "bot.py").write_text("not real source\n")

    assert list(module_sizes.iter_python_files(root)) == [kept]


def test_measure_modules_sorts_largest_first(tmp_path):
    module_sizes = _load_module_sizes()
    root = tmp_path / "src" / "kai"
    root.mkdir(parents=True)
    small = root / "small.py"
    large = root / "large.py"
    small.write_text("one\n")
    large.write_text("one\ntwo\nthree\n")

    result = module_sizes.measure_modules(root)

    assert [(row.lines, row.path) for row in result] == [(3, large), (1, small)]


def test_render_report_counts_modules_at_threshold(tmp_path):
    module_sizes = _load_module_sizes()
    first = module_sizes.ModuleSize(lines=10, path=tmp_path / "first.py")
    second = module_sizes.ModuleSize(lines=5, path=tmp_path / "second.py")

    report = module_sizes.render_report([first, second], limit=5, top=1)

    assert "threshold: 5 lines" in report
    assert "Modules at or above threshold: 2" in report
    assert "first.py" in report
    assert "second.py" not in report
