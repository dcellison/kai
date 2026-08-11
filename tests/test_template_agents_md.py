"""Tests for the tracked backend-neutral AGENTS.md template.

The template at templates/AGENTS.md is the source for every per-user
inner-agent identity file. It ships to new users via the install-time seed
step and is the upstream of the _migrate_recalled_memory_section helper.
Regressions to the structure or wording of pinned sections would propagate
into every freshly-seeded per-user copy on the next install.

Coverage focus: the `## Reading Recalled Memory` section added for the
anti-fabrication rule. The assertions pin the section's structure and the
worked-example literals against the values `_SOURCE_SHORT` actually emits
at render time, so a future regression that desyncs the example tags from
production output fails this test before reaching review.
"""

from __future__ import annotations

import re

from kai.config import PROJECT_ROOT

# Path to the tracked template. Resolved through PROJECT_ROOT so the test
# follows whatever install/dev anchoring config.py has bound rather than
# hardcoding a relative path that breaks under either layout.
_TEMPLATE = PROJECT_ROOT / "templates" / "AGENTS.md"

# The section header serves as the migration helper's sentinel. Pinning
# it here means a rename in the template fails this test before it
# silently breaks the install-time append logic in install.py.
_SECTION_HEADER = "## Reading Recalled Memory"

# The next top-level heading after the recalled-memory section. The
# section is bounded by the literal `## Behavioral Rules` line; the
# migration helper's header-bounded scan relies on this exact terminator.
_NEXT_TOP_HEADER = "## Behavioral Rules"


def _extract_section() -> str:
    """Return the recalled-memory section verbatim (header through terminator)."""
    text = _TEMPLATE.read_text()
    start = text.find(_SECTION_HEADER)
    assert start != -1, f"section header {_SECTION_HEADER!r} not found in template"
    # Search for the next `## ` from a position past the header itself,
    # not from start, so the section's own header line does not match.
    end = text.find("\n## ", start + len(_SECTION_HEADER))
    if end == -1:
        # Section runs to EOF; valid shape, included for completeness so
        # a future template that moves the section to the end of the
        # file still passes the structural checks.
        return text[start:]
    return text[start:end]


def test_section_header_present() -> None:
    """Tracked template contains the literal sentinel header."""
    assert _SECTION_HEADER in _TEMPLATE.read_text()


def test_section_ends_before_next_top_header() -> None:
    """Section is bounded by a top-level header (or EOF), not another `## `.

    Catches the regression where a future top-level section accidentally
    lands inside the recalled-memory block; the migration helper's
    header-bounded scan would copy too few lines and the appended block
    on existing per-user copies would be silently truncated.
    """
    section = _extract_section()
    # The section's own header is the first occurrence; any subsequent
    # `## ` at line start signals the terminator.
    inner = section[len(_SECTION_HEADER) :]
    assert "\n## " not in inner, (
        "recalled-memory section contains a nested top-level header; expected the next `## ` to terminate the section"
    )


def test_three_mode_labels_present() -> None:
    """Section contains the three mode labels the rule depends on."""
    section = _extract_section()
    for label in ("Citation", "Inference", "Partial match with gap"):
        assert label in section, f"mode label {label!r} missing from recalled-memory section"


def test_inference_example_hedge_present() -> None:
    """Inference worked example carries the `Memory doesn't state` hedge.

    The hedge phrasing is the load-bearing signal that distinguishes
    inference from citation: the agent says it is inferring, names what
    memory does NOT contain, and offers the bridge as a guess. Removing
    or rewording this hedge would degrade the worked example into a
    second citation example without anyone noticing in code review.
    """
    section = _extract_section()
    assert "Memory doesn't state" in section, "inference worked example must include the `Memory doesn't state` hedge"


def test_source_tag_literals_match_render_output() -> None:
    """Worked examples use the source tags `_SOURCE_SHORT` actually emits.

    `_SOURCE_SHORT` in `kai.memory` maps `extracted` -> `fact`,
    `migration` -> `fact`, `episode` -> `episode`. The agent never sees
    `extracted` in a row prefix. An earlier spec revision used
    `, extracted)` in the worked examples, which broke the pattern-match
    anchor the rule relies on; this test pins the corrected literals so
    a future regression of the same class fails before reaching production.
    """
    section = _extract_section()
    assert ", fact)" in section, (
        "citation/inference examples must use the `fact` source tag "
        "(what `_SOURCE_SHORT['extracted']` and `_SOURCE_SHORT['migration']` "
        "actually emit), not `extracted`"
    )
    assert ", episode," in section, (
        "partial-match example must use the `episode, <quality>` source-tag "
        "shape that `format_context` actually emits for episode rows"
    )


def test_no_em_or_en_dashes_in_section() -> None:
    """Section contains no em or en dashes.

    The project-wide rule is to use hyphens, semicolons, periods, or
    commas instead. Em/en dashes in operator-facing prompt content have
    bitten review rounds before; pinning the rule mechanically against
    the new section closes the recurrence path for this specific surface.
    """
    section = _extract_section()
    # The regex character class contains the literal en dash (U+2013) and
    # em dash (U+2014). The `noqa` suppresses ruff's RUF001 ambiguous-
    # character lint here; the literal characters are deliberate because
    # the test's whole purpose is to flag those exact characters in the
    # tracked template content.
    forbidden = re.findall("[–—]", section)  # noqa: RUF001
    assert not forbidden, (
        f"recalled-memory section contains em/en dashes: {forbidden!r}; "
        "use hyphens, semicolons, periods, or commas instead"
    )
