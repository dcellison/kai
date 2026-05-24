"""Snapshot tests for `_EXTRACTION_SYSTEM_PROMPT` structure.

Pins the boundary properties of the extraction prompt that this
revision targets: the fact-side IGNORE block and DURABILITY TEST are
gone, the QUALITY TEST block is in their place, the version stamp has
bumped, and the worked-example anchors are intact. Pinning these
structurally (not by exact full-prompt string) keeps the tests stable
under minor wording edits while still catching the regressions this
revision is intended to prevent.

The em/en-dash gate matches the project-wide convention and mirrors
`tests/test_template_claude_md.py::test_no_em_or_en_dashes_in_section`.
"""

from __future__ import annotations

import re

from kai.memory_extraction import _EXTRACTION_PROMPT_VERSION, _EXTRACTION_SYSTEM_PROMPT

# The new section's header literal. Pinning it here means a rename in
# the prompt fails this test before it silently breaks the production
# extractor's reliance on the QUALITY TEST as the upfront-floor gate.
_QUALITY_TEST_HEADER = "QUALITY TEST:"


def _extract_quality_test_block() -> str:
    """Return the QUALITY TEST section from header through its terminator.

    Terminator is the next ALL-CAPS section header in the prompt
    (CONFIDENCE, FORMAT, EPISODE CLASSIFICATION, etc.). The block-bounded
    scan lets the dash-detection test below operate on just the new
    content rather than the whole prompt (which has unrelated text the
    project's no-dashes rule does not apply to retroactively).
    """
    start = _EXTRACTION_SYSTEM_PROMPT.find(_QUALITY_TEST_HEADER)
    assert start != -1, f"{_QUALITY_TEST_HEADER!r} missing from prompt"
    # Terminator: the next ALL-CAPS-line section header. CONFIDENCE
    # follows QUALITY TEST in the current prompt structure; the scan
    # tolerates any future intervening ALL-CAPS section by matching
    # the first all-caps header after QUALITY TEST.
    tail = _EXTRACTION_SYSTEM_PROMPT[start + len(_QUALITY_TEST_HEADER) :]
    next_header = re.search(r"\n([A-Z][A-Z _]+):", tail)
    if next_header is None:
        # Section runs to end of prompt; valid shape for completeness.
        return _EXTRACTION_SYSTEM_PROMPT[start:]
    return _EXTRACTION_SYSTEM_PROMPT[start : start + len(_QUALITY_TEST_HEADER) + next_header.start()]


def test_quality_test_header_present():
    """Section header literal landed in the prompt."""
    assert _QUALITY_TEST_HEADER in _EXTRACTION_SYSTEM_PROMPT


def test_ignore_block_removed():
    """The fact-side IGNORE: header is gone.

    The episode-side block has its own header (`EPISODE IGNORE rules`)
    which contains the word IGNORE but is a different literal; that
    block stays per parent epic scope.
    """
    assert "IGNORE:" not in _EXTRACTION_SYSTEM_PROMPT, (
        "fact-side IGNORE: block should have been replaced by QUALITY TEST"
    )


def test_durability_test_removed():
    """DURABILITY TEST: no longer appears anywhere in the prompt.

    The fact-side DURABILITY TEST was replaced by QUALITY TEST in the
    fact-side revision; the episode-side EPISODE DURABILITY TEST was
    replaced by EPISODE QUALITY TEST in the episode-side revision.
    Asserting absence (count == 0) guards both removals; a future
    edit re-introducing the legacy header on either side fails this
    test before the stale wording reaches production.
    """
    assert _EXTRACTION_SYSTEM_PROMPT.count("DURABILITY TEST") == 0


def test_store_block_does_not_reference_durability_test():
    """The fact-side STORE block must not direct the model to a
    DURABILITY TEST that no longer exists.

    The colon-suffixed `DURABILITY TEST:` header check above is too
    narrow: a bare-form `DURABILITY TEST below` inside the STORE
    block leaves the model pointing at a section that was deleted in
    the v9 swap. The originating regression: the STORE block's
    "Apply the DURABILITY TEST below" reference survived that swap
    because the deletion range was line-based and the STORE block
    sat outside it.

    Pinning the absence of any "DURABILITY TEST" substring in the
    fact-side STORE block - regardless of suffix - closes the
    recurrence path for that orphan-pointer shape. The sibling
    `test_durability_test_removed` covers global absence; this test
    bounds itself to the fact-side STORE region so the assertion
    error message stays attributable to the original failure
    surface rather than a generic prompt-wide grep.
    """
    # Bound the search to the fact-side STORE region: from "STORE
    # these fact types:" through (but not including) "EPISODE
    # CLASSIFICATION". The bound keeps this test focused on the
    # historical orphan-pointer site even though `DURABILITY TEST`
    # is now absent everywhere (also asserted by the sibling test).
    store_idx = _EXTRACTION_SYSTEM_PROMPT.index("STORE these fact types:")
    episode_idx = _EXTRACTION_SYSTEM_PROMPT.index("EPISODE CLASSIFICATION")
    fact_side = _EXTRACTION_SYSTEM_PROMPT[store_idx:episode_idx]
    assert "DURABILITY TEST" not in fact_side, (
        "fact-side STORE/QUALITY TEST region must not reference DURABILITY TEST; "
        "the section was deleted in v9 and any reference is an orphan pointer"
    )


def test_prompt_version_bumped():
    """Version stamp matches the prompt revision."""
    assert _EXTRACTION_PROMPT_VERSION == "10"


def test_no_em_or_en_dashes_in_quality_test_block():
    """The new section uses hyphens, semicolons, periods, or commas only.

    Em/en dashes have bitten prior review rounds; mechanically pinning
    the rule against the new section closes the recurrence path for
    this surface. The block-scoped check (rather than full-prompt)
    avoids retroactively gating older sections that may contain the
    characters legitimately.

    Literal characters in the regex are the en dash (U+2013) and em
    dash (U+2014). The noqa suppresses ruff's RUF001 ambiguous-
    character lint here; the literals are deliberate because the test's
    purpose is to flag those exact characters.
    """
    block = _extract_quality_test_block()
    forbidden = re.findall("[–—]", block)  # noqa: RUF001
    assert not forbidden, (
        f"QUALITY TEST block contains em/en dashes: {forbidden!r}; use hyphens, semicolons, periods, or commas instead"
    )


def test_positive_worked_examples_present():
    """The three positive (emit) worked examples are in the block.

    Each example's distinctive opener pins the example against
    accidental rewording in a future edit. Removing or paraphrasing
    one of these would fail the test before reaching review, matching
    the v3 review's hedge-language pinning pattern in the inference
    example for spec #430.
    """
    block = _extract_quality_test_block()
    for snippet in (
        '"I prefer Celsius."',
        '"I live in Toronto."',
        '"My laptop is a 2024 M3 MacBook Pro."',
    ):
        assert snippet in block, f"positive worked example missing: {snippet}"


def test_negative_worked_examples_present():
    """The four negative (do-not-emit) worked examples are in the block.

    Same pinning rationale as the positive set. The four negatives
    cover the four classes the prior IGNORE list targeted: workflow-
    event metadata, decision-to-do, assistant self-report, and
    in-progress task state ("ephemeral state"). PR #467 review (W-2)
    added the ephemeral-state example to reconcile the spec prose
    that named four classes against the original block which only
    carried three.
    """
    block = _extract_quality_test_block()
    for snippet in (
        '"Spec X v3 was approved"',
        '"Let\'s file an issue about X."',
        '"I\'m extracting facts now"',
        '"I\'m writing the spec now."',
    ):
        assert snippet in block, f"negative worked example missing: {snippet}"
