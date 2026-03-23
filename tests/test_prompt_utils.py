"""Tests for prompt_utils.py shared prompt construction utilities."""

from kai.prompt_utils import make_boundary


class TestMakeBoundary:
    def test_unique_tokens(self):
        """Each call to make_boundary produces a different token."""
        begin1, end1 = make_boundary("TEST")
        begin2, end2 = make_boundary("TEST")
        assert begin1 != begin2
        assert end1 != end2

    def test_format_and_token_pairing(self):
        """Boundary strings follow the expected format with matching tokens."""
        begin, end = make_boundary("ISSUE_BODY")
        assert begin.startswith("--- BEGIN ISSUE_BODY ")
        assert begin.endswith(" ---")
        assert end.startswith("--- END ISSUE_BODY ")
        assert end.endswith(" ---")
        # Verify the same token appears in both begin and end
        token = begin.split()[-2]
        assert end == f"--- END ISSUE_BODY {token} ---"
