"""
Shared prompt construction utilities.

Small helpers used by multiple agent modules (review.py, triage.py) that
are too simple to warrant their own module but need to be shared to avoid
duplication of security-relevant code.
"""

import secrets


def make_boundary(label: str) -> tuple[str, str]:
    """
    Generate a pair of randomized boundary delimiters for prompt injection prevention.

    Each call produces a unique 8-character hex token, making it computationally
    infeasible for injected content to guess and forge a closing delimiter.
    Used by review.py and triage.py to wrap untrusted data in prompts.

    Args:
        label: Human-readable label for the boundary (e.g., "ISSUE_BODY").

    Returns:
        A (begin, end) tuple of delimiter strings.
    """
    token = secrets.token_hex(4)
    return (f"--- BEGIN {label} {token} ---", f"--- END {label} {token} ---")
