"""Canonical scheduled-job type identifiers and compatibility handling."""

JOB_TYPE_REMINDER = "reminder"
JOB_TYPE_AGENT = "agent"

# Accepted only as an input/storage compatibility alias. New writes use
# JOB_TYPE_AGENT so the persisted identifier does not name an implementation.
LEGACY_JOB_TYPE_AGENT = "claude"

CANONICAL_JOB_TYPES = (JOB_TYPE_REMINDER, JOB_TYPE_AGENT)


def normalize_job_type(job_type: object) -> str:
    """Return a canonical job type or reject an unknown identity."""
    if job_type == LEGACY_JOB_TYPE_AGENT:
        return JOB_TYPE_AGENT
    if job_type in CANONICAL_JOB_TYPES:
        return str(job_type)
    valid = ", ".join(CANONICAL_JOB_TYPES)
    raise ValueError(f"job_type must be one of: {valid}")
