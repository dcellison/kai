"""Tests for the protected-install OS-user trust boundary."""

import pytest

from kai.user_isolation import validate_protected_user_isolation


def test_accepts_unique_non_service_accounts():
    targets = validate_protected_user_isolation(
        [(1, "alice", "alice"), (2, "bob", "bob")],
        "kai",
    )
    assert targets == ("alice", "bob")


def test_rejects_missing_os_user():
    with pytest.raises(ValueError, match="missing required os_user"):
        validate_protected_user_isolation([(1, "alice", None)], "kai")


def test_rejects_service_account():
    with pytest.raises(ValueError, match="maps to service account 'kai'"):
        validate_protected_user_isolation([(1, "alice", "kai")], "kai")


def test_rejects_shared_os_account():
    with pytest.raises(ValueError, match="shared by multiple Telegram principals"):
        validate_protected_user_isolation(
            [(1, "alice", "shared"), (2, "bob", "shared")],
            "kai",
        )


def test_reports_all_violations_together():
    with pytest.raises(ValueError) as excinfo:
        validate_protected_user_isolation(
            [
                (1, "missing", None),
                (2, "service", "kai"),
                (3, "shared-a", "shared"),
                (4, "shared-b", "shared"),
            ],
            "kai",
        )
    message = str(excinfo.value)
    assert "missing required os_user" in message
    assert "maps to service account 'kai'" in message
    assert "shared by multiple Telegram principals" in message


def test_rejects_empty_assignment_set():
    with pytest.raises(ValueError, match="no valid interactive users"):
        validate_protected_user_isolation([], "kai")


def test_rejects_account_alias_for_service_uid():
    uids = {"kai-alias": 503}
    with pytest.raises(ValueError, match="resolves to service uid 503"):
        validate_protected_user_isolation(
            [(1, "alice", "kai-alias")],
            "kai",
            account_uid=uids.__getitem__,
            service_uid=503,
        )


def test_rejects_two_account_names_for_same_uid():
    uids = {"alice": 501, "alice-alias": 501}
    with pytest.raises(ValueError, match="OS uid 501 is shared"):
        validate_protected_user_isolation(
            [(1, "alice", "alice"), (2, "bob", "alice-alias")],
            "kai",
            account_uid=uids.__getitem__,
            service_uid=503,
        )


def test_rejects_nonexistent_target_account():
    with pytest.raises(ValueError, match="nonexistent OS account 'missing'"):
        validate_protected_user_isolation(
            [(1, "alice", "missing")],
            "kai",
            account_uid={}.__getitem__,
            service_uid=503,
        )
