"""Protected-install OS-user isolation validation.

The protected deployment runs the outer Kai daemon as a dedicated service
account and persistent conversational agents as per-user OS accounts.  Keeping
those identities distinct is a security boundary: the service account has
exact sudo permissions for protected Kai configuration that an agent must not
inherit.

Single-user repo deployments intentionally do not use this validator.  In that
mode the operator and agent are the same trust domain and no protected service
account exists.
"""

from collections.abc import Callable, Iterable

UserAssignment = tuple[int, str, str | None]


def validate_protected_user_isolation(
    assignments: Iterable[UserAssignment],
    service_user: str,
    *,
    account_uid: Callable[[str], int] | None = None,
    service_uid: int | None = None,
) -> tuple[str, ...]:
    """Validate the OS-user boundary for a protected installation.

    Every interactive Telegram principal must name an OS account, that account
    must differ from the outer service account, and two principals must not
    share an account.  Sharing would also share agent credentials, home files,
    and any data readable by that Unix identity.

    Returns the normalized target usernames in input order.  Raises
    ``ValueError`` with one actionable message containing every violation so an
    operator can repair ``users.yaml`` in one pass.  When ``account_uid`` is
    supplied, the validator also requires every target account to exist and
    compares resolved UIDs so passwd aliases cannot bypass the name checks.
    """
    service_user = service_user.strip()
    rows = list(assignments)
    errors: list[str] = []
    targets: list[str] = []
    principals_by_target: dict[str, list[str]] = {}
    principals_by_uid: dict[int, list[tuple[str, str]]] = {}

    if not rows:
        errors.append("users.yaml contains no valid interactive users")

    for telegram_id, name, raw_os_user in rows:
        principal = f"{name!r} (telegram_id={telegram_id})"
        os_user = raw_os_user.strip() if raw_os_user else ""
        if not os_user:
            errors.append(f"user {principal} is missing required os_user")
            continue
        if os_user == service_user:
            errors.append(f"user {principal} maps to service account {service_user!r}")
            continue
        targets.append(os_user)
        principals_by_target.setdefault(os_user, []).append(principal)
        if account_uid is not None:
            try:
                uid = account_uid(os_user)
            except KeyError:
                errors.append(f"user {principal} maps to nonexistent OS account {os_user!r}")
                continue
            if service_uid is not None and uid == service_uid:
                errors.append(
                    f"user {principal} maps to OS account {os_user!r}, which resolves to service uid {service_uid}"
                )
                continue
            principals_by_uid.setdefault(uid, []).append((os_user, principal))

    for os_user, principals in principals_by_target.items():
        if len(principals) > 1:
            errors.append(f"OS account {os_user!r} is shared by multiple Telegram principals: " + ", ".join(principals))

    for uid, resolved in principals_by_uid.items():
        names = {os_user for os_user, _principal in resolved}
        if len(names) > 1:
            principals = ", ".join(f"{principal} via {os_user!r}" for os_user, principal in resolved)
            errors.append(f"OS uid {uid} is shared by multiple account names and Telegram principals: {principals}")

    if errors:
        detail = "\n  - ".join(errors)
        raise ValueError(
            "Protected installation requires one unique, non-service os_user "
            "for every interactive user:\n"
            f"  - {detail}\n"
            "Assign each users.yaml entry its own existing OS account, distinct "
            f"from service account {service_user!r}."
        )

    return tuple(targets)
