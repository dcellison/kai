from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kai.workshop.domain import PrincipalId, RuntimeProfileId
from kai.workshop.principal_policies import (
    WorkshopPrincipalPolicyAccessDenied,
    WorkshopPrincipalPolicyConflict,
    WorkshopPrincipalPolicyService,
    WorkshopPrincipalPolicyValidationError,
)
from kai.workshop.storage_namespaces import (
    WorkshopPrincipalStorageNamespace,
    WorkshopPrincipalStorageRegistry,
)


def _service(tmp_path: Path) -> tuple[WorkshopPrincipalPolicyService, PrincipalId, Path]:
    principal_id = PrincipalId("prn_" + "1" * 32)
    namespace = WorkshopPrincipalStorageNamespace(
        principal_id,
        RuntimeProfileId("rtp_" + "2" * 32),
        None,
    )
    home = namespace.home_directory(tmp_path)
    home.mkdir(parents=True, mode=0o700)
    home.chmod(0o700)
    policy = home / "AGENTS.md"
    policy.write_text("# Principal Policy\n\nKeep answers concise.\n", encoding="utf-8")
    policy.chmod(0o600)
    return (
        WorkshopPrincipalPolicyService(
            tmp_path,
            WorkshopPrincipalStorageRegistry((namespace,)),
            manager=tmp_path / "missing-manager",
            reader=tmp_path / "missing-reader",
        ),
        principal_id,
        policy,
    )


@pytest.mark.asyncio
async def test_principal_policy_save_is_owner_scoped_and_revision_checked(tmp_path: Path) -> None:
    service, principal_id, policy = _service(tmp_path)
    authority = service.authority_for_principal(principal_id)
    before = await service.read(authority)

    after = await service.save(
        authority,
        expected_revision=before.revision,
        content="# Principal Policy\n\nPrefer compact diffs.\n",
    )

    assert after.content == "# Principal Policy\n\nPrefer compact diffs.\n"
    assert after.revision == hashlib.sha256(after.content.encode()).hexdigest()
    assert policy.read_text(encoding="utf-8") == after.content
    with pytest.raises(WorkshopPrincipalPolicyConflict):
        await service.save(
            authority,
            expected_revision=before.revision,
            content="# Principal Policy\n\nStale write.\n",
        )


@pytest.mark.asyncio
async def test_principal_policy_rejects_agent_identity_and_other_principals(tmp_path: Path) -> None:
    service, principal_id, _policy = _service(tmp_path)
    authority = service.authority_for_principal(principal_id)
    before = await service.read(authority)

    with pytest.raises(WorkshopPrincipalPolicyValidationError, match="Agent identity"):
        await service.save(
            authority,
            expected_revision=before.revision,
            content="# Kai\n\nKeep answers concise.\n",
        )
    with pytest.raises(WorkshopPrincipalPolicyAccessDenied):
        service.authority_for_principal(PrincipalId("prn_" + "3" * 32))
