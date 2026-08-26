"""Operator policy for the first transport-independent Workshop human."""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass

from kai.workshop.domain import RuntimeProfileId, WorkshopId

WORKSHOP_BOOTSTRAP_ENV = "KAI_WORKSHOP_BOOTSTRAP"
_PROVISIONING_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class WorkshopInitialProvisioningError(RuntimeError):
    """The protected initial-Workshop policy is missing or malformed."""


@dataclass(frozen=True, slots=True)
class WorkshopInitialProvisioning:
    """Stable operator intent for one initial canonical Workshop admin."""

    workshop_id: WorkshopId
    provisioning_key: str
    display_name: str
    role: str
    runtime_profile_id: RuntimeProfileId

    def __post_init__(self) -> None:
        if not _PROVISIONING_KEY_PATTERN.fullmatch(self.provisioning_key):
            raise WorkshopInitialProvisioningError("Initial Workshop provisioning key must be a lowercase identifier")
        if not self.display_name.strip() or len(self.display_name.strip()) > 200:
            raise WorkshopInitialProvisioningError(
                "Initial Workshop display name must contain 1 through 200 characters"
            )
        if self.role not in {"admin", "member"}:
            raise WorkshopInitialProvisioningError("Initial Workshop role must be 'admin' or 'member'")

    @classmethod
    def create(cls, display_name: str) -> WorkshopInitialProvisioning:
        workshop_id = WorkshopId.new()
        provisioning_key = "initial-admin"
        return cls(
            workshop_id=workshop_id,
            provisioning_key=provisioning_key,
            display_name=display_name.strip(),
            role="admin",
            runtime_profile_id=RuntimeProfileId.derived(
                workshop_id,
                f"initial-runtime:{provisioning_key}",
            ),
        )

    @classmethod
    def from_json(cls, value: object) -> WorkshopInitialProvisioning:
        if not isinstance(value, str) or not value.strip():
            raise WorkshopInitialProvisioningError(f"{WORKSHOP_BOOTSTRAP_ENV} must contain a versioned policy")
        encoded = value.strip()
        if not encoded.startswith("v1."):
            raise WorkshopInitialProvisioningError(f"{WORKSHOP_BOOTSTRAP_ENV} must contain a version 1 policy")
        try:
            padding = "=" * (-len(encoded[3:]) % 4)
            document = json.loads(base64.urlsafe_b64decode(encoded[3:] + padding).decode("utf-8"))
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkshopInitialProvisioningError(f"{WORKSHOP_BOOTSTRAP_ENV} is not a valid encoded policy") from exc
        if not isinstance(document, dict) or document.get("version") != 1:
            raise WorkshopInitialProvisioningError(f"{WORKSHOP_BOOTSTRAP_ENV} must be a version 1 JSON object")
        expected = {
            "version",
            "workshop_id",
            "provisioning_key",
            "display_name",
            "role",
            "runtime_profile_id",
        }
        if set(document) != expected:
            raise WorkshopInitialProvisioningError(f"{WORKSHOP_BOOTSTRAP_ENV} contains missing or unsupported fields")
        try:
            return cls(
                workshop_id=WorkshopId(str(document["workshop_id"])),
                provisioning_key=str(document["provisioning_key"]),
                display_name=str(document["display_name"]),
                role=str(document["role"]),
                runtime_profile_id=RuntimeProfileId(str(document["runtime_profile_id"])),
            )
        except (TypeError, ValueError) as exc:
            raise WorkshopInitialProvisioningError(
                f"{WORKSHOP_BOOTSTRAP_ENV} contains an invalid opaque identifier"
            ) from exc

    def to_json(self) -> str:
        payload = json.dumps(
            {
                "version": 1,
                "workshop_id": str(self.workshop_id),
                "provisioning_key": self.provisioning_key,
                "display_name": self.display_name.strip(),
                "role": self.role,
                "runtime_profile_id": str(self.runtime_profile_id),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "v1." + base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def parse_initial_provisioning(value: object) -> WorkshopInitialProvisioning | None:
    """Parse an optional initial-provisioning policy."""
    if value is None or value == "":
        return None
    return WorkshopInitialProvisioning.from_json(value)
