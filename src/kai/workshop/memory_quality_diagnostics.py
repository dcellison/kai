"""Privacy-bounded diagnostics for canonical memory extraction quality."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kai.config import (
    ModelRole,
    canonicalize_model_for_backend,
    get_model_for,
    validate_model_for_backend_policy,
)
from kai.memory_model_evaluation import EVALUATION_SCORE_VERSION, EVALUATION_VERSION
from kai.memory_quality_corpus import CORPUS_VERSION, REVIEW_VERSION
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileError,
    WorkshopRuntimeProfileRegistry,
)

_FRAGMENTATION_ALERT_RATE = 0.25
_CANDIDATE_STARVATION_RATE = 0.80
_MIN_RATE_SAMPLE = 5
_MAX_FACTS_PER_RECEIPT = 3
_REPEATED_FAILURES = 2
_PROVIDER_FAILURES = frozenset({"authentication_failure", "provider_failure", "timeout"})
_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9._:/@+\-]")
_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MemoryQualityDiagnostic:
    """One content-free snapshot rendered by installed diagnostics."""

    state: str
    configured_roles: tuple[str, ...]
    actual_roles: tuple[str, ...]
    prompt_versions: tuple[str, ...]
    attempts: int
    admitted: int
    suppressed: int
    unclassified: int
    failed: int
    stored: int
    replaced: int
    skipped: int
    classifier_positive: int
    episode_attempts: int
    episode_stored: int
    episode_failed: int
    corpus_baselines: int
    qualified_baselines: int
    comparison_scores: int
    malformed_artifacts: int
    alerts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryRouteProvenance:
    """Content-free extraction route attributable to one written memory."""

    memory_id: str
    outcome: str
    extraction_role: str
    backend: str
    provider: str
    model: str
    prompt_version: str
    receipt_id: str
    created_at: str


@dataclass(frozen=True, slots=True)
class _Receipt:
    runtime_profile_id: str
    role: str
    backend: str
    provider: str
    model: str
    prompt_version: str
    status: str
    failure_code: str | None
    candidate_count: int
    classifier_result: bool | None
    raw_count: int
    accepted_count: int
    admission: str | None
    fragmentation_rejected: int
    batch_limit: int | None
    stored_count: int
    replaced_count: int
    skipped_count: int
    created_at: str


def workshop_memory_extraction_quality_status(
    db_path: Path,
    *,
    runtime_policy_path: Path | None = None,
    default_models: Mapping[str, str] | None = None,
    quality_root: Path | None = None,
) -> str:
    """Report exact role provenance and aggregate alerts without private content."""

    prefix = "Workshop memory extraction quality:"
    try:
        snapshot = build_memory_quality_diagnostic(
            db_path,
            runtime_policy_path=runtime_policy_path,
            default_models=default_models,
            quality_root=quality_root,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    configured = ",".join(snapshot.configured_roles) or "unavailable"
    actual = ",".join(snapshot.actual_roles) or "none"
    prompts = ",".join(snapshot.prompt_versions) or "none"
    alerts = ",".join(snapshot.alerts) or "none"
    return (
        f"{prefix} {snapshot.state}; configured role defaults=[{configured}]; "
        f"actual receipt routes=[{actual}]; prompts=[{prompts}]; "
        f"fact passes={snapshot.attempts} (admitted={snapshot.admitted}, "
        f"suppressed={snapshot.suppressed}, unclassified={snapshot.unclassified}, "
        f"failed={snapshot.failed}), writes=(stored={snapshot.stored}, "
        f"replaced={snapshot.replaced}, skipped={snapshot.skipped}); "
        f"episodes=(classified={snapshot.classifier_positive}, attempted={snapshot.episode_attempts}, "
        f"stored={snapshot.episode_stored}, failed={snapshot.episode_failed}); "
        f"corpus=(schema=v{CORPUS_VERSION}, review=v{REVIEW_VERSION}, "
        f"evaluation=v{EVALUATION_VERSION}, score=v{EVALUATION_SCORE_VERSION}, "
        f"baselines={snapshot.corpus_baselines}, qualified={snapshot.qualified_baselines}, "
        f"comparisons={snapshot.comparison_scores}, malformed={snapshot.malformed_artifacts}); "
        f"alerts={alerts}; authority=aggregate/redacted"
    )


def build_memory_quality_diagnostic(
    db_path: Path,
    *,
    runtime_policy_path: Path | None = None,
    default_models: Mapping[str, str] | None = None,
    quality_root: Path | None = None,
) -> MemoryQualityDiagnostic:
    if not db_path.is_file():
        raise OSError("database unavailable")
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        tables = {
            str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        if "memory_extraction_receipts" not in tables:
            raise ValueError("memory extraction receipts unavailable")
        receipts = _load_receipts(connection)
        missing_receipts = _missing_receipt_count(connection, tables)
    finally:
        connection.close()

    profiles = _load_profiles(runtime_policy_path)
    configured = _configured_roles(profiles, default_models or {})
    actual = _actual_roles(receipts)
    prompts = _prompt_versions(receipts)
    facts = [receipt for receipt in receipts if receipt.role == "fact_extraction"]
    episodes = [receipt for receipt in receipts if receipt.role == "episode_generation"]
    admitted = sum(receipt.admission == "admitted" for receipt in facts)
    suppressed = sum(receipt.admission == "suppressed" for receipt in facts)
    unclassified = sum(receipt.admission is None for receipt in facts)
    failed = sum(receipt.status == "failed" for receipt in facts)
    stored = sum(receipt.stored_count for receipt in facts)
    replaced = sum(receipt.replaced_count for receipt in facts)
    skipped = sum(receipt.skipped_count for receipt in facts)
    classifier_positive = sum(receipt.classifier_result is True for receipt in facts)
    episode_stored = sum(receipt.stored_count for receipt in episodes)
    episode_failed = sum(receipt.status == "failed" for receipt in episodes)

    alerts: list[str] = []
    if missing_receipts:
        alerts.append(f"missing_receipts:{missing_receipts}")
    drift = _model_drift_count(receipts, profiles, default_models or {})
    if drift:
        alerts.append(f"model_drift:{drift}")
    repeated_failures = _repeated_provider_failures(receipts)
    if repeated_failures:
        alerts.append(f"repeated_provider_failures:{repeated_failures}")
    abnormal_volume = sum(
        receipt.accepted_count > _MAX_FACTS_PER_RECEIPT
        or receipt.stored_count > _MAX_FACTS_PER_RECEIPT
        or (receipt.batch_limit is not None and receipt.batch_limit > 1)
        for receipt in facts
    )

    if abnormal_volume:
        alerts.append(f"abnormal_volume:{abnormal_volume}")
    raw_facts = sum(receipt.raw_count for receipt in facts)
    fragmented = sum(receipt.fragmentation_rejected for receipt in facts)
    if raw_facts >= 10 and fragmented / raw_facts > _FRAGMENTATION_ALERT_RATE:
        alerts.append("fragmentation_high")
    accepted_passes = [receipt for receipt in facts if receipt.admission == "admitted" and receipt.accepted_count]
    starved = sum(receipt.candidate_count == 0 for receipt in accepted_passes)
    if len(accepted_passes) >= _MIN_RATE_SAMPLE and starved / len(accepted_passes) > _CANDIDATE_STARVATION_RATE:
        alerts.append("candidate_starvation")

    baselines, qualified, comparisons, malformed = _quality_artifact_counts(quality_root)
    if malformed:
        alerts.append(f"malformed_quality_artifacts:{malformed}")
    return MemoryQualityDiagnostic(
        state="active" if not alerts else "DEGRADED",
        configured_roles=configured,
        actual_roles=actual,
        prompt_versions=prompts,
        attempts=len(facts),
        admitted=admitted,
        suppressed=suppressed,
        unclassified=unclassified,
        failed=failed,
        stored=stored,
        replaced=replaced,
        skipped=skipped,
        classifier_positive=classifier_positive,
        episode_attempts=len(episodes),
        episode_stored=episode_stored,
        episode_failed=episode_failed,
        corpus_baselines=baselines,
        qualified_baselines=qualified,
        comparison_scores=comparisons,
        malformed_artifacts=malformed,
        alerts=tuple(alerts),
    )


def load_memory_route_provenance(
    connection: sqlite3.Connection,
    principal_id: str,
    *,
    limit: int,
    offset: int = 0,
) -> tuple[MemoryRouteProvenance, ...]:
    """Return content-free model provenance for one principal's memory writes."""

    if isinstance(limit, bool) or not 1 <= limit <= 5000:
        raise ValueError("limit must be between 1 and 5000")
    if isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be non-negative")
    rows = connection.execute(
        "SELECT receipt.receipt_id, receipt.extraction_role, receipt.backend, receipt.provider, "
        "receipt.model, receipt.prompt_version, receipt.storage_outcome_json, receipt.created_at "
        "FROM memory_extraction_receipts receipt "
        "JOIN runtime_profile_owners owner "
        "ON owner.runtime_profile_id = receipt.runtime_profile_id "
        "AND owner.principal_id = receipt.principal_id "
        "WHERE receipt.principal_id = ? AND receipt.status = 'completed' "
        "ORDER BY receipt.created_at DESC, receipt.receipt_id DESC LIMIT ? OFFSET ?",
        (principal_id, limit, offset),
    ).fetchall()
    provenance: list[MemoryRouteProvenance] = []
    for row in rows:
        storage = json.loads(str(row[6]))
        if not isinstance(storage, dict) or not isinstance(storage.get("decisions"), list):
            raise ValueError("malformed receipt storage outcome")
        for decision in storage["decisions"]:
            if not isinstance(decision, dict):
                raise ValueError("malformed receipt storage decision")
            outcome = decision.get("outcome")
            memory_id = decision.get("new_memory_id")
            if outcome not in {"stored", "replaced"} or not isinstance(memory_id, str) or not memory_id:
                continue
            provenance.append(
                MemoryRouteProvenance(
                    memory_id=memory_id,
                    outcome=outcome,
                    extraction_role=str(row[1]),
                    backend=str(row[2]),
                    provider=str(row[3]),
                    model=str(row[4]),
                    prompt_version=str(row[5]),
                    receipt_id=str(row[0]),
                    created_at=str(row[7]),
                )
            )
    return tuple(provenance)


def _load_receipts(connection: sqlite3.Connection) -> list[_Receipt]:
    rows = connection.execute(
        "SELECT runtime_profile_id, extraction_role, backend, provider, model, prompt_version, "
        "status, failure_code, candidate_ids_json, classifier_result, validation_outcome_json, "
        "storage_outcome_json, created_at FROM memory_extraction_receipts ORDER BY created_at, receipt_id"
    ).fetchall()
    receipts: list[_Receipt] = []
    for row in rows:
        candidates = json.loads(str(row[8]))
        validation = json.loads(str(row[10]))
        storage = json.loads(str(row[11]))
        if not isinstance(candidates, list) or not isinstance(validation, dict) or not isinstance(storage, dict):
            raise ValueError("malformed memory extraction receipt")
        policy = validation.get("policy")
        if not isinstance(policy, dict):
            policy = {}
        receipts.append(
            _Receipt(
                runtime_profile_id=str(row[0]),
                role=str(row[1]),
                backend=str(row[2]),
                provider=str(row[3]),
                model=str(row[4]),
                prompt_version=str(row[5]),
                status=str(row[6]),
                failure_code=str(row[7]) if row[7] is not None else None,
                candidate_count=len(candidates),
                classifier_result=None if row[9] is None else bool(row[9]),
                raw_count=_nonnegative_int(validation.get("raw_count")),
                accepted_count=_nonnegative_int(validation.get("accepted_count")),
                admission=str(policy["admission"]) if policy.get("admission") in {"admitted", "suppressed"} else None,
                fragmentation_rejected=_nonnegative_int(policy.get("fragmentation_rejected")),
                batch_limit=_optional_nonnegative_int(policy.get("batch_limit")),
                stored_count=_nonnegative_int(storage.get("stored_count")),
                replaced_count=_nonnegative_int(storage.get("replaced_count")),
                skipped_count=_nonnegative_int(storage.get("skipped_count")),
                created_at=str(row[12]),
            )
        )
    return receipts


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _missing_receipt_count(connection: sqlite3.Connection, tables: set[str]) -> int:
    if "workshop_post_run_effects" not in tables:
        return 0
    row = None
    if "workshop_schema_migrations" in tables:
        row = connection.execute("SELECT applied_at FROM workshop_schema_migrations WHERE version = 87").fetchone()
    if row is None or row[0] is None:
        row = connection.execute("SELECT MIN(created_at) FROM memory_extraction_receipts").fetchone()
    cutover = str(row[0]) if row is not None and row[0] is not None else None
    if cutover is None:
        return 0
    result = connection.execute(
        "SELECT COUNT(*) FROM workshop_post_run_effects effect "
        "WHERE effect.created_at >= ? AND effect.status IN ('succeeded', 'failed') "
        "AND NOT EXISTS (SELECT 1 FROM memory_extraction_receipts receipt "
        "WHERE receipt.run_id = effect.run_id AND receipt.extraction_role = 'fact_extraction')",
        (cutover,),
    ).fetchone()
    return int(result[0]) if result is not None else 0


def _load_profiles(path: Path | None) -> WorkshopRuntimeProfileRegistry | None:
    if path is None:
        return None
    try:
        return WorkshopRuntimeProfileRegistry.from_yaml(path.read_text(encoding="utf-8"))
    except (OSError, WorkshopRuntimeProfileError):
        return None


def _role_model(
    profile: ProtectedRuntimeProfile,
    backend: str,
    provider: str,
    role: ModelRole,
    defaults: Mapping[str, str],
) -> str | None:
    option = next(
        (item for item in profile.backend_options if item.backend == backend and item.provider == provider),
        None,
    )
    if option is None:
        return None
    raw = dict(option.role_models).get(role.value) or defaults.get(role.value) or get_model_for(role, backend, provider)
    model = canonicalize_model_for_backend(raw, backend)
    if validate_model_for_backend_policy(model, backend, provider, allowed_models=option.allowed_models):
        return model
    return get_model_for(role, backend, provider)


def _configured_roles(
    profiles: WorkshopRuntimeProfileRegistry | None,
    defaults: Mapping[str, str],
) -> tuple[str, ...]:
    if profiles is None:
        return ()
    values: Counter[str] = Counter()
    for profile in profiles.profiles:
        option = profile.default_backend_option
        for role in (ModelRole.MEMORY_EXTRACTION, ModelRole.MEMORY_EPISODE):
            model = _role_model(profile, option.backend, option.provider, role, defaults)
            assert model is not None
            label = "fact" if role is ModelRole.MEMORY_EXTRACTION else "episode"
            values[f"{label}:{_safe_route(option.backend, option.provider, model)}"] += 1
    return _render_counts(values)


def _actual_roles(receipts: list[_Receipt]) -> tuple[str, ...]:
    values: Counter[str] = Counter()
    for receipt in receipts:
        label = "fact" if receipt.role == "fact_extraction" else "episode"
        values[f"{label}:{_safe_route(receipt.backend, receipt.provider, receipt.model)}"] += 1
    return _render_counts(values)


def _prompt_versions(receipts: list[_Receipt]) -> tuple[str, ...]:
    values: Counter[str] = Counter()
    for receipt in receipts:
        label = "fact" if receipt.role == "fact_extraction" else "episode"
        values[f"{label}:v{_safe(receipt.prompt_version)}"] += 1
    return _render_counts(values)


def _model_drift_count(
    receipts: list[_Receipt],
    profiles: WorkshopRuntimeProfileRegistry | None,
    defaults: Mapping[str, str],
) -> int:
    if profiles is None:
        return 0
    by_id = {str(profile.profile_id): profile for profile in profiles.profiles}
    latest: dict[tuple[str, str], _Receipt] = {}
    for receipt in receipts:
        latest[(receipt.runtime_profile_id, receipt.role)] = receipt
    drift = 0
    for receipt in latest.values():
        profile = by_id.get(receipt.runtime_profile_id)
        role = ModelRole.MEMORY_EXTRACTION if receipt.role == "fact_extraction" else ModelRole.MEMORY_EPISODE
        expected = (
            _role_model(profile, receipt.backend, receipt.provider, role, defaults) if profile is not None else None
        )
        drift += int(expected is None or expected != receipt.model)
    return drift


def _repeated_provider_failures(receipts: list[_Receipt]) -> int:
    failures = 0
    by_lane: dict[tuple[str, str], list[_Receipt]] = {}
    for receipt in receipts:
        by_lane.setdefault((receipt.runtime_profile_id, receipt.role), []).append(receipt)
    for lane in by_lane.values():
        recent = lane[-5:]
        count = sum(receipt.failure_code in _PROVIDER_FAILURES for receipt in recent)
        failures += count if count >= _REPEATED_FAILURES else 0
    return failures


def _quality_artifact_counts(root: Path | None) -> tuple[int, int, int, int]:
    if root is None or not root.is_dir():
        return 0, 0, 0, 0
    baselines = qualified = comparisons = malformed = 0
    for path in root.glob("*/docs/memory-quality/*-score.json"):
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_ARTIFACT_BYTES:
                malformed += 1
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                malformed += 1
            elif document.get("artifact") == "kai_memory_quality_score" and document.get("version") == 1:
                baselines += 1
                qualified += int(document.get("qualification_ready") is True)
            elif (
                document.get("artifact") == "kai_memory_model_evaluation_score"
                and document.get("version") == EVALUATION_SCORE_VERSION
            ):
                comparisons += 1
            else:
                malformed += 1
        except (OSError, json.JSONDecodeError):
            malformed += 1
    return baselines, qualified, comparisons, malformed


def _safe(value: str) -> str:
    return _SAFE_LABEL_RE.sub("_", "_".join(value.split()))[:256]


def _safe_route(backend: str, provider: str, model: str) -> str:
    return f"{_safe(backend)}/{_safe(provider)}/{_safe(model)}"


def _render_counts(values: Counter[str]) -> tuple[str, ...]:
    return tuple(f"{key}x{count}" for key, count in sorted(values.items()))
