import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  applyMemoryTriage,
  approveSafeMemoryTriage,
  loadMemoryTriage,
  loadMemoryTriageGroups,
  previewSafeMemoryTriage,
  recommendMemoryTriage,
  saveMemoryTriageDecision,
} from "./api";
import { AuthenticationError } from "./api";
import { useConfirmation } from "./ConfirmationDialog";
import { MarkdownMessage } from "./MarkdownMessage";
import type {
  WorkshopMemoryReconciliationDisposition,
  WorkshopMemoryTriageGroup,
  WorkshopMemoryTriagePreview,
  WorkshopMemoryTriageSummary,
} from "./types";

function formatLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

function summarizeEvidence(value: string): string {
  const condensed = value.replace(/\s+/g, " ").trim();
  return condensed.length > 88 ? `${condensed.slice(0, 87)}…` : condensed;
}

interface OperatorAdmissionOutcome {
  button: string;
  label: string;
  operatorAdmits: boolean;
  verb: string;
}

function operatorAdmissionOutcome(action: Record<string, unknown>): OperatorAdmissionOutcome | null {
  switch (action.kind) {
    case "adopt_as_current":
    case "adopt_corrected":
      return {
        button: "Approve current-truth adoption",
        label: "Adopt as current truth",
        operatorAdmits: true,
        verb: "adopt as current truth",
      };
    case "keep_first_retract_rest":
      return {
        button: "Approve consolidation",
        label: "Consolidate into current truth",
        operatorAdmits: true,
        verb: "consolidate into current truth",
      };
    case "expire_all":
      return {
        button: "Approve obsolescence",
        label: "Mark obsolete",
        operatorAdmits: false,
        verb: "mark obsolete",
      };
    case "record_episode_chain":
      return {
        button: "Approve historical retention",
        label: "Retain as retrievable history",
        operatorAdmits: true,
        verb: "retain as retrievable history",
      };
    default:
      return null;
  }
}

function homogeneousOperatorAdmissionOutcome(
  groups: WorkshopMemoryTriageGroup[],
): OperatorAdmissionOutcome | null {
  if (groups.length === 0) return null;
  const first = operatorAdmissionOutcome(groups[0].proposedAction);
  if (!first) return null;
  return groups.every((group) => {
    const outcome = operatorAdmissionOutcome(group.proposedAction);
    return outcome?.verb === first.verb;
  }) ? first : null;
}

function resolutionLabel(group: WorkshopMemoryTriageGroup): string {
  return operatorAdmissionOutcome(group.proposedAction)?.label ?? formatLabel(group.resolution);
}

function requestError(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback;
}

type ExceptionDecisionChoice = "approve" | "obsolete" | "reject" | "defer";

function groupReference(groupId: string): string {
  return `Group ${groupId.replace(/^mtg_/, "").slice(0, 8)}`;
}

function initialExceptionDecision(group: WorkshopMemoryTriageGroup | null): ExceptionDecisionChoice {
  if (!group) return "defer";
  if (group.decision.disposition === "reject" || group.decision.disposition === "defer") {
    return group.decision.disposition;
  }
  const storedKind = group.decision.disposition === "approve" ? group.decision.action.kind : null;
  const recommended = group.decision.recommendation.outcome;
  const suggestedKind = storedKind ?? group.proposedAction.kind;
  if (suggestedKind === "expire_all" || recommended === "obsolete") return "obsolete";
  if (
    suggestedKind === "adopt_corrected"
    || suggestedKind === "adopt_as_current"
    || suggestedKind === "keep_first_retract_rest"
    || suggestedKind === "record_episode_chain"
    || recommended === "adopt"
    || recommended === "consolidate"
  ) {
    return "approve";
  }
  return "defer";
}

function approvedReplacement(group: WorkshopMemoryTriageGroup | null): Record<string, unknown> | null {
  if (!group || group.decision.disposition !== "approve" || group.decision.action.kind !== "adopt_corrected") {
    return null;
  }
  const replacement = group.decision.action.replacement;
  return replacement && typeof replacement === "object" && !Array.isArray(replacement)
    ? replacement as Record<string, unknown>
    : null;
}

function initialScope(group: WorkshopMemoryTriageGroup | null): "" | "global" | "project" {
  const replacementScope = approvedReplacement(group)?.scope_kind;
  if (replacementScope === "global" || replacementScope === "project") return replacementScope;
  const first = group?.evidence[0];
  if (!first || first.kind !== "fact") return "";
  if (first.scope !== "global" && first.scope !== "project") return "";
  const consistent = group.evidence.every(
    (item) => item.scope === first.scope && item.projectId === first.projectId,
  );
  return consistent ? first.scope : "";
}

function relatedRecommendationGroups(
  group: WorkshopMemoryTriageGroup,
  availableGroupIds: Set<string>,
): string[] {
  const structured = group.decision.recommendation.related_group_ids;
  const candidates = Array.isArray(structured) && structured.every((value) => typeof value === "string")
    ? structured
    : String(group.decision.recommendation.rationale ?? "").match(/mtg_[a-z0-9]+/gi) ?? [];
  return [...new Set(candidates)].filter(
    (groupId) => groupId !== group.groupId && availableGroupIds.has(groupId),
  );
}

export function MemoryReviewIcon(): React.JSX.Element {
  return (
    <svg aria-hidden="true" fill="none" focusable="false" viewBox="0 0 24 24">
      <path d="M7 4h10M7 9h10M7 14h6" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      <path d="m14.5 18 2 2 4-5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
    </svg>
  );
}

function ExceptionEditor({
  allowedProjects,
  availableGroups,
  group,
  onAuthenticationFailure,
  onSelectGroup,
  onSaved,
  readOnly,
  token,
  planId,
}: {
  allowedProjects: Array<{ displayName: string; projectId: string }>;
  availableGroups: WorkshopMemoryTriageGroup[];
  group: WorkshopMemoryTriageGroup | null;
  onAuthenticationFailure: (message: string) => void;
  onSelectGroup: (groupId: string) => void;
  onSaved: () => void;
  readOnly: boolean;
  token: string;
  planId: string | null;
}): React.JSX.Element {
  const first = group?.evidence[0] ?? null;
  const replacement = approvedReplacement(group);
  const [content, setContent] = useState(
    typeof replacement?.content === "string" ? replacement.content : first?.text ?? "",
  );
  const [decisionChoice, setDecisionChoice] = useState<ExceptionDecisionChoice>(initialExceptionDecision(group));
  const [scopeKind, setScopeKind] = useState<"" | "global" | "project">(initialScope(group));
  const [projectId, setProjectId] = useState(
    typeof replacement?.scope_key === "string" ? replacement.scope_key : first?.projectId ?? "",
  );
  const [note, setNote] = useState(group?.decision.operatorNote ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!group || !first || !planId) {
    return (
      <div className="memory-detail-empty">
        <span aria-hidden="true">◇</span>
        <h2>No exception selected</h2>
        <p>Only uncertain or conflicting memory groups require individual review.</p>
      </div>
    );
  }

  const recommendation = group.decision.recommendation;
  const corrected = content.trim() !== first.text;
  const factEvidence = group.evidence.every((item) => item.kind === "fact");
  const episodeEvidence = group.evidence.every((item) => item.kind === "episode");
  const scopeChanged = scopeKind !== first.scope || (scopeKind === "project" && projectId !== first.projectId);
  const projectAuthorized = scopeKind !== "project"
    || allowedProjects.some((project) => project.projectId === projectId);
  const validityEnd = first.validUntil ? Date.parse(first.validUntil) : null;
  const expired = validityEnd !== null && !Number.isNaN(validityEnd) && validityEnd <= Date.now();
  let approvalAction: Record<string, unknown> | null = null;
  if (decisionChoice === "approve" && episodeEvidence) {
    approvalAction = { kind: "record_episode_chain" };
  } else if (decisionChoice === "obsolete" && factEvidence) {
    approvalAction = { kind: "expire_all" };
  } else if (
    decisionChoice === "approve"
    && factEvidence
    && content.trim()
    && scopeKind
    && projectAuthorized
    && (scopeKind !== "project" || projectId)
  ) {
    const requiresCanonicalReplacement = group.evidence.length > 1
      || corrected
      || scopeChanged
      || first.migrationGaps.includes("scope")
      || expired;
    approvalAction = requiresCanonicalReplacement
      ? {
          kind: "adopt_corrected",
          source_memory_id: first.memoryId,
          replacement: {
            content: content.trim(),
            scope_kind: scopeKind,
            scope_key: scopeKind === "project" ? projectId : "",
            confidence: typeof first.confidence === "number" ? first.confidence : 0.5,
            asserted_at: first.assertedAt,
            observed_at: first.observedAt,
            valid_from: expired ? null : first.validFrom,
            valid_until: expired ? null : first.validUntil,
          },
        }
      : { kind: "adopt_as_current" };
  }
  const requiresApprovalAction = decisionChoice === "approve" || decisionChoice === "obsolete";
  const canSave = !requiresApprovalAction || approvalAction !== null;
  const disposition: Exclude<WorkshopMemoryReconciliationDisposition, "pending"> = requiresApprovalAction
    ? "approve"
    : decisionChoice;
  const approvalLabel = episodeEvidence
    ? "Retain as immutable history"
    : group.evidence.length > 1
      ? "Approve consolidation"
      : "Approve as current truth";
  let unavailableReason: string | null = null;
  if (requiresApprovalAction && approvalAction === null) {
    if (!factEvidence && !episodeEvidence) unavailableReason = "This mixed evidence group cannot be approved.";
    else if (factEvidence && !content.trim()) unavailableReason = "Enter the current fact wording before approval.";
    else if (factEvidence && !scopeKind) unavailableReason = "Choose a scope before approval.";
    else if (factEvidence && scopeKind === "project" && !projectId) {
      unavailableReason = "Choose a project before approval.";
    } else if (factEvidence && scopeKind === "project" && !projectAuthorized) {
      unavailableReason = "The selected project is not currently authorized for this runtime.";
    } else unavailableReason = "This decision is not available for the supplied evidence.";
  }
  const availableGroupIds = new Set(availableGroups.map((item) => item.groupId));
  const relatedGroups = relatedRecommendationGroups(group, availableGroupIds);

  const submit = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const action = disposition === "approve" && approvalAction ? approvalAction : group.proposedAction;
      await saveMemoryTriageDecision(token, planId, group.groupId, {
        disposition,
        action,
        operatorNote: note,
        expectedStateVersion: group.decision.stateVersion,
      });
      onSaved();
    } catch (caught) {
      if (caught instanceof AuthenticationError) onAuthenticationFailure(caught.message);
      setError(requestError(caught, "Could not save this memory decision."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="memory-reconciliation-detail" onSubmit={(event) => void submit(event)}>
      <header className="memory-detail-header">
        <div>
          <p className="overline">Exception review</p>
          <h2>{formatLabel(group.classification)}</h2>
        </div>
        <span className={`memory-review-state ${group.decision.disposition}`}>
          {group.decision.disposition}
        </span>
      </header>

      <section className="memory-detail-section">
        <p className="memory-section-label">Why this needs judgment</p>
        <p>{group.rationale}</p>
      </section>

      {Object.keys(recommendation).length > 0 && (
        <section className="memory-detail-section">
          <p className="memory-section-label">Model recommendation</p>
          <p><strong>{formatLabel(String(recommendation.outcome ?? "needs_review"))}</strong>
            {typeof recommendation.confidence === "number" && ` · ${Math.round(recommendation.confidence * 100)}% confidence`}
          </p>
          <p>{String(recommendation.rationale ?? "No rationale was returned.")}</p>
          {relatedGroups.length > 0 && (
            <div className="memory-related-groups">
              <p>Related groups</p>
              {relatedGroups.map((groupId) => (
                <button type="button" key={groupId} title={groupId} onClick={() => onSelectGroup(groupId)}>
                  {groupReference(groupId)}
                </button>
              ))}
              <p className="memory-review-explanation">
                These are separate decisions. Adopt or consolidate one canonical fact, then reject duplicates in the linked groups.
              </p>
            </div>
          )}
          <p className="memory-review-explanation">Advisory only; it has not changed memory.</p>
        </section>
      )}

      {group.priorReviewEvidence.length > 0 && (
        <section className="memory-detail-section">
          <p className="memory-section-label">Earlier raw review</p>
          {group.priorReviewEvidence.map((item) => (
            <p key={item.candidateId}>
              <strong>{formatLabel(item.disposition)}</strong>
              {item.operatorNote ? ` · ${item.operatorNote}` : " · No operator note"}
            </p>
          ))}
          <p className="memory-review-explanation">
            This earlier decision remains evidence. It was not silently converted into a grouped approval.
          </p>
        </section>
      )}

      <section className="memory-detail-section">
        <p className="memory-section-label">Evidence</p>
        {group.evidence.map((item) => (
          <article className="memory-review-evidence" key={item.memoryId}>
            <MarkdownMessage body={item.text} />
            <small>{item.kind} · {item.scope}{item.projectId ? ` · ${item.projectId}` : ""}</small>
          </article>
        ))}
      </section>

      <fieldset className="memory-review-controls" disabled={readOnly || saving}>
        {first.kind === "fact" && (
          <section className="memory-detail-section memory-review-correction">
            <p className="memory-section-label">Canonical fact</p>
            <label>Current wording
              <textarea value={content} maxLength={16384} onChange={(event) => setContent(event.target.value)} />
            </label>
            <label>Scope
              <select value={scopeKind} onChange={(event) => setScopeKind(event.target.value as "" | "global" | "project")}>
                <option value="">Choose scope</option>
                <option value="global">Global</option>
                <option value="project">Project</option>
              </select>
            </label>
            {scopeKind === "project" && (
              <label>Project
                <select value={projectId} onChange={(event) => setProjectId(event.target.value)}>
                  <option value="">Choose project</option>
                  {projectId && !allowedProjects.some((project) => project.projectId === projectId) && (
                    <option value={projectId}>{projectId} (unavailable)</option>
                  )}
                  {allowedProjects.map((project) => (
                    <option value={project.projectId} key={project.projectId}>{project.displayName}</option>
                  ))}
                </select>
              </label>
            )}
            <p className="memory-review-explanation">
              Adoption creates one current claim backed by every evidence row in this group. You may retain the wording or correct it.
            </p>
          </section>
        )}
        <section className="memory-detail-section memory-review-decision">
          <p className="memory-section-label">Review</p>
          <label>Decision
            <select value={decisionChoice} onChange={(event) => setDecisionChoice(event.target.value as ExceptionDecisionChoice)}>
              {(factEvidence || episodeEvidence) && <option value="approve">{approvalLabel}</option>}
              {factEvidence && <option value="obsolete">Mark obsolete</option>}
              <option value="reject">Reject</option>
              <option value="defer">Defer</option>
            </select>
          </label>
          {unavailableReason && <p>{unavailableReason}</p>}
          {approvalAction?.kind === "adopt_as_current" && !corrected && (
            <p>
              Approve adopts this fact unchanged as current truth.
              {group.priorReviewEvidence.length > 0 && " The earlier decision remains audit evidence."}
            </p>
          )}
          <label>Operator note
            <textarea value={note} maxLength={4096} onChange={(event) => setNote(event.target.value)} />
          </label>
          {error && <p className="memory-editor-error" role="alert">{error}</p>}
          <button type="submit" disabled={readOnly || saving || !canSave}>
            {saving ? "Saving…" : "Save decision"}
          </button>
        </section>
      </fieldset>
    </form>
  );
}

export function MemoryReconciliation({
  allowedProjects,
  detailPanelLayout,
  onAuthenticationFailure,
  onBack,
  token,
}: {
  allowedProjects: Array<{ displayName: string; projectId: string }>;
  detailPanelLayout?: {
    width: number;
    minimumWidth: number;
    maximumWidth: number;
    onKeyDown: React.KeyboardEventHandler<HTMLDivElement>;
    onPointerDown: React.PointerEventHandler<HTMLDivElement>;
    onPointerMove: React.PointerEventHandler<HTMLDivElement>;
  };
  onAuthenticationFailure: (message: string) => void;
  onBack: () => void;
  token: string;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const [triage, setTriage] = useState<WorkshopMemoryTriageSummary | null>(null);
  const [groups, setGroups] = useState<WorkshopMemoryTriageGroup[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [mutating, setMutating] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [safePreview, setSafePreview] = useState<WorkshopMemoryTriagePreview | null>(null);
  const [safePreviewGroups, setSafePreviewGroups] = useState<WorkshopMemoryTriageGroup[]>([]);

  const handleFailure = (caught: unknown, fallback: string): void => {
    if (caught instanceof AuthenticationError) onAuthenticationFailure(caught.message);
    setError(requestError(caught, fallback));
  };

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSafePreview(null);
    setSafePreviewGroups([]);
    void loadMemoryTriage(token)
      .then(async (summary) => {
        if (cancelled) return;
        setTriage(summary);
        setGroups([]);
        setSelectedId(null);
        if (!summary) return;
        const loadedGroups: WorkshopMemoryTriageGroup[] = [];
        let offset: number | null = 0;
        let latestSummary = summary;
        while (offset !== null) {
          const page = await loadMemoryTriageGroups(token, summary.planId, {
            exceptionsOnly: true,
            limit: 100,
            offset,
          });
          loadedGroups.push(...page.groups);
          latestSummary = page.triage;
          offset = page.nextOffset;
        }
        if (cancelled) return;
        setTriage(latestSummary);
        setGroups(loadedGroups);
        setSelectedId(loadedGroups[0]?.groupId ?? null);
      })
      .catch((caught) => { if (!cancelled) handleFailure(caught, "Could not load memory triage."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [refreshKey, token]);

  const selected = useMemo(
    () => groups.find((group) => group.groupId === selectedId) ?? null,
    [groups, selectedId],
  );

  const previewSafe = async (): Promise<void> => {
    if (!triage) return;
    setMutating(true);
    setError(null);
    try {
      const pendingGroups: WorkshopMemoryTriageGroup[] = [];
      let offset: number | null = 0;
      while (offset !== null) {
        const page = await loadMemoryTriageGroups(token, triage.planId, { limit: 100, offset });
        pendingGroups.push(...page.groups.filter((group) => (
          group.deterministic && group.bulkEligible && group.decision.disposition === "pending"
        )));
        offset = page.nextOffset;
      }
      const first = pendingGroups[0];
      if (!first) throw new Error("No deterministic memory groups are pending.");
      const actionSignature = JSON.stringify(first.proposedAction);
      const previewGroups = pendingGroups.filter(
        (group) => JSON.stringify(group.proposedAction) === actionSignature,
      );
      const preview = await previewSafeMemoryTriage(
        token,
        triage.planId,
        triage.reviewVersion,
        previewGroups.map((group) => group.groupId),
      );
      if (previewGroups.length !== preview.groupCount) {
        throw new Error("Kai returned an incomplete deterministic memory preview.");
      }
      setSafePreview(preview);
      setSafePreviewGroups(previewGroups);
      setReport("Review the complete grouped evidence below before approving it.");
    } catch (caught) {
      handleFailure(caught, "Could not preview deterministic memory groups.");
    } finally {
      setMutating(false);
    }
  };

  const approveSafe = async (): Promise<void> => {
    if (!safePreview) return;
    const counts = safePreview.resolutionCounts;
    const outcome = homogeneousOperatorAdmissionOutcome(safePreviewGroups);
    const accepted = await confirm(outcome
      ? `${formatLabel(outcome.verb)} for ${safePreview.memoryCount} deterministic legacy memories? ` +
        (outcome.operatorAdmits
          ? "Their original provenance will remain incomplete; your explicit approval supplies separate retrieval admission authority."
          : "Their original provenance will remain incomplete and the canonical lifecycle will exclude them as obsolete.")
      : `Approve ${safePreview.memoryCount} deterministic memories after reviewing the complete preview: ` +
        `${counts.adopt} adopt, ${counts.consolidate} consolidate, ${counts.obsolete} obsolete? ` +
        "No detected warning is not proof that a fact is true; your approval is the trust decision.");
    if (!accepted) return;
    setMutating(true);
    setError(null);
    try {
      await approveSafeMemoryTriage(token, safePreview);
      setReport(outcome
        ? `${safePreview.memoryCount} deterministic legacy memories approved to ${outcome.verb} in ${safePreview.groupCount} evidence-bound groups.`
        : `${safePreview.memoryCount} deterministic memories approved in ${safePreview.groupCount} evidence-bound groups.`);
      setRefreshKey((value) => value + 1);
    } catch (caught) {
      handleFailure(caught, "Could not approve deterministic memory groups.");
    } finally {
      setMutating(false);
    }
  };

  const analyzeExceptions = async (): Promise<void> => {
    if (!triage) return;
    setMutating(true);
    setError(null);
    try {
      const result = await recommendMemoryTriage(token, triage.planId);
      setReport(
        result.remaining === 0
          ? `The configured memory-quality model analyzed all ${triage.exceptionGroups} exception groups. Recommendations are advisory only.`
          : `The configured memory-quality model analyzed ${result.recommended} exception groups; ${result.remaining} remain. Recommendations are advisory only.`,
      );
      setRefreshKey((value) => value + 1);
    } catch (caught) {
      handleFailure(caught, "Could not analyze uncertain memory groups.");
    } finally {
      setMutating(false);
    }
  };

  const apply = async (): Promise<void> => {
    if (!triage) return;
    const accepted = await confirm(
      "Apply this grouped plan through canonical fact and episode lifecycle events? A final receipt will record every outcome.",
    );
    if (!accepted) return;
    setMutating(true);
    setError(null);
    try {
      const summary = await applyMemoryTriage(token, triage.planId, triage.reviewVersion);
      setReport(
        `${summary.operator_admitted ?? 0} operator admitted, ` +
        `${summary.adopted ?? 0} adopted by lifecycle outcome, ${summary.consolidated ?? 0} consolidated, ` +
        `${summary.obsolete ?? 0} obsolete, ${summary.deferred ?? 0} deferred, ` +
        `${summary.rejected ?? 0} rejected, ${summary.failed ?? 0} failed, ` +
        `${summary.still_unresolved ?? 0} still unresolved.` +
        // Failed items are applied but did not reach search, so they are
        // missing from recall until retried.
        ((summary.failed ?? 0) > 0
          ? ` ${summary.failed} did not reach search; retry them under Search sync in Fact review.`
          : ""),
      );
      setRefreshKey((value) => value + 1);
    } catch (caught) {
      handleFailure(caught, "Could not apply grouped memory triage.");
    } finally {
      setMutating(false);
    }
  };

  return (
    <section className="memory-workspace memory-reconciliation-workspace" aria-label="Memory reconciliation">
      <div className="memory-browser-pane">
        <header className="memory-header">
          <div><p className="breadcrumbs">Kai Workshop / Memory</p><h1>Legacy triage</h1></div>
          <button className="panel-icon-button" type="button" aria-label="Back to memories" title="Back to memories" onClick={onBack}>
            <span aria-hidden="true">←</span>
          </button>
        </header>
        <div className="memory-browser-scroll">
          {loading ? (
            <p className="memory-list-state" role="status">Building grouped memory triage…</p>
          ) : error ? (
            <div className="memory-list-state error" role="alert"><p>{error}</p><button onClick={() => setRefreshKey((value) => value + 1)}>Retry</button></div>
          ) : !triage ? (
            <div className="memory-list-state"><MemoryReviewIcon /><h2>No reconciliation audit</h2><p>Run an authorized read-only audit first.</p></div>
          ) : (
            <>
              <section className="memory-reconciliation-summary" aria-label="Triage summary">
                <div><strong>{triage.resolutionCounts.adopt}</strong><span>Retain</span></div>
                <div><strong>{triage.resolutionCounts.consolidate}</strong><span>Consolidate</span></div>
                <div><strong>{triage.resolutionCounts.obsolete}</strong><span>Obsolete</span></div>
                <div><strong>{triage.exceptionGroups}</strong><span>Need review</span></div>
              </section>
              <p className="memory-review-explanation">
                The plan partitions {triage.memoryCount} legacy memories into {triage.groupCount} non-overlapping groups.
                Only uncertain or conflicting groups appear below.
              </p>
              {report && <p className="memory-mutation-report" role="status">{report}</p>}
              <div className="memory-review-toolbar">
                <button type="button" disabled={triage.pendingDeterministicGroups === 0 || mutating || triage.status === "applied"} onClick={() => void previewSafe()}>
                  {safePreview ? "Refresh safe preview" : "Preview safe groups"}
                </button>
                <button type="button" disabled={triage.exceptionGroups === 0 || triage.recommendedGroups >= triage.exceptionGroups || mutating || triage.status === "applied"} onClick={() => void analyzeExceptions()}>
                  {triage.exceptionGroups > 0 && triage.recommendedGroups >= triage.exceptionGroups
                    ? `Exceptions analyzed (${triage.recommendedGroups})`
                    : "Analyze exceptions"}
                </button>
                <button type="button" disabled={triage.dispositionCounts.pending !== 0 || mutating || triage.status === "applied"} onClick={() => void apply()}>
                  {triage.status === "applied" ? "Applied" : "Apply plan"}
                </button>
              </div>
              {safePreview && (
                <section className="memory-detail-section memory-safe-preview" aria-label="Deterministic memory preview">
                  <p className="memory-section-label">Complete deterministic preview</p>
                  <p>
                    {safePreview.memoryCount} memories in {safePreview.groupCount} evidence-bound groups.
                    {homogeneousOperatorAdmissionOutcome(safePreviewGroups)
                      ? homogeneousOperatorAdmissionOutcome(safePreviewGroups)?.operatorAdmits
                        ? " Their original provenance remains incomplete; your explicit approval supplies separate retrieval admission authority."
                        : " Their original provenance remains incomplete and the canonical lifecycle will keep them out of retrieval."
                      : " No detected warning is proof of truth; approval is your explicit trust decision."}
                  </p>
                  <p>
                    {homogeneousOperatorAdmissionOutcome(safePreviewGroups)
                      ? `Proposed outcome: ${safePreview.memoryCount} ${homogeneousOperatorAdmissionOutcome(safePreviewGroups)?.verb}.`
                      : `Proposed outcomes: ${safePreview.resolutionCounts.adopt} adopt, ${safePreview.resolutionCounts.consolidate} consolidate, ${safePreview.resolutionCounts.obsolete} obsolete.`}
                  </p>
                  {safePreviewGroups.map((previewGroup) => (
                    <details key={previewGroup.groupId}>
                      <summary>
                        {formatLabel(previewGroup.classification)} · {summarizeEvidence(previewGroup.evidence[0]?.text ?? "Unavailable memory")} · {previewGroup.evidence.length} {previewGroup.evidence.length === 1 ? "memory" : "memories"} · {resolutionLabel(previewGroup)}
                      </summary>
                      <p>{previewGroup.rationale}</p>
                      {previewGroup.evidence.map((item) => (
                        <article className="memory-review-evidence" key={item.memoryId}>
                          <MarkdownMessage body={item.text} />
                          <small>{item.scope}{item.projectId ? ` · ${item.projectId}` : ""}</small>
                        </article>
                      ))}
                    </details>
                  ))}
                  <button type="button" disabled={mutating} onClick={() => void approveSafe()}>
                    {homogeneousOperatorAdmissionOutcome(safePreviewGroups)
                      ? homogeneousOperatorAdmissionOutcome(safePreviewGroups)?.button
                      : "Approve preview"}
                  </button>
                </section>
              )}
              {groups.length === 0 ? (
                <div className="memory-list-state"><h2>No exceptional groups</h2><p>The deterministic preview contains everything in this audit.</p></div>
              ) : (
                <div className="memory-record-list memory-reconciliation-list" role="listbox" aria-label="Memory triage exceptions">
                  {groups.map((group) => (
                    <button type="button" role="option" aria-selected={selectedId === group.groupId} key={group.groupId}
                      className={`memory-record ${selectedId === group.groupId ? "selected" : ""}`}
                      onClick={() => setSelectedId(group.groupId)}>
                      <span className={`memory-review-state ${group.decision.disposition}`}>{group.decision.disposition}</span>
                      <span className="memory-record-copy">
                        <strong>{group.evidence[0]?.text ?? "Unavailable memory"}</strong>
                        <small>{groupReference(group.groupId)} · {formatLabel(group.classification)} · {group.evidence.length} evidence row{group.evidence.length === 1 ? "" : "s"}</small>
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
      <aside className="context-pane memory-detail-pane" aria-label="Memory triage exception detail">
        {detailPanelLayout && (
          <div className="context-resize-handle" role="separator" aria-label="Resize memory triage detail"
            aria-orientation="vertical" aria-valuemin={detailPanelLayout.minimumWidth}
            aria-valuemax={detailPanelLayout.maximumWidth} aria-valuenow={detailPanelLayout.width} tabIndex={0}
            onKeyDown={detailPanelLayout.onKeyDown} onPointerDown={detailPanelLayout.onPointerDown}
            onPointerMove={detailPanelLayout.onPointerMove} />
        )}
        <ExceptionEditor
          allowedProjects={allowedProjects}
          availableGroups={groups}
          group={selected}
          key={`${triage?.planId ?? "none"}:${selected?.groupId ?? "none"}:${selected?.decision.stateVersion ?? 0}`}
          onAuthenticationFailure={onAuthenticationFailure}
          onSelectGroup={setSelectedId}
          onSaved={() => { setReport("Exception decision saved."); setRefreshKey((value) => value + 1); }}
          planId={triage?.planId ?? null}
          readOnly={triage?.status === "applied"}
          token={token}
        />
      </aside>
    </section>
  );
}
