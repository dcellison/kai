import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  applyMemoryReconciliation,
  bulkMemoryReconciliationDecision,
  loadMemoryReconciliation,
  loadMemoryReconciliationCandidates,
  saveMemoryReconciliationDecision,
} from "./api";
import { AuthenticationError } from "./api";
import { useConfirmation } from "./ConfirmationDialog";
import { MarkdownMessage } from "./MarkdownMessage";
import type {
  WorkshopMemoryReconciliationCandidate,
  WorkshopMemoryReconciliationDisposition,
  WorkshopMemoryReconciliationSummary,
} from "./types";

interface ReconciliationFilters {
  action: string;
  category: string;
  disposition: "" | WorkshopMemoryReconciliationDisposition;
  gap: string;
  kind: string;
  scope: string;
  uncertainty: string;
}

const EMPTY_FILTERS: ReconciliationFilters = {
  action: "",
  category: "",
  disposition: "",
  gap: "",
  kind: "",
  scope: "",
  uncertainty: "",
};

function formatLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

function requestError(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback;
}

function toDateTimeLocal(value: string | null): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "" : date.toISOString().slice(0, 16);
}

function toIso(value: string): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? null : date.toISOString();
}

export function MemoryReviewIcon(): React.JSX.Element {
  return (
    <svg aria-hidden="true" fill="none" focusable="false" viewBox="0 0 24 24">
      <path d="M7 4h10M7 9h10M7 14h6" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      <path d="m14.5 18 2 2 4-5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
    </svg>
  );
}

function CandidateEditor({
  auditId,
  candidate,
  onAuthenticationFailure,
  onSaved,
  readOnly,
  token,
  allowedProjects,
}: {
  auditId: string | null;
  candidate: WorkshopMemoryReconciliationCandidate | null;
  onAuthenticationFailure: (message: string) => void;
  onSaved: () => void;
  readOnly: boolean;
  token: string;
  allowedProjects: Array<{ displayName: string; projectId: string }>;
}): React.JSX.Element {
  const evidence = candidate?.evidence[0] ?? null;
  const [content, setContent] = useState(evidence?.text ?? "");
  const [scope, setScope] = useState(evidence?.scope === "project" ? "project" : "global");
  const [projectId, setProjectId] = useState(evidence?.projectId ?? "");
  const [confidence, setConfidence] = useState(String(evidence?.confidence ?? 0.5));
  const [assertedAt, setAssertedAt] = useState(toDateTimeLocal(evidence?.assertedAt ?? null));
  const [observedAt, setObservedAt] = useState(toDateTimeLocal(evidence?.observedAt ?? null));
  const [validFrom, setValidFrom] = useState(toDateTimeLocal(evidence?.validFrom ?? null));
  const [validUntil, setValidUntil] = useState(toDateTimeLocal(evidence?.validUntil ?? null));
  const [disposition, setDisposition] = useState<Exclude<WorkshopMemoryReconciliationDisposition, "pending">>(
    candidate?.decision.disposition === "pending" ? "defer" : candidate?.decision.disposition ?? "defer",
  );
  const [note, setNote] = useState(candidate?.decision.operatorNote ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!auditId || !candidate || !evidence) {
    return (
      <div className="memory-detail-empty">
        <span aria-hidden="true">◇</span>
        <h2>Select a review candidate</h2>
        <p>Choose a legacy memory to inspect its evidence and record a decision.</p>
      </div>
    );
  }

  const fact = candidate.evidence.every((item) => item.kind === "fact");
  const corrected = fact && (
    content.trim() !== evidence.text || scope !== evidence.scope ||
    (scope === "project" ? projectId !== (evidence.projectId ?? "") : evidence.projectId !== null) ||
    Number(confidence) !== evidence.confidence || validFrom !== toDateTimeLocal(evidence.validFrom) ||
    validUntil !== toDateTimeLocal(evidence.validUntil) ||
    assertedAt !== toDateTimeLocal(evidence.assertedAt) || observedAt !== toDateTimeLocal(evidence.observedAt)
  );
  const proposedKind = typeof candidate.proposedAction.kind === "string"
    ? candidate.proposedAction.kind
    : "manual_edit_required";
  const canApprove = fact ? corrected || proposedKind !== "manual_edit_required" : proposedKind === "record_episode_chain";

  const submit = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      let action = candidate.decision.action;
      if (disposition === "approve") {
        action = corrected
          ? {
              kind: "adopt_corrected",
              source_memory_id: evidence.memoryId,
              replacement: {
                content: content.trim(),
                scope_kind: scope,
                scope_key: scope === "project" ? projectId.trim() : "",
                confidence: Number(confidence),
                asserted_at: toIso(assertedAt),
                observed_at: toIso(observedAt),
                valid_from: toIso(validFrom),
                valid_until: toIso(validUntil),
              },
            }
          : candidate.proposedAction;
      }
      await saveMemoryReconciliationDecision(token, auditId, candidate.candidateId, {
        disposition,
        action,
        operatorNote: note,
        expectedStateVersion: candidate.decision.stateVersion,
      });
      onSaved();
    } catch (caught) {
      if (caught instanceof AuthenticationError) onAuthenticationFailure(caught.message);
      setError(requestError(caught, "Could not save this review decision."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="memory-reconciliation-detail" onSubmit={(event) => void submit(event)}>
      <header className="memory-detail-header">
        <div>
          <p className="overline">Legacy review</p>
          <h2>{formatLabel(candidate.category)}</h2>
        </div>
        <span className={`memory-review-state ${candidate.decision.disposition}`}>
          {candidate.decision.disposition}
        </span>
      </header>

      <section className="memory-detail-section">
        <p className="memory-section-label">Why this needs review</p>
        <p>{candidate.rationale}</p>
        <p className="memory-review-explanation">
          Missing legacy provenance means this memory requires human review. It does not mean the memory is false.
        </p>
      </section>

      <section className="memory-detail-section">
        <p className="memory-section-label">Evidence</p>
        {candidate.evidence.map((item) => (
          <article className="memory-review-evidence" key={item.memoryId}>
            <MarkdownMessage body={item.text} />
            <dl className="memory-metadata-grid">
              <div><dt>Kind</dt><dd>{item.kind}</dd></div>
              <div><dt>Scope</dt><dd>{item.scope}{item.projectId ? ` · ${item.projectId}` : ""}</dd></div>
              <div><dt>Confidence</dt><dd>{Math.round(item.confidence * 100)}%</dd></div>
              <div><dt>Uncertainty</dt><dd>{candidate.uncertainty}</dd></div>
            </dl>
            <p className="memory-review-gaps">
              Missing: {item.migrationGaps.length ? item.migrationGaps.map(formatLabel).join(", ") : "none"}
            </p>
          </article>
        ))}
      </section>

      {readOnly && <p className="memory-mutation-report" role="status">This reconciliation batch has been applied.</p>}
      <fieldset className="memory-review-controls" disabled={readOnly || saving}>
        {fact && (
          <section className="memory-detail-section memory-review-correction">
          <p className="memory-section-label">Correct before adoption</p>
          <label>Fact text
            <textarea value={content} maxLength={16384} onChange={(event) => setContent(event.target.value)} />
          </label>
          <div className="memory-review-fields">
            <label>Scope
              <select value={scope} onChange={(event) => setScope(event.target.value)}>
                <option value="global">Global</option>
                <option value="project">Project</option>
              </select>
            </label>
            {scope === "project" && (
              <label>Project
                <select value={projectId} onChange={(event) => setProjectId(event.target.value)}>
                  <option value="">Choose a permitted project</option>
                  {allowedProjects.map((project) => (
                    <option key={project.projectId} value={project.projectId}>{project.displayName}</option>
                  ))}
                </select>
              </label>
            )}
            <label>Confidence
              <input type="number" min="0" max="1" step="0.01" value={confidence} onChange={(event) => setConfidence(event.target.value)} />
            </label>
            <label>Valid from
              <input type="datetime-local" value={validFrom} onChange={(event) => setValidFrom(event.target.value)} />
            </label>
            <label>Asserted at
              <input type="datetime-local" value={assertedAt} onChange={(event) => setAssertedAt(event.target.value)} />
            </label>
            <label>Observed at
              <input type="datetime-local" value={observedAt} onChange={(event) => setObservedAt(event.target.value)} />
            </label>
            <label>Valid until
              <input type="datetime-local" value={validUntil} onChange={(event) => setValidUntil(event.target.value)} />
            </label>
          </div>
          </section>
        )}

        <section className="memory-detail-section memory-review-decision">
        <p className="memory-section-label">Decision</p>
        <label>Disposition
          <select value={disposition} onChange={(event) => setDisposition(
            event.target.value as Exclude<WorkshopMemoryReconciliationDisposition, "pending">,
          )}>
            <option value="approve" disabled={!canApprove}>Approve</option>
            <option value="reject">Reject</option>
            <option value="defer">Defer</option>
          </select>
        </label>
        {!canApprove && <p>Correct this fact before approval, or reject or defer it.</p>}
        <label>Operator note
          <textarea value={note} maxLength={4096} onChange={(event) => setNote(event.target.value)} />
        </label>
        {error && <p className="memory-editor-error" role="alert">{error}</p>}
        <button type="submit" disabled={readOnly || saving || (disposition === "approve" && !canApprove)}>
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
  const [audit, setAudit] = useState<WorkshopMemoryReconciliationSummary | null>(null);
  const [candidates, setCandidates] = useState<WorkshopMemoryReconciliationCandidate[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [filters, setFilters] = useState<ReconciliationFilters>(EMPTY_FILTERS);
  const [draftFilters, setDraftFilters] = useState<ReconciliationFilters>(EMPTY_FILTERS);
  const [nextOffset, setNextOffset] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [mutating, setMutating] = useState(false);

  const handleFailure = (caught: unknown, fallback: string): void => {
    if (caught instanceof AuthenticationError) onAuthenticationFailure(caught.message);
    setError(requestError(caught, fallback));
  };

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void loadMemoryReconciliation(token)
      .then(async (summary) => {
        if (cancelled) return;
        setAudit(summary);
        setCandidates([]);
        setSelectedId(null);
        setSelectedIds(new Set());
        if (!summary) return;
        const page = await loadMemoryReconciliationCandidates(token, summary.auditId, {
          category: filters.category || undefined,
          action: filters.action || undefined,
          disposition: filters.disposition || undefined,
          kind: filters.kind || undefined,
          gap: filters.gap || undefined,
          scope: filters.scope || undefined,
          uncertainty: filters.uncertainty || undefined,
          limit: 25,
        });
        if (cancelled) return;
        setAudit(page.audit);
        setCandidates(page.candidates);
        setNextOffset(page.nextOffset);
        setSelectedId(page.candidates[0]?.candidateId ?? null);
      })
      .catch((caught) => { if (!cancelled) handleFailure(caught, "Could not load memory reconciliation."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [filters, refreshKey, token]);

  const selected = useMemo(
    () => candidates.find((candidate) => candidate.candidateId === selectedId) ?? null,
    [candidates, selectedId],
  );

  const loadMore = async (): Promise<void> => {
    if (!audit || nextOffset === null) return;
    setLoadingMore(true);
    try {
      const page = await loadMemoryReconciliationCandidates(token, audit.auditId, {
        category: filters.category || undefined,
        action: filters.action || undefined,
        disposition: filters.disposition || undefined,
        kind: filters.kind || undefined,
        gap: filters.gap || undefined,
        scope: filters.scope || undefined,
        uncertainty: filters.uncertainty || undefined,
        offset: nextOffset,
        limit: 25,
      });
      setAudit(page.audit);
      setCandidates((current) => [...current, ...page.candidates]);
      setNextOffset(page.nextOffset);
    } catch (caught) {
      handleFailure(caught, "Could not load more reconciliation candidates.");
    } finally {
      setLoadingMore(false);
    }
  };

  const bulk = async (disposition: "reject" | "defer"): Promise<void> => {
    if (!audit || selectedIds.size === 0) return;
    const accepted = await confirm(
      `${formatLabel(disposition)} ${selectedIds.size} explicitly selected reconciliation candidates?`,
    );
    if (!accepted) return;
    setMutating(true);
    setError(null);
    try {
      const result = await bulkMemoryReconciliationDecision(token, audit.auditId, {
        candidateIds: [...selectedIds],
        disposition,
        operatorNote: `Bulk ${disposition} from the Workshop reconciliation queue.`,
        expectedReviewVersion: audit.reviewVersion,
      });
      setReport(`${result.changed} candidates marked ${disposition}.`);
      setRefreshKey((value) => value + 1);
    } catch (caught) {
      handleFailure(caught, "Could not save the bulk reconciliation decision.");
    } finally {
      setMutating(false);
    }
  };

  const apply = async (): Promise<void> => {
    if (!audit || audit.dispositionCounts.pending !== 0) return;
    const accepted = await confirm(
      "Apply every approved memory decision through canonical lifecycle events? Rejected and deferred evidence remains unchanged.",
    );
    if (!accepted) return;
    setMutating(true);
    setError(null);
    try {
      await applyMemoryReconciliation(token, audit.auditId, audit.reviewVersion);
      setReport("Reviewed memory decisions were applied and receipted.");
      setRefreshKey((value) => value + 1);
    } catch (caught) {
      handleFailure(caught, "Could not apply memory reconciliation.");
    } finally {
      setMutating(false);
    }
  };

  return (
    <section className="memory-workspace memory-reconciliation-workspace" aria-label="Memory reconciliation">
      <div className="memory-browser-pane">
        <header className="memory-header">
          <div>
            <p className="breadcrumbs">Kai Workshop / Memory</p>
            <h1>Legacy review</h1>
          </div>
          <button className="panel-icon-button" type="button" aria-label="Back to memories" title="Back to memories" onClick={onBack}>
            <span aria-hidden="true">←</span>
          </button>
        </header>

        <div className="memory-browser-scroll">
          {loading ? (
            <p className="memory-list-state" role="status">Loading reconciliation review…</p>
          ) : error ? (
            <div className="memory-list-state error" role="alert"><p>{error}</p><button onClick={() => setRefreshKey((value) => value + 1)}>Retry</button></div>
          ) : !audit ? (
            <div className="memory-list-state">
              <MemoryReviewIcon />
              <h2>No reconciliation audit</h2>
              <p>Run a read-only authorized audit before reviewing legacy memory here.</p>
            </div>
          ) : (
            <>
              <section className="memory-reconciliation-summary" aria-label="Reconciliation progress">
                <div><strong>{audit.candidateCount}</strong><span>Candidates</span></div>
                <div><strong>{audit.dispositionCounts.pending}</strong><span>Pending</span></div>
                <div><strong>{audit.dispositionCounts.approve}</strong><span>Approved</span></div>
                <div><strong>{audit.dispositionCounts.reject + audit.dispositionCounts.defer}</strong><span>Not adopted</span></div>
              </section>
              <p className="memory-review-explanation">
                Legacy provenance gaps require review; they do not establish that a memory is false.
              </p>
              <form className="memory-reconciliation-filters" onSubmit={(event) => {
                event.preventDefault();
                setFilters(draftFilters);
              }}>
                <label>Reason<select value={draftFilters.category} onChange={(event) => setDraftFilters((value) => ({ ...value, category: event.target.value }))}>
                  <option value="">Every reason</option>
                  {Object.keys(audit.categoryCounts).map((value) => <option key={value} value={value}>{formatLabel(value)}</option>)}
                </select></label>
                <label>Action<select value={draftFilters.action} onChange={(event) => setDraftFilters((value) => ({ ...value, action: event.target.value }))}>
                  <option value="">Every action</option>
                  {Object.keys(audit.actionCounts).map((value) => <option key={value} value={value}>{formatLabel(value)}</option>)}
                </select></label>
                <label>Missing provenance<select value={draftFilters.gap} onChange={(event) => setDraftFilters((value) => ({ ...value, gap: event.target.value }))}>
                  <option value="">Every gap</option>
                  {Object.keys(audit.gapCounts).map((value) => <option key={value} value={value}>{formatLabel(value)}</option>)}
                </select></label>
                <label>State<select value={draftFilters.disposition} onChange={(event) => setDraftFilters((value) => ({ ...value, disposition: event.target.value as ReconciliationFilters["disposition"] }))}>
                  <option value="">Every state</option><option value="pending">Pending</option><option value="approve">Approved</option><option value="reject">Rejected</option><option value="defer">Deferred</option>
                </select></label>
                <label>Kind<select value={draftFilters.kind} onChange={(event) => setDraftFilters((value) => ({ ...value, kind: event.target.value }))}>
                  <option value="">Facts and episodes</option><option value="fact">Facts</option><option value="episode">Episodes</option>
                </select></label>
                <label>Scope<select value={draftFilters.scope} onChange={(event) => setDraftFilters((value) => ({ ...value, scope: event.target.value }))}>
                  <option value="">Every scope</option><option value="global">Global</option><option value="project">Project</option>
                </select></label>
                <label>Uncertainty<select value={draftFilters.uncertainty} onChange={(event) => setDraftFilters((value) => ({ ...value, uncertainty: event.target.value }))}>
                  <option value="">Every level</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option>
                </select></label>
                <div><button type="submit">Apply filters</button><button type="button" className="quiet-button" onClick={() => { setDraftFilters(EMPTY_FILTERS); setFilters(EMPTY_FILTERS); }}>Clear</button></div>
              </form>
              {report && <p className="memory-mutation-report" role="status">{report}</p>}
              <div className="memory-review-toolbar">
                <strong>{selectedIds.size} selected</strong>
                <button type="button" disabled={!selectedIds.size || mutating} onClick={() => void bulk("defer")}>Defer selected</button>
                <button type="button" className="danger" disabled={!selectedIds.size || mutating} onClick={() => void bulk("reject")}>Reject selected</button>
                <button type="button" disabled={audit.dispositionCounts.pending !== 0 || audit.status === "applied" || mutating} onClick={() => void apply()}>
                  {audit.status === "applied" ? "Applied" : "Apply reviewed batch"}
                </button>
              </div>
              <div className="memory-record-list memory-reconciliation-list" role="listbox" aria-label="Reconciliation candidates">
                {candidates.map((candidate) => (
                  <div className="memory-record-row selecting" key={candidate.candidateId}>
                    <input type="checkbox" aria-label={`Select ${candidate.evidence[0]?.text ?? candidate.candidateId}`} checked={selectedIds.has(candidate.candidateId)} onChange={() => setSelectedIds((current) => {
                      const next = new Set(current); if (next.has(candidate.candidateId)) next.delete(candidate.candidateId); else next.add(candidate.candidateId); return next;
                    })} />
                    <button type="button" role="option" aria-selected={selectedId === candidate.candidateId} className={`memory-record ${selectedId === candidate.candidateId ? "selected" : ""}`} onClick={() => setSelectedId(candidate.candidateId)}>
                      <span className={`memory-review-state ${candidate.decision.disposition}`}>{candidate.decision.disposition}</span>
                      <span className="memory-record-copy"><strong>{candidate.evidence[0]?.text ?? "Unavailable memory"}</strong><small>{formatLabel(candidate.category)} · {candidate.uncertainty} uncertainty</small></span>
                    </button>
                  </div>
                ))}
              </div>
              {nextOffset !== null && <button className="memory-load-more" type="button" disabled={loadingMore} onClick={() => void loadMore()}>{loadingMore ? "Loading…" : "Load more candidates"}</button>}
            </>
          )}
        </div>
      </div>
      <aside className="context-pane memory-detail-pane" aria-label="Reconciliation candidate detail">
        {detailPanelLayout && (
          <div
            className="context-resize-handle"
            role="separator"
            aria-label="Resize memory reconciliation detail"
            aria-orientation="vertical"
            aria-valuemin={detailPanelLayout.minimumWidth}
            aria-valuemax={detailPanelLayout.maximumWidth}
            aria-valuenow={detailPanelLayout.width}
            tabIndex={0}
            onKeyDown={detailPanelLayout.onKeyDown}
            onPointerDown={detailPanelLayout.onPointerDown}
            onPointerMove={detailPanelLayout.onPointerMove}
          />
        )}
        <CandidateEditor
          allowedProjects={allowedProjects}
          auditId={audit?.auditId ?? null}
          candidate={selected}
          key={`${audit?.auditId ?? "none"}:${selected?.candidateId ?? "none"}:${selected?.decision.stateVersion ?? 0}`}
          onAuthenticationFailure={onAuthenticationFailure}
          readOnly={audit?.status === "applied"}
          onSaved={() => {
            setReport("Review decision saved.");
            setRefreshKey((value) => value + 1);
          }}
          token={token}
        />
      </aside>
    </section>
  );
}
