/*
 * Owner fact review: settle unresolved fact conflicts, restore forgotten facts,
 * and retry facts or episodes whose search sync failed.
 *
 * None of these have a usable vector row (a conflict deletes the old row and
 * never projects the new one; forgetting or expiry deletes it; a failed sync
 * never wrote or rewrote it), so the Memory explorer cannot list them. This view reads them from the canonical review
 * endpoints instead and follows the reconciliation view's layout: a list on
 * the left and a resizable detail pane on the right.
 *
 * Every decision goes through an inline confirmation rather than the shared
 * confirmation dialog, because the owner may add a note that is recorded as
 * part of the lifecycle reason.
 */
import { useCallback, useEffect, useState } from "react";

import {
  AuthenticationError,
  MemoryConflictChangedError,
  loadForgottenMemories,
  loadMemoryConflict,
  loadMemoryConflicts,
  loadMemoryProjectionAudit,
  loadMemoryProjections,
  resolveMemoryConflict,
  restoreForgottenMemory,
  retryMemoryProjections,
} from "./api";
import { MarkdownMessage } from "./MarkdownMessage";
import type {
  WorkshopMemoryConflictSummary,
  WorkshopMemoryForgottenFact,
  WorkshopMemoryLifecycle,
  WorkshopMemoryLifecycleRevision,
  WorkshopMemoryProjectionAudit,
  WorkshopMemoryProjectionFailure,
  WorkshopMemoryProjectionReview,
} from "./types";

// Matches the server's note limit so the field cannot submit a note the
// server would reject.
const MAX_NOTE_CHARACTERS = 500;

type ReviewTab = "conflicts" | "forgotten" | "sync";

// Error classes are stored as short codes; say what the owner can act on.
function syncErrorLabel(code: string | null): string {
  switch (code) {
    case "LifecycleProjectionReadError":
      return "The search index could not be read";
    case "FactLifecycleProjectionFailed":
    case "EpisodeHistoryProjectionFailed":
      return "The search index refused the write";
    default:
      return code ?? "Unknown error";
  }
}

function syncOutcome(result: { failed: number; retried: number; succeeded: number }): string {
  if (result.retried === 0) return "Nothing needed a retry.";
  const still = result.failed > 0 ? ` ${result.failed} still failing.` : "";
  return `Retried ${result.retried}: ${result.succeeded} back in search.${still}`;
}

// The decision awaiting confirmation. Keeping a revision names the side the
// owner picked; restoring always restores the listed revision.
type PendingDecision = { kind: "keep"; revisionId: string } | { kind: "restore" } | null;

interface DetailPanelLayout {
  maximumWidth: number;
  minimumWidth: number;
  width: number;
  onKeyDown: React.KeyboardEventHandler<HTMLDivElement>;
  onPointerDown: React.PointerEventHandler<HTMLDivElement>;
  onPointerMove: React.PointerEventHandler<HTMLDivElement>;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function scopeLabel(scope: { kind: string; key: string | null }): string {
  return scope.kind === "project" && scope.key ? `Project ${scope.key}` : "Global";
}

// Extraction revisions record the model that proposed them; owner edits and
// restores carry none, so an empty source means the owner wrote it.
function sourceLabel(revision: WorkshopMemoryLifecycleRevision): string {
  const parts = [revision.provider, revision.model].filter((part): part is string => Boolean(part));
  return parts.length > 0 ? parts.join(" · ") : "Operator";
}

function admissionLabel(value: string | undefined): string {
  return value ? value.replaceAll("_", " ") : "unknown";
}

// Oldest first, so the side that used to be current reads first.
function unresolvedRevisions(lifecycle: WorkshopMemoryLifecycle | null): WorkshopMemoryLifecycleRevision[] {
  if (!lifecycle) return [];
  return lifecycle.revisions
    .filter((revision) => revision.state === "unresolved_conflict")
    .sort((left, right) => left.storedAt.localeCompare(right.storedAt));
}

export function FactReview({
  detailPanelLayout,
  onAuthenticationFailure,
  onBack,
  onChanged,
  onOpenMemory,
  token,
}: {
  detailPanelLayout?: DetailPanelLayout;
  onAuthenticationFailure: (message: string) => void;
  onBack: () => void;
  // Called after a successful decision so the explorer can refresh its
  // statistics and conflict badge.
  onChanged: () => void;
  onOpenMemory: (memoryId: string) => void;
  token: string;
}): React.JSX.Element {
  const [tab, setTab] = useState<ReviewTab>("conflicts");
  const [conflicts, setConflicts] = useState<WorkshopMemoryConflictSummary[]>([]);
  const [conflictTotal, setConflictTotal] = useState(0);
  const [forgotten, setForgotten] = useState<WorkshopMemoryForgottenFact[]>([]);
  const [forgottenTotal, setForgottenTotal] = useState(0);
  const [sync, setSync] = useState<WorkshopMemoryProjectionReview | null>(null);
  // The index check reads the owner's whole search index, so it runs only
  // when asked and its result is kept until the next check.
  const [audit, setAudit] = useState<WorkshopMemoryProjectionAudit | null>(null);
  const [auditing, setAuditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedClaimId, setSelectedClaimId] = useState<string | null>(null);
  const [detail, setDetail] = useState<WorkshopMemoryLifecycle | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailKey, setDetailKey] = useState(0);
  const [pending, setPending] = useState<PendingDecision>(null);
  const [note, setNote] = useState("");
  const [mutating, setMutating] = useState(false);
  const [decisionError, setDecisionError] = useState<string | null>(null);
  // The outcome of the last decision, kept after the claim leaves the list
  // so the owner can still open the fact in the explorer.
  const [report, setReport] = useState<{ memoryId: string | null; text: string } | null>(null);

  const failureMessage = useCallback((caught: unknown, fallback: string): string => {
    if (caught instanceof AuthenticationError) onAuthenticationFailure(caught.message);
    return caught instanceof Error ? caught.message : fallback;
  }, [onAuthenticationFailure]);

  // Every list loads together so each tab can show its count up front.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void Promise.all([loadMemoryConflicts(token), loadForgottenMemories(token), loadMemoryProjections(token)])
      .then(([conflictList, forgottenList, syncStatus]) => {
        if (cancelled) return;
        setConflicts(conflictList.items);
        setConflictTotal(conflictList.total);
        setForgotten(forgottenList.items);
        setForgottenTotal(forgottenList.total);
        setSync(syncStatus);
      })
      .catch((caught: unknown) => {
        if (!cancelled) setError(failureMessage(caught, "Could not load fact review."));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [failureMessage, refreshKey, token]);

  // Conflict detail comes from the server so every revision shows its full
  // content and provenance; forgotten facts render from the list item.
  useEffect(() => {
    setDetail(null);
    setDetailError(null);
    if (tab !== "conflicts" || !selectedClaimId) return;
    let cancelled = false;
    void loadMemoryConflict(token, selectedClaimId)
      .then((lifecycle) => {
        if (!cancelled) setDetail(lifecycle);
      })
      .catch((caught: unknown) => {
        if (!cancelled) setDetailError(failureMessage(caught, "Could not load this conflict."));
      });
    return () => { cancelled = true; };
  }, [detailKey, failureMessage, selectedClaimId, tab, token]);

  const select = (claimId: string): void => {
    setSelectedClaimId(claimId);
    setPending(null);
    setNote("");
    setDecisionError(null);
  };

  const switchTab = (next: ReviewTab): void => {
    setTab(next);
    setSelectedClaimId(null);
    setPending(null);
    setNote("");
    setDecisionError(null);
  };

  const selectedConflict = conflicts.find((item) => item.claimId === selectedClaimId) ?? null;
  const selectedForgotten = forgotten.find((item) => item.claimId === selectedClaimId) ?? null;
  const selectedSync = sync?.failed.find((item) => item.id === selectedClaimId) ?? null;
  const sides = unresolvedRevisions(detail);
  const keptSide = pending?.kind === "keep"
    ? sides.find((revision) => revision.revisionId === pending.revisionId) ?? null
    : null;

  const confirmDecision = async (): Promise<void> => {
    if (!pending || !selectedClaimId) return;
    setMutating(true);
    setDecisionError(null);
    try {
      const outcome = pending.kind === "keep"
        ? await resolveMemoryConflict(token, selectedClaimId, {
            keepRevisionId: pending.revisionId,
            // The server refuses the decision if the unresolved set changed
            // since this detail loaded, so the owner only settles what they saw.
            expectedRevisionIds: sides.map((revision) => revision.revisionId),
            note: note.trim(),
          })
        : await restoreForgottenMemory(token, selectedClaimId, {
            revisionId: selectedForgotten?.revisionId ?? "",
            note: note.trim(),
          });
      setReport({
        memoryId: outcome.memoryId,
        text: pending.kind === "keep"
          ? "Kept the selected version. It is back in recall."
          : "Restored the fact. It is current from now, with no end date.",
      });
      setSelectedClaimId(null);
      setPending(null);
      setNote("");
      setRefreshKey((value) => value + 1);
      onChanged();
    } catch (caught) {
      if (caught instanceof MemoryConflictChangedError) {
        // Show the fresh state instead of leaving a decision the server
        // will keep refusing.
        setPending(null);
        setDecisionError("This conflict changed since you opened it. The latest versions are shown below.");
        setDetailKey((value) => value + 1);
        setRefreshKey((value) => value + 1);
      } else {
        setDecisionError(failureMessage(caught, "Could not save this decision."));
      }
    } finally {
      setMutating(false);
    }
  };

  const confirmation = pending && (
    <section className="memory-detail-section memory-review-decision fact-review-confirmation" aria-label="Confirm decision">
      <p className="memory-section-label">{pending.kind === "keep" ? "Keep this version" : "Restore this fact"}</p>
      <p>
        {pending.kind === "keep"
          ? "The kept version becomes the current fact. The other version will be kept in history as superseded."
          : "The fact will be current from now, with no end date. Its earlier history stays as it is."}
      </p>
      {keptSide?.admissionAuthority === "quarantined" && (
        <p className="fact-review-warning" role="note">
          This version has incomplete provenance. It will stay out of recall until its admission is reviewed.
        </p>
      )}
      <label>Note (optional)
        <textarea value={note} maxLength={MAX_NOTE_CHARACTERS} onChange={(event) => setNote(event.target.value)} />
      </label>
      <div className="memory-review-toolbar">
        <button type="button" disabled={mutating} onClick={() => void confirmDecision()}>
          {mutating ? "Saving…" : pending.kind === "keep" ? "Keep this version" : "Restore"}
        </button>
        <button type="button" disabled={mutating} onClick={() => { setPending(null); setNote(""); }}>Cancel</button>
      </div>
    </section>
  );

  // Retrying is harmless to repeat (only failed items reset), so it runs
  // without a confirmation step. With no item, everything failed retries.
  const retrySync = async (item: WorkshopMemoryProjectionFailure | null): Promise<void> => {
    setMutating(true);
    setDecisionError(null);
    try {
      const result = await retryMemoryProjections(
        token,
        item === null ? {} : item.kind === "fact" ? { claimIds: [item.id] } : { episodeIds: [item.id] },
      );
      setReport({ memoryId: null, text: syncOutcome(result) });
      setSelectedClaimId(null);
      setRefreshKey((value) => value + 1);
      onChanged();
    } catch (caught) {
      setDecisionError(failureMessage(caught, "Could not retry search sync."));
    } finally {
      setMutating(false);
    }
  };

  const checkIndex = async (): Promise<void> => {
    setAuditing(true);
    setDecisionError(null);
    try {
      setAudit(await loadMemoryProjectionAudit(token));
    } catch (caught) {
      setDecisionError(failureMessage(caught, "Could not check the search index."));
    } finally {
      setAuditing(false);
    }
  };

  const syncList = (sync?.failed ?? []).map((item) => (
    <button type="button" role="option" aria-selected={selectedClaimId === item.id} key={`${item.kind}:${item.id}`}
      className={`memory-record ${selectedClaimId === item.id ? "selected" : ""}`}
      onClick={() => select(item.id)}>
      <span className="memory-review-state failed">not in search</span>
      <span className="memory-record-copy">
        <strong>{item.preview}</strong>
        <small>{item.kind === "fact" ? "Fact" : "Episode"} · {item.operation} · last tried {formatDate(item.updatedAt)}</small>
      </span>
    </button>
  ));

  const list = tab === "sync" ? syncList : tab === "conflicts"
    ? conflicts.map((item) => (
      <button type="button" role="option" aria-selected={selectedClaimId === item.claimId} key={item.claimId}
        className={`memory-record ${selectedClaimId === item.claimId ? "selected" : ""}`}
        onClick={() => select(item.claimId)}>
        <span className="memory-review-state unresolved_conflict">conflict</span>
        <span className="memory-record-copy">
          <strong>{item.revisions.map((revision) => revision.preview).join(" / ")}</strong>
          <small>{scopeLabel(item.scope)} · opened {formatDate(item.openedAt)}</small>
        </span>
      </button>
    ))
    : forgotten.map((item) => (
      <button type="button" role="option" aria-selected={selectedClaimId === item.claimId} key={item.claimId}
        className={`memory-record ${selectedClaimId === item.claimId ? "selected" : ""}`}
        onClick={() => select(item.claimId)}>
        <span className={`memory-review-state ${item.state}`}>{item.state}</span>
        <span className="memory-record-copy">
          <strong>{item.preview}</strong>
          <small>{scopeLabel(item.scope)} · {formatDate(item.changedAt)}</small>
        </span>
      </button>
    ));

  return (
    <section className="memory-workspace memory-reconciliation-workspace" aria-label="Fact review">
      <div className="memory-browser-pane">
        <header className="memory-header">
          <div><p className="breadcrumbs">Kai Workshop / Memory</p><h1>Fact review</h1></div>
          <button className="panel-icon-button" type="button" aria-label="Back to memories" title="Back to memories" onClick={onBack}>
            <span aria-hidden="true">←</span>
          </button>
        </header>
        <div className="memory-browser-scroll">
          <div className="fact-review-tabs" role="tablist" aria-label="Fact review lists">
            <button type="button" role="tab" aria-selected={tab === "conflicts"} onClick={() => switchTab("conflicts")}>
              Conflicts <span>{conflictTotal}</span>
            </button>
            <button type="button" role="tab" aria-selected={tab === "forgotten"} onClick={() => switchTab("forgotten")}>
              Forgotten <span>{forgottenTotal}</span>
            </button>
            <button type="button" role="tab" aria-selected={tab === "sync"} onClick={() => switchTab("sync")}>
              Search sync <span>{sync?.failedTotal ?? 0}</span>
            </button>
          </div>
          {tab === "sync" && !loading && !error && (
            <section className="fact-review-sync-summary" aria-label="Search sync summary">
              <p className="memory-review-explanation">
                These facts and episodes are saved, but their search copy failed to update, so Kai cannot recall them yet.
                {sync && sync.blocked > 0 && ` ${sync.blocked} later change${sync.blocked === 1 ? " is" : "s are"} waiting behind them.`}
              </p>
              <div className="memory-review-toolbar">
                <button type="button" disabled={mutating || (sync?.failedTotal ?? 0) === 0} onClick={() => void retrySync(null)}>
                  Retry all
                </button>
                <button type="button" disabled={auditing} onClick={() => void checkIndex()}>
                  {auditing ? "Checking…" : "Check search index"}
                </button>
              </div>
              {audit && (
                <p className="memory-review-explanation" role="note">
                  {audit.orphan.length + audit.unknown.length === 0 && audit.duplicate === 0
                    ? "The search index matches your memory."
                    : `The search index has ${audit.orphan.length} leftover and ${audit.unknown.length} unknown row(s)` +
                      `${audit.duplicate > 0 ? `, and ${audit.duplicate} item(s) with duplicates` : ""}. ` +
                      "Kai never recalls them; they only take up space."}
                </p>
              )}
              {decisionError && !selectedSync && <p className="memory-editor-error" role="alert">{decisionError}</p>}
            </section>
          )}
          {report && (
            <p className="memory-mutation-report" role="status">
              {report.text}
              {report.memoryId && (
                <> <button type="button" className="fact-review-link" onClick={() => onOpenMemory(report.memoryId as string)}>
                  Open in memory
                </button></>
              )}
            </p>
          )}
          {loading ? (
            <p className="memory-list-state" role="status">Loading fact review…</p>
          ) : error ? (
            <div className="memory-list-state error" role="alert"><p>{error}</p><button onClick={() => setRefreshKey((value) => value + 1)}>Retry</button></div>
          ) : list.length === 0 ? (
            <div className="memory-list-state">
              <h2>{tab === "conflicts" ? "No open conflicts" : tab === "forgotten" ? "No forgotten facts" : "Everything is in search"}</h2>
              <p>
                {tab === "conflicts"
                  ? "Conflicts appear when an extracted update is not certain enough to replace a stored fact."
                  : tab === "forgotten"
                    ? "Facts you forget, or that expire, appear here and can be restored."
                    : "Facts and episodes appear here if their search copy fails to update."}
              </p>
            </div>
          ) : (
            <div className="memory-record-list memory-reconciliation-list" role="listbox"
              aria-label={tab === "conflicts" ? "Unresolved conflicts" : tab === "forgotten" ? "Forgotten facts" : "Search sync failures"}>
              {list}
            </div>
          )}
        </div>
      </div>
      <aside className="context-pane memory-detail-pane" aria-label="Fact review detail">
        {detailPanelLayout && (
          <div className="context-resize-handle" role="separator" aria-label="Resize fact review detail"
            aria-orientation="vertical" aria-valuemin={detailPanelLayout.minimumWidth}
            aria-valuemax={detailPanelLayout.maximumWidth} aria-valuenow={detailPanelLayout.width} tabIndex={0}
            onKeyDown={detailPanelLayout.onKeyDown} onPointerDown={detailPanelLayout.onPointerDown}
            onPointerMove={detailPanelLayout.onPointerMove} />
        )}
        <div className="memory-reconciliation-detail">
          {tab === "conflicts" && selectedConflict ? (
            <>
              <header className="memory-detail-header">
                <div><p className="overline">Unresolved conflict</p><h2>Which version is true?</h2></div>
                <span className="memory-review-state unresolved_conflict">conflict</span>
              </header>
              <p className="memory-review-explanation">
                Neither version is recalled until you keep one. If neither wording is quite right, keep the closer one and edit it in memory.
              </p>
              {decisionError && <p className="memory-editor-error" role="alert">{decisionError}</p>}
              {detailError ? (
                <p className="memory-editor-error" role="alert">{detailError}</p>
              ) : !detail ? (
                <p className="memory-list-state" role="status">Loading conflict…</p>
              ) : (
                <div className="fact-review-sides">
                  {sides.map((revision) => (
                    <article className="memory-review-evidence" key={revision.revisionId} aria-label="Competing version">
                      <MarkdownMessage body={revision.content} />
                      <dl className="fact-review-facts">
                        <dt>Stored</dt><dd>{formatDate(revision.storedAt)}</dd>
                        <dt>Source</dt><dd>{sourceLabel(revision)}</dd>
                        <dt>Confidence</dt>
                        <dd>{typeof revision.confidence === "number" ? `${Math.round(revision.confidence * 100)}%` : "Not recorded"}</dd>
                        <dt>Reason</dt><dd>{revision.reason}</dd>
                        <dt>Admission</dt><dd>{admissionLabel(revision.admissionAuthority)}</dd>
                      </dl>
                      <button type="button" disabled={mutating}
                        onClick={() => { setPending({ kind: "keep", revisionId: revision.revisionId }); setDecisionError(null); }}>
                        Keep this version
                      </button>
                    </article>
                  ))}
                </div>
              )}
              {confirmation}
            </>
          ) : tab === "forgotten" && selectedForgotten ? (
            <>
              <header className="memory-detail-header">
                <div><p className="overline">{selectedForgotten.state === "expired" ? "Expired fact" : "Forgotten fact"}</p><h2>Restore to current?</h2></div>
                <span className={`memory-review-state ${selectedForgotten.state}`}>{selectedForgotten.state}</span>
              </header>
              <section className="memory-detail-section">
                <MarkdownMessage body={selectedForgotten.preview} />
                <dl className="fact-review-facts">
                  <dt>Scope</dt><dd>{scopeLabel(selectedForgotten.scope)}</dd>
                  <dt>{selectedForgotten.state === "expired" ? "Expired" : "Forgotten"}</dt>
                  <dd>{formatDate(selectedForgotten.changedAt)}</dd>
                  <dt>Reason</dt><dd>{selectedForgotten.reason}</dd>
                </dl>
              </section>
              {decisionError && <p className="memory-editor-error" role="alert">{decisionError}</p>}
              {!pending && (
                <button type="button" disabled={mutating} onClick={() => { setPending({ kind: "restore" }); setDecisionError(null); }}>
                  Restore
                </button>
              )}
              {confirmation}
            </>
          ) : tab === "sync" && selectedSync ? (
            <>
              <header className="memory-detail-header">
                <div><p className="overline">{selectedSync.kind === "fact" ? "Fact" : "Episode"} not in search</p><h2>Retry search sync?</h2></div>
                <span className="memory-review-state failed">failed</span>
              </header>
              <section className="memory-detail-section">
                <MarkdownMessage body={selectedSync.preview} />
                <dl className="fact-review-facts">
                  <dt>Step</dt><dd>{selectedSync.operation}</dd>
                  <dt>Attempts</dt><dd>{selectedSync.attempts}</dd>
                  <dt>Error</dt><dd>{syncErrorLabel(selectedSync.errorCode)}</dd>
                  <dt>Last tried</dt><dd>{formatDate(selectedSync.updatedAt)}</dd>
                </dl>
              </section>
              <p className="memory-review-explanation">
                The saved {selectedSync.kind} is unchanged. Retrying only updates its search copy{selectedSync.kind === "fact" ? ", then runs any later changes that were waiting on it" : ""}.
              </p>
              {decisionError && <p className="memory-editor-error" role="alert">{decisionError}</p>}
              <button type="button" disabled={mutating} onClick={() => void retrySync(selectedSync)}>
                {mutating ? "Retrying…" : "Retry"}
              </button>
            </>
          ) : (
            <div className="memory-list-state">
              <p>
                {tab === "conflicts"
                  ? "Select a conflict to compare its versions."
                  : tab === "forgotten"
                    ? "Select a fact to see why it left current truth."
                    : "Select an item to see why it is not in search."}
              </p>
            </div>
          )}
        </div>
      </aside>
    </section>
  );
}
