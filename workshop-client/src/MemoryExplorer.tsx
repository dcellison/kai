import {
  FormEvent,
  KeyboardEvent as ReactKeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  AuthenticationError,
  MemoryRevisionConflictError,
  createMemoryFact,
  deleteMemories,
  deleteMemory,
  editMemory,
  loadMemoryDetail,
  loadMemoryRecords,
  loadMemorySource,
  loadMemoryStats,
  moveMemoriesScope,
  moveMemoryScope,
  searchMemories,
} from "./api";
import { MarkdownMessage } from "./MarkdownMessage";
import { ConfirmationProvider, useConfirmation } from "./ConfirmationDialog";
import type {
  WorkshopMemoryDetail,
  WorkshopMemoryEpisodeFields,
  WorkshopMemoryFilters,
  WorkshopMemoryMutationBatch,
  WorkshopMemoryProjectOption,
  WorkshopMemoryRecord,
  WorkshopMemoryScope,
  WorkshopMemorySearchHit,
  WorkshopMemorySourceContext,
  WorkshopMemoryStats,
} from "./types";

interface ExplorerFilters {
  kind: "" | "fact" | "episode";
  projectId: string;
  scope: "" | "global" | "project" | "task";
  tag: string;
}

const EMPTY_FILTERS: ExplorerFilters = {
  kind: "",
  projectId: "",
  scope: "",
  tag: "",
};

function apiFilters(filters: ExplorerFilters): WorkshopMemoryFilters {
  return {
    kind: filters.kind || undefined,
    projectId: filters.projectId || undefined,
    scope: filters.scope || undefined,
    tag: filters.tag.trim() || undefined,
  };
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(date);
}

function scopeDescription(scope: WorkshopMemoryScope): string {
  if (!scope.retrievable) {
    switch (scope.exclusionReason) {
      case "legacy_scope_quarantined":
        return "Quarantined until its legacy scope is reviewed.";
      case "invalid_scope_quarantined":
        return "Quarantined because its stored scope is invalid.";
      case "project_id_mismatch":
        return `Unavailable outside project ${scope.projectId ?? "unknown"}.`;
      case "project_scope_not_allowed":
        return `Available only when project ${scope.projectId ?? "unknown"} is active.`;
      case "task_scope_not_supported":
        return "Task-scoped memories are not currently recalled.";
      default:
        return "Not currently eligible for recall.";
    }
  }
  if (scope.scope === "project") {
    return `Available while project ${scope.projectId ?? "unknown"} is active.`;
  }
  if (scope.scope === "task") {
    return "Available in its assigned task context.";
  }
  return "Available across your projects.";
}

function sourceReason(reason: string | null): string {
  switch (reason) {
    case "explicit_creation":
      return "This memory was saved explicitly, so no source conversation was linked.";
    case "legacy_source":
      return "This memory predates canonical source links.";
    case "canonical_source_missing":
      return "The canonical source conversation is no longer available.";
    case "invalid_provenance":
      return "The stored source reference is incomplete or invalid.";
    case "source_not_authorized":
      return "Your session cannot read the referenced source conversation.";
    default:
      return "No canonical source conversation is available for this memory.";
  }
}

function episodeLabel(key: string): string {
  return key
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function splitEditorValues(value: string): string[] {
  return value
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function editorRequestId(): string {
  return typeof globalThis.crypto?.randomUUID === "function"
    ? globalThis.crypto.randomUUID()
    : `memory-editor-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function MemoryEditorDialog({
  allowedProjects,
  detail,
  onAuthenticationFailure,
  onClose,
  onReload,
  onSaved,
  token,
}: {
  allowedProjects: WorkshopMemoryProjectOption[];
  detail: WorkshopMemoryDetail | null;
  onAuthenticationFailure: (message: string) => void;
  onClose: () => void;
  onReload: () => void;
  onSaved: (record: WorkshopMemoryDetail, message: string) => void;
  token: string;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const episode = detail?.episode;
  const [content, setContent] = useState(detail?.content ?? "");
  const [tags, setTags] = useState((episode?.tags ?? detail?.tags ?? []).join(", "));
  const [target, setTarget] = useState("global");
  const [goal, setGoal] = useState(episode?.goal ?? "");
  const [context, setContext] = useState(episode?.context ?? "");
  const [approach, setApproach] = useState(episode?.approach ?? "");
  const [outcome, setOutcome] = useState(episode?.outcome ?? "");
  const [outcomeQuality, setOutcomeQuality] = useState<WorkshopMemoryEpisodeFields["outcomeQuality"]>(
    episode?.outcomeQuality ?? "success",
  );
  const [lessons, setLessons] = useState(episode?.lessons ?? "");
  const [actors, setActors] = useState((episode?.actors ?? []).join(", "));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);
  const [requestId] = useState(editorRequestId);
  const editing = detail !== null;
  const isEpisode = detail?.kind === "episode";
  const dirty = editing
    ? isEpisode
      ? goal !== (episode?.goal ?? "") || context !== (episode?.context ?? "") ||
        approach !== (episode?.approach ?? "") || outcome !== (episode?.outcome ?? "") ||
        outcomeQuality !== (episode?.outcomeQuality ?? "success") ||
        lessons !== (episode?.lessons ?? "") || actors !== (episode?.actors ?? []).join(", ") ||
        tags !== (episode?.tags ?? []).join(", ")
      : content !== detail.content || tags !== detail.tags.join(", ")
    : Boolean(content.trim() || tags.trim() || target !== "global");

  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent): void => {
      event.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const closeSafely = async (): Promise<void> => {
    if (dirty && !await confirm("Discard your unsaved memory changes?")) return;
    onClose();
  };

  const submit = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    setConflict(false);
    try {
      if (!detail) {
        const parsedTarget = target.startsWith("project:")
          ? { scope: "project" as const, projectId: target.slice("project:".length) }
          : { scope: "global" as const };
        const result = await createMemoryFact(token, {
          content,
          tags: splitEditorValues(tags),
          target: parsedTarget,
          requestId,
        });
        onSaved(result.record, result.created ? "Explicit memory created." : "Memory was already created.");
      } else if (isEpisode) {
        const result = await editMemory(token, {
          kind: "episode",
          memoryId: detail.memoryId,
          revision: detail.revision,
          requestId,
          episode: {
            actors: splitEditorValues(actors),
            approach,
            context,
            goal,
            lessons: lessons.trim() || null,
            outcome,
            outcomeQuality,
            tags: splitEditorValues(tags),
          },
        });
        onSaved(result.record, result.idempotentReplay ? "Memory was already saved." : "Episode updated and re-indexed.");
      } else {
        const result = await editMemory(token, {
          kind: "fact",
          memoryId: detail.memoryId,
          revision: detail.revision,
          requestId,
          content,
          tags: splitEditorValues(tags),
        });
        onSaved(result.record, result.idempotentReplay ? "Memory was already saved." : "Fact updated and re-indexed.");
      }
    } catch (caught) {
      if (caught instanceof MemoryRevisionConflictError) {
        setConflict(true);
        setError("This memory changed after you opened it. Reload the latest revision before saving again.");
      } else if (caught instanceof AuthenticationError) {
        onAuthenticationFailure(caught.message);
        setError(caught.message);
      } else {
        setError(caught instanceof Error ? caught.message : "Could not save this memory.");
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="memory-confirm-backdrop memory-editor-backdrop">
      <section className="memory-editor" role="dialog" aria-modal="true" aria-labelledby="memory-editor-title">
        <header>
          <div>
            <p className="overline">{editing ? "Edit memory" : "New explicit memory"}</p>
            <h2 id="memory-editor-title">
              {editing ? `Correct ${isEpisode ? "episode" : "fact"}` : "Create fact"}
            </h2>
          </div>
          <button type="button" className="quiet-button" disabled={saving} onClick={() => void closeSafely()}>Close</button>
        </header>
        <form onSubmit={(event) => void submit(event)}>
          {isEpisode ? (
            <>
              <label>Goal <span>affects semantic retrieval</span>
                <textarea required maxLength={20000} value={goal} onChange={(event) => setGoal(event.target.value)} />
              </label>
              <label>Context <span>affects semantic retrieval</span>
                <textarea required maxLength={20000} value={context} onChange={(event) => setContext(event.target.value)} />
              </label>
              <label>Approach
                <textarea required maxLength={20000} value={approach} onChange={(event) => setApproach(event.target.value)} />
              </label>
              <label>Outcome
                <textarea required maxLength={20000} value={outcome} onChange={(event) => setOutcome(event.target.value)} />
              </label>
              <div className="memory-editor-grid">
                <label>Outcome quality
                  <select value={outcomeQuality} onChange={(event) => setOutcomeQuality(
                    event.target.value as WorkshopMemoryEpisodeFields["outcomeQuality"],
                  )}>
                    <option value="success">Success</option>
                    <option value="partial">Partial</option>
                    <option value="failure">Failure</option>
                  </select>
                </label>
                <label>Actors <span>comma-separated</span>
                  <input maxLength={4096} value={actors} onChange={(event) => setActors(event.target.value)} />
                </label>
              </div>
              <label>Lessons <span>optional</span>
                <textarea maxLength={20000} value={lessons} onChange={(event) => setLessons(event.target.value)} />
              </label>
            </>
          ) : (
            <label>Content <span>affects semantic retrieval</span>
              <textarea required maxLength={100000} value={content} onChange={(event) => setContent(event.target.value)} />
            </label>
          )}
          <label>Tags <span>comma- or line-separated</span>
            <input maxLength={4096} value={tags} onChange={(event) => setTags(event.target.value)} />
          </label>
          {!editing && (
            <label>Recall scope
              <select value={target} onChange={(event) => setTarget(event.target.value)}>
                <option value="global">Global scope</option>
                {allowedProjects.map((project) => (
                  <option key={project.projectId} value={`project:${project.projectId}`}>
                    {project.displayName}
                  </option>
                ))}
              </select>
            </label>
          )}
          {editing && (
            <p className="memory-editor-provenance">
              Source provenance and authority fields remain immutable. Saving regenerates the searchable vector.
            </p>
          )}
          {error && <p className="memory-editor-error" role="alert">{error}</p>}
          <footer>
            {conflict && (
              <button type="button" className="quiet-button" onClick={onReload}>Reload latest</button>
            )}
            <button type="button" className="quiet-button" disabled={saving} onClick={() => void closeSafely()}>Cancel</button>
            <button type="submit" disabled={saving || (editing && !dirty)}>
              {saving ? "Saving…" : editing ? "Save correction" : "Create memory"}
            </button>
          </footer>
        </form>
      </section>
    </div>
  );
}

function MemoryDetailPane({
  allowedProjects,
  busy,
  detail,
  error,
  loading,
  onDelete,
  onEdit,
  onMove,
  source,
  sourceError,
}: {
  allowedProjects: WorkshopMemoryProjectOption[];
  busy: boolean;
  detail: WorkshopMemoryDetail | null;
  error: string | null;
  loading: boolean;
  onDelete: (memoryId: string) => void;
  onEdit: (detail: WorkshopMemoryDetail) => void;
  onMove: (memoryId: string, target: string) => void;
  source: WorkshopMemorySourceContext | null;
  sourceError: string | null;
}): React.JSX.Element {
  const [target, setTarget] = useState("global");
  if (loading) {
    return <p className="memory-detail-state" role="status">Loading memory details…</p>;
  }
  if (error) {
    return <p className="memory-detail-state error" role="alert">{error}</p>;
  }
  if (!detail) {
    return (
      <div className="memory-detail-empty">
        <span aria-hidden="true">◇</span>
        <h2>Select a memory</h2>
        <p>Choose a fact or episode to inspect its content, scope, recall preview, and source.</p>
      </div>
    );
  }

  return (
    <div className="memory-detail-scroll">
      <header className="memory-detail-header">
        <div>
          <p className="overline">Memory detail</p>
          <h2>{detail.kind === "episode" ? "Episode" : "Fact"}</h2>
        </div>
        <span className={`memory-kind ${detail.kind}`}>{detail.kind}</span>
      </header>

      <section className="memory-detail-section">
        <p className="memory-section-label">Stored memory</p>
        <MarkdownMessage body={detail.content} />
      </section>

      {detail.episode && Object.keys(detail.episode).length > 0 && (
        <section className="memory-detail-section">
          <p className="memory-section-label">Episode structure</p>
          <dl className="memory-episode-fields">
            {Object.entries(detail.episode).map(([key, value]) => (
              <div key={key}>
                <dt>{episodeLabel(key)}</dt>
                <dd><MarkdownMessage body={Array.isArray(value) ? value.join(", ") : value ?? ""} /></dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      <section className="memory-detail-section scope-card">
        <p className="memory-section-label">Recall scope</p>
        <strong>{detail.scope.scope === "project" ? "Project memory" : `${episodeLabel(detail.scope.scope)} memory`}</strong>
        <p>{scopeDescription(detail.scope)}</p>
        <dl className="memory-metadata-grid">
          <div><dt>Speaker</dt><dd>{detail.speaker}</dd></div>
          <div><dt>Confidence</dt><dd>{Math.round(detail.confidence * 100)}%</dd></div>
          <div><dt>Source</dt><dd>{detail.source}</dd></div>
          <div><dt>Updated</dt><dd>{formatDate(detail.updatedAt)}</dd></div>
        </dl>
      </section>

      <section className="memory-detail-section memory-management">
        <p className="memory-section-label">Manage memory</p>
        <label>
          Move to
          <select value={target} onChange={(event) => setTarget(event.target.value)}>
            <option value="global">Global scope</option>
            {allowedProjects.map((project) => (
              <option key={project.projectId} value={`project:${project.projectId}`}>
                {project.displayName}
              </option>
            ))}
          </select>
        </label>
        <div>
          <button type="button" disabled={busy} onClick={() => onEdit(detail)}>
            Edit memory…
          </button>
          <button type="button" disabled={busy} onClick={() => onMove(detail.memoryId, target)}>
            Move memory…
          </button>
          <button
            className="danger"
            type="button"
            disabled={busy}
            onClick={() => onDelete(detail.memoryId)}
          >
            Forget memory…
          </button>
        </div>
      </section>

      <section className="memory-detail-section recall-preview">
        <p className="memory-section-label">Agent recall preview</p>
        <p>
          This compact record—not the rich stored episode above—is what Kai can
          place in an agent prompt when this memory is recalled.
        </p>
        <pre>{detail.compactRecall}</pre>
      </section>

      <section className="memory-detail-section">
        <p className="memory-section-label">Source conversation</p>
        {sourceError ? (
          <p className="memory-source-unavailable" role="alert">{sourceError}</p>
        ) : !source ? (
          <p className="memory-source-unavailable" role="status">Loading source context…</p>
        ) : source.status === "unavailable" ? (
          <p className="memory-source-unavailable">{sourceReason(source.reason)}</p>
        ) : (
          <div className="memory-source-context">
            {source.source && (
              <article>
                <header>
                  <strong>{source.source.authorDisplayName}</strong>
                  <time>{formatDate(source.source.createdAt)}</time>
                </header>
                <MarkdownMessage body={source.source.body} />
              </article>
            )}
            {source.result && (
              <article className="agent">
                <header>
                  <strong>{source.result.authorDisplayName}</strong>
                  <time>{formatDate(source.result.createdAt)}</time>
                </header>
                <MarkdownMessage body={source.result.body} />
              </article>
            )}
          </div>
        )}
      </section>

      {detail.tags.length > 0 && (
        <section className="memory-detail-section">
          <p className="memory-section-label">Tags</p>
          <div className="memory-tags">
            {detail.tags.map((tag) => <span key={tag}>{tag}</span>)}
          </div>
        </section>
      )}
    </div>
  );
}

function MemoryExplorerContent({
  initialMemoryId,
  onAuthenticationFailure,
  onClose,
  onForget,
  onSelectMemory,
  token,
}: {
  initialMemoryId: string | null;
  onAuthenticationFailure: (message: string) => void;
  onClose: () => void;
  onForget: () => void;
  onSelectMemory: (memoryId: string | null) => void;
  token: string;
}): React.JSX.Element {
  const [stats, setStats] = useState<WorkshopMemoryStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [filterDraft, setFilterDraft] = useState<ExplorerFilters>(EMPTY_FILTERS);
  const [filters, setFilters] = useState<ExplorerFilters>(EMPTY_FILTERS);
  const [queryDraft, setQueryDraft] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [browseOrder, setBrowseOrder] = useState<"newest" | "oldest">("newest");
  const [searchOrder, setSearchOrder] = useState<"relevance" | "newest" | "oldest">("relevance");
  const [records, setRecords] = useState<WorkshopMemoryRecord[]>([]);
  const [searchHits, setSearchHits] = useState<WorkshopMemorySearchHit[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [detailRefreshKey, setDetailRefreshKey] = useState(0);
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(initialMemoryId);
  const [detail, setDetail] = useState<WorkshopMemoryDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(initialMemoryId !== null);
  const [source, setSource] = useState<WorkshopMemorySourceContext | null>(null);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [bulkTarget, setBulkTarget] = useState("global");
  const [pendingMutation, setPendingMutation] = useState<{
    ids: string[];
    operation: "move_scope" | "delete";
    target?: string;
  } | null>(null);
  const [mutationRunning, setMutationRunning] = useState(false);
  const [mutationReport, setMutationReport] = useState<string | null>(null);
  const [editorDetail, setEditorDetail] = useState<WorkshopMemoryDetail | "create" | null>(null);
  const recordRefs = useRef<Array<HTMLButtonElement | null>>([]);

  useEffect(() => {
    setSelectedMemoryId(initialMemoryId);
  }, [initialMemoryId]);

  const handleError = useCallback((caught: unknown, fallback: string): string => {
    if (caught instanceof AuthenticationError) {
      onAuthenticationFailure(caught.message);
      return caught.message;
    }
    return caught instanceof Error ? caught.message : fallback;
  }, [onAuthenticationFailure]);

  useEffect(() => {
    let cancelled = false;
    setStatsError(null);
    void loadMemoryStats(token)
      .then((value) => {
        if (!cancelled) setStats(value);
      })
      .catch((caught) => {
        if (!cancelled) {
          setStatsError(handleError(caught, "Could not load memory statistics."));
        }
      });
    return () => { cancelled = true; };
  }, [handleError, refreshKey, token]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setListError(null);
    setNextCursor(null);
    const request = searchQuery
      ? searchMemories(token, searchQuery, {
          ...apiFilters(filters),
          limit: 50,
        })
      : loadMemoryRecords(token, {
          ...apiFilters(filters),
          limit: 50,
          order: browseOrder,
        });
    void request
      .then((value) => {
        if (cancelled) return;
        if ("hits" in value) {
          setSearchHits(value.hits);
          setRecords([]);
        } else {
          setRecords(value.records);
          setSearchHits([]);
          setNextCursor(value.nextCursor);
        }
      })
      .catch((caught) => {
        if (!cancelled) {
          setListError(handleError(caught, "Could not load memories."));
          setRecords([]);
          setSearchHits([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [browseOrder, filters, handleError, refreshKey, searchQuery, token]);

  const displayedRecords = useMemo(() => {
    if (!searchQuery) return records;
    const hits = [...searchHits];
    if (searchOrder !== "relevance") {
      hits.sort((left, right) => {
        const comparison = left.record.updatedAt.localeCompare(right.record.updatedAt);
        return searchOrder === "newest" ? -comparison : comparison;
      });
    }
    return hits.map((hit) => hit.record);
  }, [records, searchHits, searchOrder, searchQuery]);

  useEffect(() => {
    if (!selectedMemoryId && displayedRecords[0]) {
      setSelectedMemoryId(displayedRecords[0].memoryId);
      onSelectMemory(displayedRecords[0].memoryId);
    }
  }, [displayedRecords, onSelectMemory, selectedMemoryId]);

  useEffect(() => {
    if (!selectedMemoryId) {
      setDetail(null);
      setSource(null);
      setDetailLoading(false);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    setDetailError(null);
    setSourceError(null);
    setDetail(null);
    setSource(null);
    void Promise.allSettled([
      loadMemoryDetail(token, selectedMemoryId),
      loadMemorySource(token, selectedMemoryId),
    ]).then(([detailResult, sourceResult]) => {
      if (cancelled) return;
      if (detailResult.status === "fulfilled") {
        setDetail(detailResult.value);
      } else {
        setDetailError(handleError(detailResult.reason, "Could not load this memory."));
      }
      if (sourceResult.status === "fulfilled") {
        setSource(sourceResult.value);
      } else {
        setSourceError(handleError(sourceResult.reason, "Could not load source context."));
      }
      setDetailLoading(false);
    });
    return () => { cancelled = true; };
  }, [detailRefreshKey, handleError, selectedMemoryId, token]);

  const projectIds = useMemo(() => {
    if (!stats) return [];
    return Object.keys(stats.byScope)
      .filter((scope) => scope.startsWith("project:"))
      .map((scope) => scope.slice("project:".length))
      .filter(Boolean)
      .sort();
  }, [stats]);

  const allowedProjects = stats?.allowedProjects ?? [];

  const selectMemory = useCallback((memoryId: string): void => {
    setSelectedMemoryId(memoryId);
    onSelectMemory(memoryId);
  }, [onSelectMemory]);

  const resetSelectionAndResults = (): void => {
    setSelectionMode(false);
    setSelectedMemoryId(null);
    setRecords([]);
    setSearchHits([]);
    setNextCursor(null);
    setSelectedIds(new Set());
    onSelectMemory(null);
  };

  const toggleSelected = (memoryId: string): void => {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(memoryId)) next.delete(memoryId);
      else next.add(memoryId);
      return next;
    });
  };

  const cancelSelection = (): void => {
    setSelectionMode(false);
    setSelectedIds(new Set());
  };

  const mutationSummary = (batch: WorkshopMemoryMutationBatch): string => {
    const counts = { succeeded: 0, not_found: 0, stale: 0, failed: 0 };
    for (const result of batch.results) counts[result.outcome] += 1;
    const parts = [`${counts.succeeded} succeeded`];
    if (counts.stale) parts.push(`${counts.stale} became stale`);
    if (counts.not_found) parts.push(`${counts.not_found} no longer existed`);
    if (counts.failed) parts.push(`${counts.failed} failed`);
    return `${batch.operation === "delete" ? "Forget" : "Scope change"}: ${parts.join(", ")}.`;
  };

  const runMutation = async (): Promise<void> => {
    if (!pendingMutation) return;
    setMutationRunning(true);
    setMutationReport(null);
    try {
      let batch: WorkshopMemoryMutationBatch;
      if (pendingMutation.operation === "delete") {
        batch = pendingMutation.ids.length === 1
          ? await deleteMemory(token, pendingMutation.ids[0]!)
          : await deleteMemories(token, pendingMutation.ids);
      } else {
        const target = pendingMutation.target ?? "global";
        const parsedTarget = target.startsWith("project:")
          ? { scope: "project" as const, projectId: target.slice("project:".length) }
          : { scope: "global" as const };
        batch = pendingMutation.ids.length === 1
          ? await moveMemoryScope(token, pendingMutation.ids[0]!, parsedTarget)
          : await moveMemoriesScope(token, pendingMutation.ids, parsedTarget);
      }
      setMutationReport(mutationSummary(batch));
      const reconciled = new Set(
        batch.results
          .filter((result) => result.outcome !== "failed")
          .map((result) => result.memoryId),
      );
      setSelectedIds((current) => new Set([...current].filter((id) => !reconciled.has(id))));
      if (batch.results.every((result) => result.outcome !== "failed")) {
        setSelectionMode(false);
        setSelectedIds(new Set());
      }
      if (pendingMutation.operation === "delete") {
        setRecords((current) => current.filter((record) => !reconciled.has(record.memoryId)));
        setSearchHits((current) => current.filter((hit) => !reconciled.has(hit.record.memoryId)));
      }
      if (selectedMemoryId && reconciled.has(selectedMemoryId) && pendingMutation.operation === "delete") {
        setSelectedMemoryId(null);
        onSelectMemory(null);
      }
      setPendingMutation(null);
      setRefreshKey((value) => value + 1);
      setDetailRefreshKey((value) => value + 1);
    } catch (caught) {
      setMutationReport(handleError(caught, "Could not update memories."));
      setPendingMutation(null);
    } finally {
      setMutationRunning(false);
    }
  };

  const recordsForConfirmation = pendingMutation?.ids.map((memoryId) => ({
    memoryId,
    preview: displayedRecords.find((record) => record.memoryId === memoryId)?.preview
      ?? (detail?.memoryId === memoryId ? detail.preview : memoryId),
  })) ?? [];

  const loadMore = async (): Promise<void> => {
    if (!nextCursor || loadingMore || searchQuery) return;
    setLoadingMore(true);
    setListError(null);
    try {
      const page = await loadMemoryRecords(token, {
        ...apiFilters(filters),
        cursor: nextCursor,
        limit: 50,
        order: browseOrder,
      });
      setRecords((current) => [...current, ...page.records]);
      setNextCursor(page.nextCursor);
    } catch (caught) {
      setListError(handleError(caught, "Could not load more memories."));
    } finally {
      setLoadingMore(false);
    }
  };

  const applyFilters = (event: FormEvent): void => {
    event.preventDefault();
    resetSelectionAndResults();
    setFilters({ ...filterDraft, tag: filterDraft.tag.trim() });
  };

  const submitSearch = (event: FormEvent): void => {
    event.preventDefault();
    resetSelectionAndResults();
    setSearchQuery(queryDraft.trim());
  };

  const clearFilters = (): void => {
    resetSelectionAndResults();
    setFilterDraft(EMPTY_FILTERS);
    setFilters(EMPTY_FILTERS);
  };

  const handleRecordKey = (
    event: ReactKeyboardEvent<HTMLButtonElement>,
    index: number,
  ): void => {
    const nextIndex = event.key === "ArrowDown"
      ? Math.min(index + 1, displayedRecords.length - 1)
      : event.key === "ArrowUp"
        ? Math.max(index - 1, 0)
        : event.key === "Home"
          ? 0
          : event.key === "End"
            ? displayedRecords.length - 1
            : -1;
    if (nextIndex < 0 || nextIndex === index) return;
    event.preventDefault();
    const nextRecord = displayedRecords[nextIndex];
    if (nextRecord) {
      if (!selectionMode) selectMemory(nextRecord.memoryId);
      recordRefs.current[nextIndex]?.focus();
    }
  };

  const resultCount = searchQuery ? searchHits.length : records.length;

  return (
    <section className="memory-workspace" aria-label="Memory workspace">
      <div className="memory-browser-pane">
        <header className="memory-header">
          <div>
            <p className="breadcrumbs">Kai Workshop / Memory</p>
            <h1>Memory</h1>
          </div>
          <div className="memory-header-actions">
            <button type="button" onClick={() => setEditorDetail("create")}>Add fact…</button>
            <button className="quiet-button memory-mobile-back" type="button" onClick={onClose}>
              Back to conversation
            </button>
            <button className="quiet-button" type="button" onClick={onForget}>
              Forget session
            </button>
          </div>
        </header>

        <div className="memory-browser-scroll">
          <section className="memory-stats" aria-label="Memory statistics">
            {stats ? (
              <>
                <div><strong>{stats.total}</strong><span>Total memories</span></div>
                <div><strong>{stats.facts}</strong><span>Facts</span></div>
                <div><strong>{stats.episodes}</strong><span>Episodes</span></div>
              </>
            ) : statsError ? (
              <button type="button" className="memory-retry" onClick={() => setRefreshKey((value) => value + 1)}>
                Statistics unavailable · Retry
              </button>
            ) : (
              <p role="status">Loading corpus summary…</p>
            )}
          </section>

          <form className="memory-search" onSubmit={submitSearch} role="search">
            <label htmlFor="memory-query">Search memories</label>
            <div>
              <input
                id="memory-query"
                type="search"
                maxLength={2000}
                value={queryDraft}
                onChange={(event) => setQueryDraft(event.target.value)}
                placeholder="Describe what you remember…"
              />
              <button type="submit">Search</button>
              {searchQuery && (
                <button
                  type="button"
                  className="quiet-button"
                  onClick={() => {
                    resetSelectionAndResults();
                    setQueryDraft("");
                    setSearchQuery("");
                  }}
                >
                  Clear search
                </button>
              )}
            </div>
          </form>

          <form className="memory-filters" onSubmit={applyFilters}>
            <label>
              Kind
              <select
                value={filterDraft.kind}
                onChange={(event) => setFilterDraft((current) => ({
                  ...current,
                  kind: event.target.value as ExplorerFilters["kind"],
                }))}
              >
                <option value="">Facts and episodes</option>
                <option value="fact">Facts</option>
                <option value="episode">Episodes</option>
              </select>
            </label>
            <label>
              Scope
              <select
                value={filterDraft.scope}
                onChange={(event) => setFilterDraft((current) => ({
                  ...current,
                  scope: event.target.value as ExplorerFilters["scope"],
                }))}
              >
                <option value="">Every scope</option>
                <option value="global">Global</option>
                <option value="project">Project</option>
                <option value="task">Task</option>
              </select>
            </label>
            <label>
              Project
              <select
                value={filterDraft.projectId}
                onChange={(event) => setFilterDraft((current) => ({
                  ...current,
                  projectId: event.target.value,
                  scope: event.target.value ? "project" : current.scope,
                }))}
              >
                <option value="">Every project</option>
                {projectIds.map((projectId) => (
                  <option value={projectId} key={projectId}>{projectId}</option>
                ))}
              </select>
            </label>
            <label>
              Tag
              <input
                value={filterDraft.tag}
                maxLength={128}
                onChange={(event) => setFilterDraft((current) => ({
                  ...current,
                  tag: event.target.value,
                }))}
                placeholder="e.g. preference"
              />
            </label>
            <div className="memory-filter-actions">
              <button type="submit">Apply filters</button>
              <button type="button" className="quiet-button" onClick={clearFilters}>Clear filters</button>
            </div>
          </form>

          <div className="memory-results-heading">
            <div>
              <p className="overline">{searchQuery ? "Semantic matches" : "Memory corpus"}</p>
              <h2>{loading ? "Loading…" : `${resultCount} ${searchQuery ? "matches" : "loaded"}`}</h2>
            </div>
            <div className="memory-results-controls">
              <label>
                Sort
                {searchQuery ? (
                  <select value={searchOrder} onChange={(event) => setSearchOrder(event.target.value as typeof searchOrder)}>
                    <option value="relevance">Relevance</option>
                    <option value="newest">Newest first</option>
                    <option value="oldest">Oldest first</option>
                  </select>
                ) : (
                  <select value={browseOrder} onChange={(event) => setBrowseOrder(event.target.value as typeof browseOrder)}>
                    <option value="newest">Newest first</option>
                    <option value="oldest">Oldest first</option>
                  </select>
                )}
              </label>
              <button
                type="button"
                className="quiet-button memory-selection-toggle"
                disabled={loading || displayedRecords.length === 0 || mutationRunning}
                onClick={() => {
                  if (selectionMode) cancelSelection();
                  else {
                    setSelectedIds(new Set());
                    setSelectionMode(true);
                  }
                }}
              >
                {selectionMode ? "Cancel selection" : "Select memories"}
              </button>
            </div>
          </div>

          {mutationReport && <p className="memory-mutation-report" role="status">{mutationReport}</p>}

          {selectionMode && (
            <section className="memory-bulk-actions" aria-label="Selected memory actions">
              <strong>{selectedIds.size} selected</strong>
              <select
                aria-label="Move selected memories to"
                value={bulkTarget}
                onChange={(event) => setBulkTarget(event.target.value)}
              >
                <option value="global">Global scope</option>
                {allowedProjects.map((project) => (
                  <option key={project.projectId} value={`project:${project.projectId}`}>
                    {project.displayName}
                  </option>
                ))}
              </select>
              <button
                type="button"
                disabled={selectedIds.size === 0 || mutationRunning}
                onClick={() => setPendingMutation({
                  ids: [...selectedIds], operation: "move_scope", target: bulkTarget,
                })}
              >
                Move selected…
              </button>
              <button
                className="danger"
                type="button"
                disabled={selectedIds.size === 0 || mutationRunning}
                onClick={() => setPendingMutation({ ids: [...selectedIds], operation: "delete" })}
              >
                Forget selected…
              </button>
            </section>
          )}

          {listError ? (
            <div className="memory-list-state error" role="alert">
              <p>{listError}</p>
              <button type="button" onClick={() => setRefreshKey((value) => value + 1)}>Retry</button>
            </div>
          ) : loading ? (
            <p className="memory-list-state" role="status">Loading memories…</p>
          ) : displayedRecords.length === 0 ? (
            <div className="memory-list-state">
              <h3>No memories found</h3>
              <p>Clear search or filters to broaden this view.</p>
            </div>
          ) : (
            <div className="memory-record-list" role="listbox" aria-label="Memory results">
              {displayedRecords.map((record, index) => (
                <div className={`memory-record-row ${selectionMode ? "selecting" : ""}`} key={record.memoryId}>
                  {selectionMode && (
                    <input
                      type="checkbox"
                      aria-label={`Select ${record.preview || record.memoryId}`}
                      checked={selectedIds.has(record.memoryId)}
                      onChange={() => toggleSelected(record.memoryId)}
                    />
                  )}
                  <button
                  key={record.memoryId}
                  ref={(element) => { recordRefs.current[index] = element; }}
                  type="button"
                  role="option"
                  aria-selected={selectionMode
                    ? selectedIds.has(record.memoryId)
                    : selectedMemoryId === record.memoryId}
                  className={`memory-record ${(
                    selectionMode ? selectedIds.has(record.memoryId) : selectedMemoryId === record.memoryId
                  ) ? "selected" : ""}`}
                  onClick={() => {
                    if (selectionMode) toggleSelected(record.memoryId);
                    else selectMemory(record.memoryId);
                  }}
                  onKeyDown={(event) => handleRecordKey(event, index)}
                >
                  <span className={`memory-kind ${record.kind}`}>{record.kind}</span>
                  <span className="memory-record-copy">
                    <strong>{record.preview || "Untitled memory"}</strong>
                    <small>{scopeDescription(record.scope)}</small>
                    {record.tags.length > 0 && (
                      <span className="memory-tags compact">
                        {record.tags.slice(0, 3).map((tag) => <span key={tag}>{tag}</span>)}
                      </span>
                    )}
                  </span>
                  <time>{formatDate(record.updatedAt)}</time>
                  </button>
                </div>
              ))}
            </div>
          )}

          {nextCursor && !searchQuery && !listError && (
            <button
              className="memory-load-more"
              type="button"
              disabled={loadingMore}
              onClick={() => void loadMore()}
            >
              {loadingMore ? "Loading…" : "Load more memories"}
            </button>
          )}
        </div>
      </div>

      <aside
        className="memory-detail-pane"
        aria-label={selectionMode ? "Memory selection" : "Memory detail"}
      >
        {selectionMode ? (
          <div className="memory-selection-summary">
            <span aria-hidden="true">◇</span>
            <p className="overline">Batch selection</p>
            <h2>
              {selectedIds.size === 0
                ? "Choose memories"
                : `${selectedIds.size} ${selectedIds.size === 1 ? "memory" : "memories"} selected`}
            </h2>
            <p>
              Select the memories you want to move or forget. Cancel selection to return to the current memory details.
            </p>
          </div>
        ) : (
          <MemoryDetailPane
            allowedProjects={allowedProjects}
            busy={mutationRunning}
            detail={detail}
            error={detailError}
            loading={detailLoading}
            onDelete={(memoryId) => setPendingMutation({ ids: [memoryId], operation: "delete" })}
            onEdit={(record) => setEditorDetail(record)}
            onMove={(memoryId, target) => setPendingMutation({
              ids: [memoryId], operation: "move_scope", target,
            })}
            source={source}
            sourceError={sourceError}
          />
        )}
      </aside>

      {pendingMutation && (
        <div className="memory-confirm-backdrop">
          <section className="memory-confirm" role="dialog" aria-modal="true" aria-labelledby="memory-confirm-title">
            <p className="overline">Confirm memory change</p>
            <h2 id="memory-confirm-title">
              {pendingMutation.operation === "delete"
                ? `Forget ${pendingMutation.ids.length} ${pendingMutation.ids.length === 1 ? "memory" : "memories"}?`
                : `Move ${pendingMutation.ids.length} ${pendingMutation.ids.length === 1 ? "memory" : "memories"}?`}
            </h2>
            <p>
              {pendingMutation.operation === "delete"
                ? "These records will be permanently removed from recall and cannot be restored here."
                : `Their recall scope will change to ${pendingMutation.target?.startsWith("project:")
                  ? `project ${pendingMutation.target.slice("project:".length)}`
                  : "global"}.`}
            </p>
            <ul>
              {recordsForConfirmation.map((record) => (
                <li key={record.memoryId}>{record.preview}</li>
              ))}
            </ul>
            <div>
              <button type="button" className="quiet-button" disabled={mutationRunning} onClick={() => setPendingMutation(null)}>
                Cancel
              </button>
              <button
                type="button"
                className={pendingMutation.operation === "delete" ? "danger" : ""}
                disabled={mutationRunning}
                onClick={() => void runMutation()}
              >
                {mutationRunning ? "Applying…" : pendingMutation.operation === "delete" ? "Forget permanently" : "Confirm move"}
              </button>
            </div>
          </section>
        </div>
      )}
      {editorDetail && (
        <MemoryEditorDialog
          allowedProjects={allowedProjects}
          detail={editorDetail === "create" ? null : editorDetail}
          onAuthenticationFailure={onAuthenticationFailure}
          onClose={() => setEditorDetail(null)}
          onReload={() => {
            setEditorDetail(null);
            setDetailRefreshKey((value) => value + 1);
            setRefreshKey((value) => value + 1);
          }}
          onSaved={(record, message) => {
            setEditorDetail(null);
            setDetail(record);
            setSelectedMemoryId(record.memoryId);
            onSelectMemory(record.memoryId);
            setMutationReport(message);
            setDetailRefreshKey((value) => value + 1);
            setRefreshKey((value) => value + 1);
          }}
          token={token}
        />
      )}
    </section>
  );
}

export function MemoryExplorer(
  props: React.ComponentProps<typeof MemoryExplorerContent>,
): React.JSX.Element {
  return (
    <ConfirmationProvider>
      <MemoryExplorerContent {...props} />
    </ConfirmationProvider>
  );
}
