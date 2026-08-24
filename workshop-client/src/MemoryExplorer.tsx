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
  loadMemoryDetail,
  loadMemoryRecords,
  loadMemorySource,
  loadMemoryStats,
  searchMemories,
} from "./api";
import { MarkdownMessage } from "./MarkdownMessage";
import type {
  WorkshopMemoryDetail,
  WorkshopMemoryFilters,
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

function MemoryDetailPane({
  detail,
  error,
  loading,
  source,
  sourceError,
}: {
  detail: WorkshopMemoryDetail | null;
  error: string | null;
  loading: boolean;
  source: WorkshopMemorySourceContext | null;
  sourceError: string | null;
}): React.JSX.Element {
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
                <dd><MarkdownMessage body={value} /></dd>
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

export function MemoryExplorer({
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
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(initialMemoryId);
  const [detail, setDetail] = useState<WorkshopMemoryDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(initialMemoryId !== null);
  const [source, setSource] = useState<WorkshopMemorySourceContext | null>(null);
  const [sourceError, setSourceError] = useState<string | null>(null);
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
  }, [handleError, selectedMemoryId, token]);

  const projectIds = useMemo(() => {
    if (!stats) return [];
    return Object.keys(stats.byScope)
      .filter((scope) => scope.startsWith("project:"))
      .map((scope) => scope.slice("project:".length))
      .filter(Boolean)
      .sort();
  }, [stats]);

  const selectMemory = useCallback((memoryId: string): void => {
    setSelectedMemoryId(memoryId);
    onSelectMemory(memoryId);
  }, [onSelectMemory]);

  const resetSelectionAndResults = (): void => {
    setSelectedMemoryId(null);
    setRecords([]);
    setSearchHits([]);
    setNextCursor(null);
    onSelectMemory(null);
  };

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
      selectMemory(nextRecord.memoryId);
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
          </div>

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
                <button
                  key={record.memoryId}
                  ref={(element) => { recordRefs.current[index] = element; }}
                  type="button"
                  role="option"
                  aria-selected={selectedMemoryId === record.memoryId}
                  className={`memory-record ${selectedMemoryId === record.memoryId ? "selected" : ""}`}
                  onClick={() => selectMemory(record.memoryId)}
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

      <aside className="memory-detail-pane" aria-label="Memory detail">
        <MemoryDetailPane
          detail={detail}
          error={detailError}
          loading={detailLoading}
          source={source}
          sourceError={sourceError}
        />
      </aside>
    </section>
  );
}
