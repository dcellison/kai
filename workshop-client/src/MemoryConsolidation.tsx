/*
 * Consolidate several legacy facts into one canonical fact during triage.
 *
 * Legacy memory often holds the same fact in several wordings. Instead of
 * approving one and rejecting each duplicate separately, the owner selects
 * the facts that say the same thing, picks or writes the final wording and
 * scope, and saves the whole selection as one consolidation. Saving marks
 * every selected fact as part of it; applying the plan records one current
 * fact that cites all of them, and the rest are never recalled again.
 *
 * Facts are shown by their full text, never by internal ids. The selection
 * starts from the facts a model recommendation linked (or the saved
 * consolidation's facts), and any other pending fact can be added by search.
 */
import { useMemo, useState } from "react";

import { AuthenticationError, cancelMemoryConsolidation, stageMemoryConsolidation } from "./api";
import { useConfirmation } from "./ConfirmationDialog";
import type { WorkshopMemoryConsolidation, WorkshopMemoryTriageGroup } from "./types";

// Search results stay short enough to scan; refine the query for more.
const MAX_SEARCH_RESULTS = 8;

function factText(group: WorkshopMemoryTriageGroup): string {
  return group.evidence[0]?.text ?? "Unavailable fact";
}

function scopeText(group: WorkshopMemoryTriageGroup): string {
  const first = group.evidence[0];
  if (!first || first.migrationGaps.includes("scope")) return "Scope not recorded";
  return first.scope === "project" && first.projectId ? `Project ${first.projectId}` : "Global";
}

// The scope to start from: the one every selected fact agrees on, or none
// when any fact lacks a recorded scope or they disagree.
function sharedScope(groups: WorkshopMemoryTriageGroup[]): { kind: "" | "global" | "project"; key: string } {
  const scopes = new Set(
    groups.map((group) => {
      const first = group.evidence[0];
      if (!first || first.migrationGaps.includes("scope")) return "";
      return first.scope === "project" ? `project:${first.projectId ?? ""}` : "global";
    }),
  );
  if (scopes.size !== 1) return { kind: "", key: "" };
  const [only] = [...scopes];
  if (only === "global") return { kind: "global", key: "" };
  if (only.startsWith("project:") && only.length > "project:".length) return { kind: "project", key: only.slice(8) };
  return { kind: "", key: "" };
}

// A fact can join this consolidation when it is a plain fact group that is
// not already part of a different consolidation.
function eligible(group: WorkshopMemoryTriageGroup, consolidationId: string | null): boolean {
  return (
    group.evidence.length > 0 &&
    group.evidence.every((item) => item.kind === "fact") &&
    (group.consolidationId === null || group.consolidationId === consolidationId)
  );
}

export function ConsolidationEditor({
  allowedProjects,
  anchor,
  availableGroups,
  consolidation,
  onAuthenticationFailure,
  onClose,
  onSaved,
  planId,
  token,
}: {
  allowedProjects: Array<{ displayName: string; projectId: string }>;
  anchor: WorkshopMemoryTriageGroup;
  availableGroups: WorkshopMemoryTriageGroup[];
  consolidation: WorkshopMemoryConsolidation | null;
  onAuthenticationFailure: (message: string) => void;
  onClose: () => void;
  onSaved: (message: string) => void;
  planId: string;
  token: string;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const byId = useMemo(() => new Map(availableGroups.map((group) => [group.groupId, group])), [availableGroups]);
  const ownId = consolidation?.consolidationId ?? null;
  const [selectedIds, setSelectedIds] = useState<string[]>(() => {
    const initial = consolidation?.groupIds ?? [anchor.groupId, ...anchor.relatedGroupIds];
    return [...new Set(initial)].filter((id) => {
      const group = byId.get(id);
      return group !== undefined && eligible(group, ownId);
    });
  });
  const selected = selectedIds.map((id) => byId.get(id)).filter((group): group is WorkshopMemoryTriageGroup => !!group);
  const initialScope = consolidation
    ? { kind: consolidation.scopeKind, key: consolidation.scopeKey ?? "" }
    : sharedScope(selected);
  const [content, setContent] = useState(consolidation?.content ?? factText(anchor));
  const [sourceMemoryId, setSourceMemoryId] = useState(
    consolidation?.sourceMemoryId ?? anchor.evidence[0]?.memoryId ?? "",
  );
  const [scopeKind, setScopeKind] = useState<"" | "global" | "project">(initialScope.kind);
  const [projectId, setProjectId] = useState(initialScope.key);
  const [note, setNote] = useState(consolidation?.operatorNote ?? "");
  const [query, setQuery] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const results = query.trim()
    ? availableGroups
      .filter(
        (group) =>
          !selectedIds.includes(group.groupId) &&
          eligible(group, ownId) &&
          factText(group).toLowerCase().includes(query.trim().toLowerCase()),
      )
      .slice(0, MAX_SEARCH_RESULTS)
    : [];
  const sourceSelected = selected.some((group) => group.evidence.some((item) => item.memoryId === sourceMemoryId));
  const projectAuthorized = scopeKind !== "project" || allowedProjects.some((project) => project.projectId === projectId);
  let blocker: string | null = null;
  if (selected.length < 2) blocker = "Select at least two facts to consolidate.";
  else if (!content.trim()) blocker = "Enter the consolidated fact's wording.";
  else if (!sourceSelected) blocker = "Choose the wording of one of the selected facts to start from.";
  else if (!scopeKind) blocker = "Choose a scope for the consolidated fact.";
  else if (scopeKind === "project" && !projectAuthorized) blocker = "Choose a project you currently have access to.";
  const preview = `${selected.length} selected facts → 1 current fact; ${Math.max(selected.length - 1, 0)} duplicates excluded`;

  const failure = (caught: unknown, fallback: string): void => {
    if (caught instanceof AuthenticationError) onAuthenticationFailure(caught.message);
    // The draft stays as it is, so the owner can fix the problem and retry.
    setError(caught instanceof Error ? caught.message : fallback);
  };

  const toggle = (groupId: string): void => {
    setSelectedIds((current) =>
      current.includes(groupId) ? current.filter((id) => id !== groupId) : [...current, groupId],
    );
  };

  const useWording = (group: WorkshopMemoryTriageGroup): void => {
    setContent(factText(group));
    setSourceMemoryId(group.evidence[0]?.memoryId ?? "");
  };

  const save = async (): Promise<void> => {
    if (blocker) return;
    const accepted = await confirm(
      `${consolidation ? "Update" : "Save"} this consolidation? ${preview}. ` +
        "Applying the plan keeps the other facts only as evidence; Kai will not recall them.",
    );
    if (!accepted) return;
    setSaving(true);
    setError(null);
    try {
      await stageMemoryConsolidation(token, planId, {
        canonical: { content: content.trim(), scopeKey: scopeKind === "project" ? projectId : null, scopeKind: scopeKind as "global" | "project", sourceMemoryId },
        ...(consolidation ? { consolidationId: consolidation.consolidationId, expectedRevision: consolidation.revision } : {}),
        expectedStateVersions: Object.fromEntries(selected.map((group) => [group.groupId, group.decision.stateVersion])),
        groupIds: selected.map((group) => group.groupId),
        operatorNote: note,
      });
      onSaved(`Consolidation saved: ${preview}.`);
    } catch (caught) {
      failure(caught, "Could not save the consolidation.");
    } finally {
      setSaving(false);
    }
  };

  const cancel = async (): Promise<void> => {
    if (!consolidation) return;
    const accepted = await confirm("Cancel this consolidation? Each fact gets back the decision it had before.");
    if (!accepted) return;
    setSaving(true);
    setError(null);
    try {
      await cancelMemoryConsolidation(token, planId, consolidation.consolidationId, consolidation.revision);
      onSaved("Consolidation cancelled; each fact has its earlier decision again.");
    } catch (caught) {
      failure(caught, "Could not cancel the consolidation.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="memory-reconciliation-detail memory-consolidation" aria-label="Consolidate facts">
      <header className="memory-detail-header">
        <div>
          <p className="overline">{consolidation ? "Saved consolidation" : "Consolidate facts"}</p>
          <h2>Several facts, one current fact</h2>
        </div>
        <button type="button" className="panel-icon-button" aria-label="Back to the fact" title="Back to the fact" onClick={onClose}>
          <span aria-hidden="true">←</span>
        </button>
      </header>

      <section className="memory-detail-section">
        <p className="memory-section-label">Selected facts</p>
        <p className="memory-review-explanation">
          Keep checked the facts that say the same thing. Use one's wording as a starting point, then edit it below.
        </p>
        <ul className="memory-consolidation-facts">
          {selected.map((group) => (
            <li key={group.groupId}>
              <label>
                <input type="checkbox" checked onChange={() => toggle(group.groupId)} />
                <span>{factText(group)}</span>
              </label>
              <small>{scopeText(group)}</small>
              <button
                type="button"
                className="quiet-button"
                aria-pressed={group.evidence[0]?.memoryId === sourceMemoryId}
                onClick={() => useWording(group)}
              >
                {group.evidence[0]?.memoryId === sourceMemoryId ? "Wording source" : "Use this wording"}
              </button>
            </li>
          ))}
        </ul>
        <label className="memory-consolidation-search">
          Add another fact
          <input type="search" value={query} placeholder="Search pending facts" onChange={(event) => setQuery(event.target.value)} />
        </label>
        {results.length > 0 && (
          <ul className="memory-consolidation-facts" aria-label="Matching facts">
            {results.map((group) => (
              <li key={group.groupId}>
                <label>
                  <input type="checkbox" checked={false} onChange={() => toggle(group.groupId)} />
                  <span>{factText(group)}</span>
                </label>
                <small>{scopeText(group)}</small>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="memory-detail-section memory-review-correction">
        <p className="memory-section-label">Consolidated fact</p>
        <label>Wording
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
              {allowedProjects.map((project) => (
                <option value={project.projectId} key={project.projectId}>{project.displayName}</option>
              ))}
            </select>
          </label>
        )}
        <label>Note (optional)
          <textarea value={note} maxLength={4096} onChange={(event) => setNote(event.target.value)} />
        </label>
      </section>

      <section className="memory-detail-section memory-review-decision">
        <p className="memory-consolidation-preview" role="status">{preview}</p>
        {blocker && <p className="memory-review-explanation">{blocker}</p>}
        {error && <p className="memory-editor-error" role="alert">{error}</p>}
        <div className="memory-review-toolbar">
          <button type="button" disabled={saving || blocker !== null} onClick={() => void save()}>
            {saving ? "Saving…" : consolidation ? "Update consolidation" : "Save consolidation"}
          </button>
          {consolidation && (
            <button type="button" className="danger" disabled={saving} onClick={() => void cancel()}>
              Cancel consolidation
            </button>
          )}
        </div>
      </section>
    </section>
  );
}
