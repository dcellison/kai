import { useEffect, useMemo, useState } from "react";

import {
  loadPreferenceDocument,
  loadPrincipalPolicyDocument,
  loadRunContextManifests,
  PrincipalPolicyRevisionConflictError,
  PreferenceRevisionConflictError,
  savePreferenceDocument,
  savePrincipalPolicyDocument,
} from "./api";
import type {
  WorkshopContextInvalidation,
  WorkshopContextManifest,
  WorkshopContextSource,
  WorkshopSession,
} from "./types";

function PencilIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.75"
      aria-hidden="true"
    >
      <path d="M4 20h4L19 9l-4-4L4 16v4Z" />
      <path d="m13.5 6.5 4 4" />
    </svg>
  );
}

function CheckIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.75"
      aria-hidden="true"
    >
      <path d="m5 12 4 4L19 6" />
    </svg>
  );
}

function CloseIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.75"
      aria-hidden="true"
    >
      <path d="m6 6 12 12M18 6 6 18" />
    </svg>
  );
}

function readable(value: string): string {
  return value.replaceAll("_", " ");
}

function abbreviatedRevision(value: string | null): string {
  if (!value) return "none";
  return value.length > 24 ? `${value.slice(0, 12)}…${value.slice(-8)}` : value;
}

function SourceEditor({
  session,
  source,
  onCancel,
  onSaved,
}: {
  session: WorkshopSession;
  source: WorkshopContextSource;
  onCancel: () => void;
  onSaved: (invalidation: WorkshopContextInvalidation | undefined) => Promise<void>;
}): React.JSX.Element {
  const [content, setContent] = useState("");
  const [revision, setRevision] = useState("");
  const [maximum, setMaximum] = useState(0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const policy = source.inspection.editTarget === "principal_policy";

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    const request = policy
      ? loadPrincipalPolicyDocument(session)
      : loadPreferenceDocument(session);
    void request.then(
      (document) => {
        if (!active) return;
        setContent(document.content);
        setRevision(document.revision);
        setMaximum(document.maxBytes);
      },
      (caught: unknown) => {
        if (active) setError(caught instanceof Error ? caught.message : "Could not load this source.");
      },
    ).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [policy, session]);

  const save = async (): Promise<void> => {
    setSaving(true);
    setError(null);
    try {
      const document = policy
        ? await savePrincipalPolicyDocument(session, content, revision)
        : await savePreferenceDocument(session, content, revision);
      setRevision(document.revision);
      await onSaved(document.contextInvalidation);
      onCancel();
    } catch (caught) {
      if (
        caught instanceof PrincipalPolicyRevisionConflictError ||
        caught instanceof PreferenceRevisionConflictError
      ) {
        setError("This source changed elsewhere. Close the editor, reload, and review the current revision.");
      } else {
        setError(caught instanceof Error ? caught.message : "Could not save this source.");
      }
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <p role="status">Loading source…</p>;
  return (
    <div className="context-source-editor">
      <textarea
        autoFocus
        aria-label={`Edit ${source.inspection.title}`}
        value={content}
        onChange={(event) => setContent(event.target.value)}
        rows={12}
        maxLength={maximum || undefined}
      />
      <div className="context-source-editor-actions">
        <span>{new TextEncoder().encode(content).length.toLocaleString()} / {maximum.toLocaleString()} bytes</span>
        <button
          className="panel-icon-button"
          type="button"
          aria-label={`Cancel editing ${source.inspection.title}`}
          title="Cancel"
          onClick={onCancel}
          disabled={saving}
        >
          <CloseIcon />
        </button>
        <button
          className="panel-icon-button"
          type="button"
          aria-label={`Save ${source.inspection.title}`}
          title="Save"
          onClick={() => void save()}
          disabled={saving || new TextEncoder().encode(content).length > maximum}
        >
          <CheckIcon />
        </button>
      </div>
      {error && <p className="settings-error" role="alert">{error}</p>}
    </div>
  );
}

function SourceCard({
  session,
  source,
  editing,
  onEdit,
  onNavigate,
  onReload,
}: {
  session: WorkshopSession;
  source: WorkshopContextSource;
  editing: boolean;
  onEdit: () => void;
  onNavigate: (source: WorkshopContextSource) => void;
  onReload: (invalidation: WorkshopContextInvalidation | undefined) => Promise<void>;
}): React.JSX.Element {
  const inspection = source.inspection;
  const inlineEditor = inspection.editTarget === "principal_policy" ||
    inspection.editTarget === "personal_preferences";
  return (
    <article className={`context-source-card ${inspection.previewState} ${inspection.freshness}`}>
      <header>
        <div>
          <h4>{inspection.title}</h4>
          <p>{inspection.description}</p>
        </div>
        {inspection.editable && (!inlineEditor || !editing) && (
          <button
            className="panel-icon-button"
            type="button"
            aria-label={`Edit ${inspection.title}`}
            title={`Edit ${inspection.title}`}
            onClick={inlineEditor ? onEdit : () => onNavigate(source)}
          >
            <PencilIcon />
          </button>
        )}
      </header>
      {editing && inlineEditor ? (
        <SourceEditor session={session} source={source} onCancel={onEdit} onSaved={onReload} />
      ) : (
        <>
          <dl className="context-source-facts">
            <div><dt>Owner</dt><dd>{readable(source.ownerKind)}{source.ownerId ? ` · ${abbreviatedRevision(source.ownerId)}` : ""}</dd></div>
            <div><dt>Source</dt><dd>{inspection.sourceReference}</dd></div>
            <div><dt>Scope</dt><dd>{readable(source.scope)}</dd></div>
            <div><dt>Authority</dt><dd>{readable(source.authorityClass)}</dd></div>
            <div><dt>Trust</dt><dd>{readable(source.trustClass)}</dd></div>
            <div><dt>Delivery</dt><dd>{readable(source.deliveryRole)}</dd></div>
            <div><dt>Refresh</dt><dd>{readable(source.refreshClass)}</dd></div>
            <div><dt>State</dt><dd>{readable(source.state)} · {readable(source.reason)}</dd></div>
            <div><dt>Run revision</dt><dd title={source.revision ?? undefined}>{abbreviatedRevision(source.revision)}</dd></div>
            <div><dt>Current</dt><dd title={inspection.currentRevision ?? undefined}>{readable(inspection.freshness)}{inspection.currentRevision ? ` · ${abbreviatedRevision(inspection.currentRevision)}` : ""}</dd></div>
            <div><dt>Rendered</dt><dd>{source.renderedBytes.toLocaleString()}{source.budgetBytes ? ` / ${source.budgetBytes.toLocaleString()}` : ""} bytes</dd></div>
          </dl>
          {source.historyBoundary !== null && <p className="context-source-note">Conversation boundary: event {source.historyBoundary}</p>}
          {source.nativeInstructionSources.length > 0 && (
            <ul className="context-native-sources">
              {source.nativeInstructionSources.map((native) => (
                <li key={`${native.filename}:${native.pathSha256}`}>
                  {native.filename} · {readable(native.scope)} · path {abbreviatedRevision(native.pathSha256)}
                </li>
              ))}
            </ul>
          )}
          {inspection.preview !== null ? (
            <details className="context-source-preview">
              <summary>{inspection.previewState === "exact" ? "Rendered preview" : "Redacted preview"}</summary>
              <pre>{inspection.preview}</pre>
              {inspection.previewReason && <p>{inspection.previewReason}</p>}
            </details>
          ) : (
            <p className="context-source-note">{inspection.previewReason ?? "No rendered preview is available."}</p>
          )}
          {inspection.changeEffect && (
            <p className="context-source-effect">
              Changes apply on {inspection.changeEffect === "next_turn" ? "the next turn" : "provider-session refresh"}.
            </p>
          )}
        </>
      )}
    </article>
  );
}

export function ContextStackInspector({
  session,
  runId,
  onNavigate,
}: {
  session: WorkshopSession;
  runId: string | null;
  onNavigate: (source: WorkshopContextSource) => void;
}): React.JSX.Element {
  const [manifest, setManifest] = useState<WorkshopContextManifest | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingKind, setEditingKind] = useState<string | null>(null);
  const [updateNotice, setUpdateNotice] = useState<string | null>(null);

  const load = async (): Promise<void> => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    try {
      const snapshot = await loadRunContextManifests(session, runId);
      setManifest(snapshot.manifests.at(-1) ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load this run's context stack.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setManifest(null);
    setEditingKind(null);
    setUpdateNotice(null);
    if (runId) void load();
  }, [runId, session.channelId, session.token]);

  const changedCount = useMemo(
    () => manifest?.sources.filter((source) => source.inspection.freshness === "changed").length ?? 0,
    [manifest],
  );

  const reloadAfterSave = async (
    invalidation: WorkshopContextInvalidation | undefined,
  ): Promise<void> => {
    if (!invalidation) {
      setUpdateNotice("Saved. The new revision applies according to this source's refresh policy.");
    } else if (invalidation.state === "failed") {
      setUpdateNotice("Saved, but targeted context refresh failed. The affected lane will retry on its next turn.");
    } else if (invalidation.state === "pending") {
      setUpdateNotice(
        `Saved. ${invalidation.applied} lane${invalidation.applied === 1 ? "" : "s"} refreshed; ` +
        `${invalidation.pending} will refresh after current work finishes.`,
      );
    } else if (invalidation.state === "unchanged") {
      setUpdateNotice("No content changed; no context lane needed a refresh.");
    } else {
      setUpdateNotice(
        `Saved. Context refreshed in ${invalidation.applied} lane${invalidation.applied === 1 ? "" : "s"}.`,
      );
    }
    await load();
  };

  if (!runId) return <p className="trace-empty">No run context is available yet.</p>;
  if (loading && !manifest) return <p role="status">Loading context stack…</p>;
  if (error) return <p className="settings-error" role="alert">{error}</p>;
  if (!manifest) return <p className="trace-empty">No context manifest was recorded for this run.</p>;
  return (
    <div className="context-stack-inspector">
      <div className="context-stack-summary">
        <p><strong>{manifest.backend}{manifest.provider ? ` · ${manifest.provider}` : ""}</strong> · {manifest.model}</p>
        <p>{manifest.sources.length} canonical sources · {changedCount} changed since this run</p>
        <code title={manifest.manifestSha256}>Manifest {abbreviatedRevision(manifest.manifestSha256)}</code>
      </div>
      {updateNotice && <p className="context-source-update" role="status">{updateNotice}</p>}
      <ol className="context-source-list">
        {manifest.sources.map((source) => (
          <li key={source.kind}>
            <SourceCard
              session={session}
              source={source}
              editing={editingKind === source.kind}
              onEdit={() => setEditingKind((current) => current === source.kind ? null : source.kind)}
              onNavigate={onNavigate}
              onReload={reloadAfterSave}
            />
          </li>
        ))}
      </ol>
    </div>
  );
}
