import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  AuthenticationError,
  ChannelAccessError,
  loadReviewJobs,
  submitReviewJob,
} from "./api";
import type { WorkshopReviewJobs, WorkshopSession } from "./types";

function operationId(): string {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return `workshop-review:${Array.from(bytes, (item) => item.toString(16).padStart(2, "0")).join("")}`;
}

function parsePullRequestReference(value: string): { repository: string | null; number: number } | null {
  const trimmed = value.trim();
  const url = /^https:\/\/github\.com\/([^/]+)\/([^/]+)\/pull\/(\d+)(?:[/?#].*)?$/i.exec(trimmed);
  if (url) return { repository: `${url[1]}/${url[2]}`.toLocaleLowerCase(), number: Number(url[3]) };
  const explicit = /^([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)#(\d+)$/.exec(trimmed);
  if (explicit) return { repository: explicit[1].toLocaleLowerCase(), number: Number(explicit[2]) };
  const number = /^#?(\d+)$/.exec(trimmed);
  return number ? { repository: null, number: Number(number[1]) } : null;
}

export function ReviewDialog({
  onAuthenticationFailure,
  onChannelAccessFailure,
  onClose,
  onSubmitted,
  session,
}: {
  onAuthenticationFailure: (message: string) => void;
  onChannelAccessFailure: (message: string) => void;
  onClose: () => void;
  onSubmitted: () => void;
  session: WorkshopSession;
}): React.JSX.Element {
  const [snapshot, setSnapshot] = useState<WorkshopReviewJobs | null>(null);
  const [reference, setReference] = useState("");
  const [repository, setRepository] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const parsed = useMemo(() => parsePullRequestReference(reference), [reference]);

  useEffect(() => {
    let cancelled = false;
    void loadReviewJobs(session).then((result) => {
      if (cancelled) return;
      setSnapshot(result);
      setRepository(result.submission.inferredRepository ?? (result.submission.repositories.length === 1
        ? result.submission.repositories[0]
        : ""));
    }).catch((caught) => {
      if (cancelled) return;
      if (caught instanceof AuthenticationError) onAuthenticationFailure(caught.message);
      else if (caught instanceof ChannelAccessError) onChannelAccessFailure(caught.message);
      else setError(caught instanceof Error ? caught.message : "Could not load review options.");
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [onAuthenticationFailure, onChannelAccessFailure, session]);

  const resolvedRepository = parsed?.repository ?? (repository || null);
  const valid = parsed !== null && resolvedRepository !== null &&
    snapshot?.submission.repositories.includes(resolvedRepository) === true;

  const submit = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    if (!parsed || !resolvedRepository || !valid) return;
    setBusy(true);
    setError(null);
    try {
      await submitReviewJob(session, resolvedRepository, parsed.number, operationId());
      onSubmitted();
      onClose();
    } catch (caught) {
      if (caught instanceof AuthenticationError) onAuthenticationFailure(caught.message);
      else if (caught instanceof ChannelAccessError) onChannelAccessFailure(caught.message);
      else setError(caught instanceof Error ? caught.message : "Could not start the review.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="action-palette-backdrop" onMouseDown={onClose}>
      <section
        aria-labelledby="review-dialog-title"
        aria-modal="true"
        className="review-dialog"
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
      >
        <header>
          <div><p className="overline">GitHub</p><h2 id="review-dialog-title">Review pull request</h2></div>
          <button className="panel-icon-button" type="button" aria-label="Close review" title="Close" onClick={onClose}>
            <span aria-hidden="true">×</span>
          </button>
        </header>
        {loading ? <p role="status">Loading authorized repositories…</p> : snapshot ? (
          <form onSubmit={(event) => void submit(event)}>
            <label htmlFor="review-reference">Pull request</label>
            <input
              autoFocus
              id="review-reference"
              placeholder="GitHub URL, owner/repository#123, or 123"
              value={reference}
              onChange={(event) => setReference(event.target.value)}
            />
            <label htmlFor="review-repository">Authorized repository</label>
            <select
              id="review-repository"
              disabled={parsed !== null && parsed.repository !== null}
              value={resolvedRepository ?? ""}
              onChange={(event) => setRepository(event.target.value)}
            >
              <option value="">Choose a repository</option>
              {snapshot.submission.repositories.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
            <div className="review-context">
              <strong>Active workspace</strong>
              <span>{snapshot.submission.activeWorkspace}</span>
              <small>{snapshot.submission.activeWorkspaceRepository
                ? `GitHub repository: ${snapshot.submission.activeWorkspaceRepository}`
                : "No GitHub repository was detected for this workspace."}</small>
            </div>
            {parsed?.repository && !snapshot.submission.repositories.includes(parsed.repository) && (
              <p className="form-error" role="alert">That repository is not authorized for review.</p>
            )}
            {error && <p className="form-error" role="alert">{error}</p>}
            <div className="form-actions">
              <button className="primary-button" type="submit" disabled={busy || !valid}>
                {busy ? "Starting…" : "Start review"}
              </button>
            </div>
          </form>
        ) : <p className="form-error" role="alert">{error ?? "Review options are unavailable."}</p>}
      </section>
    </div>
  );
}
