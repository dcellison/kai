import { useCallback, useEffect, useState } from "react";

import {
  AuthenticationError,
  ChannelAccessError,
  downloadReviewArtifact,
  loadReviewArtifact,
  loadReviewJobs,
  mutateReviewJob,
} from "./api";
import { ActivityWorkspaceHeader } from "./ActivityWorkspaceHeader";
import { useConfirmation } from "./ConfirmationDialog";
import type {
  ConnectionState,
  WorkshopReviewJob,
  WorkshopReviewJobs,
  WorkshopSession,
} from "./types";

function PullRequestIcon(): React.JSX.Element {
  return (
    <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
      <circle cx="6" cy="5" r="2" stroke="currentColor" strokeWidth="1.8" />
      <circle cx="6" cy="19" r="2" stroke="currentColor" strokeWidth="1.8" />
      <circle cx="18" cy="5" r="2" stroke="currentColor" strokeWidth="1.8" />
      <path d="M6 7v10M18 7v3a5 5 0 0 1-5 5H9" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}

function reviewStatusLabel(status: WorkshopReviewJob["status"]): string {
  return {
    pending: "Queued",
    executing: "Running",
    succeeded: "Completed",
    failed: "Failed",
    cancelled: "Cancelled",
    timed_out: "Timed out",
  }[status];
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function operationId(): string {
  return globalThis.crypto?.randomUUID
    ? `workshop-review:${globalThis.crypto.randomUUID()}`
    : `workshop-review:${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function ReviewsWorkspace({
  connection,
  onAuthenticationFailure,
  onChannelAccessFailure,
  onReviewPullRequest,
  revision,
  session,
  workshopName,
}: {
  connection: ConnectionState;
  onAuthenticationFailure: (message: string) => void;
  onChannelAccessFailure: (message: string) => void;
  onReviewPullRequest: () => void;
  revision: number;
  session: WorkshopSession;
  workshopName: string;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const [reviews, setReviews] = useState<WorkshopReviewJobs | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFailure = useCallback((caught: unknown, fallback: string): void => {
    if (caught instanceof AuthenticationError) {
      onAuthenticationFailure(caught.message);
      return;
    }
    if (caught instanceof ChannelAccessError) {
      onChannelAccessFailure(caught.message);
      return;
    }
    setError(caught instanceof Error ? caught.message : fallback);
  }, [onAuthenticationFailure, onChannelAccessFailure]);

  const refresh = useCallback(async (showLoading = true): Promise<void> => {
    if (showLoading) setLoading(true);
    setError(null);
    try {
      setReviews(await loadReviewJobs(session));
    } catch (caught) {
      handleFailure(caught, "Could not load pull-request reviews.");
    } finally {
      if (showLoading) setLoading(false);
    }
  }, [handleFailure, session]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (revision > 0) void refresh(false);
  }, [refresh, revision]);

  useEffect(() => {
    if (!reviews?.jobs.some((job) => job.status === "pending" || job.status === "executing")) {
      return;
    }
    const timer = window.setTimeout(() => void refresh(false), 2_000);
    return () => window.clearTimeout(timer);
  }, [refresh, reviews]);

  const changeReview = async (
    job: WorkshopReviewJob,
    action: "cancel" | "retry",
  ): Promise<void> => {
    if (
      action === "cancel" &&
      !await confirm(`Cancel the review of ${job.repository}#${job.pullRequestNumber}?`)
    ) {
      return;
    }
    setBusyId(job.reviewJobId);
    setError(null);
    try {
      await mutateReviewJob(session, job.reviewJobId, action, operationId());
      await refresh(false);
    } catch (caught) {
      handleFailure(caught, `Could not ${action} the pull-request review.`);
    } finally {
      setBusyId(null);
    }
  };

  const openArtifact = async (job: WorkshopReviewJob): Promise<void> => {
    setBusyId(job.reviewJobId);
    setError(null);
    try {
      const url = URL.createObjectURL(await loadReviewArtifact(session, job.reviewJobId));
      const opened = window.open(url, "_blank", "noopener,noreferrer");
      if (!opened) throw new Error("The browser blocked the review artifact window.");
      window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (caught) {
      handleFailure(caught, "Could not open the review artifact.");
    } finally {
      setBusyId(null);
    }
  };

  const activeCount = reviews?.jobs.filter(
    (job) => job.status === "pending" || job.status === "executing",
  ).length ?? 0;

  return (
    <section className="reviews-workspace" aria-label="Reviews">
      <ActivityWorkspaceHeader
        actions={(
          <button className="quiet-button" type="button" onClick={onReviewPullRequest}>
            Review pull request
          </button>
        )}
        connection={connection}
        symbol={<PullRequestIcon />}
        title="Reviews"
        workshopName={workshopName}
      />

      <div className="reviews-summary" aria-live="polite">
        <strong>{reviews?.jobs.length ?? 0}</strong>
        <span>{reviews?.jobs.length === 1 ? "review" : "reviews"}</span>
        {activeCount > 0 && <span>· {activeCount} active</span>}
      </div>

      {error && <p className="reviews-error" role="alert">{error}</p>}
      {loading ? (
        <p className="reviews-empty">Loading reviews…</p>
      ) : !reviews || reviews.jobs.length === 0 ? (
        <div className="reviews-empty">
          <PullRequestIcon />
          <p>No on-demand reviews yet.</p>
        </div>
      ) : (
        <ol className="review-list">
          {reviews.jobs.map((job) => (
            <li key={job.reviewJobId}>
              <div className="review-copy">
                <strong>{job.repository}#{job.pullRequestNumber}</strong>
                <small>{reviewStatusLabel(job.status)} · {formatDate(job.updatedAt)}</small>
                {job.lastErrorCode && (
                  <small>Error: {job.lastErrorCode.replaceAll("_", " ")}</small>
                )}
                {job.artifact && job.artifact.warningCount > 0 && (
                  <details className="review-warnings">
                    <summary className="review-warning">
                      {job.artifact.warningCount} collection warning
                      {job.artifact.warningCount === 1 ? "" : "s"}
                    </summary>
                    <ul>
                      {job.artifact.warnings.map((warning, index) => (
                        <li key={`${warning.source}-${index}`}>
                          <strong>{warning.source.replaceAll("_", " ")}</strong>: {warning.message}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
              <div className="review-actions">
                {job.artifact && (
                  <>
                    <button
                      className="quiet-button"
                      type="button"
                      disabled={busyId === job.reviewJobId}
                      onClick={() => void openArtifact(job)}
                    >
                      Open
                    </button>
                    <button
                      className="quiet-button"
                      type="button"
                      disabled={busyId === job.reviewJobId}
                      onClick={() => void downloadReviewArtifact(session, job).catch((caught) =>
                        handleFailure(caught, "Could not download the review artifact."))}
                    >
                      Download
                    </button>
                  </>
                )}
                {(job.status === "pending" || job.status === "executing") && (
                  <button
                    className="quiet-button"
                    type="button"
                    disabled={busyId === job.reviewJobId}
                    onClick={() => void changeReview(job, "cancel")}
                  >
                    Cancel
                  </button>
                )}
                {(job.status === "failed" || job.status === "cancelled" || job.status === "timed_out") && (
                  <button
                    className="quiet-button"
                    type="button"
                    disabled={busyId === job.reviewJobId}
                    onClick={() => void changeReview(job, "retry")}
                  >
                    Retry
                  </button>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
