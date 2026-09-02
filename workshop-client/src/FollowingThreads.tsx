import { useState } from "react";

import type { WorkshopFollowedThread } from "./types";
import type { FollowedThreadClientState } from "./useFollowedThreads";

function formatTime(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(parsed);
}

function BellIcon({ followed = false }: { followed?: boolean }): React.JSX.Element {
  return (
    <svg
      aria-hidden="true"
      fill={followed ? "currentColor" : "none"}
      focusable="false"
      viewBox="0 0 24 24"
    >
      <path
        d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

export function FollowingThreads({
  following,
  onClose,
  onOpen,
}: {
  following: FollowedThreadClientState & {
    unfollow: (thread: WorkshopFollowedThread) => Promise<void>;
  };
  onClose: () => void;
  onOpen: (thread: WorkshopFollowedThread) => boolean;
}): React.JSX.Element {
  const [navigationError, setNavigationError] = useState<string | null>(null);
  return (
    <section className="following-workspace" aria-label="Following">
      <header className="following-header">
        <div>
          <p className="overline">Workspace</p>
          <h1>Following</h1>
        </div>
        <button
          className="panel-icon-button"
          type="button"
          aria-label="Back to conversation"
          title="Back to conversation"
          onClick={onClose}
        >
          <span aria-hidden="true">←</span>
        </button>
      </header>

      <div className="following-summary" aria-live="polite">
        <strong>{following.threads.length}</strong>
        <span>{following.threads.length === 1 ? "followed thread" : "followed threads"}</span>
        {following.totalUnread > 0 && (
          <span>· {following.totalUnread}{following.totalUnread >= 1000 ? "+" : ""} unread</span>
        )}
      </div>

      <div className="following-notices">
        {following.error && <p className="following-error" role="alert">{following.error}</p>}
        {navigationError && <p className="following-error" role="alert">{navigationError}</p>}
      </div>

      {following.loading ? (
        <p className="following-empty">Loading followed threads…</p>
      ) : following.threads.length === 0 ? (
        <div className="following-empty">
          <BellIcon />
          <p>No followed threads.</p>
        </div>
      ) : (
        <ol className="following-list">
          {following.threads.map((thread) => {
            const rootId = thread.state.threadRootId;
            const unread = thread.state.unreadCount;
            return (
              <li className={unread > 0 ? "unread" : "read"} key={rootId}>
                <button
                  className="following-source-link"
                  type="button"
                  onClick={() => {
                    setNavigationError(null);
                    if (!onOpen(thread)) {
                      setNavigationError("You no longer have access to that thread's channel.");
                    }
                  }}
                >
                  <span className="following-channel">
                    # {thread.channelName ?? "Channel"}
                    {thread.channelArchived && <small>Archived</small>}
                  </span>
                  <span className="following-root">
                    <strong>{thread.rootAuthorDisplayName}</strong>
                    <span>{thread.rootExcerpt || "Attachment or empty message"}</span>
                  </span>
                  <small className="following-latest">
                    {thread.latestReplyCreatedAt && thread.latestReplyAuthorDisplayName
                      ? `Latest reply by ${thread.latestReplyAuthorDisplayName} · ${formatTime(thread.latestReplyCreatedAt)}`
                      : `Started ${formatTime(thread.rootCreatedAt)}`}
                  </small>
                  {unread > 0 && (
                    <span className="following-unread">
                      {unread}{thread.state.unreadCountCapped ? "+" : ""} unread
                    </span>
                  )}
                </button>
                <button
                  className="panel-icon-button following-unfollow-button followed"
                  type="button"
                  aria-label={`Unfollow thread by ${thread.rootAuthorDisplayName}`}
                  aria-pressed="true"
                  title="Unfollow thread"
                  disabled={following.pendingThreadIds.has(rootId)}
                  onClick={() => void following.unfollow(thread)}
                >
                  <BellIcon followed />
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
