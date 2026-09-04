import { useState } from "react";

import type { WorkshopHumanNotification } from "./types";
import type { HumanNotificationState } from "./useHumanNotifications";
import { HumanAvatar } from "./HumanAvatar";

const INACTIVE_HUMAN_AVATAR = { active: false, stateVersion: 0, url: null } as const;

function formatMentionTime(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(parsed);
}

export function MentionsInbox({
  inbox,
  onClose,
  onOpen,
}: {
  inbox: HumanNotificationState & {
    loadMore: () => void;
    markAllRead: () => Promise<void>;
    markVisibleRead: (messageId: string) => Promise<void>;
    setRead: (notification: WorkshopHumanNotification, read: boolean) => Promise<void>;
  };
  onClose: () => void;
  onOpen: (notification: WorkshopHumanNotification) => boolean;
}): React.JSX.Element {
  const [navigationError, setNavigationError] = useState<string | null>(null);
  return (
    <section className="mentions-workspace" aria-label="Mentions">
      <header className="mentions-header">
        <div>
          <p className="overline">Personal inbox</p>
          <h1>Mentions</h1>
        </div>
        <div className="mentions-header-actions">
          {inbox.counts.unread > 0 && (
            <button
              className="quiet-button"
              type="button"
              disabled={inbox.pending}
              onClick={() => void inbox.markAllRead()}
            >
              {inbox.counts.unread > 100 ? "Mark next 100 read" : "Mark all read"}
            </button>
          )}
          <button
            className="panel-icon-button"
            type="button"
            aria-label="Back to conversation"
            title="Back to conversation"
            onClick={onClose}
          >
            <span aria-hidden="true">←</span>
          </button>
        </div>
      </header>

      <div className="mentions-summary" aria-live="polite">
        <strong>{inbox.counts.unread}</strong>
        <span>{inbox.counts.unread === 1 ? "unread mention" : "unread mentions"}</span>
      </div>

      <div className="mentions-notices">
        {inbox.error && <p className="mentions-error" role="alert">{inbox.error}</p>}
        {navigationError && <p className="mentions-error" role="alert">{navigationError}</p>}
      </div>
      {inbox.loading ? (
        <p className="mentions-empty">Loading mentions…</p>
      ) : inbox.notifications.length === 0 ? (
        <div className="mentions-empty">
          <span aria-hidden="true">@</span>
          <p>No mentions yet.</p>
        </div>
      ) : (
        <ol className="mentions-list">
          {inbox.notifications.map((notification) => (
            <li className={notification.read ? "read" : "unread"} key={notification.notificationId}>
              <button
                className="mention-source-link"
                type="button"
                onClick={() => {
                  setNavigationError(null);
                  if (!onOpen(notification)) {
                    setNavigationError(
                      "You no longer have access to that mention's source channel.",
                    );
                    return;
                  }
                }}
              >
                <HumanAvatar
                  avatar={notification.sourceAuthorAvatar ?? INACTIVE_HUMAN_AVATAR}
                  className="mention-avatar"
                  displayName={notification.sourceAuthorDisplayName}
                  principalId={notification.sourceAuthorPrincipalId}
                />
                <span className="mention-copy">
                  <span>
                    <strong>{notification.sourceAuthorDisplayName}</strong>
                    {" mentioned you in "}
                    <b>{notification.channelName ?? "a channel"}</b>
                  </span>
                  <small>
                    {notification.sourceThreadRootId ? "Thread reply" : "Channel message"}
                    {" · "}{formatMentionTime(notification.createdAt)}
                  </small>
                </span>
              </button>
              <button
                className="mention-state-button"
                type="button"
                disabled={inbox.pending}
                aria-label={notification.read ? "Mark mention unread" : "Mark mention read"}
                title={notification.read ? "Mark unread" : "Mark read"}
                onClick={() => void inbox.setRead(notification, !notification.read)}
              >
                <span aria-hidden="true">{notification.read ? "○" : "●"}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
      {inbox.hasMore && (
        <button
          className="mentions-load-more"
          type="button"
          disabled={inbox.loadingMore}
          onClick={inbox.loadMore}
        >
          {inbox.loadingMore ? "Loading…" : "Load older mentions"}
        </button>
      )}
    </section>
  );
}
