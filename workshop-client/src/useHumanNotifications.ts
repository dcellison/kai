import { useCallback, useEffect, useRef, useState } from "react";

import {
  AuthenticationError,
  loadHumanNotificationCounts,
  loadHumanNotifications,
  markHumanNotificationRead,
  markHumanNotificationsRead,
  markHumanNotificationUnread,
  ResynchronizationRequired,
  streamHumanNotifications,
} from "./api";
import type {
  WorkshopHumanNotification,
  WorkshopHumanNotificationCounts,
} from "./types";

const RECONNECT_DELAY_MS = 2000;
const EMPTY_COUNTS: WorkshopHumanNotificationCounts = {
  read: 0,
  total: 0,
  unread: 0,
  unreadByChannel: {},
};

function operationId(prefix: string): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return `${prefix}-${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function waitForRetry(signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timeout = window.setTimeout(resolve, RECONNECT_DELAY_MS);
    signal.addEventListener("abort", () => {
      window.clearTimeout(timeout);
      resolve();
    }, { once: true });
  });
}

function mergeNotification(
  current: WorkshopHumanNotification[],
  incoming: WorkshopHumanNotification,
): WorkshopHumanNotification[] {
  const next = current.filter(
    (notification) => notification.notificationId !== incoming.notificationId,
  );
  next.push(incoming);
  return next.sort(
    (left, right) => right.createdEventPosition - left.createdEventPosition,
  );
}

export interface HumanNotificationState {
  counts: WorkshopHumanNotificationCounts;
  error: string | null;
  hasMore: boolean;
  loading: boolean;
  loadingMore: boolean;
  notifications: WorkshopHumanNotification[];
  pending: boolean;
}

export function useHumanNotifications(
  token: string,
  onAuthenticationFailure: (message: string) => void,
): HumanNotificationState & {
  loadMore: () => void;
  markAllRead: () => Promise<void>;
  setRead: (notification: WorkshopHumanNotification, read: boolean) => Promise<void>;
} {
  const [state, setState] = useState<HumanNotificationState>({
    counts: EMPTY_COUNTS,
    error: null,
    hasMore: false,
    loading: true,
    loadingMore: false,
    notifications: [],
    pending: false,
  });
  const cursorRef = useRef<string | null>(null);
  const loadMorePendingRef = useRef(false);
  const countsRefreshRef = useRef<number | null>(null);

  const refreshCounts = useCallback(async (signal?: AbortSignal): Promise<void> => {
    const counts = await loadHumanNotificationCounts(token, signal);
    setState((current) => ({ ...current, counts }));
  }, [token]);

  const reload = useCallback(async (signal?: AbortSignal): Promise<number> => {
    const page = await loadHumanNotifications(token, { limit: 50 }, signal);
    cursorRef.current = page.nextCursor;
    setState((current) => ({
      ...current,
      counts: page.counts,
      error: null,
      hasMore: page.nextCursor !== null,
      loading: false,
      notifications: page.notifications,
    }));
    return page.throughPosition;
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
    let lastEventId: string | null = null;
    let needsReload = true;
    const scheduleCountsRefresh = (): void => {
      if (countsRefreshRef.current !== null) return;
      countsRefreshRef.current = window.setTimeout(() => {
        countsRefreshRef.current = null;
        void refreshCounts(controller.signal).catch((caught: unknown) => {
          if (caught instanceof AuthenticationError) {
            onAuthenticationFailure(caught.message);
          }
        });
      }, 100);
    };
    const synchronize = async (): Promise<void> => {
      while (!controller.signal.aborted) {
        try {
          if (needsReload) {
            lastEventId = String(await reload(controller.signal));
            needsReload = false;
          }
          await streamHumanNotifications(
            token,
            lastEventId,
            {
              onChanged: ({ notification }, eventId) => {
                lastEventId = eventId;
                setState((current) => ({
                  ...current,
                  notifications: mergeNotification(current.notifications, notification),
                }));
                scheduleCountsRefresh();
              },
              onConnected: () => undefined,
            },
            controller.signal,
          );
          if (!controller.signal.aborted) {
            await waitForRetry(controller.signal);
          }
        } catch (caught) {
          if (controller.signal.aborted) return;
          if (caught instanceof AuthenticationError) {
            onAuthenticationFailure(caught.message);
            return;
          }
          if (caught instanceof ResynchronizationRequired) {
            lastEventId = null;
            needsReload = true;
          } else {
            setState((current) => ({
              ...current,
              error: current.loading
                ? caught instanceof Error ? caught.message : "Could not load mentions."
                : current.error,
              loading: false,
            }));
          }
          await waitForRetry(controller.signal);
        }
      }
    };
    void synchronize();
    return () => {
      controller.abort();
      if (countsRefreshRef.current !== null) {
        window.clearTimeout(countsRefreshRef.current);
        countsRefreshRef.current = null;
      }
    };
  }, [onAuthenticationFailure, refreshCounts, reload, token]);

  const loadMore = useCallback((): void => {
    const cursor = cursorRef.current;
    if (!cursor || loadMorePendingRef.current) return;
    loadMorePendingRef.current = true;
    setState((current) => ({ ...current, loadingMore: true }));
    void loadHumanNotifications(token, { cursor, limit: 50 }).then(
      (page) => {
        cursorRef.current = page.nextCursor;
        setState((current) => {
          const known = new Set(current.notifications.map((item) => item.notificationId));
          return {
            ...current,
            counts: page.counts,
            error: null,
            hasMore: page.nextCursor !== null,
            loadingMore: false,
            notifications: [
              ...current.notifications,
              ...page.notifications.filter((item) => !known.has(item.notificationId)),
            ],
          };
        });
        loadMorePendingRef.current = false;
      },
      (caught: unknown) => {
        loadMorePendingRef.current = false;
        if (caught instanceof AuthenticationError) {
          onAuthenticationFailure(caught.message);
          return;
        }
        setState((current) => ({
          ...current,
          error: caught instanceof Error ? caught.message : "Could not load more mentions.",
          loadingMore: false,
        }));
      },
    );
  }, [onAuthenticationFailure, token]);

  const setRead = useCallback(async (
    notification: WorkshopHumanNotification,
    read: boolean,
  ): Promise<void> => {
    setState((current) => ({ ...current, error: null, pending: true }));
    try {
      const result = read
        ? await markHumanNotificationRead(
            token,
            notification.notificationId,
            notification.stateVersion,
            operationId("mention-read"),
          )
        : await markHumanNotificationUnread(
            token,
            notification.notificationId,
            notification.stateVersion,
            operationId("mention-unread"),
          );
      const counts = await loadHumanNotificationCounts(token);
      setState((current) => ({
        ...current,
        counts,
        notifications: mergeNotification(current.notifications, result.notification),
        pending: false,
      }));
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        onAuthenticationFailure(caught.message);
        return;
      }
      setState((current) => ({
        ...current,
        error: caught instanceof Error ? caught.message : "Could not update this mention.",
        pending: false,
      }));
    }
  }, [onAuthenticationFailure, token]);

  const markAllRead = useCallback(async (): Promise<void> => {
    setState((current) => ({ ...current, error: null, pending: true }));
    try {
      const unread = await loadHumanNotifications(token, { limit: 100, unreadOnly: true });
      if (unread.notifications.length > 0) {
        const mutations = await markHumanNotificationsRead(
          token,
          unread.notifications.map((notification) => ({
            expectedStateVersion: notification.stateVersion,
            notificationId: notification.notificationId,
          })),
          operationId("mentions-read"),
        );
        setState((current) => ({
          ...current,
          notifications: mutations.reduce(
            (items, mutation) => mergeNotification(items, mutation.notification),
            current.notifications,
          ),
        }));
      }
      const counts = await loadHumanNotificationCounts(token);
      setState((current) => ({ ...current, counts, pending: false }));
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        onAuthenticationFailure(caught.message);
        return;
      }
      setState((current) => ({
        ...current,
        error: caught instanceof Error ? caught.message : "Could not mark mentions read.",
        pending: false,
      }));
    }
  }, [onAuthenticationFailure, token]);

  return { ...state, loadMore, markAllRead, setRead };
}
