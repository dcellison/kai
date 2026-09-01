import { useCallback, useEffect, useRef, useState } from "react";

import {
  AuthenticationError,
  loadHumanNotificationCounts,
  loadHumanNotifications,
  markHumanNotificationRead,
  markHumanNotificationsRead,
  markHumanNotificationUnread,
} from "./api";
import type {
  WorkshopHumanNotification,
  WorkshopHumanNotificationCounts,
} from "./types";
import type { WorkshopPrincipalEvents } from "./usePrincipalEvents";

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
  principalEvents: WorkshopPrincipalEvents,
  onAuthenticationFailure: (message: string) => void,
): HumanNotificationState & {
  loadMore: () => void;
  markAllRead: () => Promise<void>;
  markVisibleRead: (messageId: string) => Promise<void>;
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
  const visibleReadPendingRef = useRef(new Set<string>());
  const subscribePrincipalEvents = principalEvents.subscribe;

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
      notifications: current.notifications
        .filter((item) => item.lastEventPosition > page.throughPosition)
        .reduce(mergeNotification, page.notifications),
    }));
    return page.throughPosition;
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
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
    const requestReload = (): void => {
      void reload(controller.signal).catch((caught: unknown) => {
        if (controller.signal.aborted) return;
        if (caught instanceof AuthenticationError) {
          onAuthenticationFailure(caught.message);
          return;
        }
        setState((current) => ({
          ...current,
          error: current.loading
            ? caught instanceof Error ? caught.message : "Could not load mentions."
            : current.error,
          loading: false,
        }));
      });
    };
    const unsubscribe = subscribePrincipalEvents((event) => {
      if (event.kind === "synchronize") {
        requestReload();
        return;
      }
      let changed = false;
      for (const change of event.batch.changes) {
        for (const { notification } of change.notificationChanges) {
          changed = true;
          setState((current) => ({
            ...current,
            notifications: mergeNotification(current.notifications, notification),
          }));
        }
      }
      if (changed) scheduleCountsRefresh();
    });
    requestReload();
    return () => {
      unsubscribe();
      controller.abort();
      if (countsRefreshRef.current !== null) {
        window.clearTimeout(countsRefreshRef.current);
        countsRefreshRef.current = null;
      }
    };
  }, [onAuthenticationFailure, refreshCounts, reload, subscribePrincipalEvents]);

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

  const markVisibleRead = useCallback(async (messageId: string): Promise<void> => {
    const unread = state.notifications.filter(
      (notification) =>
        !notification.read &&
        notification.sourceMessageId === messageId &&
        !visibleReadPendingRef.current.has(notification.notificationId),
    );
    if (unread.length === 0) {
      return;
    }
    for (const notification of unread) {
      visibleReadPendingRef.current.add(notification.notificationId);
    }
    try {
      const mutations = await Promise.all(
        unread.map((notification) => markHumanNotificationRead(
          token,
          notification.notificationId,
          notification.stateVersion,
          operationId("mention-viewed"),
        )),
      );
      const counts = await loadHumanNotificationCounts(token);
      setState((current) => ({
        ...current,
        counts,
        notifications: mutations.reduce(
          (items, mutation) => mergeNotification(items, mutation.notification),
          current.notifications,
        ),
      }));
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        onAuthenticationFailure(caught.message);
        return;
      }
      setState((current) => ({
        ...current,
        error: caught instanceof Error
          ? caught.message
          : "Could not mark the displayed mention read.",
      }));
    } finally {
      for (const notification of unread) {
        visibleReadPendingRef.current.delete(notification.notificationId);
      }
    }
  }, [onAuthenticationFailure, state.notifications, token]);

  return { ...state, loadMore, markAllRead, markVisibleRead, setRead };
}
