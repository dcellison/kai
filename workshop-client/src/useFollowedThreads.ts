import { useCallback, useEffect, useRef, useState } from "react";

import {
  AuthenticationError,
  ChannelAccessError,
  loadFollowedThreads,
  setThreadFollowed,
  ThreadUnreadConflictError,
} from "./api";
import type { WorkshopFollowedThread } from "./types";
import type { WorkshopPrincipalEvents } from "./usePrincipalEvents";

function operationId(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return `following-unfollow-${Array.from(bytes, (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("")}`;
}

export interface FollowedThreadClientState {
  error: string | null;
  loading: boolean;
  pendingThreadIds: Set<string>;
  threads: WorkshopFollowedThread[];
  totalUnread: number;
}

export function useFollowedThreads(
  token: string,
  principalEvents: WorkshopPrincipalEvents,
  onAuthenticationFailure: (message: string) => void,
): FollowedThreadClientState & {
  unfollow: (thread: WorkshopFollowedThread) => Promise<void>;
} {
  const [state, setState] = useState<FollowedThreadClientState>({
    error: null,
    loading: true,
    pendingThreadIds: new Set(),
    threads: [],
    totalUnread: 0,
  });
  const reloadTimerRef = useRef<number | null>(null);
  const subscribePrincipalEvents = principalEvents.subscribe;

  const reload = useCallback(async (signal?: AbortSignal): Promise<void> => {
    const snapshot = await loadFollowedThreads(token, signal);
    setState((current) => ({
      ...current,
      error: null,
      loading: false,
      threads: snapshot.threads,
      totalUnread: snapshot.threads.reduce(
        (total, thread) => Math.min(1000, total + thread.state.unreadCount),
        0,
      ),
    }));
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
    const requestReload = (): void => {
      void reload(controller.signal).catch((caught: unknown) => {
        if (controller.signal.aborted) return;
        if (caught instanceof AuthenticationError) {
          onAuthenticationFailure(caught.message);
          return;
        }
        setState((current) => ({
          ...current,
          error: caught instanceof Error ? caught.message : "Could not load followed threads.",
          loading: false,
        }));
      });
    };
    const scheduleReload = (): void => {
      if (reloadTimerRef.current !== null) return;
      reloadTimerRef.current = window.setTimeout(() => {
        reloadTimerRef.current = null;
        requestReload();
      }, 50);
    };
    const unsubscribe = subscribePrincipalEvents((event) => {
      if (event.kind === "synchronize") {
        requestReload();
        return;
      }
      if (event.batch.changes.some(
        (change) => change.threadChanges.length > 0 || change.unreadChanges.length > 0,
      )) {
        scheduleReload();
      }
    });
    requestReload();
    return () => {
      unsubscribe();
      controller.abort();
      if (reloadTimerRef.current !== null) {
        window.clearTimeout(reloadTimerRef.current);
        reloadTimerRef.current = null;
      }
    };
  }, [onAuthenticationFailure, reload, subscribePrincipalEvents]);

  const unfollow = useCallback(async (thread: WorkshopFollowedThread): Promise<void> => {
    const rootId = thread.state.threadRootId;
    setState((current) => ({
      ...current,
      error: null,
      pendingThreadIds: new Set(current.pendingThreadIds).add(rootId),
    }));
    try {
      const mutation = await setThreadFollowed(
        { channelId: thread.state.channelId, token },
        rootId,
        false,
        thread.state.stateVersion,
        operationId(),
      );
      setState((current) => {
        const pending = new Set(current.pendingThreadIds);
        pending.delete(rootId);
        const threads = mutation.state.followed
          ? current.threads
          : current.threads.filter((candidate) => candidate.state.threadRootId !== rootId);
        return {
          ...current,
          pendingThreadIds: pending,
          threads,
          totalUnread: threads.reduce(
            (total, candidate) => Math.min(1000, total + candidate.state.unreadCount),
            0,
          ),
        };
      });
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        onAuthenticationFailure(caught.message);
        return;
      }
      if (caught instanceof ThreadUnreadConflictError || caught instanceof ChannelAccessError) {
        try {
          await reload();
        } catch {
          // The useful error below remains visible; the principal event stream
          // will make another bounded synchronization attempt.
        }
      }
      setState((current) => {
        const pending = new Set(current.pendingThreadIds);
        pending.delete(rootId);
        return {
          ...current,
          error: caught instanceof Error ? caught.message : "Could not unfollow this thread.",
          pendingThreadIds: pending,
        };
      });
    }
  }, [onAuthenticationFailure, reload, token]);

  return { ...state, unfollow };
}
