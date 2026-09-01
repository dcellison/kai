import { useCallback, useEffect, useRef, useState } from "react";

import {
  advanceChannelReadPosition,
  AuthenticationError,
  ChannelAccessError,
  ChannelReadPositionConflictError,
  loadChannelUnread,
} from "./api";
import type {
  TimelineMessage,
  WorkshopChannelUnreadState,
  WorkshopSession,
} from "./types";
import type { WorkshopPrincipalEvents } from "./usePrincipalEvents";

function operationId(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return `channel-read-${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function indexed(
  channels: WorkshopChannelUnreadState[],
): Record<string, WorkshopChannelUnreadState> {
  return Object.fromEntries(channels.map((state) => [state.channelId, state]));
}

export interface ChannelUnreadClientState {
  byChannel: Record<string, WorkshopChannelUnreadState>;
  error: string | null;
  loading: boolean;
  totalUnread: number;
  totalUnreadCapped: boolean;
}

export function useChannelUnread(
  token: string,
  principalEvents: WorkshopPrincipalEvents,
  onAuthenticationFailure: (message: string) => void,
  onChannelAccessFailure: (message: string) => void,
): ChannelUnreadClientState & {
  advanceVisible: (session: WorkshopSession, message: TimelineMessage) => void;
} {
  const [state, setState] = useState<ChannelUnreadClientState>({
    byChannel: {},
    error: null,
    loading: true,
    totalUnread: 0,
    totalUnreadCapped: false,
  });
  const stateRef = useRef(state);
  const pendingRef = useRef(new Map<string, TimelineMessage>());
  const runningRef = useRef(new Set<string>());
  const subscribePrincipalEvents = principalEvents.subscribe;
  stateRef.current = state;

  const reload = useCallback(async (signal?: AbortSignal): Promise<number> => {
    const snapshot = await loadChannelUnread(token, signal);
    const next = {
      byChannel: indexed(snapshot.channels),
      error: null,
      loading: false,
      totalUnread: snapshot.totalUnread,
      totalUnreadCapped: snapshot.totalUnreadCapped,
    };
    stateRef.current = next;
    setState(next);
    return snapshot.throughPosition;
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
          error: current.loading
            ? caught instanceof Error ? caught.message : "Could not load unread messages."
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
      for (const change of event.batch.changes) {
        for (const { state: changed } of change.unreadChanges) {
          setState((current) => {
            const previous = current.byChannel[changed.channelId];
            const nextTotal = Math.max(
              0,
              current.totalUnread - (previous?.unreadCount ?? 0) + changed.unreadCount,
            );
            const next = {
              ...current,
              byChannel: { ...current.byChannel, [changed.channelId]: changed },
              totalUnread: nextTotal,
            };
            stateRef.current = next;
            return next;
          });
        }
      }
    });
    requestReload();
    return () => {
      unsubscribe();
      controller.abort();
    };
  }, [onAuthenticationFailure, reload, subscribePrincipalEvents]);

  const advanceVisible = useCallback((session: WorkshopSession, message: TimelineMessage): void => {
    const current = stateRef.current.byChannel[session.channelId];
    if (
      !current ||
      message.channelId !== session.channelId ||
      message.threadRootId !== null ||
      message.eventPosition <= current.readThroughEventPosition
    ) return;
    const pending = pendingRef.current.get(session.channelId);
    if (!pending || pending.eventPosition < message.eventPosition) {
      pendingRef.current.set(session.channelId, message);
    }
    if (runningRef.current.has(session.channelId)) return;
    runningRef.current.add(session.channelId);

    const drain = async (): Promise<void> => {
      try {
        while (pendingRef.current.has(session.channelId)) {
          const target = pendingRef.current.get(session.channelId);
          pendingRef.current.delete(session.channelId);
          if (!target) continue;
          let latest = stateRef.current.byChannel[session.channelId];
          if (!latest || target.eventPosition <= latest.readThroughEventPosition) continue;
          try {
            const mutation = await advanceChannelReadPosition(
              session,
              target.messageId,
              latest.stateVersion,
              operationId(),
            );
            setState((currentState) => {
              const previous = currentState.byChannel[session.channelId];
              const next = {
                ...currentState,
                byChannel: {
                  ...currentState.byChannel,
                  [session.channelId]: mutation.state,
                },
                totalUnread: Math.max(
                  0,
                  currentState.totalUnread -
                    (previous?.unreadCount ?? 0) + mutation.state.unreadCount,
                ),
              };
              stateRef.current = next;
              return next;
            });
          } catch (caught) {
            if (caught instanceof AuthenticationError) {
              onAuthenticationFailure(caught.message);
              return;
            }
            if (caught instanceof ChannelAccessError) {
              onChannelAccessFailure(caught.message);
              return;
            }
            if (caught instanceof ChannelReadPositionConflictError) {
              await reload();
              latest = stateRef.current.byChannel[session.channelId];
              if (latest && target.eventPosition > latest.readThroughEventPosition) {
                pendingRef.current.set(session.channelId, target);
              }
              continue;
            }
            setState((currentState) => ({
              ...currentState,
              error: caught instanceof Error
                ? caught.message
                : "Could not update unread messages.",
            }));
          }
        }
      } finally {
        runningRef.current.delete(session.channelId);
      }
    };
    void drain();
  }, [onAuthenticationFailure, onChannelAccessFailure, reload]);

  return { ...state, advanceVisible };
}
