import { useCallback, useEffect, useRef, useState } from "react";

import {
  advanceChannelReadPosition,
  AuthenticationError,
  ChannelAccessError,
  ChannelReadPositionConflictError,
  loadChannelUnread,
  ResynchronizationRequired,
  streamChannelUnread,
} from "./api";
import type {
  TimelineMessage,
  WorkshopChannelUnreadState,
  WorkshopSession,
} from "./types";

const RECONNECT_DELAY_MS = 2000;

function operationId(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return `channel-read-${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
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
    let cursor: string | null = null;
    let needsReload = true;
    const synchronize = async (): Promise<void> => {
      while (!controller.signal.aborted) {
        try {
          if (needsReload) {
            cursor = String(await reload(controller.signal));
            needsReload = false;
          }
          await streamChannelUnread(
            token,
            cursor,
            {
              onChanged: ({ state: changed }, eventId) => {
                cursor = eventId;
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
            cursor = null;
            needsReload = true;
          } else {
            setState((current) => ({
              ...current,
              error: current.loading
                ? caught instanceof Error ? caught.message : "Could not load unread messages."
                : current.error,
              loading: false,
            }));
          }
        }
        await waitForRetry(controller.signal);
      }
    };
    void synchronize();
    return () => controller.abort();
  }, [onAuthenticationFailure, reload, token]);

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
