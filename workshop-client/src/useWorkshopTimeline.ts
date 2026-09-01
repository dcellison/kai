import { useCallback, useEffect, useRef, useState } from "react";

import {
  AuthenticationError,
  ChannelAccessError,
  loadEarlierTimeline,
  loadTimeline,
  ResynchronizationRequired,
  streamTimeline,
} from "./api";
import type {
  ConnectionState,
  TimelineMessage,
  WorkshopRunActivity,
  WorkshopRunPreview,
  WorkshopRunTraceSignal,
  WorkshopSession,
  WorkshopMessageReaction,
} from "./types";

const RECONNECT_DELAY_MS = 2000;

function waitForRetry(signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timeout = window.setTimeout(resolve, RECONNECT_DELAY_MS);
    signal.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timeout);
        resolve();
      },
      { once: true },
    );
  });
}

function appendUnique(
  messages: TimelineMessage[],
  incoming: TimelineMessage,
): TimelineMessage[] {
  if (messages.some((message) => message.messageId === incoming.messageId)) {
    return messages;
  }
  return [...messages, incoming].sort(
    (left, right) => left.eventPosition - right.eventPosition,
  );
}

function prependUnique(
  messages: TimelineMessage[],
  earlier: TimelineMessage[],
): TimelineMessage[] {
  const known = new Set(messages.map((message) => message.messageId));
  const fresh = earlier.filter((message) => !known.has(message.messageId));
  if (fresh.length === 0) {
    return messages;
  }
  return [...fresh, ...messages].sort(
    (left, right) => left.eventPosition - right.eventPosition,
  );
}

function replaceReactions(
  messages: TimelineMessage[],
  messageId: string,
  reactions: WorkshopMessageReaction[],
): TimelineMessage[] {
  let changed = false;
  const next = messages.map((message) => {
    if (message.messageId !== messageId) {
      return message;
    }
    changed = true;
    return { ...message, reactions };
  });
  return changed ? next : messages;
}

export interface EarlierHistoryState {
  available: boolean;
  loading: boolean;
  error: string | null;
}

export interface LaterHistoryState {
  available: boolean;
  loading: boolean;
  error: string | null;
}

// Everything a loadEarlier call needs from the snapshot it extends. Held
// in a ref so the callback's identity survives snapshot reloads, with the
// generation stamp guarding against a fetch outliving its snapshot.
interface EarlierFetchContext {
  session: WorkshopSession;
  signal: AbortSignal;
  throughPosition: number;
  generation: number;
}

export function useWorkshopTimeline(
  session: WorkshopSession | null,
  active: boolean,
  startMessageId: string | null,
  onAuthenticationFailure: (message: string) => void,
  onChannelAccessFailure: (message: string) => void,
): {
  connection: ConnectionState;
  messages: TimelineMessage[];
  threadMessages: TimelineMessage[];
  reactionUpdates: Record<string, WorkshopMessageReaction[]>;
  runActivity: WorkshopRunActivity | null;
  runPreview: WorkshopRunPreview | null;
  runTrace: WorkshopRunTraceSignal | null;
  earlier: EarlierHistoryState;
  later: LaterHistoryState;
  loadEarlier: () => void;
  loadLater: () => void;
  jumpLatest: () => void;
  updateReactions: (
    messageId: string,
    reactions: WorkshopMessageReaction[],
  ) => void;
} {
  const [messages, setMessages] = useState<TimelineMessage[]>([]);
  const [threadMessages, setThreadMessages] = useState<TimelineMessage[]>([]);
  const [reactionUpdates, setReactionUpdates] = useState<
    Record<string, WorkshopMessageReaction[]>
  >({});
  const [runActivity, setRunActivity] = useState<WorkshopRunActivity | null>(null);
  const [runPreview, setRunPreview] = useState<WorkshopRunPreview | null>(null);
  const [runTrace, setRunTrace] = useState<WorkshopRunTraceSignal | null>(null);
  const [connection, setConnection] = useState<ConnectionState>({
    label: "Waiting",
    tone: "connecting",
  });
  const [earlier, setEarlier] = useState<EarlierHistoryState>({
    available: false,
    loading: false,
    error: null,
  });
  const [later, setLater] = useState<LaterHistoryState>({
    available: false,
    loading: false,
    error: null,
  });
  const earlierContextRef = useRef<EarlierFetchContext | null>(null);
  const earlierCursorRef = useRef<string | null>(null);
  const earlierLoadingRef = useRef(false);
  const laterContextRef = useRef<EarlierFetchContext | null>(null);
  const laterCursorRef = useRef<string | null>(null);
  const laterLoadingRef = useRef(false);
  const generationRef = useRef(0);
  const forceTailChannelRef = useRef<string | null>(null);
  const [reloadVersion, setReloadVersion] = useState(0);

  const jumpLatest = useCallback((): void => {
    if (!session) return;
    forceTailChannelRef.current = session.channelId;
    setReloadVersion((version) => version + 1);
  }, [session]);

  const updateReactions = useCallback(
    (messageId: string, reactions: WorkshopMessageReaction[]): void => {
      setMessages((current) => replaceReactions(current, messageId, reactions));
      setThreadMessages((current) =>
        replaceReactions(current, messageId, reactions),
      );
      setReactionUpdates((current) => ({ ...current, [messageId]: reactions }));
    },
    [],
  );

  const loadEarlier = useCallback((): void => {
    const context = earlierContextRef.current;
    const cursor = earlierCursorRef.current;
    if (!context || context.signal.aborted || earlierLoadingRef.current || cursor === null) {
      return;
    }
    earlierLoadingRef.current = true;
    setEarlier({ available: true, loading: true, error: null });
    void loadEarlierTimeline(context.session, cursor, context.throughPosition, context.signal).then(
      (page) => {
        // A snapshot reload (channel switch, resynchronization) between
        // request and response makes this page part of a window that no
        // longer exists; drop it before touching ANY state, the loading
        // guard included. Each snapshot resets the guard for its own
        // generation, and a stale settlement resetting it here would let
        // a duplicate in-flight fetch slip past.
        if (context.signal.aborted || generationRef.current !== context.generation) {
          return;
        }
        earlierLoadingRef.current = false;
        earlierCursorRef.current = page.previousCursor;
        setMessages((current) => prependUnique(current, page.messages));
        setEarlier({ available: page.previousCursor !== null, loading: false, error: null });
      },
      (caught: unknown) => {
        if (context.signal.aborted || generationRef.current !== context.generation) {
          return;
        }
        earlierLoadingRef.current = false;
        setEarlier({
          available: true,
          loading: false,
          error:
            caught instanceof Error ? caught.message : "Could not load earlier messages.",
        });
      },
    );
  }, []);

  const loadLater = useCallback((): void => {
    const context = laterContextRef.current;
    const cursor = laterCursorRef.current;
    if (!context || context.signal.aborted || laterLoadingRef.current || cursor === null) {
      return;
    }
    laterLoadingRef.current = true;
    setLater({ available: true, loading: true, error: null });
    void loadEarlierTimeline(context.session, cursor, context.throughPosition, context.signal).then(
      (page) => {
        if (context.signal.aborted || generationRef.current !== context.generation) return;
        laterLoadingRef.current = false;
        laterCursorRef.current = page.nextCursor ?? null;
        setMessages((current) => {
          const next = current;
          return page.messages.reduce(appendUnique, next);
        });
        setLater({
          available: laterCursorRef.current !== null,
          loading: false,
          error: null,
        });
      },
      (caught: unknown) => {
        if (context.signal.aborted || generationRef.current !== context.generation) return;
        laterLoadingRef.current = false;
        setLater({
          available: true,
          loading: false,
          error: caught instanceof Error ? caught.message : "Could not load newer messages.",
        });
      },
    );
  }, []);

  useEffect(() => {
    if (!active || !session) {
      generationRef.current += 1;
      earlierContextRef.current = null;
      earlierCursorRef.current = null;
      earlierLoadingRef.current = false;
      laterContextRef.current = null;
      laterCursorRef.current = null;
      laterLoadingRef.current = false;
      setEarlier({ available: false, loading: false, error: null });
      setLater({ available: false, loading: false, error: null });
      setMessages([]);
      setThreadMessages([]);
      setReactionUpdates({});
      setRunActivity(null);
      setRunPreview(null);
      setRunTrace(null);
      setConnection({ label: "Waiting", tone: "connecting" });
      return;
    }

    const controller = new AbortController();
    const { signal } = controller;
    setRunActivity(null);
    setRunPreview(null);
    setRunTrace(null);
    // Runs already seen terminal on this connection. A preview event that
    // races the terminal batch must never resurrect a finished bubble.
    const terminalRunIds = new Set<string>();
    let lastEventId = "0";
    let needsSnapshot = true;
    let knownMessageIds = new Set<string>();

    const synchronize = async (): Promise<void> => {
      while (!signal.aborted) {
        try {
          if (needsSnapshot) {
            setConnection({ label: "Loading history", tone: "connecting" });
            const forceTail = forceTailChannelRef.current === session.channelId;
            const effectiveStartMessageId = forceTail ? null : startMessageId;
            if (forceTail) {
              forceTailChannelRef.current = null;
            }
            const snapshot = effectiveStartMessageId === null
              ? await loadTimeline(session, signal)
              : await loadTimeline(session, signal, effectiveStartMessageId);
            if (signal.aborted) {
              return;
            }
            lastEventId = String(snapshot.throughPosition);
            knownMessageIds = new Set(
              snapshot.messages.map((message) => message.messageId),
            );
            setMessages(snapshot.messages.filter((message) => message.threadRootId === null));
            setThreadMessages([]);
            setReactionUpdates({});
            needsSnapshot = false;
            // Every snapshot starts a fresh backward-paging window; a
            // resynchronization deliberately collapses back to the tail,
            // discarding any earlier pages the reader had expanded.
            generationRef.current += 1;
            earlierContextRef.current = {
              session,
              signal,
              throughPosition: snapshot.throughPosition,
              generation: generationRef.current,
            };
            laterContextRef.current = earlierContextRef.current;
            earlierCursorRef.current = snapshot.previousCursor;
            laterCursorRef.current = snapshot.nextCursor ?? null;
            earlierLoadingRef.current = false;
            laterLoadingRef.current = false;
            setEarlier({
              available: snapshot.previousCursor !== null,
              loading: false,
              error: null,
            });
            setLater({
              available: laterCursorRef.current !== null,
              loading: false,
              error: null,
            });
          }

          setConnection({ label: "Connecting", tone: "connecting" });
          // Previews are ephemeral per-process server state. After a server
          // restart the registry's sequence numbering starts over, so the
          // high-water mark held here would silently discard every preview
          // from the new process. Each connection attempt starts clean; the
          // stream re-sends the current preview immediately on connect.
          setRunPreview(null);
          await streamTimeline(
            session,
            lastEventId,
            {
              onConnected: () => {
                setConnection({ label: "Live", tone: "connected" });
              },
              onMessage: (message, eventId) => {
                lastEventId = eventId;
                if (knownMessageIds.has(message.messageId)) {
                  return;
                }
                knownMessageIds.add(message.messageId);
                // The canonical assistant message replaces any streaming
                // preview; SQLite remains authoritative.
                if (message.authorKind === "agent") {
                  setRunPreview(null);
                }
                if (message.threadRootId === null) {
                  setMessages((current) => appendUnique(current, message));
                } else {
                  setThreadMessages((current) => appendUnique(current, message));
                }
              },
              onReactions: (messageId, reactions, eventId) => {
                lastEventId = eventId;
                updateReactions(messageId, reactions);
              },
              onRunActivity: (activity, eventId) => {
                lastEventId = eventId;
                if (activity.run.terminalAt !== null) {
                  terminalRunIds.add(activity.run.runId);
                  setRunPreview((current) =>
                    current && current.runId === activity.run.runId ? null : current,
                  );
                }
                setRunActivity(activity);
              },
              onRunPreview: (preview) => {
                if (terminalRunIds.has(preview.runId)) {
                  return;
                }
                setRunPreview((current) =>
                  current &&
                  current.runId === preview.runId &&
                  current.sequence >= preview.sequence
                    ? current
                    : preview,
                );
              },
              // Unlike previews, traces are durable, so a doorbell for a
              // terminal run is meaningful; the card decides whether it
              // is following that run.
              onRunTrace: (trace) => {
                setRunTrace((current) =>
                  current && current.runId === trace.runId && current.seq >= trace.seq
                    ? current
                    : trace,
                );
              },
            },
            signal,
          );
          if (!signal.aborted) {
            setConnection({ label: "Reconnecting", tone: "disconnected" });
            await waitForRetry(signal);
          }
        } catch (error) {
          if (signal.aborted || (error instanceof DOMException && error.name === "AbortError")) {
            return;
          }
          if (error instanceof AuthenticationError) {
            onAuthenticationFailure(error.message);
            return;
          }
          if (error instanceof ChannelAccessError) {
            onChannelAccessFailure(error.message);
            return;
          }
          if (error instanceof ResynchronizationRequired) {
            needsSnapshot = true;
            setConnection({ label: "Resynchronizing", tone: "connecting" });
            continue;
          }
          setConnection({ label: "Reconnecting", tone: "disconnected" });
          await waitForRetry(signal);
        }
      }
    };

    void synchronize();
    return () => controller.abort();
  }, [
    active,
    onAuthenticationFailure,
    onChannelAccessFailure,
    reloadVersion,
    session,
    startMessageId,
    updateReactions,
  ]);

  return {
    connection,
    messages,
    threadMessages,
    reactionUpdates,
    runActivity,
    runPreview,
    runTrace,
    earlier,
    later,
    loadEarlier,
    loadLater,
    jumpLatest,
    updateReactions,
  };
}
