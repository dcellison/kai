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

export interface EarlierHistoryState {
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
  onAuthenticationFailure: (message: string) => void,
  onChannelAccessFailure: (message: string) => void,
): {
  connection: ConnectionState;
  messages: TimelineMessage[];
  threadMessages: TimelineMessage[];
  runActivity: WorkshopRunActivity | null;
  runPreview: WorkshopRunPreview | null;
  runTrace: WorkshopRunTraceSignal | null;
  earlier: EarlierHistoryState;
  loadEarlier: () => void;
} {
  const [messages, setMessages] = useState<TimelineMessage[]>([]);
  const [threadMessages, setThreadMessages] = useState<TimelineMessage[]>([]);
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
  const earlierContextRef = useRef<EarlierFetchContext | null>(null);
  const earlierCursorRef = useRef<string | null>(null);
  const earlierLoadingRef = useRef(false);
  const generationRef = useRef(0);

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

  useEffect(() => {
    if (!active || !session) {
      generationRef.current += 1;
      earlierContextRef.current = null;
      earlierCursorRef.current = null;
      earlierLoadingRef.current = false;
      setEarlier({ available: false, loading: false, error: null });
      setMessages([]);
      setThreadMessages([]);
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
            const snapshot = await loadTimeline(session, signal);
            if (signal.aborted) {
              return;
            }
            lastEventId = String(snapshot.throughPosition);
            knownMessageIds = new Set(
              snapshot.messages.map((message) => message.messageId),
            );
            setMessages(snapshot.messages.filter((message) => message.threadRootId === null));
            setThreadMessages([]);
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
            earlierCursorRef.current = snapshot.previousCursor;
            earlierLoadingRef.current = false;
            setEarlier({
              available: snapshot.previousCursor !== null,
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
    session,
  ]);

  return {
    connection,
    messages,
    threadMessages,
    runActivity,
    runPreview,
    runTrace,
    earlier,
    loadEarlier,
  };
}
