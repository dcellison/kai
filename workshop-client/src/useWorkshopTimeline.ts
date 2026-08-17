import { useEffect, useState } from "react";

import {
  AuthenticationError,
  ChannelAccessError,
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

export function useWorkshopTimeline(
  session: WorkshopSession | null,
  active: boolean,
  onAuthenticationFailure: (message: string) => void,
  onChannelAccessFailure: (message: string) => void,
): {
  connection: ConnectionState;
  messages: TimelineMessage[];
  runActivity: WorkshopRunActivity | null;
  runPreview: WorkshopRunPreview | null;
  runTrace: WorkshopRunTraceSignal | null;
} {
  const [messages, setMessages] = useState<TimelineMessage[]>([]);
  const [runActivity, setRunActivity] = useState<WorkshopRunActivity | null>(null);
  const [runPreview, setRunPreview] = useState<WorkshopRunPreview | null>(null);
  const [runTrace, setRunTrace] = useState<WorkshopRunTraceSignal | null>(null);
  const [connection, setConnection] = useState<ConnectionState>({
    label: "Waiting",
    tone: "connecting",
  });

  useEffect(() => {
    if (!active || !session) {
      setMessages([]);
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
            setMessages(snapshot.messages);
            needsSnapshot = false;
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
                setMessages((current) => appendUnique(current, message));
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

  return { connection, messages, runActivity, runPreview, runTrace };
}
