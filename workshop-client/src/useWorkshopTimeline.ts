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
} {
  const [messages, setMessages] = useState<TimelineMessage[]>([]);
  const [runActivity, setRunActivity] = useState<WorkshopRunActivity | null>(null);
  const [connection, setConnection] = useState<ConnectionState>({
    label: "Waiting",
    tone: "connecting",
  });

  useEffect(() => {
    if (!active || !session) {
      setMessages([]);
      setRunActivity(null);
      setConnection({ label: "Waiting", tone: "connecting" });
      return;
    }

    const controller = new AbortController();
    const { signal } = controller;
    setRunActivity(null);
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
                setMessages((current) => appendUnique(current, message));
              },
              onRunActivity: (activity, eventId) => {
                lastEventId = eventId;
                setRunActivity(activity);
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

  return { connection, messages, runActivity };
}
