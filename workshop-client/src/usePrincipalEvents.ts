import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  AuthenticationError,
  ResynchronizationRequired,
  streamPrincipalEvents,
} from "./api";
import type {
  ConnectionState,
  WorkshopPrincipalEventBatch,
} from "./types";

const RECONNECT_DELAY_MS = 2000;

export type WorkshopPrincipalEvent =
  | { kind: "batch"; batch: WorkshopPrincipalEventBatch }
  | { kind: "synchronize" };

export interface WorkshopPrincipalEvents {
  connection: ConnectionState;
  subscribe: (listener: (event: WorkshopPrincipalEvent) => void) => () => void;
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

export function usePrincipalEvents(
  token: string,
  onAuthenticationFailure: (message: string) => void,
): WorkshopPrincipalEvents {
  const [connection, setConnection] = useState<ConnectionState>({
    label: "Connecting",
    tone: "connecting",
  });
  const synchronizedRef = useRef(false);
  const listenersRef = useRef(new Set<(event: WorkshopPrincipalEvent) => void>());
  const authenticationFailureRef = useRef(onAuthenticationFailure);
  authenticationFailureRef.current = onAuthenticationFailure;

  const publish = useCallback((event: WorkshopPrincipalEvent): void => {
    for (const listener of listenersRef.current) listener(event);
  }, []);

  const subscribe = useCallback((listener: (event: WorkshopPrincipalEvent) => void): (() => void) => {
    listenersRef.current.add(listener);
    if (synchronizedRef.current) listener({ kind: "synchronize" });
    return () => listenersRef.current.delete(listener);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let lastEventId: string | null = null;
    let awaitingCheckpoint = true;
    const synchronize = async (): Promise<void> => {
      while (!controller.signal.aborted) {
        try {
          synchronizedRef.current = false;
          setConnection({ label: "Connecting", tone: "connecting" });
          await streamPrincipalEvents(
            token,
            lastEventId,
            {
              onBatch: (batch, eventId) => {
                lastEventId = eventId;
                if (awaitingCheckpoint) {
                  awaitingCheckpoint = false;
                  synchronizedRef.current = true;
                  publish({ kind: "synchronize" });
                }
                publish({ batch, kind: "batch" });
              },
              onConnected: () => {
                setConnection({ label: "Live", tone: "connected" });
              },
            },
            controller.signal,
          );
        } catch (caught) {
          if (controller.signal.aborted) return;
          synchronizedRef.current = false;
          setConnection({ label: "Connecting", tone: "connecting" });
          if (caught instanceof AuthenticationError) {
            authenticationFailureRef.current(caught.message);
            return;
          }
          if (caught instanceof ResynchronizationRequired) {
            lastEventId = null;
            awaitingCheckpoint = true;
          }
        }
        await waitForRetry(controller.signal);
      }
    };
    void synchronize();
    return () => {
      synchronizedRef.current = false;
      controller.abort();
    };
  }, [publish, token]);

  return useMemo(() => ({ connection, subscribe }), [connection, subscribe]);
}
