import {
  FormEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import {
  AuthenticationError,
  cancelRun,
  ChannelAccessError,
  loadRun,
  redeemEnrollment,
  submitCommand,
} from "./api";
import type {
  CommandSubmissionResult,
  ConnectionState,
  TimelineMessage,
  WorkshopRun,
  WorkshopSession,
} from "./types";
import { CHANNEL_PATTERN } from "./types";
import { useWorkshopTimeline } from "./useWorkshopTimeline";

const SESSION_KEY = "kai.workshop.read-session.v1";
const ACTIVE_RUN_KEY = "kai.workshop.active-run.v1";
const TIMELINE_FOLLOW_DISTANCE_PX = 96;

function restoreSession(): WorkshopSession | null {
  try {
    const stored: unknown = JSON.parse(sessionStorage.getItem(SESSION_KEY) ?? "null");
    if (
      typeof stored === "object" &&
      stored !== null &&
      "token" in stored &&
      "channelId" in stored &&
      typeof stored.token === "string" &&
      stored.token.length > 0 &&
      typeof stored.channelId === "string" &&
      CHANNEL_PATTERN.test(stored.channelId)
    ) {
      return { channelId: stored.channelId, token: stored.token };
    }
  } catch {
    // Malformed tab-local state has no authority.
  }
  sessionStorage.removeItem(SESSION_KEY);
  return null;
}

function storeSession(session: WorkshopSession): void {
  sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

function forgetStoredSession(): void {
  sessionStorage.removeItem(SESSION_KEY);
  sessionStorage.removeItem(ACTIVE_RUN_KEY);
}

function restoreActiveRunId(channelId: string): string | null {
  try {
    const stored: unknown = JSON.parse(sessionStorage.getItem(ACTIVE_RUN_KEY) ?? "null");
    if (
      typeof stored === "object" &&
      stored !== null &&
      "channelId" in stored &&
      "runId" in stored &&
      stored.channelId === channelId &&
      typeof stored.runId === "string" &&
      stored.runId.startsWith("run_")
    ) {
      return stored.runId;
    }
  } catch {
    // Malformed tab-local state has no authority.
  }
  sessionStorage.removeItem(ACTIVE_RUN_KEY);
  return null;
}

function storeActiveRun(channelId: string, runId: string): void {
  sessionStorage.setItem(ACTIVE_RUN_KEY, JSON.stringify({ channelId, runId }));
}

type ActiveWorkshopRun = WorkshopRun & { status: "accepted" | "started" };

function isRunActive(run: WorkshopRun | null): run is ActiveWorkshopRun {
  return run?.status === "accepted" || run?.status === "started";
}

function isNearTimelineBottom(element: HTMLDivElement): boolean {
  return (
    element.scrollHeight - element.scrollTop - element.clientHeight <=
    TIMELINE_FOLLOW_DISTANCE_PX
  );
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return "";
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function EnrollmentView({
  existingSession,
  initialChannelId,
  notice,
  onForget,
  onOpen,
}: {
  existingSession: boolean;
  initialChannelId: string;
  notice: string | null;
  onForget: () => void;
  onOpen: (input: {
    channelId: string;
    deviceDisplayName: string;
    enrollmentToken: string;
  }) => Promise<void>;
}): React.JSX.Element {
  const [channelId, setChannelId] = useState(initialChannelId);
  const [deviceDisplayName, setDeviceDisplayName] = useState("Workshop browser");
  const [enrollmentToken, setEnrollmentToken] = useState("");
  const [error, setError] = useState<string | null>(notice);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setError(null);
    const normalizedChannelId = channelId.trim();
    if (!CHANNEL_PATTERN.test(normalizedChannelId)) {
      setError("Enter the complete Workshop channel ID supplied by the operator.");
      return;
    }
    if (
      !existingSession &&
      (!deviceDisplayName.trim() || !enrollmentToken.trim())
    ) {
      setError("Device name, channel ID, and enrollment token are required.");
      return;
    }
    setBusy(true);
    try {
      await onOpen({
        channelId: normalizedChannelId,
        deviceDisplayName: deviceDisplayName.trim(),
        enrollmentToken: enrollmentToken.trim(),
      });
    } catch (caught) {
      setEnrollmentToken("");
      setError(caught instanceof Error ? caught.message : "Enrollment failed.");
      setBusy(false);
    }
  };

  return (
    <main className="enrollment-page">
      <div className="enrollment-glow" aria-hidden="true" />
      <header className="enrollment-brand">
        <span className="brand-mark">K</span>
        <span>Kai Workshop</span>
        <span className="preview-badge">Workshop preview</span>
      </header>

      <section className="enrollment-layout">
        <div className="enrollment-intro">
          <p className="overline">A shared place for meaningful work</p>
          <h1>People and agents, working in the same room.</h1>
          <p className="lede">
            Open the canonical conversation already shared with Telegram. Your
            history and live updates come from Kai—not from this browser.
          </p>

          <ol className="threshold-steps" aria-label="How Workshop connects">
            <li>
              <span>01</span>
              <div>
                <strong>Enroll this device</strong>
                <p>Use a short-lived grant issued by the Kai operator.</p>
              </div>
            </li>
            <li>
              <span>02</span>
              <div>
                <strong>Join the channel</strong>
                <p>Read one canonical history across every client.</p>
              </div>
            </li>
            <li>
              <span>03</span>
              <div>
                <strong>Stay in step</strong>
                <p>New human and agent messages arrive live.</p>
              </div>
            </li>
          </ol>
        </div>

        <div className="enrollment-card">
          <div className="card-heading">
            <span className="section-number">01</span>
            <div>
              <p className="overline">Secure client enrollment</p>
              <h2>Open a Workshop channel</h2>
            </div>
          </div>

          {existingSession ? (
            <p className="session-hint">
              Enrollment is complete for this tab. Correct the channel ID, or
              forget the session and enroll again.
            </p>
          ) : (
            <p className="card-copy">
              The session credential remains in this tab only and is never
              written to the URL or permanent browser storage.
            </p>
          )}

          <form onSubmit={(event) => void submit(event)} noValidate>
            {!existingSession && (
              <>
                <label htmlFor="device-name">Device name</label>
                <input
                  id="device-name"
                  name="device-name"
                  value={deviceDisplayName}
                  onChange={(event) => setDeviceDisplayName(event.target.value)}
                  maxLength={100}
                  autoComplete="off"
                  required
                />

                <label htmlFor="enrollment-token">Enrollment token</label>
                <input
                  id="enrollment-token"
                  name="enrollment-token"
                  type="password"
                  value={enrollmentToken}
                  onChange={(event) => setEnrollmentToken(event.target.value)}
                  maxLength={512}
                  placeholder="kai_ws_enroll_v1.…"
                  autoComplete="off"
                  spellCheck={false}
                  required
                />
              </>
            )}

            <label htmlFor="channel-id">Channel ID</label>
            <input
              id="channel-id"
              name="channel-id"
              value={channelId}
              onChange={(event) => setChannelId(event.target.value)}
              maxLength={64}
              placeholder="chn_…"
              autoComplete="off"
              spellCheck={false}
              required
            />

            <div className="form-actions">
              <button className="primary-button" type="submit" disabled={busy}>
                {busy ? "Opening…" : "Open channel"}
              </button>
              {existingSession && (
                <button className="quiet-button" type="button" onClick={onForget}>
                  Forget session
                </button>
              )}
            </div>
            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
          </form>
        </div>
      </section>

      <footer className="enrollment-footer">
        <span>Server-authoritative collaboration</span>
        <span>·</span>
        <span>Telegram and Workshop in one history</span>
      </footer>
    </main>
  );
}

function ConnectionIndicator({
  connection,
}: {
  connection: ConnectionState;
}): React.JSX.Element {
  return (
    <span className={`connection-indicator ${connection.tone}`} role="status">
      <span className="connection-dot" aria-hidden="true" />
      {connection.label}
    </span>
  );
}

function MessageItem({ message }: { message: TimelineMessage }): React.JSX.Element {
  const isAgent = message.authorKind === "agent";
  const displayName = message.authorDisplayName || "Unknown author";
  return (
    <li className={`message-row ${isAgent ? "agent" : "human"}`}>
      <span className="message-avatar" aria-hidden="true">
        {displayName.slice(0, 1).toUpperCase()}
      </span>
      <article>
        <header className="message-meta">
          <strong>{displayName}</strong>
          <time dateTime={message.createdAt}>
            {formatTimestamp(message.createdAt)}
          </time>
        </header>
        <p>{message.body}</p>
      </article>
    </li>
  );
}

function createClientMessageId(): string {
  if (typeof globalThis.crypto?.getRandomValues !== "function") {
    throw new Error("This browser cannot create secure command identities.");
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  const token = Array.from(bytes, (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
  return `browser-${token}`;
}

function WorkshopView({
  channelId,
  connection,
  messages,
  onForget,
  onCancelRun,
  onLoadRun,
  onSubmitCommand,
}: {
  channelId: string;
  connection: ConnectionState;
  messages: TimelineMessage[];
  onForget: () => void;
  onCancelRun: (runId: string) => Promise<WorkshopRun>;
  onLoadRun: (runId: string) => Promise<WorkshopRun>;
  onSubmitCommand: (
    clientMessageId: string,
    body: string,
  ) => Promise<CommandSubmissionResult>;
}): React.JSX.Element {
  const [draft, setDraft] = useState("");
  const [pendingMessageId, setPendingMessageId] = useState<string | null>(null);
  const [submissionError, setSubmissionError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [activeRun, setActiveRun] = useState<WorkshopRun | null>(null);
  const [unseenMessageCount, setUnseenMessageCount] = useState(0);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const timelineChannelRef = useRef(channelId);
  const timelineInitializedRef = useRef(false);
  const timelineFollowRef = useRef(true);
  const latestMessagePositionRef = useRef(0);
  const agentName =
    messages.find((message) => message.authorKind === "agent")
      ?.authorDisplayName || "Agent";
  const humanName =
    messages.find((message) => message.authorKind === "human")
      ?.authorDisplayName || "You";

  useEffect(() => {
    const restoredRunId = restoreActiveRunId(channelId);
    if (!restoredRunId) {
      return;
    }
    let cancelled = false;
    void onLoadRun(restoredRunId)
      .then((run) => {
        if (!cancelled) {
          setActiveRun(run);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          sessionStorage.removeItem(ACTIVE_RUN_KEY);
          setSubmissionError(
            caught instanceof Error ? caught.message : "Could not restore the active run.",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [channelId, onLoadRun]);

  useEffect(() => {
    if (!isRunActive(activeRun)) {
      if (activeRun) {
        sessionStorage.removeItem(ACTIVE_RUN_KEY);
      }
      return;
    }
    storeActiveRun(channelId, activeRun.runId);
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void onLoadRun(activeRun.runId)
        .then((run) => {
          if (!cancelled) {
            setActiveRun(run);
          }
        })
        .catch((caught: unknown) => {
          if (!cancelled) {
            setSubmissionError(
              caught instanceof Error ? caught.message : "Could not inspect the active run.",
            );
          }
        });
    }, 1000);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [activeRun, channelId, onLoadRun]);

  useLayoutEffect(() => {
    const timeline = timelineRef.current;
    if (!timeline) {
      return;
    }
    if (timelineChannelRef.current !== channelId) {
      timelineChannelRef.current = channelId;
      timelineInitializedRef.current = false;
      timelineFollowRef.current = true;
      latestMessagePositionRef.current = 0;
      setUnseenMessageCount(0);
    }

    const latestPosition = messages.reduce(
      (position, message) => Math.max(position, message.eventPosition),
      0,
    );
    if (!timelineInitializedRef.current) {
      if (messages.length === 0) {
        return;
      }
      timeline.scrollTop = timeline.scrollHeight;
      timelineInitializedRef.current = true;
      timelineFollowRef.current = true;
      latestMessagePositionRef.current = latestPosition;
      setUnseenMessageCount(0);
      return;
    }

    const addedMessages = messages.filter(
      (message) => message.eventPosition > latestMessagePositionRef.current,
    ).length;
    latestMessagePositionRef.current = Math.max(
      latestMessagePositionRef.current,
      latestPosition,
    );
    if (addedMessages === 0) {
      return;
    }
    if (timelineFollowRef.current) {
      timeline.scrollTop = timeline.scrollHeight;
      setUnseenMessageCount(0);
    } else {
      setUnseenMessageCount((count) => count + addedMessages);
    }
  }, [channelId, messages]);

  const handleTimelineScroll = (): void => {
    const timeline = timelineRef.current;
    if (!timeline || !timelineInitializedRef.current) {
      return;
    }
    const shouldFollow = isNearTimelineBottom(timeline);
    timelineFollowRef.current = shouldFollow;
    if (shouldFollow) {
      setUnseenMessageCount(0);
    }
  };

  const followLatestMessage = (): void => {
    const timeline = timelineRef.current;
    if (!timeline) {
      return;
    }
    timelineFollowRef.current = true;
    timeline.scrollTop = timeline.scrollHeight;
    setUnseenMessageCount(0);
  };

  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const body = draft.trim();
    if (!body || submitting || isRunActive(activeRun)) {
      return;
    }
    setSubmissionError(null);
    setSubmitting(true);
    try {
      const clientMessageId = pendingMessageId ?? createClientMessageId();
      setPendingMessageId(clientMessageId);
      const result = await onSubmitCommand(clientMessageId, body);
      setDraft("");
      setPendingMessageId(null);
      setActiveRun(result.run);
    } catch (caught) {
      setSubmissionError(
        caught instanceof Error ? caught.message : "Kai could not run this command.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  const stopRun = async (): Promise<void> => {
    if (!activeRun || !isRunActive(activeRun) || stopping) {
      return;
    }
    setSubmissionError(null);
    setStopping(true);
    try {
      setActiveRun(await onCancelRun(activeRun.runId));
    } catch (caught) {
      setSubmissionError(
        caught instanceof Error ? caught.message : "Could not stop this run.",
      );
    } finally {
      setStopping(false);
    }
  };

  return (
    <main className="workshop-app">
      <aside className="workspace-rail" aria-label="Workshop switcher">
        <span className="rail-brand">K</span>
        <button className="rail-item active" type="button" aria-label="Current Workshop">
          WS
        </button>
        <span className="rail-spacer" />
        <span className="rail-status" title="Kai connected" aria-label="Kai connected" />
      </aside>

      <aside className="channel-sidebar" aria-label="Workshop navigation">
        <header className="sidebar-header">
          <div>
            <p className="overline">Kai Workshop</p>
            <h1>Current Workshop</h1>
          </div>
          <span className="read-only-chip">Connected</span>
        </header>

        <nav>
          <p className="nav-heading">Channels</p>
          <button className="channel-link active" type="button">
            <span>#</span>
            <span>conversation</span>
            <span className="live-pip" aria-label="Live" />
          </button>

          <p className="nav-heading">Agents</p>
          <button className="agent-link" type="button">
            <span className="mini-avatar">
              {agentName.slice(0, 1).toUpperCase()}
            </span>
            <span>
              <strong>{agentName}</strong>
              <small>coding agent</small>
            </span>
          </button>
        </nav>

        <footer className="sidebar-footer">
          <span className="mini-avatar human">
            {humanName.slice(0, 1).toUpperCase()}
          </span>
          <span>
            <strong>{humanName}</strong>
            <small>Human collaborator</small>
          </span>
        </footer>
      </aside>

      <section className="conversation-pane">
        <header className="conversation-header">
          <div>
            <p className="breadcrumbs">Current Workshop / Channels</p>
            <h2># conversation</h2>
          </div>
          <div className="conversation-actions">
            <ConnectionIndicator connection={connection} />
            <button className="quiet-button" type="button" onClick={onForget}>
              Forget session
            </button>
          </div>
        </header>

        <div
          ref={timelineRef}
          className="timeline-wrap"
          aria-label="Conversation timeline"
          onScroll={handleTimelineScroll}
        >
          <div className="channel-introduction">
            <span className="channel-symbol">#</span>
            <div>
              <p className="overline">Canonical conversation</p>
              <h3>Welcome to this channel</h3>
              <p>
                This channel is shared across Workshop and Telegram. Messages
                below come from Kai’s durable conversation history.
              </p>
            </div>
          </div>

          {messages.length === 0 ? (
            <p className="empty-timeline">No messages yet. New activity will appear here.</p>
          ) : (
            <ol className="message-list" aria-live="polite">
              {messages.map((message) => (
                <MessageItem key={message.messageId} message={message} />
              ))}
            </ol>
          )}
        </div>

        <footer className="composer-preview">
          {unseenMessageCount > 0 && (
            <button
              className="new-messages-button"
              type="button"
              onClick={followLatestMessage}
            >
              {unseenMessageCount === 1
                ? "1 new message"
                : `${unseenMessageCount} new messages`}
            </button>
          )}
          <form className="composer-form" onSubmit={(event) => void submit(event)}>
            <textarea
              aria-label="Message Kai"
              value={draft}
              onChange={(event) => {
                setDraft(event.target.value);
                if (!submitting) {
                  setPendingMessageId(null);
                }
              }}
              maxLength={50000}
              placeholder="Message Kai…"
              rows={3}
            />
            <button
              type="submit"
              disabled={submitting || isRunActive(activeRun) || !draft.trim()}
            >
              {submitting ? "Sending…" : "Send"}
            </button>
            {isRunActive(activeRun) && (
              <button
                className="stop-button"
                type="button"
                disabled={stopping}
                onClick={() => void stopRun()}
              >
                {stopping ? "Stopping…" : "Stop"}
              </button>
            )}
          </form>
          {submissionError && (
            <p className="composer-error" role="alert">{submissionError}</p>
          )}
          <span className="composer-mode" role="status">
            {activeRun
              ? `Workshop run: ${activeRun.status}`
              : "Canonical Workshop command"}
          </span>
        </footer>
      </section>

      <aside className="context-pane" aria-label="Channel context">
        <header>
          <p className="overline">Channel context</p>
          <h2># conversation</h2>
        </header>

        <section className="context-section">
          <span className="section-number">01</span>
          <h3>Connection</h3>
          <ConnectionIndicator connection={connection} />
          <p>History and new messages are synchronized directly with Kai.</p>
        </section>

        <section className="context-section">
          <span className="section-number">02</span>
          <h3>Canonical identity</h3>
          <code title={channelId}>{channelId}</code>
          <p>The channel—not a Telegram chat—is the collaboration boundary.</p>
        </section>

        <section className="context-section future-section">
          <span className="section-number">03</span>
          <h3>Coming into view</h3>
          <ul>
            <li>Threads and run inspection</li>
            <li>Agent activity and approvals</li>
            <li>Projects and shared artifacts</li>
          </ul>
        </section>
      </aside>
    </main>
  );
}

export default function App(): React.JSX.Element {
  const [session, setSession] = useState<WorkshopSession | null>(() =>
    restoreSession(),
  );
  const [view, setView] = useState<"enrollment" | "workshop">(() =>
    sessionStorage.getItem(SESSION_KEY) ? "workshop" : "enrollment",
  );
  const [notice, setNotice] = useState<string | null>(null);

  const forgetSession = useCallback((message: string | null = null): void => {
    forgetStoredSession();
    setSession(null);
    setNotice(message);
    setView("enrollment");
  }, []);

  const correctChannel = useCallback((message: string): void => {
    setNotice(message);
    setView("enrollment");
  }, []);

  const handleAuthenticationFailure = useCallback(
    (message: string): void => forgetSession(message),
    [forgetSession],
  );

  const { connection, messages } = useWorkshopTimeline(
    session,
    view === "workshop",
    handleAuthenticationFailure,
    correctChannel,
  );

  const openChannel = async ({
    channelId,
    deviceDisplayName,
    enrollmentToken,
  }: {
    channelId: string;
    deviceDisplayName: string;
    enrollmentToken: string;
  }): Promise<void> => {
    const token =
      session?.token ??
      (await redeemEnrollment(enrollmentToken, deviceDisplayName));
    const nextSession = { channelId, token };
    storeSession(nextSession);
    setSession(nextSession);
    setNotice(null);
    setView("workshop");
  };

  const runCommand = async (
    clientMessageId: string,
    body: string,
  ): Promise<CommandSubmissionResult> => {
    if (!session) {
      throw new Error("Workshop session unavailable.");
    }
    try {
      return await submitCommand(session, clientMessageId, body);
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        forgetSession(caught.message);
      } else if (caught instanceof ChannelAccessError) {
        correctChannel(caught.message);
      }
      throw caught;
    }
  };

  const inspectRun = useCallback(
    async (runId: string): Promise<WorkshopRun> => {
      if (!session) {
        throw new Error("Workshop session unavailable.");
      }
      try {
        return await loadRun(session, runId);
      } catch (caught) {
        if (caught instanceof AuthenticationError) {
          forgetSession(caught.message);
        } else if (caught instanceof ChannelAccessError) {
          correctChannel(caught.message);
        }
        throw caught;
      }
    },
    [correctChannel, forgetSession, session],
  );

  const stopRun = useCallback(
    async (runId: string): Promise<WorkshopRun> => {
      if (!session) {
        throw new Error("Workshop session unavailable.");
      }
      try {
        return await cancelRun(session, runId);
      } catch (caught) {
        if (caught instanceof AuthenticationError) {
          forgetSession(caught.message);
        } else if (caught instanceof ChannelAccessError) {
          correctChannel(caught.message);
        }
        throw caught;
      }
    },
    [correctChannel, forgetSession, session],
  );

  if (view === "enrollment") {
    return (
      <EnrollmentView
        key={`${session ? "correction" : "fresh"}:${notice ?? ""}`}
        existingSession={session !== null}
        initialChannelId={session?.channelId ?? ""}
        notice={notice}
        onForget={() => forgetSession()}
        onOpen={openChannel}
      />
    );
  }

  return (
    <WorkshopView
      channelId={session?.channelId ?? ""}
      connection={connection}
      messages={messages}
      onForget={() => forgetSession()}
      onCancelRun={stopRun}
      onLoadRun={inspectRun}
      onSubmitCommand={runCommand}
    />
  );
}
