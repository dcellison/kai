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
  loadNavigation,
  loadRun,
  redeemEnrollment,
  submitCommand,
} from "./api";
import type {
  CommandSubmissionResult,
  ConnectionState,
  TimelineMessage,
  WorkshopRun,
  WorkshopRunActivity,
  WorkshopChannelSummary,
  WorkshopNavigation,
  WorkshopSession,
  WorkshopSummary,
} from "./types";
import { CHANNEL_PATTERN } from "./types";
import { useWorkshopTimeline } from "./useWorkshopTimeline";

const SESSION_KEY = "kai.workshop.read-session.v1";
const ACTIVE_RUN_KEY = "kai.workshop.active-run.v1";
const DRAFTS_KEY = "kai.workshop.drafts.v1";
const VIEWPORTS_KEY = "kai.workshop.timeline-viewports.v1";
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
  sessionStorage.removeItem(DRAFTS_KEY);
  sessionStorage.removeItem(VIEWPORTS_KEY);
}

function restoreActiveRunId(channelId: string): string | null {
  try {
    const stored: unknown = JSON.parse(sessionStorage.getItem(ACTIVE_RUN_KEY) ?? "null");
    if (
      typeof stored === "object" &&
      stored !== null &&
      channelId in stored &&
      typeof (stored as Record<string, unknown>)[channelId] === "string" &&
      String((stored as Record<string, unknown>)[channelId]).startsWith("run_")
    ) {
      return String((stored as Record<string, unknown>)[channelId]);
    }
  } catch {
    // Malformed tab-local state has no authority.
  }
  return null;
}

function storeActiveRun(channelId: string, runId: string | null): void {
  let stored: Record<string, string> = {};
  try {
    const value: unknown = JSON.parse(sessionStorage.getItem(ACTIVE_RUN_KEY) ?? "{}");
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      stored = Object.fromEntries(
        Object.entries(value).filter((entry): entry is [string, string] =>
          typeof entry[1] === "string" && entry[1].startsWith("run_"),
        ),
      );
    }
  } catch {
    // Replace malformed tab-local state with the current channel state.
  }
  if (runId) {
    stored[channelId] = runId;
  } else {
    delete stored[channelId];
  }
  sessionStorage.setItem(ACTIVE_RUN_KEY, JSON.stringify(stored));
}

function restoreDraft(channelId: string): string {
  try {
    const stored: unknown = JSON.parse(sessionStorage.getItem(DRAFTS_KEY) ?? "{}");
    if (typeof stored === "object" && stored !== null && channelId in stored) {
      const draft = (stored as Record<string, unknown>)[channelId];
      return typeof draft === "string" && draft.length <= 50000 ? draft : "";
    }
  } catch {
    // Malformed tab-local state has no authority.
  }
  return "";
}

function storeDraft(channelId: string, draft: string): void {
  let stored: Record<string, string> = {};
  try {
    const value: unknown = JSON.parse(sessionStorage.getItem(DRAFTS_KEY) ?? "{}");
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      stored = Object.fromEntries(
        Object.entries(value).filter((entry): entry is [string, string] =>
          typeof entry[1] === "string" && entry[1].length <= 50000,
        ),
      );
    }
  } catch {
    // Replace malformed tab-local state with the current channel draft.
  }
  if (draft) {
    stored[channelId] = draft;
  } else {
    delete stored[channelId];
  }
  sessionStorage.setItem(DRAFTS_KEY, JSON.stringify(stored));
}

interface StoredTimelineViewport {
  follow: boolean;
  scrollTop: number;
}

function restoreTimelineViewport(channelId: string): StoredTimelineViewport | null {
  try {
    const stored: unknown = JSON.parse(sessionStorage.getItem(VIEWPORTS_KEY) ?? "{}");
    const viewport =
      typeof stored === "object" && stored !== null
        ? (stored as Record<string, unknown>)[channelId]
        : null;
    if (
      typeof viewport === "object" &&
      viewport !== null &&
      "follow" in viewport &&
      "scrollTop" in viewport &&
      typeof viewport.follow === "boolean" &&
      typeof viewport.scrollTop === "number" &&
      Number.isFinite(viewport.scrollTop) &&
      viewport.scrollTop >= 0
    ) {
      return { follow: viewport.follow, scrollTop: viewport.scrollTop };
    }
  } catch {
    // Malformed tab-local state has no authority.
  }
  return null;
}

function storeTimelineViewport(
  channelId: string,
  viewport: StoredTimelineViewport,
): void {
  let stored: Record<string, StoredTimelineViewport> = {};
  try {
    const value: unknown = JSON.parse(sessionStorage.getItem(VIEWPORTS_KEY) ?? "{}");
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      stored = value as Record<string, StoredTimelineViewport>;
    }
  } catch {
    // Replace malformed tab-local state with the current channel viewport.
  }
  stored[channelId] = viewport;
  sessionStorage.setItem(VIEWPORTS_KEY, JSON.stringify(stored));
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

function formatRunDuration(run: WorkshopRun, now: number): string {
  const started = new Date(run.startedAt ?? run.acceptedAt).valueOf();
  const ended = run.terminalAt ? new Date(run.terminalAt).valueOf() : now;
  if (!Number.isFinite(started) || !Number.isFinite(ended)) {
    return "Duration unavailable";
  }
  const totalSeconds = Math.max(0, Math.floor((ended - started) / 1000));
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds}s`;
}

function runStatusCopy(run: WorkshopRun): string {
  if (run.status === "accepted") {
    return run.cancellationRequestedAt
      ? "Stopping when the agent reaches a safe boundary."
      : "Queued for the configured agent.";
  }
  if (run.status === "started") {
    return run.cancellationRequestedAt
      ? "Stopping when the agent reaches a safe boundary."
      : "The agent is working on this request.";
  }
  if (run.status === "completed") {
    return "The agent completed this request.";
  }
  if (run.status === "cancelled") {
    return "This request was cancelled.";
  }
  const actionableFailures: Record<string, string> = {
    authentication_expired: "Sign in to the configured backend again, then retry.",
    authentication_required: "Sign in to the configured backend, then retry.",
    backend_crashed: "The agent process stopped unexpectedly. Retry this request.",
    execution_interrupted: "Kai was interrupted while the agent was working. Retry this request.",
    model_unavailable: "The configured model is unavailable. Choose an available model, then retry.",
    no_response: "The agent ended without a response. Retry this request.",
    provider_unavailable: "The provider is temporarily unavailable. Try again later.",
    quota_exhausted: "The configured account has no usage allowance remaining.",
    transient: "The provider failed temporarily. Try this request again.",
  };
  return (
    (run.terminalCode && actionableFailures[run.terminalCode]) ||
    "The agent could not complete this request. Retry it or ask the operator to inspect Kai."
  );
}

function EnrollmentView({
  existingSession,
  notice,
  onForget,
  onOpen,
}: {
  existingSession: boolean;
  notice: string | null;
  onForget: () => void;
  onOpen: (input: {
    deviceDisplayName: string;
    enrollmentToken: string;
  }) => Promise<void>;
}): React.JSX.Element {
  const [deviceDisplayName, setDeviceDisplayName] = useState("Workshop browser");
  const [enrollmentToken, setEnrollmentToken] = useState("");
  const [error, setError] = useState<string | null>(notice);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setError(null);
    if (
      !existingSession &&
      (!deviceDisplayName.trim() || !enrollmentToken.trim())
    ) {
      setError("Device name and enrollment token are required.");
      return;
    }
    setBusy(true);
    try {
      await onOpen({
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
              Enrollment is complete for this tab. Retry Workshop discovery,
              or forget the session and enroll again.
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

            <div className="form-actions">
              <button className="primary-button" type="submit" disabled={busy}>
                {busy ? "Opening…" : "Open Workshop"}
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

function channelDisplayName(channel: WorkshopChannelSummary): string {
  const name = channel.name?.trim();
  if (name) {
    return name;
  }
  if (channel.kind === "notification") {
    return "Notifications";
  }
  if (channel.kind === "group") {
    return "Group";
  }
  return "Conversation";
}

function workshopInitials(name: string): string {
  const initials = name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");
  return initials || "WS";
}

function WorkshopView({
  channel,
  connection,
  messages,
  navigation,
  runActivity,
  workshop,
  onForget,
  onCancelRun,
  onLoadRun,
  onSelectChannel,
  onSelectWorkshop,
  onSubmitCommand,
}: {
  channel: WorkshopChannelSummary;
  connection: ConnectionState;
  messages: TimelineMessage[];
  navigation: WorkshopNavigation;
  runActivity: WorkshopRunActivity | null;
  workshop: WorkshopSummary;
  onForget: () => void;
  onCancelRun: (runId: string) => Promise<WorkshopRun>;
  onLoadRun: (runId: string) => Promise<WorkshopRun>;
  onSelectChannel: (channelId: string) => void;
  onSelectWorkshop: (workshopId: string) => void;
  onSubmitCommand: (
    clientMessageId: string,
    body: string,
  ) => Promise<CommandSubmissionResult>;
}): React.JSX.Element {
  const channelId = channel.channelId;
  const channelName = channelDisplayName(channel);
  const [draft, setDraft] = useState(() => restoreDraft(channelId));
  const [pendingMessageId, setPendingMessageId] = useState<string | null>(null);
  const [submissionError, setSubmissionError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [activeRun, setActiveRun] = useState<WorkshopRun | null>(null);
  const [runClock, setRunClock] = useState(() => Date.now());
  const [unseenMessageCount, setUnseenMessageCount] = useState(0);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const timelineChannelRef = useRef(channelId);
  const timelineInitializedRef = useRef(false);
  const timelineFollowRef = useRef(true);
  const latestMessagePositionRef = useRef(0);
  const latestRunActivityRef = useRef<WorkshopRunActivity | null>(runActivity);
  const humanName = navigation.principal.displayName || "You";

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
          storeActiveRun(channelId, null);
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
    latestRunActivityRef.current = runActivity;
    if (!runActivity) {
      return;
    }
    setActiveRun(runActivity.run);
  }, [runActivity]);

  useEffect(() => {
    if (isRunActive(activeRun)) {
      storeActiveRun(channelId, activeRun.runId);
      return;
    }
    if (activeRun) {
      storeActiveRun(channelId, null);
    }
  }, [activeRun, channelId]);

  useEffect(() => {
    if (!isRunActive(activeRun)) {
      return;
    }
    setRunClock(Date.now());
    const timer = window.setInterval(() => setRunClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [activeRun]);

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
      const restoredViewport = restoreTimelineViewport(channelId);
      timeline.scrollTop = restoredViewport?.follow === false
        ? Math.min(restoredViewport.scrollTop, timeline.scrollHeight)
        : timeline.scrollHeight;
      timelineInitializedRef.current = true;
      timelineFollowRef.current = restoredViewport?.follow ?? true;
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
    storeTimelineViewport(channelId, {
      follow: shouldFollow,
      scrollTop: timeline.scrollTop,
    });
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
    storeTimelineViewport(channelId, {
      follow: true,
      scrollTop: timeline.scrollTop,
    });
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
      storeDraft(channelId, "");
      setPendingMessageId(null);
      const streamed = latestRunActivityRef.current;
      setActiveRun(
        streamed?.run.runId === result.run.runId ? streamed.run : result.run,
      );
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
        {navigation.workshops.map((availableWorkshop) => (
          <button
            className={`rail-item ${availableWorkshop.workshopId === workshop.workshopId ? "active" : ""}`}
            type="button"
            aria-label={availableWorkshop.name}
            title={availableWorkshop.name}
            onClick={() => onSelectWorkshop(availableWorkshop.workshopId)}
            key={availableWorkshop.workshopId}
          >
            {workshopInitials(availableWorkshop.name)}
          </button>
        ))}
        <span className="rail-spacer" />
        <span className="rail-status" title="Kai connected" aria-label="Kai connected" />
      </aside>

      <aside className="channel-sidebar" aria-label="Workshop navigation">
        <header className="sidebar-header">
          <div>
            <p className="overline">Kai Workshop</p>
            <h1>{workshop.name}</h1>
          </div>
          <span className="read-only-chip">{workshop.role}</span>
        </header>

        <nav>
          <p className="nav-heading">Channels</p>
          {workshop.channels
            .filter((availableChannel) => availableChannel.kind !== "notification")
            .map((availableChannel) => (
              <button
                className={`channel-link ${availableChannel.channelId === channelId ? "active" : ""}`}
                type="button"
                onClick={() => onSelectChannel(availableChannel.channelId)}
                key={availableChannel.channelId}
              >
                <span>#</span>
                <span>{channelDisplayName(availableChannel)}</span>
                {availableChannel.channelId === channelId && (
                  <span className="live-pip" aria-label="Live" />
                )}
              </button>
            ))}

          {workshop.channels.some(
            (availableChannel) => availableChannel.kind === "notification",
          ) && (
            <>
              <p className="nav-heading">Notifications</p>
              {workshop.channels
                .filter((availableChannel) => availableChannel.kind === "notification")
                .map((availableChannel) => (
                  <button
                    className={`channel-link notification ${availableChannel.channelId === channelId ? "active" : ""}`}
                    type="button"
                    onClick={() => onSelectChannel(availableChannel.channelId)}
                    key={availableChannel.channelId}
                  >
                    <span>!</span>
                    <span>{channelDisplayName(availableChannel)}</span>
                    {availableChannel.channelId === channelId && (
                      <span className="live-pip" aria-label="Live" />
                    )}
                  </button>
                ))}
            </>
          )}

          <p className="nav-heading">Agents</p>
          {channel.agents.map((agent) => (
            <div className="agent-link" key={agent.agentId}>
              <span className="mini-avatar">
                {agent.name.slice(0, 1).toUpperCase()}
              </span>
              <span>
                <strong>{agent.name}</strong>
                <small>coding agent</small>
              </span>
            </div>
          ))}
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
            <p className="breadcrumbs">{workshop.name} / {channel.kind === "notification" ? "Notifications" : "Channels"}</p>
            <h2>{channel.kind === "notification" ? "!" : "#"} {channelName}</h2>
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
              <h3>Welcome to {channelName}</h3>
              <p>
                {channel.kind === "notification"
                  ? "This outbound channel records notifications delivered by Kai."
                  : "Messages below come from Kai’s durable conversation history across every connected client."}
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
          {activeRun && (
            <section
              className={`run-activity ${activeRun.status}`}
              aria-label="Agent run activity"
              aria-live="polite"
            >
              <div>
                <p className="run-activity-title">Agent run</p>
                <p className="run-activity-copy">{runStatusCopy(activeRun)}</p>
              </div>
              <div className="run-activity-state">
                <strong>{activeRun.status}</strong>
                <span>{formatRunDuration(activeRun, runClock)}</span>
              </div>
            </section>
          )}
          {channel.canSubmitCommands ? (
            <form className="composer-form" onSubmit={(event) => void submit(event)}>
              <textarea
                aria-label="Message Kai"
                value={draft}
                onChange={(event) => {
                  const nextDraft = event.target.value;
                  setDraft(nextDraft);
                  storeDraft(channelId, nextDraft);
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
          ) : (
            <p className="read-only-channel-notice">
              This channel is outbound-only. Kai records delivery here, but it
              does not accept conversation commands.
            </p>
          )}
          {submissionError && (
            <p className="composer-error" role="alert">{submissionError}</p>
          )}
          {!activeRun && channel.canSubmitCommands && (
            <span className="composer-mode" role="status">
              Canonical Workshop command
            </span>
          )}
        </footer>
      </section>

      <aside className="context-pane" aria-label="Channel context">
        <header>
          <p className="overline">Channel context</p>
          <h2>{channel.kind === "notification" ? "!" : "#"} {channelName}</h2>
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

        <section className="context-section">
          <span className="section-number">03</span>
          <h3>Channel authority</h3>
          <p>
            {channel.canSubmitCommands
              ? "You can read this channel and submit commands to its assigned agent."
              : "You can read this outbound channel; command submission is disabled."}
          </p>
        </section>

        <section className="context-section future-section">
          <span className="section-number">04</span>
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

function findNavigationChannel(
  navigation: WorkshopNavigation,
  channelId: string,
): { channel: WorkshopChannelSummary; workshop: WorkshopSummary } | null {
  for (const workshop of navigation.workshops) {
    const channel = workshop.channels.find(
      (availableChannel) => availableChannel.channelId === channelId,
    );
    if (channel) {
      return { channel, workshop };
    }
  }
  return null;
}

function preferredNavigationChannel(
  navigation: WorkshopNavigation,
  preferredChannelId: string | null,
): { channel: WorkshopChannelSummary; workshop: WorkshopSummary } | null {
  if (preferredChannelId) {
    const preferred = findNavigationChannel(navigation, preferredChannelId);
    if (preferred) {
      return preferred;
    }
  }
  for (const workshop of navigation.workshops) {
    const channel = workshop.channels.find(
      (availableChannel) => availableChannel.canSubmitCommands,
    );
    if (channel) {
      return { channel, workshop };
    }
  }
  for (const workshop of navigation.workshops) {
    if (workshop.channels[0]) {
      return { channel: workshop.channels[0], workshop };
    }
  }
  return null;
}

function ActiveWorkshopClient({
  navigation,
  session,
  onAuthenticationFailure,
  onChannelAccessFailure,
  onForget,
  onSelectChannel,
  onSelectWorkshop,
}: {
  navigation: WorkshopNavigation;
  session: WorkshopSession;
  onAuthenticationFailure: (message: string) => void;
  onChannelAccessFailure: (message: string) => void;
  onForget: () => void;
  onSelectChannel: (channelId: string) => void;
  onSelectWorkshop: (workshopId: string) => void;
}): React.JSX.Element {
  const selected = findNavigationChannel(navigation, session.channelId);
  const { connection, messages, runActivity } = useWorkshopTimeline(
    session,
    selected !== null,
    onAuthenticationFailure,
    onChannelAccessFailure,
  );
  const withAccessHandling = useCallback(
    async <Result,>(operation: () => Promise<Result>): Promise<Result> => {
      try {
        return await operation();
      } catch (caught) {
        if (caught instanceof AuthenticationError) {
          onAuthenticationFailure(caught.message);
        } else if (caught instanceof ChannelAccessError) {
          onChannelAccessFailure(caught.message);
        }
        throw caught;
      }
    },
    [onAuthenticationFailure, onChannelAccessFailure],
  );
  const loadSelectedRun = useCallback(
    (runId: string) => withAccessHandling(() => loadRun(session, runId)),
    [session, withAccessHandling],
  );
  const cancelSelectedRun = useCallback(
    (runId: string) => withAccessHandling(() => cancelRun(session, runId)),
    [session, withAccessHandling],
  );
  const submitSelectedCommand = useCallback(
    (clientMessageId: string, body: string) =>
      withAccessHandling(() => submitCommand(session, clientMessageId, body)),
    [session, withAccessHandling],
  );
  if (!selected) {
    return <main className="loading-workshop">Workshop access changed.</main>;
  }

  return (
    <WorkshopView
      channel={selected.channel}
      connection={connection}
      messages={messages}
      navigation={navigation}
      runActivity={runActivity}
      workshop={selected.workshop}
      onForget={onForget}
      onCancelRun={cancelSelectedRun}
      onLoadRun={loadSelectedRun}
      onSelectChannel={onSelectChannel}
      onSelectWorkshop={onSelectWorkshop}
      onSubmitCommand={submitSelectedCommand}
    />
  );
}

export default function App(): React.JSX.Element {
  const [session, setSession] = useState<WorkshopSession | null>(() =>
    restoreSession(),
  );
  const [navigation, setNavigation] = useState<WorkshopNavigation | null>(null);
  const [view, setView] = useState<"enrollment" | "workshop">(() =>
    sessionStorage.getItem(SESSION_KEY) ? "workshop" : "enrollment",
  );
  const [notice, setNotice] = useState<string | null>(null);

  const forgetSession = useCallback((message: string | null = null): void => {
    forgetStoredSession();
    setSession(null);
    setNavigation(null);
    setNotice(message);
    setView("enrollment");
  }, []);

  const handleAuthenticationFailure = useCallback(
    (message: string): void => forgetSession(message),
    [forgetSession],
  );

  const adoptNavigation = useCallback(
    (
      token: string,
      discovered: WorkshopNavigation,
      preferredChannelId: string | null,
    ): void => {
      const selected = preferredNavigationChannel(discovered, preferredChannelId);
      if (!selected) {
        throw new Error("This Workshop account has no accessible channels.");
      }
      const nextSession = { channelId: selected.channel.channelId, token };
      storeSession(nextSession);
      setSession(nextSession);
      setNavigation(discovered);
      setNotice(null);
      setView("workshop");
    },
    [],
  );

  useEffect(() => {
    if (view !== "workshop" || !session || navigation) {
      return;
    }
    let cancelled = false;
    void loadNavigation(session.token)
      .then((discovered) => {
        if (!cancelled) {
          adoptNavigation(session.token, discovered, session.channelId);
        }
      })
      .catch((caught: unknown) => {
        if (cancelled) {
          return;
        }
        if (caught instanceof AuthenticationError) {
          forgetSession(caught.message);
          return;
        }
        setNotice(
          caught instanceof Error
            ? caught.message
            : "Could not load Workshop navigation.",
        );
        setView("enrollment");
      });
    return () => {
      cancelled = true;
    };
  }, [adoptNavigation, forgetSession, navigation, session, view]);

  const openChannel = async ({
    deviceDisplayName,
    enrollmentToken,
  }: {
    deviceDisplayName: string;
    enrollmentToken: string;
  }): Promise<void> => {
    const token =
      session?.token ??
      (await redeemEnrollment(enrollmentToken, deviceDisplayName));
    const discovered = await loadNavigation(token);
    adoptNavigation(token, discovered, session?.channelId ?? null);
  };

  const refreshChannelAccess = useCallback(async (message: string): Promise<void> => {
    if (!session) {
      return;
    }
    try {
      const discovered = await loadNavigation(session.token);
      adoptNavigation(session.token, discovered, session.channelId);
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        forgetSession(caught.message);
        return;
      }
      setNotice(
        caught instanceof Error ? caught.message : message,
      );
      setNavigation(null);
      setView("enrollment");
    }
  }, [adoptNavigation, forgetSession, session]);

  const selectChannel = (channelId: string): void => {
    if (!session || !navigation || !findNavigationChannel(navigation, channelId)) {
      return;
    }
    const nextSession = { ...session, channelId };
    storeSession(nextSession);
    setSession(nextSession);
  };

  const selectWorkshop = (workshopId: string): void => {
    const workshop = navigation?.workshops.find(
      (availableWorkshop) => availableWorkshop.workshopId === workshopId,
    );
    const channel =
      workshop?.channels.find((availableChannel) => availableChannel.canSubmitCommands) ??
      workshop?.channels[0];
    if (channel) {
      selectChannel(channel.channelId);
    }
  };

  if (view === "enrollment") {
    return (
      <EnrollmentView
        key={`${session ? "correction" : "fresh"}:${notice ?? ""}`}
        existingSession={session !== null}
        notice={notice}
        onForget={() => forgetSession()}
        onOpen={openChannel}
      />
    );
  }

  if (!session || !navigation) {
    return <main className="loading-workshop">Opening Kai Workshop…</main>;
  }

  return (
    <ActiveWorkshopClient
      key={session.channelId}
      navigation={navigation}
      session={session}
      onAuthenticationFailure={handleAuthenticationFailure}
      onChannelAccessFailure={(message) => void refreshChannelAccess(message)}
      onForget={() => forgetSession()}
      onSelectChannel={selectChannel}
      onSelectWorkshop={selectWorkshop}
    />
  );
}
