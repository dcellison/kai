import {
  CSSProperties,
  FormEvent,
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  AuthenticationError,
  cancelRun,
  ChannelAccessError,
  createChannel,
  dismissChannelAgent,
  loadAppearancePreferences,
  loadNavigation,
  loadNotificationPreferences,
  loadArtifactBlob,
  loadRun,
  loadRunTrace,
  loadSettingsWorkspace,
  loadThreadTimeline,
  redeemEnrollment,
  setMessageReaction,
  submitCommand,
  switchWorkspace,
} from "./api";
import type {
  CommandSubmissionResult,
  ConnectionState,
  TimelineMessage,
  ThreadTimelineSnapshot,
  WorkshopRun,
  WorkshopRunActivity,
  WorkshopRunPreview,
  WorkshopRunTracePage,
  WorkshopRunTraceSignal,
  WorkshopChannelSummary,
  WorkshopNavigation,
  WorkshopNotificationPreferences,
  WorkshopSession,
  WorkshopSettingsWorkspace,
  WorkshopSummary,
  WorkshopArtifactSummary,
  WorkshopAgentSummary,
  WorkshopAppearancePreferences,
  WorkshopReaction,
} from "./types";
import { CHANNEL_PATTERN } from "./types";
import { RunTraceCard } from "./RunTraceCard";
import { useRunTrace } from "./useRunTrace";
import { useWorkshopTimeline } from "./useWorkshopTimeline";
import type { EarlierHistoryState } from "./useWorkshopTimeline";
import { MarkdownMessage } from "./MarkdownMessage";
import { startArtifactDownload } from "./artifactDownload";
import { MemoryExplorer } from "./MemoryExplorer";
import { SettingsWorkspace } from "./SettingsWorkspace";
import { applyWorkshopTheme, clearWorkshopThemeHint } from "./theme";
import { ConfirmationProvider, useConfirmation } from "./ConfirmationDialog";

const BROWSER_CREDENTIAL_KEY = "kai.workshop.client-credential.v1";
const TAB_CHANNEL_KEY = "kai.workshop.active-channel.v1";
const LEGACY_SESSION_KEY = "kai.workshop.read-session.v1";
const ACTIVE_RUN_KEY = "kai.workshop.active-run.v1";
const DRAFTS_KEY = "kai.workshop.drafts.v1";
const VIEWPORTS_KEY = "kai.workshop.timeline-viewports.v1";
const SIDEBAR_LAYOUT_KEY = "kai.workshop.sidebar-layout.v4";
const TIMELINE_FOLLOW_DISTANCE_PX = 96;
const UI_SCALE = 1.5;
const MIN_SIDEBAR_WIDTH_PX = 176 * UI_SCALE;
const DEFAULT_SIDEBAR_WIDTH_PX = MIN_SIDEBAR_WIDTH_PX;
const MAX_SIDEBAR_WIDTH_PX = 420 * UI_SCALE;
const COLLAPSED_SIDEBAR_WIDTH_PX = 56 * UI_SCALE;
const MEMORY_ID_PATTERN = /^[A-Za-z0-9_-]{1,256}$/;

type WorkshopDestination =
  | { kind: "conversation" }
  | { kind: "memory"; memoryId: string | null }
  | { kind: "settings" };

function destinationFromLocation(): WorkshopDestination {
  const parameters = new URLSearchParams(window.location.search);
  if (parameters.get("view") === "settings") {
    return { kind: "settings" };
  }
  if (parameters.get("view") !== "memory") {
    return { kind: "conversation" };
  }
  const memoryId = parameters.get("memory");
  return {
    kind: "memory",
    memoryId: memoryId && MEMORY_ID_PATTERN.test(memoryId) ? memoryId : null,
  };
}

function writeDestination(
  destination: WorkshopDestination,
  mode: "push" | "replace",
): void {
  const url = new URL(window.location.href);
  url.searchParams.delete("view");
  url.searchParams.delete("memory");
  if (destination.kind === "memory") {
    url.searchParams.set("view", "memory");
    if (destination.memoryId) {
      url.searchParams.set("memory", destination.memoryId);
    }
  } else if (destination.kind === "settings") {
    url.searchParams.set("view", "settings");
  }
  window.history[mode === "push" ? "pushState" : "replaceState"](
    null,
    "",
    `${url.pathname}${url.search}${url.hash}`,
  );
}

interface StoredSidebarLayout {
  collapsed: boolean;
  width: number;
}

function clampSidebarWidth(width: number): number {
  return Math.min(
    MAX_SIDEBAR_WIDTH_PX,
    Math.max(MIN_SIDEBAR_WIDTH_PX, width),
  );
}

function restoreSidebarLayout(): StoredSidebarLayout {
  try {
    const stored: unknown = JSON.parse(
      sessionStorage.getItem(SIDEBAR_LAYOUT_KEY) ?? "null",
    );
    if (
      typeof stored === "object" &&
      stored !== null &&
      "collapsed" in stored &&
      "width" in stored &&
      typeof stored.collapsed === "boolean" &&
      typeof stored.width === "number" &&
      Number.isFinite(stored.width)
    ) {
      return {
        collapsed: stored.collapsed,
        width: clampSidebarWidth(stored.width),
      };
    }
  } catch {
    // Malformed layout state has no authority.
  }
  return { collapsed: false, width: DEFAULT_SIDEBAR_WIDTH_PX };
}

function storeSidebarLayout(layout: StoredSidebarLayout): void {
  sessionStorage.setItem(SIDEBAR_LAYOUT_KEY, JSON.stringify(layout));
}

const CONTEXT_LAYOUT_KEY = "kai.workshop.context-layout.v4";
const MIN_CONTEXT_WIDTH_PX = 240 * UI_SCALE;
const DEFAULT_CONTEXT_WIDTH_PX = MIN_CONTEXT_WIDTH_PX;
const MAX_CONTEXT_WIDTH_PX = 560 * UI_SCALE;

function clampContextWidth(width: number): number {
  return Math.min(
    MAX_CONTEXT_WIDTH_PX,
    Math.max(MIN_CONTEXT_WIDTH_PX, width),
  );
}

function restoreContextWidth(): number {
  try {
    const stored: unknown = JSON.parse(
      sessionStorage.getItem(CONTEXT_LAYOUT_KEY) ?? "null",
    );
    if (
      typeof stored === "object" &&
      stored !== null &&
      "width" in stored &&
      typeof stored.width === "number" &&
      Number.isFinite(stored.width)
    ) {
      return clampContextWidth(stored.width);
    }
  } catch {
    // Malformed layout state has no authority.
  }
  return DEFAULT_CONTEXT_WIDTH_PX;
}

function storeContextWidth(width: number): void {
  sessionStorage.setItem(CONTEXT_LAYOUT_KEY, JSON.stringify({ width }));
}

interface RestoredWorkshopAccess {
  channelId: string | null;
  token: string;
}

function parseLegacySession(value: string | null): WorkshopSession | null {
  try {
    const stored: unknown = JSON.parse(value ?? "null");
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
    // Malformed legacy state has no authority.
  }
  return null;
}

function restoreBrowserCredential(value: string | null): string | null {
  try {
    const stored: unknown = JSON.parse(value ?? "null");
    if (
      typeof stored === "object" &&
      stored !== null &&
      "token" in stored &&
      typeof stored.token === "string" &&
      stored.token.length > 0
    ) {
      return stored.token;
    }
  } catch {
    // Malformed browser-scoped credentials have no authority.
  }
  return null;
}

function restoreTabChannel(): string | null {
  try {
    const stored: unknown = JSON.parse(
      sessionStorage.getItem(TAB_CHANNEL_KEY) ?? "null",
    );
    if (
      typeof stored === "object" &&
      stored !== null &&
      "channelId" in stored &&
      typeof stored.channelId === "string" &&
      CHANNEL_PATTERN.test(stored.channelId)
    ) {
      return stored.channelId;
    }
  } catch {
    // Malformed tab-local channel state has no authority.
  }
  sessionStorage.removeItem(TAB_CHANNEL_KEY);
  return null;
}

function restoreWorkshopAccess(): RestoredWorkshopAccess | null {
  let token = restoreBrowserCredential(
    localStorage.getItem(BROWSER_CREDENTIAL_KEY),
  );
  let channelId = restoreTabChannel();
  const legacy = parseLegacySession(
    sessionStorage.getItem(LEGACY_SESSION_KEY),
  );

  if (!token && legacy) {
    token = legacy.token;
    channelId ??= legacy.channelId;
    localStorage.setItem(
      BROWSER_CREDENTIAL_KEY,
      JSON.stringify({ token: legacy.token }),
    );
  } else if (token && legacy?.token === token) {
    channelId ??= legacy.channelId;
  }

  if (localStorage.getItem(BROWSER_CREDENTIAL_KEY) && !token) {
    localStorage.removeItem(BROWSER_CREDENTIAL_KEY);
  }
  sessionStorage.removeItem(LEGACY_SESSION_KEY);

  return token ? { channelId, token } : null;
}

function storeWorkshopAccess(session: WorkshopSession): void {
  localStorage.setItem(
    BROWSER_CREDENTIAL_KEY,
    JSON.stringify({ token: session.token }),
  );
  sessionStorage.setItem(
    TAB_CHANNEL_KEY,
    JSON.stringify({ channelId: session.channelId }),
  );
  sessionStorage.removeItem(LEGACY_SESSION_KEY);
}

function clearTabSessionState(): void {
  sessionStorage.removeItem(TAB_CHANNEL_KEY);
  sessionStorage.removeItem(LEGACY_SESSION_KEY);
  sessionStorage.removeItem(ACTIVE_RUN_KEY);
  sessionStorage.removeItem(DRAFTS_KEY);
  sessionStorage.removeItem(VIEWPORTS_KEY);
}

function forgetStoredSession(): void {
  localStorage.removeItem(BROWSER_CREDENTIAL_KEY);
  clearTabSessionState();
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
              Enrollment is complete for this browser profile. Retry Workshop
              discovery, or forget the session and enroll again.
            </p>
          ) : (
            <p className="card-copy">
              The session credential remains in this browser profile for this
              Kai origin and is never written to the URL.
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

const MESSAGE_REACTIONS: {
  label: string;
  reaction: WorkshopReaction;
  symbol: string;
}[] = [
  { label: "Thumbs up", reaction: "thumbs_up", symbol: "👍" },
  { label: "Heart", reaction: "heart", symbol: "♥" },
  { label: "Laugh", reaction: "laugh", symbol: "😄" },
  { label: "Celebrate", reaction: "celebrate", symbol: "🎉" },
  { label: "Eyes", reaction: "eyes", symbol: "👀" },
  { label: "Done", reaction: "check", symbol: "✓" },
];

function MessageItem({
  message,
  notification = false,
  onDownloadArtifact,
  onLoadArtifact,
  onOpenThread,
  onSetReaction,
}: {
  message: TimelineMessage;
  notification?: boolean;
  onDownloadArtifact: (artifactId: string) => void;
  onLoadArtifact: (artifactId: string) => Promise<Blob>;
  onOpenThread?: (messageId: string) => void;
  onSetReaction?: (
    messageId: string,
    reaction: WorkshopReaction,
    active: boolean,
  ) => Promise<void>;
}): React.JSX.Element {
  const [reactionPickerOpen, setReactionPickerOpen] = useState(false);
  const [reactionPending, setReactionPending] = useState<WorkshopReaction | null>(null);
  const [reactionError, setReactionError] = useState<string | null>(null);
  const isAgent = message.authorKind === "agent";
  const displayName = message.authorDisplayName || "Unknown author";
  const reactions = message.reactions ?? [];
  const setReaction = async (
    reaction: WorkshopReaction,
    active: boolean,
  ): Promise<void> => {
    if (!onSetReaction || reactionPending !== null) {
      return;
    }
    setReactionPending(reaction);
    setReactionError(null);
    try {
      await onSetReaction(message.messageId, reaction, active);
      setReactionPickerOpen(false);
    } catch (caught) {
      setReactionError(
        caught instanceof Error ? caught.message : "Could not update this reaction.",
      );
    } finally {
      setReactionPending(null);
    }
  };
  if (notification) {
    return (
      <li className="notification-row">
        <span className="notification-source" aria-hidden="true">GH</span>
        <article>
          <header className="message-meta">
            <strong>GitHub</strong>
            <time dateTime={message.createdAt}>
              {formatTimestamp(message.createdAt)}
            </time>
          </header>
          <MarkdownMessage body={message.body} mentions={message.mentions} />
          {message.artifacts.map((artifact) => (
            <ArtifactAttachment
              artifact={artifact}
              key={artifact.artifactId}
              onDownload={onDownloadArtifact}
              onLoad={onLoadArtifact}
            />
          ))}
        </article>
      </li>
    );
  }
  return (
    <li className={`message-row ${isAgent ? "agent" : "human"}`}>
      <span className="message-avatar" aria-hidden="true">
        {displayName.slice(0, 1).toUpperCase()}
      </span>
      <article>
        {(onOpenThread || onSetReaction) && (
          <div className="message-actions" role="group" aria-label={`Actions for message from ${displayName}`}>
            {onSetReaction && (
              <div className="reaction-picker-anchor">
                <button
                  className="message-action-button"
                  type="button"
                  aria-label="Add reaction"
                  aria-expanded={reactionPickerOpen}
                  title="Add reaction"
                  onClick={() => {
                    setReactionError(null);
                    setReactionPickerOpen((open) => !open);
                  }}
                >
                  <AddReactionIcon />
                </button>
                {reactionPickerOpen && (
                  <div
                    className="reaction-picker"
                    role="menu"
                    aria-label="Choose a reaction"
                    onKeyDown={(event) => {
                      if (event.key === "Escape") {
                        event.preventDefault();
                        setReactionPickerOpen(false);
                      }
                    }}
                  >
                    {MESSAGE_REACTIONS.map((option) => {
                      const active = reactions.some(
                        (reaction) => reaction.reaction === option.reaction && reaction.reactedByViewer,
                      );
                      return (
                        <button
                          type="button"
                          role="menuitemcheckbox"
                          aria-checked={active}
                          aria-label={`${active ? "Remove" : "Add"} ${option.label} reaction`}
                          disabled={reactionPending !== null}
                          key={option.reaction}
                          onClick={() => void setReaction(option.reaction, !active)}
                        >
                          {option.symbol}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
            {onOpenThread && (
              <button
                className="message-action-button"
                type="button"
                aria-label="Reply to message"
                title="Reply"
                onClick={() => onOpenThread(message.messageId)}
              >
                <ReplyIcon />
              </button>
            )}
          </div>
        )}
        <header className="message-meta">
          <strong>{displayName}</strong>
          <time dateTime={message.createdAt}>
            {formatTimestamp(message.createdAt)}
          </time>
        </header>
        <MarkdownMessage body={message.body} mentions={message.mentions} />
        {message.artifacts.map((artifact) => (
          <ArtifactAttachment
            artifact={artifact}
            key={artifact.artifactId}
            onDownload={onDownloadArtifact}
            onLoad={onLoadArtifact}
          />
        ))}
        {(reactions.length > 0 || (onOpenThread && message.replyCount > 0)) && (
          <div className="message-engagement" role="group" aria-label="Message engagement">
            {reactions.length > 0 && (
              <div className="message-reactions" aria-label="Message reactions">
                {reactions.map((reaction) => {
                  const option = MESSAGE_REACTIONS.find(
                    (candidate) => candidate.reaction === reaction.reaction,
                  );
                  if (!option) {
                    return null;
                  }
                  return (
                    <button
                      className={reaction.reactedByViewer ? "active" : ""}
                      type="button"
                      aria-label={`${option.label}: ${reaction.count}. ${reaction.reactedByViewer ? "Remove your reaction" : "Add your reaction"}`}
                      aria-pressed={reaction.reactedByViewer}
                      disabled={!onSetReaction || reactionPending !== null}
                      key={reaction.reaction}
                      onClick={() => void setReaction(reaction.reaction, !reaction.reactedByViewer)}
                    >
                      <span aria-hidden="true">{option.symbol}</span>
                      <span>{reaction.count}</span>
                    </button>
                  );
                })}
              </div>
            )}
            {onOpenThread && message.replyCount > 0 && (
              <button
                className="thread-summary"
                type="button"
                aria-label={`Open thread with ${message.replyCount} ${message.replyCount === 1 ? "reply" : "replies"}`}
                title="Open thread"
                onClick={() => onOpenThread(message.messageId)}
              >
                <span>
                  {message.replyCount} {message.replyCount === 1 ? "reply" : "replies"}
                </span>
              </button>
            )}
          </div>
        )}
        {reactionError && <p className="reaction-error" role="alert">{reactionError}</p>}
      </article>
    </li>
  );
}

const SAFE_INLINE_IMAGE_TYPES = new Set([
  "image/gif",
  "image/jpeg",
  "image/png",
  "image/webp",
]);

function ArtifactAttachment({
  artifact,
  onDownload,
  onLoad,
}: {
  artifact: WorkshopArtifactSummary;
  onDownload: (artifactId: string) => void;
  onLoad: (artifactId: string) => Promise<Blob>;
}): React.JSX.Element {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inline = SAFE_INLINE_IMAGE_TYPES.has(artifact.mediaType) ||
    artifact.mediaType.startsWith("audio/");

  useEffect(() => {
    if (!inline) {
      return;
    }
    let cancelled = false;
    let createdUrl: string | null = null;
    void onLoad(artifact.artifactId)
      .then((blob) => {
        if (cancelled) {
          return;
        }
        createdUrl = URL.createObjectURL(blob);
        setObjectUrl(createdUrl);
      })
      .catch(() => {
        if (!cancelled) {
          setError("Attachment preview unavailable.");
        }
      });
    return () => {
      cancelled = true;
      if (createdUrl) {
        URL.revokeObjectURL(createdUrl);
      }
    };
  }, [artifact.artifactId, inline, onLoad]);

  const download = (): void => {
    setError(null);
    try {
      onDownload(artifact.artifactId);
    } catch {
      setError("Could not download this attachment.");
    }
  };

  return (
    <section className="message-artifact">
      {objectUrl && SAFE_INLINE_IMAGE_TYPES.has(artifact.mediaType) && (
        <img src={objectUrl} alt={artifact.originalFilename ?? "Attached image"} />
      )}
      {objectUrl && artifact.mediaType.startsWith("audio/") && (
        <audio src={objectUrl} controls preload="metadata" />
      )}
      <div className="artifact-meta">
        <span>{artifact.originalFilename ?? "Attachment"}</span>
        <small>{Math.max(1, Math.ceil(artifact.byteSize / 1024))} KB</small>
        <button type="button" onClick={download}>Download</button>
      </div>
      {error && <p className="artifact-error" role="alert">{error}</p>}
    </section>
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
  if (channel.kind === "direct") {
    const participantNames = channel.participants
      .map((participant) => participant.displayName.trim())
      .filter(Boolean);
    if (participantNames.length > 0) {
      return participantNames.join(", ");
    }
    if (channel.agents.length === 1 && channel.agents[0].name.trim()) {
      return channel.agents[0].name.trim();
    }
  }
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

function channelSymbol(channel: WorkshopChannelSummary): string {
  if (channel.kind === "notification") {
    return "!";
  }
  if (channel.kind === "direct") {
    return "@";
  }
  return "#";
}

function workshopRoleLabel(role: string): string {
  if (role === "admin") {
    return "Workshop administrator";
  }
  if (role === "member") {
    return "Workshop member";
  }
  return `Workshop ${role}`;
}

interface ChannelCreationRequest {
  agentIds: string[];
  name: string;
  originChannelId: string | null;
}

interface MentionCandidate {
  displayName: string;
  kind: "agent" | "human";
  principalId: string;
}

interface MentionTrigger {
  end: number;
  query: string;
  start: number;
}

function findMentionTrigger(value: string, caret: number): MentionTrigger | null {
  const prefix = value.slice(0, caret);
  const match = /(?:^|\s)@([^@\n]*)$/.exec(prefix);
  if (!match) {
    return null;
  }
  const query = match[1];
  return {
    end: caret,
    query,
    start: caret - query.length - 1,
  };
}

function ChannelCreationDialog({
  agents,
  initialAgentIds,
  originChannelId,
  originName,
  onCancel,
  onCreate,
}: {
  agents: WorkshopAgentSummary[];
  initialAgentIds: string[];
  originChannelId: string | null;
  originName: string | null;
  onCancel: () => void;
  onCreate: (input: ChannelCreationRequest) => Promise<void>;
}): React.JSX.Element {
  const [name, setName] = useState("");
  const [selectedAgentIds, setSelectedAgentIds] = useState<string[]>(() =>
    initialAgentIds.length > 0
      ? initialAgentIds
      : agents[0]
        ? [agents[0].agentId]
        : [],
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const normalizedName = name.trim();
    if (!normalizedName || selectedAgentIds.length === 0) {
      setError("A channel name and at least one agent are required.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onCreate({
        agentIds: selectedAgentIds,
        name: normalizedName,
        originChannelId,
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not create this channel.");
      setBusy(false);
    }
  };
  return (
    <div className="modal-backdrop">
      <section className="channel-creation-dialog" role="dialog" aria-modal="true" aria-labelledby="create-channel-title">
        <p className="overline">New conversation space</p>
        <h2 id="create-channel-title">Create channel</h2>
        {originName && <p>Start from <strong>{originName}</strong>.</p>}
        <form onSubmit={(event) => void submit(event)}>
          <label htmlFor="channel-name">Channel name</label>
          <input
            id="channel-name"
            autoFocus
            maxLength={200}
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <fieldset>
            <legend>Agents</legend>
            {agents.map((agent) => (
              <label className="channel-agent-choice" key={agent.agentId}>
                <input
                  type="checkbox"
                  checked={selectedAgentIds.includes(agent.agentId)}
                  onChange={(event) =>
                    setSelectedAgentIds((current) =>
                      event.target.checked
                        ? [...current, agent.agentId]
                        : current.filter((agentId) => agentId !== agent.agentId),
                    )
                  }
                />
                <span>{agent.name}</span>
              </label>
            ))}
          </fieldset>
          {error && <p className="form-error" role="alert">{error}</p>}
          <div className="form-actions">
            <button className="primary-button" type="submit" disabled={busy}>
              {busy ? "Creating…" : "Create channel"}
            </button>
            <button className="quiet-button" type="button" onClick={onCancel} disabled={busy}>
              Cancel
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

function PaperclipIcon(): React.JSX.Element {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      focusable="false"
      viewBox="0 0 24 24"
    >
      <path
        d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function SendIcon(): React.JSX.Element {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      focusable="false"
      viewBox="0 0 24 24"
    >
      <path
        d="m22 2-7 20-4-9-9-4Z"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
      <path
        d="M22 2 11 13"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function ReplyIcon(): React.JSX.Element {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      focusable="false"
      viewBox="0 0 24 24"
    >
      <path
        d="M20 15a3 3 0 0 1-3 3H9l-5 3V7a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3z"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function AddReactionIcon(): React.JSX.Element {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      focusable="false"
      viewBox="0 0 24 24"
    >
      <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2" />
      <path
        d="M8 9h.01M14 9h.01M7.5 13.5c1 1.5 2.2 2.2 3.5 2.2s2.5-.7 3.5-2.2M19 15v6M16 18h6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function ThreadPane({
  channelName,
  liveMessages,
  onClose,
  onDownloadArtifact,
  onLoadArtifact,
  onLoadThread,
  onSetReaction,
  onSubmitCommand,
  reactionUpdates,
  rootMessage,
  runActive,
}: {
  channelName: string;
  liveMessages: TimelineMessage[];
  onClose: () => void;
  onDownloadArtifact: (artifactId: string) => void;
  onLoadArtifact: (artifactId: string) => Promise<Blob>;
  onLoadThread: (rootMessageId: string, cursor: string | null, signal?: AbortSignal) => Promise<ThreadTimelineSnapshot>;
  onSetReaction: (
    messageId: string,
    reaction: WorkshopReaction,
    active: boolean,
  ) => Promise<void>;
  onSubmitCommand: (clientMessageId: string, body: string, artifact: File | null, threadRootId: string | null) => Promise<CommandSubmissionResult>;
  reactionUpdates: Record<string, TimelineMessage["reactions"]>;
  rootMessage: TimelineMessage;
  runActive: boolean;
}): React.JSX.Element {
  const [snapshot, setSnapshot] = useState<ThreadTimelineSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [draft, setDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [pendingMessageId, setPendingMessageId] = useState<string | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setSnapshot(null);
    setError(null);
    void onLoadThread(rootMessage.messageId, null, controller.signal).then(
      setSnapshot,
      (caught: unknown) => {
        if (!controller.signal.aborted) {
          setError(caught instanceof Error ? caught.message : "Could not load this thread.");
        }
      },
    );
    return () => controller.abort();
  }, [onLoadThread, rootMessage.messageId]);

  useLayoutEffect(() => {
    const composer = composerRef.current;
    if (!composer) {
      return;
    }
    const borderHeight = composer.offsetHeight - composer.clientHeight;
    composer.style.height = "auto";
    composer.style.height = `${composer.scrollHeight + borderHeight}px`;
  }, [draft, rootMessage.messageId]);

  const replies = useMemo(() => {
    const byId = new Map<string, TimelineMessage>();
    for (const message of snapshot?.messages ?? []) {
      byId.set(message.messageId, message);
    }
    for (const message of liveMessages) {
      if (message.threadRootId === rootMessage.messageId) {
        byId.set(message.messageId, message);
      }
    }
    return Array.from(byId.values()).sort(
      (left, right) => left.eventPosition - right.eventPosition,
    );
  }, [liveMessages, rootMessage.messageId, snapshot?.messages]);

  const loadMore = async (): Promise<void> => {
    if (!snapshot?.nextCursor || loadingMore) {
      return;
    }
    setLoadingMore(true);
    setError(null);
    try {
      const page = await onLoadThread(rootMessage.messageId, snapshot.nextCursor);
      setSnapshot((current) => current && ({
        ...current,
        messages: [...current.messages, ...page.messages],
        nextCursor: page.nextCursor,
      }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load more replies.");
    } finally {
      setLoadingMore(false);
    }
  };

  const submitReply = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const body = draft.trim();
    if (!body || submitting || runActive) {
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const clientMessageId = pendingMessageId ?? createClientMessageId();
      setPendingMessageId(clientMessageId);
      await onSubmitCommand(clientMessageId, body, null, rootMessage.messageId);
      setDraft("");
      setPendingMessageId(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Kai could not send this reply.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="thread-pane">
      <header className="thread-header">
        <div>
          <p className="overline">Thread in {channelName}</p>
          <h2>{replies.length} {replies.length === 1 ? "reply" : "replies"}</h2>
        </div>
        <button className="quiet-button" type="button" onClick={onClose}>Close</button>
      </header>
      <div className="thread-scroll">
        <ol className="thread-message-list">
          <MessageItem
            message={(() => {
              const message = snapshot?.root ?? rootMessage;
              return {
                ...message,
                reactions: reactionUpdates[message.messageId] ?? message.reactions,
              };
            })()}
            onDownloadArtifact={onDownloadArtifact}
            onLoadArtifact={onLoadArtifact}
            onSetReaction={onSetReaction}
          />
          {replies.map((message) => (
            <MessageItem
              key={message.messageId}
              message={{
                ...message,
                reactions: reactionUpdates[message.messageId] ?? message.reactions,
              }}
              onDownloadArtifact={onDownloadArtifact}
              onLoadArtifact={onLoadArtifact}
              onSetReaction={onSetReaction}
            />
          ))}
        </ol>
        {!snapshot && !error && <p className="thread-state">Loading replies…</p>}
        {snapshot?.nextCursor && (
          <button
            className="timeline-earlier-button"
            type="button"
            disabled={loadingMore}
            onClick={() => void loadMore()}
          >
            {loadingMore ? "Loading…" : "Load more replies"}
          </button>
        )}
      </div>
      <form className="composer-form thread-composer" onSubmit={(event) => void submitReply(event)}>
        <textarea
          ref={composerRef}
          aria-label={`Reply in ${channelName}`}
          maxLength={50000}
          placeholder="Reply…"
          rows={1}
          value={draft}
          disabled={submitting || runActive}
          onChange={(event) => {
            setDraft(event.target.value);
            if (!submitting) {
              setPendingMessageId(null);
            }
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
        />
        <button
          className="composer-icon-button send-button"
          type="submit"
          aria-busy={submitting}
          aria-label={submitting ? "Sending reply…" : "Send reply"}
          title={submitting ? "Sending reply…" : "Send reply"}
          disabled={!draft.trim() || submitting || runActive}
        >
          <SendIcon />
        </button>
      </form>
      {error && <p className="thread-error" role="alert">{error}</p>}
    </div>
  );
}

function WorkshopView({
  channel,
  connection,
  earlier,
  messages,
  threadMessages,
  memoryDestination,
  memoryToken,
  settingsDestination,
  settingsRuntimeLabel,
  settingsSession,
  navigation,
  runActivity,
  runPreview,
  runTrace,
  reactionUpdates,
  workshop,
  onForget,
  onCancelRun,
  onCreateChannel,
  onDismissAgent,
  onDownloadArtifact,
  onLoadEarlier,
  onLoadArtifact,
  onLoadRun,
  onLoadRunTrace,
  onLoadSettingsWorkspace,
  onLoadThread,
  onMemoryAuthenticationFailure,
  onOpenMemory,
  onOpenSettings,
  onSelectMemory,
  onSelectChannel,
  onSetReaction,
  onSubmitCommand,
  onSwitchWorkspace,
  onSettingsDirtyChange,
  onSettingsAccessFailure,
}: {
  channel: WorkshopChannelSummary;
  connection: ConnectionState;
  earlier: EarlierHistoryState;
  messages: TimelineMessage[];
  threadMessages: TimelineMessage[];
  memoryDestination: { memoryId: string | null } | null;
  memoryToken: string;
  settingsDestination: boolean;
  settingsRuntimeLabel: string;
  settingsSession: WorkshopSession;
  navigation: WorkshopNavigation;
  runActivity: WorkshopRunActivity | null;
  runPreview: WorkshopRunPreview | null;
  runTrace: WorkshopRunTraceSignal | null;
  reactionUpdates: Record<string, TimelineMessage["reactions"]>;
  workshop: WorkshopSummary;
  onForget: () => void;
  onCancelRun: (runId: string) => Promise<WorkshopRun>;
  onCreateChannel: (input: ChannelCreationRequest) => Promise<void>;
  onDismissAgent: (agentId: string, clientDismissalId: string) => Promise<void>;
  onDownloadArtifact: (artifactId: string) => void;
  onLoadEarlier: () => void;
  onLoadArtifact: (artifactId: string) => Promise<Blob>;
  onLoadRun: (runId: string) => Promise<WorkshopRun>;
  onLoadRunTrace: (runId: string, afterSeq: number) => Promise<WorkshopRunTracePage>;
  onLoadSettingsWorkspace: () => Promise<WorkshopSettingsWorkspace>;
  onLoadThread: (
    rootMessageId: string,
    cursor: string | null,
    signal?: AbortSignal,
  ) => Promise<ThreadTimelineSnapshot>;
  onMemoryAuthenticationFailure: (message: string) => void;
  onOpenMemory: () => void;
  onOpenSettings: () => void;
  onSelectMemory: (memoryId: string | null) => void;
  onSelectChannel: (channelId: string) => void;
  onSetReaction: (
    messageId: string,
    reaction: WorkshopReaction,
    active: boolean,
  ) => Promise<void>;
  onSubmitCommand: (
    clientMessageId: string,
    body: string,
    artifact: File | null,
    threadRootId: string | null,
  ) => Promise<CommandSubmissionResult>;
  onSwitchWorkspace: (
    path: string,
    revision: string,
  ) => Promise<WorkshopSettingsWorkspace>;
  onSettingsDirtyChange: (dirty: boolean) => void;
  onSettingsAccessFailure: (message: string) => void;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const channelId = channel.channelId;
  const memoryOpen = memoryDestination !== null;
  const settingsOpen = settingsDestination;
  const auxiliaryWorkspaceOpen = memoryOpen || settingsOpen;
  const channelName = channelDisplayName(channel);
  const symbol = channelSymbol(channel);
  const [draft, setDraft] = useState(() => restoreDraft(channelId));
  const [pendingMessageId, setPendingMessageId] = useState<string | null>(null);
  const [submissionError, setSubmissionError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [selectedArtifact, setSelectedArtifact] = useState<File | null>(null);
  const [stopping, setStopping] = useState(false);
  const [activeRun, setActiveRun] = useState<WorkshopRun | null>(null);
  const [runClock, setRunClock] = useState(() => Date.now());
  const [unseenMessageCount, setUnseenMessageCount] = useState(0);
  // Render-state mirror of timelineFollowRef's negation. The ref is
  // deliberately render-free for scroll-frequency updates; the
  // jump-to-latest button needs a re-render when follow disengages, so
  // this state is set alongside every follow-ref write. Same-value
  // setState is a React no-op, keeping the per-scroll cost nil.
  const [awayFromBottom, setAwayFromBottom] = useState(false);
  const [sidebarLayout, setSidebarLayout] = useState(restoreSidebarLayout);
  const [resizingSidebar, setResizingSidebar] = useState(false);
  const [contextWidth, setContextWidth] = useState(restoreContextWidth);
  const [resizingContext, setResizingContext] = useState(false);
  const [settingsWorkspace, setSettingsWorkspace] =
    useState<WorkshopSettingsWorkspace | null>(null);
  const [settingsWorkspaceError, setSettingsWorkspaceError] =
    useState<string | null>(null);
  const [switchingWorkspace, setSwitchingWorkspace] = useState(false);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [channelCreation, setChannelCreation] = useState<{
    initialAgentIds: string[];
    originChannelId: string | null;
    originName: string | null;
  } | null>(null);
  const [dismissedAgents, setDismissedAgents] = useState<Record<string, number>>({});
  const [dismissingAgentId, setDismissingAgentId] = useState<string | null>(null);
  const [engagementClock, setEngagementClock] = useState(() => Date.now());
  const [mentionTrigger, setMentionTrigger] = useState<MentionTrigger | null>(null);
  const [mentionSelection, setMentionSelection] = useState(0);
  const [notificationPreferences, setNotificationPreferences] =
    useState<WorkshopNotificationPreferences | null>(null);
  const [threadRootMessageId, setThreadRootMessageId] = useState<string | null>(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const pendingComposerCaretRef = useRef<number | null>(null);
  const artifactInputRef = useRef<HTMLInputElement | null>(null);
  const sidebarRef = useRef<HTMLElement | null>(null);
  const profileMenuRef = useRef<HTMLElement | null>(null);
  const sidebarResizeStartRef = useRef({ pointerX: 0, width: 0 });
  const contextResizeStartRef = useRef({ pointerX: 0, width: 0 });
  const timelineChannelRef = useRef(channelId);
  const timelineInitializedRef = useRef(false);
  const timelineFollowRef = useRef(true);
  const latestMessagePositionRef = useRef(0);
  const earliestMessagePositionRef = useRef(0);
  // Viewport captured when "load earlier" is clicked, so the prepended
  // page can be offset out of view instead of shoving the reader down.
  const earlierAnchorRef = useRef<{ scrollHeight: number; scrollTop: number } | null>(null);
  const latestRunActivityRef = useRef<WorkshopRunActivity | null>(runActivity);
  const humanName = navigation.principal.displayName || "You";
  const humanRole = workshopRoleLabel(workshop.role);
  const threadRootMessage = useMemo(
    () => messages.find((message) => message.messageId === threadRootMessageId) ?? null,
    [messages, threadRootMessageId],
  );
  const displayedMessages = useMemo(
    () => messages.map((message) => {
      const liveReplies = threadMessages.filter(
        (candidate) => candidate.threadRootId === message.messageId,
      );
      if (liveReplies.length === 0) {
        return message;
      }
      return {
        ...message,
        replyCount: message.replyCount + liveReplies.length,
        latestReplyAt: liveReplies[liveReplies.length - 1].createdAt,
      };
    }),
    [messages, threadMessages],
  );

  useEffect(() => {
    setThreadRootMessageId(null);
  }, [channelId]);
  const availableAgents = useMemo(() => {
    const unique = new Map<string, WorkshopAgentSummary>();
    for (const availableChannel of workshop.channels) {
      for (const agent of availableChannel.agents) {
        unique.set(agent.agentId, agent);
      }
    }
    return Array.from(unique.values()).sort((left, right) =>
      left.name.localeCompare(right.name),
    );
  }, [workshop.channels]);
  const mentionCandidates = useMemo(() => {
    if (channel.kind !== "group" || !mentionTrigger) {
      return [];
    }
    const members = new Map<string, MentionCandidate>();
    for (const agent of channel.agents) {
      members.set(agent.principalId, {
        displayName: agent.name,
        kind: "agent",
        principalId: agent.principalId,
      });
    }
    for (const participant of channel.participants) {
      if (
        participant.principalId === navigation.principal.principalId ||
        (participant.kind !== "agent" && participant.kind !== "human")
      ) {
        continue;
      }
      members.set(participant.principalId, {
        displayName: participant.displayName,
        kind: participant.kind,
        principalId: participant.principalId,
      });
    }
    const query = mentionTrigger.query.trim().toLocaleLowerCase();
    return Array.from(members.values())
      .filter(
        (candidate) =>
          !query || candidate.displayName.toLocaleLowerCase().startsWith(query),
      )
      .sort((left, right) => left.displayName.localeCompare(right.displayName));
  }, [channel, mentionTrigger, navigation.principal.principalId]);

  useEffect(() => {
    if (channel.kind !== "notification") {
      setNotificationPreferences(null);
      return;
    }
    let active = true;
    void loadNotificationPreferences(settingsSession)
      .then((snapshot) => {
        if (active) {
          setNotificationPreferences(snapshot);
        }
      })
      .catch(() => {
        if (active) {
          setNotificationPreferences(null);
        }
      });
    return () => {
      active = false;
    };
  }, [channel.kind, settingsSession]);
  // The inspected run: the channel's active run when one exists, else
  // the most recently settled run (activeRun keeps its terminal value
  // and is seeded from replayed lifecycle events on mount).
  const inspectedRunId = activeRun?.runId ?? null;
  const {
    entries: traceEntries,
    failed: traceFailed,
    loaded: traceLoaded,
  } = useRunTrace(inspectedRunId, runTrace, onLoadRunTrace);

  useEffect(() => {
    storeSidebarLayout(sidebarLayout);
  }, [sidebarLayout]);

  useEffect(() => {
    storeContextWidth(contextWidth);
  }, [contextWidth]);

  useEffect(() => {
    if (channel.kind !== "group") {
      return;
    }
    const timer = window.setInterval(() => setEngagementClock(Date.now()), 30000);
    return () => window.clearInterval(timer);
  }, [channel.kind]);

  const engagedAgents = useMemo(() => {
    if (channel.kind !== "group") {
      return [];
    }
    return channel.agents.filter((agent) => {
      const dismissedAt = dismissedAgents[agent.agentId] ?? 0;
      let engagedUntil = dismissedAt
        ? 0
        : agent.engagedUntil
          ? Date.parse(agent.engagedUntil)
          : agent.engaged
            ? engagementClock
            : 0;
      for (const message of messages) {
        const createdAt = Date.parse(message.createdAt);
        if (!Number.isFinite(createdAt) || createdAt <= dismissedAt) {
          continue;
        }
        if (
          message.authorPrincipalId === agent.principalId ||
          message.mentions.some(
            (mention) =>
              mention.kind === "agent" &&
              mention.principalId === agent.principalId,
          )
        ) {
          engagedUntil = Math.max(engagedUntil, createdAt + 900000);
        }
      }
      return engagedUntil > engagementClock && engagedUntil > dismissedAt;
    });
  }, [channel.agents, channel.kind, dismissedAgents, engagementClock, messages]);

  const dismissAgent = async (agentId: string): Promise<void> => {
    if (dismissingAgentId) {
      return;
    }
    setDismissingAgentId(agentId);
    setSubmissionError(null);
    try {
      await onDismissAgent(agentId, createClientMessageId());
      setDismissedAgents((current) => ({ ...current, [agentId]: Date.now() }));
      setEngagementClock(Date.now());
    } catch (caught) {
      setSubmissionError(
        caught instanceof Error ? caught.message : "Could not dismiss this agent.",
      );
    } finally {
      setDismissingAgentId(null);
    }
  };

  const forgetCurrentSession = async (): Promise<void> => {
    setProfileMenuOpen(false);
    if (await confirm("Forget this browser session? You will need to enroll again.")) {
      onForget();
    }
  };

  const updateMentionTrigger = (value: string, caret: number): void => {
    setMentionTrigger(findMentionTrigger(value, caret));
    setMentionSelection(0);
  };

  const insertMention = (candidate: MentionCandidate): void => {
    if (!mentionTrigger) {
      return;
    }
    const insertion = `@${candidate.displayName} `;
    const nextDraft =
      draft.slice(0, mentionTrigger.start) +
      insertion +
      draft.slice(mentionTrigger.end);
    const nextCaret = mentionTrigger.start + insertion.length;
    setDraft(nextDraft);
    storeDraft(channelId, nextDraft);
    setPendingMessageId(null);
    setMentionTrigger(null);
    pendingComposerCaretRef.current = nextCaret;
  };

  useEffect(() => {
    if (!profileMenuOpen) {
      return;
    }
    const closeOnPointer = (event: MouseEvent): void => {
      if (!profileMenuRef.current?.contains(event.target as Node)) {
        setProfileMenuOpen(false);
      }
    };
    const closeOnEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key === "Escape") {
        setProfileMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", closeOnPointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnPointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [profileMenuOpen]);

  useEffect(() => {
    let cancelled = false;
    setSettingsWorkspace(null);
    setSettingsWorkspaceError(null);
    if (!channel.canSubmitCommands) {
      return () => {
        cancelled = true;
      };
    }
    void onLoadSettingsWorkspace()
      .then((snapshot) => {
        if (!cancelled) {
          setSettingsWorkspace(snapshot);
        }
      })
      .catch((caught) => {
        if (!cancelled) {
          setSettingsWorkspaceError(
            caught instanceof Error
              ? caught.message
              : "Could not load settings and workspaces.",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [channel.canSubmitCommands, channelId, onLoadSettingsWorkspace]);

  const selectWorkspace = async (path: string): Promise<void> => {
    if (!settingsWorkspace || path === settingsWorkspace.workspace) {
      return;
    }
    setSettingsWorkspaceError(null);
    setSwitchingWorkspace(true);
    try {
      setSettingsWorkspace(
        await onSwitchWorkspace(path, settingsWorkspace.revision),
      );
      setActiveRun(null);
    } catch (caught) {
      setSettingsWorkspaceError(
        caught instanceof Error ? caught.message : "Could not switch workspace.",
      );
    } finally {
      setSwitchingWorkspace(false);
    }
  };

  // The composer rests at a single line and grows with its content, so the
  // whole draft (including a restored per-channel draft) stays visible up to
  // the stylesheet's max-height cap, past which the textarea scrolls.
  useLayoutEffect(() => {
    const composer = composerRef.current;
    if (!composer) {
      return;
    }
    // scrollHeight never reports less than the current height, so the
    // textarea collapses to auto first; otherwise deleting lines would
    // leave it stuck at its tallest size. scrollHeight excludes borders,
    // which the border-box height must include.
    const borderHeight = composer.offsetHeight - composer.clientHeight;
    composer.style.height = "auto";
    composer.style.height = `${composer.scrollHeight + borderHeight}px`;
    if (pendingComposerCaretRef.current !== null) {
      const caret = pendingComposerCaretRef.current;
      pendingComposerCaretRef.current = null;
      composer.focus();
      composer.setSelectionRange(caret, caret);
    }
  }, [draft, channelId]);

  useEffect(() => {
    if (!resizingSidebar) {
      return;
    }
    const finishResize = (): void => setResizingSidebar(false);
    window.addEventListener("pointerup", finishResize, { once: true });
    window.addEventListener("pointercancel", finishResize, { once: true });
    return () => {
      window.removeEventListener("pointerup", finishResize);
      window.removeEventListener("pointercancel", finishResize);
    };
  }, [resizingSidebar]);

  useEffect(() => {
    if (!resizingContext) {
      return;
    }
    const finishResize = (): void => setResizingContext(false);
    window.addEventListener("pointerup", finishResize, { once: true });
    window.addEventListener("pointercancel", finishResize, { once: true });
    return () => {
      window.removeEventListener("pointerup", finishResize);
      window.removeEventListener("pointercancel", finishResize);
    };
  }, [resizingContext]);

  const beginSidebarResize = (
    event: ReactPointerEvent<HTMLDivElement>,
  ): void => {
    if (sidebarLayout.collapsed) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    sidebarResizeStartRef.current = {
      pointerX: event.clientX,
      width: sidebarLayout.width,
    };
    setResizingSidebar(true);
  };

  const resizeSidebar = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (!resizingSidebar) {
      return;
    }
    const delta = event.clientX - sidebarResizeStartRef.current.pointerX;
    setSidebarLayout((layout) => ({
      ...layout,
      width: clampSidebarWidth(sidebarResizeStartRef.current.width + delta),
    }));
  };

  const resizeSidebarFromKeyboard = (
    event: ReactKeyboardEvent<HTMLDivElement>,
  ): void => {
    const direction = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
    if (direction === 0 && event.key !== "Home" && event.key !== "End") {
      return;
    }
    event.preventDefault();
    setSidebarLayout((layout) => ({
      collapsed: false,
      width:
        event.key === "Home"
          ? MIN_SIDEBAR_WIDTH_PX
          : event.key === "End"
            ? MAX_SIDEBAR_WIDTH_PX
            : clampSidebarWidth(layout.width + direction * 16 * UI_SCALE),
    }));
  };

  const toggleSidebar = (): void => {
    setSidebarLayout((layout) => ({
      ...layout,
      collapsed: !layout.collapsed,
    }));
  };

  const beginContextResize = (
    event: ReactPointerEvent<HTMLDivElement>,
  ): void => {
    event.currentTarget.setPointerCapture(event.pointerId);
    contextResizeStartRef.current = {
      pointerX: event.clientX,
      width: contextWidth,
    };
    setResizingContext(true);
  };

  const resizeContext = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (!resizingContext) {
      return;
    }
    // The pane sits on the right, so dragging the separator left widens it.
    const delta = contextResizeStartRef.current.pointerX - event.clientX;
    setContextWidth(clampContextWidth(contextResizeStartRef.current.width + delta));
  };

  const resizeContextFromKeyboard = (
    event: ReactKeyboardEvent<HTMLDivElement>,
  ): void => {
    const direction = event.key === "ArrowLeft" ? 1 : event.key === "ArrowRight" ? -1 : 0;
    if (direction === 0 && event.key !== "Home" && event.key !== "End") {
      return;
    }
    event.preventDefault();
    setContextWidth((width) =>
      event.key === "Home"
        ? MIN_CONTEXT_WIDTH_PX
        : event.key === "End"
          ? MAX_CONTEXT_WIDTH_PX
          : clampContextWidth(width + direction * 16 * UI_SCALE),
    );
  };

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
      earliestMessagePositionRef.current = 0;
      earlierAnchorRef.current = null;
      setUnseenMessageCount(0);
      setAwayFromBottom(false);
    }

    const latestPosition = messages.reduce(
      (position, message) => Math.max(position, message.eventPosition),
      0,
    );
    // Messages arrive sorted, so the first one is the window's oldest.
    const earliestPosition = messages[0]?.eventPosition ?? 0;
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
      // Derived from the clamped position, not the stored flag: the
      // restored window is the latest page only, so a position saved
      // with earlier pages loaded can clamp to (or land near) the
      // bottom, where a jump-to-latest button over a fully visible
      // timeline would be noise.
      setAwayFromBottom(!isNearTimelineBottom(timeline));
      latestMessagePositionRef.current = latestPosition;
      earliestMessagePositionRef.current = earliestPosition;
      setUnseenMessageCount(0);
      return;
    }

    // An earlier page grew the content above the viewport; restore the
    // reader's position by the height that appeared. Consumed only when
    // the window's oldest message actually moved back, so a live append
    // racing the click cannot misapply the offset.
    const anchor = earlierAnchorRef.current;
    if (
      anchor &&
      messages.length > 0 &&
      earliestPosition < earliestMessagePositionRef.current
    ) {
      earlierAnchorRef.current = null;
      timeline.scrollTop = anchor.scrollTop + (timeline.scrollHeight - anchor.scrollHeight);
    }
    earliestMessagePositionRef.current = earliestPosition;

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

  useLayoutEffect(() => {
    const timeline = timelineRef.current;
    if (
      !timeline ||
      !timelineInitializedRef.current ||
      !timelineFollowRef.current
    ) {
      return;
    }

    // The activity card and the composer live below the scrollable timeline.
    // Adding or changing the card, or the composer growing with the draft,
    // reduces the timeline viewport, which moves the effective bottom even
    // when no message was appended. Keep following only when the reader was
    // already following; a deliberate historical position remains untouched.
    // The composer's auto-grow layout effect is declared earlier in this
    // component, so the textarea has its new size before this measurement.
    timeline.scrollTop = timeline.scrollHeight;
    storeTimelineViewport(channelId, {
      follow: true,
      scrollTop: timeline.scrollTop,
    });
  }, [activeRun, channelId, draft, runPreview]);

  const handleTimelineScroll = (): void => {
    const timeline = timelineRef.current;
    if (!timeline || !timelineInitializedRef.current) {
      return;
    }
    const shouldFollow = isNearTimelineBottom(timeline);
    timelineFollowRef.current = shouldFollow;
    setAwayFromBottom(!shouldFollow);
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
    setAwayFromBottom(false);
  };

  const handleLoadEarlier = (): void => {
    const timeline = timelineRef.current;
    if (timeline) {
      earlierAnchorRef.current = {
        scrollHeight: timeline.scrollHeight,
        scrollTop: timeline.scrollTop,
      };
    }
    onLoadEarlier();
  };

  // A failed earlier-page fetch never prepends, so its captured viewport
  // must not linger and misfire on the next successful one.
  useEffect(() => {
    if (earlier.error) {
      earlierAnchorRef.current = null;
    }
  }, [earlier.error]);

  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const body = draft.trim();
    if ((!body && !selectedArtifact) || submitting || isRunActive(activeRun)) {
      return;
    }
    setSubmissionError(null);
    setSubmitting(true);
    try {
      const clientMessageId = pendingMessageId ?? createClientMessageId();
      setPendingMessageId(clientMessageId);
      const result = await onSubmitCommand(clientMessageId, body, selectedArtifact, null);
      setDraft("");
      storeDraft(channelId, "");
      setPendingMessageId(null);
      setSelectedArtifact(null);
      if (artifactInputRef.current) {
        artifactInputRef.current.value = "";
      }
      const streamed = latestRunActivityRef.current;
      if (result.run) {
        setActiveRun(
          streamed?.run.runId === result.run.runId ? streamed.run : result.run,
        );
      }
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
    <main
      className={`workshop-app ${memoryOpen ? "memory-open" : ""} ${settingsOpen ? "settings-open" : ""} ${sidebarLayout.collapsed ? "sidebar-collapsed" : ""} ${resizingSidebar || resizingContext ? "pane-resizing" : ""}`}
      style={{
        "--channel-sidebar-width": `${
          sidebarLayout.collapsed
            ? COLLAPSED_SIDEBAR_WIDTH_PX
            : sidebarLayout.width
        }px`,
        "--context-pane-width": `${contextWidth}px`,
      } as CSSProperties}
    >
      <aside
        ref={sidebarRef}
        className={`channel-sidebar ${sidebarLayout.collapsed ? "collapsed" : ""}`}
        aria-label="Workshop navigation"
      >
        <header className="sidebar-header">
          <div className="sidebar-title">
            <p className="overline">Kai Workshop</p>
          </div>
          <div className="sidebar-header-actions">
            <button
              className="sidebar-toggle"
              type="button"
              aria-label={sidebarLayout.collapsed ? "Expand navigation" : "Collapse navigation"}
              aria-expanded={!sidebarLayout.collapsed}
              title={sidebarLayout.collapsed ? "Expand navigation" : "Collapse navigation"}
              onClick={toggleSidebar}
            >
              <span aria-hidden="true">{sidebarLayout.collapsed ? "›" : "‹"}</span>
            </button>
          </div>
        </header>

        <nav>
          <p className="nav-heading">Workspace</p>
          <button
            className={`channel-link memory-link ${memoryOpen ? "active" : ""}`}
            type="button"
            aria-label="Memory"
            title="Memory"
            onClick={onOpenMemory}
          >
            <span aria-hidden="true">◇</span>
            <span>Memory</span>
            {memoryOpen && <span className="live-pip" aria-label="Open" />}
          </button>

          {workshop.channels.some(
            (availableChannel) => availableChannel.kind === "direct",
          ) && (
            <>
              <p className="nav-heading">Direct messages</p>
              {workshop.channels
                .filter((availableChannel) => availableChannel.kind === "direct")
                .map((availableChannel) => (
                  <button
                    className={`channel-link ${!auxiliaryWorkspaceOpen && availableChannel.channelId === channelId ? "active" : ""}`}
                    type="button"
                    aria-label={channelDisplayName(availableChannel)}
                    title={channelDisplayName(availableChannel)}
                    onClick={() => onSelectChannel(availableChannel.channelId)}
                    key={availableChannel.channelId}
                  >
                    <span>{channelSymbol(availableChannel)}</span>
                    <span>{channelDisplayName(availableChannel)}</span>
                    {!auxiliaryWorkspaceOpen && availableChannel.channelId === channelId && (
                      <span className="live-pip" aria-label="Live" />
                    )}
                  </button>
                ))}
            </>
          )}

          {!sidebarLayout.collapsed && (
            <div className="nav-heading-row">
              <p className="nav-heading">Channels</p>
              <button
                className="nav-add-button"
                type="button"
                aria-label="Create channel"
                title="Create channel"
                onClick={() =>
                  setChannelCreation({
                    initialAgentIds: [],
                    originChannelId: null,
                    originName: null,
                  })
                }
              >
                <span aria-hidden="true" />
              </button>
            </div>
          )}
          {workshop.channels
            .filter((availableChannel) => availableChannel.kind === "group")
            .map((availableChannel) => (
              <button
                className={`channel-link ${!auxiliaryWorkspaceOpen && availableChannel.channelId === channelId ? "active" : ""}`}
                type="button"
                aria-label={channelDisplayName(availableChannel)}
                title={channelDisplayName(availableChannel)}
                onClick={() => onSelectChannel(availableChannel.channelId)}
                key={availableChannel.channelId}
              >
                <span>{channelSymbol(availableChannel)}</span>
                <span>{channelDisplayName(availableChannel)}</span>
                {!auxiliaryWorkspaceOpen && availableChannel.channelId === channelId && (
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
                    className={`channel-link notification ${!auxiliaryWorkspaceOpen && availableChannel.channelId === channelId ? "active" : ""}`}
                    type="button"
                    aria-label={channelDisplayName(availableChannel)}
                    title={channelDisplayName(availableChannel)}
                    onClick={() => onSelectChannel(availableChannel.channelId)}
                    key={availableChannel.channelId}
                  >
                    <span>!</span>
                    <span>{channelDisplayName(availableChannel)}</span>
                    {!auxiliaryWorkspaceOpen && availableChannel.channelId === channelId && (
                      <span className="live-pip" aria-label="Live" />
                    )}
                  </button>
                ))}
            </>
          )}

          {channel.agents.length > 0 && (
            <>
              <p className="nav-heading">Agents</p>
              {channel.agents.map((agent) => (
                <div
                  className={`agent-link ${engagedAgents.some((engaged) => engaged.agentId === agent.agentId) ? "engaged" : ""}`}
                  title={agent.name}
                  key={agent.agentId}
                >
                  <span className="mini-avatar">
                    {agent.name.slice(0, 1).toUpperCase()}
                  </span>
                  <span>
                    <strong>{agent.name}</strong>
                    <small>
                      {engagedAgents.some(
                        (engaged) => engaged.agentId === agent.agentId,
                      )
                        ? "awake"
                        : "coding agent"}
                    </small>
                  </span>
                </div>
              ))}
            </>
          )}
        </nav>

        <footer
          ref={profileMenuRef}
          className={`sidebar-footer ${settingsOpen ? "active" : ""}`}
        >
          {profileMenuOpen && (
            <div className="profile-menu" role="menu" aria-label="Profile menu">
              <p className="overline">Personal</p>
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setProfileMenuOpen(false);
                  onOpenSettings();
                }}
              >
                <span aria-hidden="true">⚙</span>
                <span><strong>Settings</strong><small>Preferences and runtime</small></span>
              </button>
              <div className="profile-menu-separator" />
              <button
                className="forget-session-menu-item"
                type="button"
                role="menuitem"
                onClick={() => void forgetCurrentSession()}
              >
                <span aria-hidden="true">↪</span>
                <span><strong>Forget session</strong><small>Enroll this browser again</small></span>
              </button>
            </div>
          )}
          <button
            className="profile-trigger"
            type="button"
            aria-haspopup="menu"
            aria-expanded={profileMenuOpen}
            aria-label={`${humanName} profile`}
            title={`${humanName} — ${humanRole}`}
            onClick={() => setProfileMenuOpen((open) => !open)}
          >
            <span className="mini-avatar human">
              {humanName.slice(0, 1).toUpperCase()}
            </span>
            <span className="profile-copy">
              <strong>{humanName}</strong>
              <small>{humanRole}</small>
            </span>
            <span className="profile-chevron" aria-hidden="true">⌃</span>
          </button>
        </footer>

        {!sidebarLayout.collapsed && (
          <div
            className="sidebar-resize-handle"
            role="separator"
            aria-label="Resize navigation"
            aria-orientation="vertical"
            aria-valuemin={MIN_SIDEBAR_WIDTH_PX}
            aria-valuemax={MAX_SIDEBAR_WIDTH_PX}
            aria-valuenow={sidebarLayout.width}
            tabIndex={0}
            onKeyDown={resizeSidebarFromKeyboard}
            onPointerDown={beginSidebarResize}
            onPointerMove={resizeSidebar}
          />
        )}
      </aside>

      {settingsOpen ? (
        <SettingsWorkspace
          onAuthenticationFailure={onMemoryAuthenticationFailure}
          onChannelAccessFailure={onSettingsAccessFailure}
          onClose={() => onSelectChannel(channelId)}
          onDirtyChange={onSettingsDirtyChange}
          isAdministrator={workshop.role === "admin"}
          principalName={humanName}
          roleLabel={humanRole}
          runtimeLabel={settingsRuntimeLabel}
          runActive={isRunActive(activeRun)}
          session={settingsSession}
        />
      ) : memoryDestination ? (
        <MemoryExplorer
          initialMemoryId={memoryDestination.memoryId}
          onAuthenticationFailure={onMemoryAuthenticationFailure}
          onClose={() => onSelectChannel(channelId)}
          onSelectMemory={onSelectMemory}
          token={memoryToken}
        />
      ) : (
        <>
      <section className="conversation-pane">
        <header className="conversation-header">
          <div>
            <p className="breadcrumbs">
              {workshop.name} / {channel.kind === "notification"
                ? "Notifications"
                : channel.kind === "direct"
                  ? "Direct messages"
                  : "Channels"}
            </p>
            <h2>{symbol} {channelName}</h2>
          </div>
          <div className="conversation-actions">
            <ConnectionIndicator connection={connection} />
            <button
              className="quiet-button mobile-settings-button"
              type="button"
              onClick={onOpenSettings}
            >
              Settings
            </button>
          </div>
        </header>

        <div
          ref={timelineRef}
          className="timeline-wrap"
          aria-label="Conversation timeline"
          onScroll={handleTimelineScroll}
        >
          <div className={`channel-introduction ${channel.kind === "notification" ? "notification" : ""}`}>
            <span className="channel-symbol">{symbol}</span>
            <div>
              <p className="overline">
                {channel.kind === "notification"
                  ? "Durable notification feed"
                  : channel.kind === "direct"
                    ? "Direct conversation"
                    : "Canonical conversation"}
              </p>
              <h3>Welcome to {channelName}</h3>
              <p>
                {channel.kind === "notification"
                  ? "Authenticated GitHub activity appears here live and is delivered to every configured client."
                  : "Messages below come from Kai’s durable conversation history across every connected client."}
              </p>
              {channel.kind === "notification" && notificationPreferences && (
                <p className="notification-routing-summary">
                  Active delivery: {notificationPreferences.preferences
                    .map((item) => `${item.displayName} → ${item.destinationName}`)
                    .join(" · ")}
                </p>
              )}
            </div>
          </div>

          {earlier.available && (
            <div className="timeline-earlier">
              <button
                type="button"
                className="timeline-earlier-button"
                onClick={handleLoadEarlier}
                disabled={earlier.loading}
              >
                {earlier.loading ? "Loading earlier messages…" : "Load earlier messages"}
              </button>
              {earlier.error && <p className="timeline-earlier-error">{earlier.error}</p>}
            </div>
          )}

          {messages.length === 0 ? (
            <p className="empty-timeline">No messages yet. New activity will appear here.</p>
          ) : (
            <ol
              className={`message-list ${channel.kind === "notification" ? "notification-feed" : ""}`}
              aria-live="polite"
            >
              {displayedMessages.map((message) => (
                <MessageItem
                  key={message.messageId}
                  message={message}
                  notification={channel.kind === "notification"}
                  onDownloadArtifact={onDownloadArtifact}
                  onLoadArtifact={onLoadArtifact}
                  onSetReaction={onSetReaction}
                  onOpenThread={
                    channel.kind === "group" ? setThreadRootMessageId : undefined
                  }
                />
              ))}
              {runPreview && channel.kind !== "notification" && (
                /* The preview payload carries no author; direct channels
                   have a single agent today, so the first agent's name is
                   the correct attribution until previews learn authorship. */
                <li className="message-row agent run-preview">
                  <span className="message-avatar" aria-hidden="true">
                    {(channel.agents[0]?.name ?? "Agent").slice(0, 1).toUpperCase()}
                  </span>
                  <article>
                    <header className="message-meta">
                      <strong>{channel.agents[0]?.name ?? "Agent"}</strong>
                      <span className="run-preview-label">writing</span>
                    </header>
                    <MarkdownMessage body={runPreview.text} />
                  </article>
                </li>
              )}
            </ol>
          )}
        </div>

        <footer className="composer-preview">
          {(awayFromBottom || unseenMessageCount > 0) && (
            <button
              className="new-messages-button"
              type="button"
              onClick={followLatestMessage}
              aria-label={
                unseenMessageCount > 0 ? undefined : "Jump to latest messages"
              }
            >
              {unseenMessageCount === 0
                ? "Jump to latest"
                : unseenMessageCount === 1
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
              <input
                ref={artifactInputRef}
                className="artifact-input"
                type="file"
                aria-label="Attach a file"
                onChange={(event) => {
                  setSelectedArtifact(event.target.files?.[0] ?? null);
                  setPendingMessageId(null);
                }}
              />
              <button
                className="attach-button composer-icon-button"
                type="button"
                aria-label="Attach"
                title="Attach a file"
                disabled={submitting || isRunActive(activeRun)}
                onClick={() => artifactInputRef.current?.click()}
              >
                <PaperclipIcon />
              </button>
              <textarea
                ref={composerRef}
                aria-label={`Message ${channelName}`}
                value={draft}
                onChange={(event) => {
                  const nextDraft = event.target.value;
                  setDraft(nextDraft);
                  storeDraft(channelId, nextDraft);
                  updateMentionTrigger(
                    nextDraft,
                    event.target.selectionStart ?? nextDraft.length,
                  );
                  if (!submitting) {
                    setPendingMessageId(null);
                  }
                }}
                onKeyDown={(event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
                  if (mentionTrigger && mentionCandidates.length > 0) {
                    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                      event.preventDefault();
                      const direction = event.key === "ArrowDown" ? 1 : -1;
                      setMentionSelection((selection) =>
                        (selection + direction + mentionCandidates.length) %
                        mentionCandidates.length,
                      );
                      return;
                    }
                    if (event.key === "Escape") {
                      event.preventDefault();
                      setMentionTrigger(null);
                      return;
                    }
                    if (event.key === "Tab" || (event.key === "Enter" && !event.shiftKey)) {
                      event.preventDefault();
                      insertMention(
                        mentionCandidates[
                          Math.min(mentionSelection, mentionCandidates.length - 1)
                        ],
                      );
                      return;
                    }
                  }
                  // Enter sends the draft; Shift+Enter inserts a newline. An
                  // Enter that confirms an IME composition must never send,
                  // so composing keystrokes are left to the editor untouched.
                  // WebKit delivers the composition-confirming Enter after
                  // compositionend with isComposing already false; it is
                  // recognizable only by the legacy 229 keyCode, so that
                  // value bails as well.
                  if (
                    event.key !== "Enter" ||
                    event.shiftKey ||
                    event.nativeEvent.isComposing ||
                    event.nativeEvent.keyCode === 229
                  ) {
                    return;
                  }
                  event.preventDefault();
                  // Submitting the surrounding form keeps every send on the
                  // submit() path, so its guards (empty draft, in-flight
                  // command, active run) apply to keyboard sends as well.
                  event.currentTarget.form?.requestSubmit();
                }}
                maxLength={50000}
                placeholder={`Message ${channelName}…`}
                rows={1}
              />
              {mentionTrigger && mentionCandidates.length > 0 && (
                <div
                  className="mention-autocomplete"
                  role="listbox"
                  aria-label="Channel members"
                >
                  {mentionCandidates.map((candidate, index) => (
                    <button
                      className={index === mentionSelection ? "selected" : ""}
                      type="button"
                      role="option"
                      aria-label={`@${candidate.displayName} — ${candidate.kind}`}
                      aria-selected={index === mentionSelection}
                      key={candidate.principalId}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => insertMention(candidate)}
                    >
                      <strong>@{candidate.displayName}</strong>
                      <span>{candidate.kind}</span>
                    </button>
                  ))}
                </div>
              )}
              <button
                className="send-button composer-icon-button"
                type="submit"
                aria-busy={submitting}
                aria-label={submitting ? "Sending…" : "Send"}
                title={submitting ? "Sending…" : "Send"}
                disabled={
                  submitting ||
                  isRunActive(activeRun) ||
                  (!draft.trim() && !selectedArtifact)
                }
              >
                <SendIcon />
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
              {channel.kind === "notification"
                ? "This channel is outbound-only. Kai records delivery here, but it does not accept conversation commands."
                : "Sending messages from Workshop is not available for this conversation yet."}
            </p>
          )}
          {submissionError && (
            <p className="composer-error" role="alert">{submissionError}</p>
          )}
          {selectedArtifact && (
            <div className="selected-artifact">
              <span>{selectedArtifact.name}</span>
              <button
                type="button"
                onClick={() => {
                  setSelectedArtifact(null);
                  setPendingMessageId(null);
                  if (artifactInputRef.current) {
                    artifactInputRef.current.value = "";
                  }
                }}
              >
                Remove
              </button>
            </div>
          )}
          {!activeRun && channel.canSubmitCommands && (
            <span className="composer-mode" role="status">
              Canonical Workshop command
            </span>
          )}
        </footer>
      </section>

      <aside className="context-pane" aria-label="Channel context">
        <div
          className="context-resize-handle"
          role="separator"
          aria-label="Resize channel context"
          aria-orientation="vertical"
          aria-valuemin={MIN_CONTEXT_WIDTH_PX}
          aria-valuemax={MAX_CONTEXT_WIDTH_PX}
          aria-valuenow={contextWidth}
          tabIndex={0}
          onKeyDown={resizeContextFromKeyboard}
          onPointerDown={beginContextResize}
          onPointerMove={resizeContext}
        />
        {threadRootMessage ? (
          <ThreadPane
            channelName={channelName}
            liveMessages={threadMessages}
            onClose={() => setThreadRootMessageId(null)}
            onDownloadArtifact={onDownloadArtifact}
            onLoadArtifact={onLoadArtifact}
            onLoadThread={onLoadThread}
            onSetReaction={onSetReaction}
            onSubmitCommand={onSubmitCommand}
            reactionUpdates={reactionUpdates}
            rootMessage={threadRootMessage}
            runActive={isRunActive(activeRun)}
          />
        ) : (
        <div className="context-scroll">
          <header>
            <p className="overline">Channel context</p>
            <h2>{symbol} {channelName}</h2>
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

          {channel.kind === "group" && channel.agents.length > 0 && (
            <section className="context-section agent-attention-section">
              <span className="section-number">04</span>
              <h3>Agent attention</h3>
              <ul>
                {channel.agents.map((agent) => {
                  const engaged = engagedAgents.some(
                    (candidate) => candidate.agentId === agent.agentId,
                  );
                  return (
                    <li key={agent.agentId}>
                      <span>
                        <strong>{agent.name}</strong>
                        <small>{engaged ? "Awake in this channel" : "Not engaged"}</small>
                      </span>
                      {engaged && (
                        <button
                          className="quiet-button"
                          type="button"
                          disabled={dismissingAgentId === agent.agentId}
                          onClick={() => void dismissAgent(agent.agentId)}
                        >
                          {dismissingAgentId === agent.agentId
                            ? "Dismissing…"
                            : "Dismiss"}
                        </button>
                      )}
                    </li>
                  );
                })}
              </ul>
            </section>
          )}

          <section className="context-section trace-section">
            <span className="section-number">
              {channel.kind === "group" && channel.agents.length > 0 ? "05" : "04"}
            </span>
            <h3>Runtime and workspace</h3>
            {settingsWorkspace ? (
              <div className="runtime-settings">
                <p>
                  <strong>{settingsWorkspace.backend}</strong>
                  {settingsWorkspace.provider
                    ? ` · ${settingsWorkspace.provider}`
                    : ""}
                </p>
                <p>
                  Model: <code>{settingsWorkspace.model.value}</code>
                  <br />
                  Timeout: {settingsWorkspace.timeoutSeconds.value}s
                </p>
                <label htmlFor={`workspace-${channelId}`}>Workspace</label>
                <select
                  id={`workspace-${channelId}`}
                  value={settingsWorkspace.workspace}
                  disabled={switchingWorkspace || isRunActive(activeRun)}
                  onChange={(event) => void selectWorkspace(event.target.value)}
                >
                  {settingsWorkspace.workspaces.map((workspaceOption) => (
                    <option key={workspaceOption.path} value={workspaceOption.path}>
                      {workspaceOption.name}
                    </option>
                  ))}
                </select>
                <p className="settings-source">
                  Model: {settingsWorkspace.model.source}; timeout:{" "}
                  {settingsWorkspace.timeoutSeconds.source}
                </p>
              </div>
            ) : settingsWorkspaceError ? (
              <p className="settings-error">{settingsWorkspaceError}</p>
            ) : channel.canSubmitCommands ? (
              <p>Loading runtime settings…</p>
            ) : (
              <p>No agent runtime is assigned to this channel.</p>
            )}
          </section>

          <section className="context-section trace-section">
            <span className="section-number">
              {channel.kind === "group" && channel.agents.length > 0 ? "06" : "05"}
            </span>
            <h3>Run inspector</h3>
            <RunTraceCard
              entries={traceEntries}
              failed={traceFailed}
              loaded={traceLoaded}
              runId={inspectedRunId}
            />
          </section>

        </div>
        )}
      </aside>
      {channelCreation && (
        <ChannelCreationDialog
          agents={availableAgents}
          initialAgentIds={channelCreation.initialAgentIds}
          originChannelId={channelCreation.originChannelId}
          originName={channelCreation.originName}
          onCancel={() => setChannelCreation(null)}
          onCreate={async (input) => {
            await onCreateChannel(input);
            setChannelCreation(null);
          }}
        />
      )}
        </>
      )}
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
  destination,
  navigation,
  session,
  onAuthenticationFailure,
  onChannelAccessFailure,
  onCreateChannel,
  onForget,
  onOpenMemory,
  onOpenSettings,
  onSelectChannel,
  onSelectMemory,
  onSettingsDirtyChange,
}: {
  destination: WorkshopDestination;
  navigation: WorkshopNavigation;
  session: WorkshopSession;
  onAuthenticationFailure: (message: string) => void;
  onChannelAccessFailure: (message: string) => void;
  onCreateChannel: (input: ChannelCreationRequest) => Promise<void>;
  onForget: () => void;
  onOpenMemory: () => void;
  onOpenSettings: () => void;
  onSelectChannel: (channelId: string) => void;
  onSelectMemory: (memoryId: string | null) => void;
  onSettingsDirtyChange: (dirty: boolean) => void;
}): React.JSX.Element {
  const selected = findNavigationChannel(navigation, session.channelId);
  const availableChannels = navigation.workshops.flatMap(
    (availableWorkshop) => availableWorkshop.channels,
  );
  // Personal settings belong to the principal's direct runtime lane. A
  // mention-gated group may accept commands without exposing that channel
  // through the direct-runtime settings API.
  const settingsChannel =
    availableChannels.find(
      (availableChannel) =>
        availableChannel.kind === "direct" &&
        availableChannel.canSubmitCommands,
    ) ??
    availableChannels.find(
      (availableChannel) => availableChannel.canSubmitCommands,
    ) ??
    selected?.channel;
  const settingsSession = useMemo(
    () => ({
      channelId: settingsChannel?.channelId ?? session.channelId,
      token: session.token,
    }),
    [session.channelId, session.token, settingsChannel?.channelId],
  );
  const {
    connection,
    messages,
    threadMessages,
    reactionUpdates,
    runActivity,
    runPreview,
    runTrace,
    earlier,
    loadEarlier,
    updateReactions,
  } =
    useWorkshopTimeline(
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
  const loadSelectedRunTrace = useCallback(
    (runId: string, afterSeq: number) =>
      withAccessHandling(() => loadRunTrace(session, runId, afterSeq)),
    [session, withAccessHandling],
  );
  const cancelSelectedRun = useCallback(
    (runId: string) => withAccessHandling(() => cancelRun(session, runId)),
    [session, withAccessHandling],
  );
  const submitSelectedCommand = useCallback(
    (
      clientMessageId: string,
      body: string,
      artifact: File | null,
      threadRootId: string | null,
    ) =>
      withAccessHandling(() =>
        submitCommand(session, clientMessageId, body, artifact, threadRootId)
      ),
    [session, withAccessHandling],
  );
  const loadSelectedThread = useCallback(
    (rootMessageId: string, cursor: string | null, signal?: AbortSignal) =>
      withAccessHandling(() =>
        loadThreadTimeline(session, rootMessageId, cursor, signal)
      ),
    [session, withAccessHandling],
  );
  const loadSelectedArtifact = useCallback(
    (artifactId: string) =>
      withAccessHandling(() => loadArtifactBlob(session, artifactId)),
    [session, withAccessHandling],
  );
  const downloadSelectedArtifact = useCallback(
    (artifactId: string) => startArtifactDownload(session, artifactId),
    [session],
  );
  const loadSelectedSettingsWorkspace = useCallback(async () => {
    try {
      return await loadSettingsWorkspace(settingsSession);
    } catch (caught) {
      // Runtime settings are auxiliary channel context. A canonical channel
      // can remain readable and live even when it has no direct-runtime
      // settings authority, so only authentication failure is global here.
      if (caught instanceof AuthenticationError) {
        onAuthenticationFailure(caught.message);
      }
      throw caught;
    }
  }, [onAuthenticationFailure, settingsSession]);
  const switchSelectedWorkspace = useCallback(
    (path: string, revision: string) =>
      withAccessHandling(() => switchWorkspace(settingsSession, path, revision)),
    [settingsSession, withAccessHandling],
  );
  const dismissSelectedAgent = useCallback(
    (agentId: string, clientDismissalId: string) =>
      withAccessHandling(() =>
        dismissChannelAgent(session, agentId, clientDismissalId),
      ),
    [session, withAccessHandling],
  );
  const setSelectedMessageReaction = useCallback(
    (messageId: string, reaction: WorkshopReaction, active: boolean) =>
      withAccessHandling(async () => {
        const reactions = await setMessageReaction(
          session,
          messageId,
          reaction,
          active,
        );
        updateReactions(messageId, reactions);
      }),
    [session, updateReactions, withAccessHandling],
  );
  if (!selected) {
    return <main className="loading-workshop">Workshop access changed.</main>;
  }

  return (
    <WorkshopView
      channel={selected.channel}
      connection={connection}
      earlier={earlier}
      messages={messages}
      threadMessages={threadMessages}
      memoryDestination={destination.kind === "memory" ? destination : null}
      memoryToken={session.token}
      settingsDestination={destination.kind === "settings"}
      settingsRuntimeLabel={settingsChannel ? channelDisplayName(settingsChannel) : "assigned runtime"}
      settingsSession={settingsSession}
      navigation={navigation}
      onLoadEarlier={loadEarlier}
      onDownloadArtifact={downloadSelectedArtifact}
      onLoadArtifact={loadSelectedArtifact}
      runActivity={runActivity}
      runPreview={runPreview}
      runTrace={runTrace}
      reactionUpdates={reactionUpdates}
      workshop={selected.workshop}
      onForget={onForget}
      onCancelRun={cancelSelectedRun}
      onCreateChannel={onCreateChannel}
      onDismissAgent={dismissSelectedAgent}
      onLoadRun={loadSelectedRun}
      onLoadRunTrace={loadSelectedRunTrace}
      onLoadSettingsWorkspace={loadSelectedSettingsWorkspace}
      onLoadThread={loadSelectedThread}
      onMemoryAuthenticationFailure={onAuthenticationFailure}
      onOpenMemory={onOpenMemory}
      onOpenSettings={onOpenSettings}
      onSelectMemory={onSelectMemory}
      onSelectChannel={onSelectChannel}
      onSetReaction={setSelectedMessageReaction}
      onSubmitCommand={submitSelectedCommand}
      onSwitchWorkspace={switchSelectedWorkspace}
      onSettingsAccessFailure={onChannelAccessFailure}
      onSettingsDirtyChange={onSettingsDirtyChange}
    />
  );
}

function WorkshopApp(): React.JSX.Element {
  const confirm = useConfirmation();
  const [access, setAccess] = useState<RestoredWorkshopAccess | null>(() =>
    restoreWorkshopAccess(),
  );
  const [session, setSession] = useState<WorkshopSession | null>(null);
  const [navigation, setNavigation] = useState<WorkshopNavigation | null>(null);
  const [view, setView] = useState<"enrollment" | "workshop">(() =>
    access ? "workshop" : "enrollment",
  );
  const [notice, setNotice] = useState<string | null>(null);
  const [destination, setDestination] = useState<WorkshopDestination>(
    destinationFromLocation,
  );
  const [settingsDirty, setSettingsDirty] = useState(false);

  useEffect(() => {
    const synchronizeBrowserCredential = (event: StorageEvent): void => {
      if (
        event.storageArea !== localStorage ||
        event.key !== BROWSER_CREDENTIAL_KEY
      ) {
        return;
      }
      const token = restoreBrowserCredential(event.newValue);
      if (!token) {
        clearTabSessionState();
        clearWorkshopThemeHint();
        setAccess(null);
        setSession(null);
        setNavigation(null);
        setNotice("Workshop session was forgotten in another tab.");
        setView("enrollment");
        setSettingsDirty(false);
        return;
      }
      if (token === access?.token) {
        return;
      }
      setAccess({ channelId: restoreTabChannel(), token });
      setSession(null);
      setNavigation(null);
      setNotice(null);
      setView("workshop");
      setSettingsDirty(false);
    };
    window.addEventListener("storage", synchronizeBrowserCredential);
    return () =>
      window.removeEventListener("storage", synchronizeBrowserCredential);
  }, [access?.token]);

  useEffect(() => {
    const restoreDestination = async (): Promise<void> => {
      const restored = destinationFromLocation();
      if (
        destination.kind === "settings" &&
        restored.kind !== "settings" &&
        settingsDirty &&
        !await confirm("Discard unsaved preference changes?")
      ) {
        writeDestination({ kind: "settings" }, "push");
        return;
      }
      setDestination(restored);
    };
    const handlePopState = (): void => { void restoreDestination(); };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [confirm, destination, settingsDirty]);

  const forgetSession = useCallback((message: string | null = null): void => {
    forgetStoredSession();
    clearWorkshopThemeHint();
    setAccess(null);
    setSession(null);
    setNavigation(null);
    setNotice(message);
    setView("enrollment");
    setSettingsDirty(false);
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
      appearance: WorkshopAppearancePreferences,
    ): void => {
      const selected = preferredNavigationChannel(discovered, preferredChannelId);
      if (!selected) {
        throw new Error("This Workshop account has no accessible channels.");
      }
      const nextSession = { channelId: selected.channel.channelId, token };
      storeWorkshopAccess(nextSession);
      setAccess({ channelId: nextSession.channelId, token });
      setSession(nextSession);
      setNavigation(discovered);
      applyWorkshopTheme(appearance.themeId);
      setNotice(null);
      setView("workshop");
    },
    [],
  );

  useEffect(() => {
    if (view !== "workshop" || !access || navigation) {
      return;
    }
    let cancelled = false;
    void Promise.all([
      loadNavigation(access.token),
      loadAppearancePreferences({ token: access.token }),
    ])
      .then(([discovered, appearance]) => {
        if (!cancelled) {
          adoptNavigation(
            access.token,
            discovered,
            access.channelId,
            appearance,
          );
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
  }, [access, adoptNavigation, forgetSession, navigation, view]);

  const openChannel = async ({
    deviceDisplayName,
    enrollmentToken,
  }: {
    deviceDisplayName: string;
    enrollmentToken: string;
  }): Promise<void> => {
    const token =
      access?.token ??
      (await redeemEnrollment(enrollmentToken, deviceDisplayName));
    const [discovered, appearance] = await Promise.all([
      loadNavigation(token),
      loadAppearancePreferences({ token }),
    ]);
    adoptNavigation(token, discovered, access?.channelId ?? null, appearance);
  };

  const refreshChannelAccess = useCallback(async (message: string): Promise<void> => {
    if (!session) {
      return;
    }
    try {
      const [discovered, appearance] = await Promise.all([
        loadNavigation(session.token),
        loadAppearancePreferences({ token: session.token }),
      ]);
      adoptNavigation(session.token, discovered, session.channelId, appearance);
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
  const handleChannelAccessFailure = useCallback(
    (message: string): void => {
      void refreshChannelAccess(message);
    },
    [refreshChannelAccess],
  );

  const selectChannel = async (channelId: string): Promise<void> => {
    if (!session || !navigation || !findNavigationChannel(navigation, channelId)) {
      return;
    }
    if (
      destination.kind === "settings" &&
      settingsDirty &&
      !await confirm("Discard unsaved preference changes?")
    ) {
      return;
    }
    const nextSession = { ...session, channelId };
    storeWorkshopAccess(nextSession);
    setSession(nextSession);
    const nextDestination: WorkshopDestination = { kind: "conversation" };
    setDestination(nextDestination);
    writeDestination(nextDestination, "push");
  };

  const openMemory = async (): Promise<void> => {
    if (
      destination.kind === "settings" &&
      settingsDirty &&
      !await confirm("Discard unsaved preference changes?")
    ) {
      return;
    }
    const nextDestination: WorkshopDestination = { kind: "memory", memoryId: null };
    setDestination(nextDestination);
    writeDestination(nextDestination, "push");
  };

  const openSettings = (): void => {
    const nextDestination: WorkshopDestination = { kind: "settings" };
    setDestination(nextDestination);
    writeDestination(nextDestination, "push");
  };

  const selectMemory = useCallback((memoryId: string | null): void => {
    const nextDestination: WorkshopDestination = { kind: "memory", memoryId };
    setDestination(nextDestination);
    writeDestination(nextDestination, "replace");
  }, []);

  const createWorkshopChannel = async (
    input: ChannelCreationRequest,
  ): Promise<void> => {
    if (!session) {
      throw new Error("Workshop is not connected.");
    }
    try {
      const channelId = await createChannel(session.token, input);
      const [discovered, appearance] = await Promise.all([
        loadNavigation(session.token),
        loadAppearancePreferences({ token: session.token }),
      ]);
      adoptNavigation(session.token, discovered, channelId, appearance);
      const nextDestination: WorkshopDestination = { kind: "conversation" };
      setDestination(nextDestination);
      writeDestination(nextDestination, "push");
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        handleAuthenticationFailure(caught.message);
      } else if (caught instanceof ChannelAccessError) {
        handleChannelAccessFailure(caught.message);
      }
      throw caught;
    }
  };

  if (view === "enrollment") {
    return (
      <EnrollmentView
        key={`${access ? "correction" : "fresh"}:${notice ?? ""}`}
        existingSession={access !== null}
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
      destination={destination}
      navigation={navigation}
      session={session}
      onAuthenticationFailure={handleAuthenticationFailure}
      onChannelAccessFailure={handleChannelAccessFailure}
      onCreateChannel={createWorkshopChannel}
      onForget={() => forgetSession()}
      onOpenMemory={() => void openMemory()}
      onOpenSettings={openSettings}
      onSelectChannel={(channelId) => void selectChannel(channelId)}
      onSelectMemory={selectMemory}
      onSettingsDirtyChange={setSettingsDirty}
    />
  );
}

export default function App(): React.JSX.Element {
  return (
    <ConfirmationProvider>
      <WorkshopApp />
    </ConfirmationProvider>
  );
}
