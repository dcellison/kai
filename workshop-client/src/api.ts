import type {
  CommandSubmissionResult,
  TimelineMessage,
  TimelineSnapshot,
  WorkshopRun,
  WorkshopRunActivity,
  WorkshopRunPreview,
  WorkshopNavigation,
  WorkshopRunStatus,
  WorkshopRunTraceEntry,
  WorkshopRunTraceKind,
  WorkshopRunTracePage,
  WorkshopRunTraceSignal,
  WorkshopRunTransition,
  WorkshopSession,
} from "./types";
import {
  AGENT_PATTERN,
  CHANNEL_PATTERN,
  PRINCIPAL_PATTERN,
  WORKSHOP_PATTERN,
} from "./types";

export class AuthenticationError extends Error {}
export class ChannelAccessError extends Error {}
export class ResynchronizationRequired extends Error {}

interface StreamEvent {
  data: string;
  eventId: string | null;
  eventName: string;
}

interface StreamHandlers {
  onConnected: () => void;
  onMessage: (message: TimelineMessage, eventId: string) => void;
  onRunActivity: (activity: WorkshopRunActivity, eventId: string) => void;
  onRunPreview: (preview: WorkshopRunPreview) => void;
  onRunTrace: (signal: WorkshopRunTraceSignal) => void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function safeErrorMessage(payload: unknown, fallback: string): string {
  if (!isRecord(payload) || !isRecord(payload.error)) {
    return fallback;
  }
  const message = payload.error.message;
  return typeof message === "string" && message.length <= 200
    ? message
    : fallback;
}

async function responsePayload(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function authorizedFetch(
  session: WorkshopSession,
  path: string,
  options: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${session.token}`);
  const response = await fetch(path, {
    ...options,
    cache: "no-store",
    headers,
  });
  if (response.status === 401) {
    throw new AuthenticationError(
      "This Workshop session expired or was revoked.",
    );
  }
  if (response.status === 403) {
    throw new ChannelAccessError(
      "This session cannot access that Workshop channel.",
    );
  }
  return response;
}

function parseMessage(value: unknown, channelId: string): TimelineMessage | null {
  if (!isRecord(value)) {
    return null;
  }
  const {
    author_display_name: authorDisplayName,
    author_kind: authorKind,
    body,
    channel_id: messageChannelId,
    created_at: createdAt,
    event_position: eventPosition,
    message_id: messageId,
  } = value;
  if (
    typeof authorDisplayName !== "string" ||
    typeof authorKind !== "string" ||
    typeof body !== "string" ||
    messageChannelId !== channelId ||
    typeof createdAt !== "string" ||
    !Number.isSafeInteger(eventPosition) ||
    typeof messageId !== "string"
  ) {
    return null;
  }
  return {
    authorDisplayName,
    authorKind,
    body,
    channelId,
    createdAt,
    eventPosition: eventPosition as number,
    messageId,
  };
}

const RUN_STATUSES = new Set<WorkshopRunStatus>([
  "accepted",
  "started",
  "completed",
  "failed",
  "cancelled",
]);

const RUN_TRANSITIONS = new Set<WorkshopRunTransition>([
  "run.accepted",
  "run.started",
  "run.cancellation_requested",
  "run.completed",
  "run.failed",
  "run.cancelled",
]);

function parseRun(value: unknown, channelId: string): WorkshopRun | null {
  if (!isRecord(value)) {
    return null;
  }
  const {
    accepted_at: acceptedAt,
    cancellation_requested_at: cancellationRequestedAt,
    channel_id: runChannelId,
    result_message_id: resultMessageId,
    run_id: runId,
    started_at: startedAt,
    status,
    terminal_at: terminalAt,
    terminal_code: terminalCode,
  } = value;
  if (
    typeof acceptedAt !== "string" ||
    (cancellationRequestedAt !== null && typeof cancellationRequestedAt !== "string") ||
    runChannelId !== channelId ||
    (resultMessageId !== null && typeof resultMessageId !== "string") ||
    typeof runId !== "string" ||
    (startedAt !== null && typeof startedAt !== "string") ||
    typeof status !== "string" ||
    !RUN_STATUSES.has(status as WorkshopRunStatus) ||
    (terminalAt !== null && typeof terminalAt !== "string") ||
    (terminalCode !== null && typeof terminalCode !== "string")
  ) {
    return null;
  }
  return {
    acceptedAt,
    cancellationRequestedAt,
    channelId,
    resultMessageId,
    runId,
    startedAt,
    status: status as WorkshopRunStatus,
    terminalAt,
    terminalCode,
  };
}

export async function redeemEnrollment(
  enrollmentToken: string,
  deviceDisplayName: string,
): Promise<string> {
  const response = await fetch("/v1/client/enrollment/redeem", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      device_display_name: deviceDisplayName,
      enrollment_token: enrollmentToken,
    }),
    cache: "no-store",
  });
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    !isRecord(payload.session) ||
    typeof payload.session.token !== "string" ||
    payload.session.token.length === 0
  ) {
    throw new Error(safeErrorMessage(payload, "Enrollment failed."));
  }
  return payload.session.token;
}

export async function loadNavigation(token: string): Promise<WorkshopNavigation> {
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/navigation",
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(
      safeErrorMessage(payload, "Could not load Workshop navigation."),
    );
  }
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !isRecord(payload.principal) ||
    typeof payload.principal.principal_id !== "string" ||
    !PRINCIPAL_PATTERN.test(payload.principal.principal_id) ||
    typeof payload.principal.display_name !== "string" ||
    !Array.isArray(payload.workshops)
  ) {
    throw new Error("Kai returned unsupported Workshop navigation.");
  }

  const workshops = payload.workshops.map((rawWorkshop) => {
    if (
      !isRecord(rawWorkshop) ||
      typeof rawWorkshop.workshop_id !== "string" ||
      !WORKSHOP_PATTERN.test(rawWorkshop.workshop_id) ||
      typeof rawWorkshop.name !== "string" ||
      typeof rawWorkshop.role !== "string" ||
      !Array.isArray(rawWorkshop.channels)
    ) {
      throw new Error("Kai returned unsupported Workshop navigation.");
    }
    const channels = rawWorkshop.channels.map((rawChannel) => {
      if (
        !isRecord(rawChannel) ||
        typeof rawChannel.channel_id !== "string" ||
        !CHANNEL_PATTERN.test(rawChannel.channel_id) ||
        (rawChannel.name !== null && typeof rawChannel.name !== "string") ||
        !["direct", "group", "notification"].includes(
          String(rawChannel.kind),
        ) ||
        typeof rawChannel.role !== "string" ||
        typeof rawChannel.can_submit_commands !== "boolean" ||
        !Array.isArray(rawChannel.agents) ||
        !Array.isArray(rawChannel.participants)
      ) {
        throw new Error("Kai returned unsupported Workshop navigation.");
      }
      const agents = rawChannel.agents.map((rawAgent) => {
        if (
          !isRecord(rawAgent) ||
          typeof rawAgent.agent_id !== "string" ||
          !AGENT_PATTERN.test(rawAgent.agent_id) ||
          typeof rawAgent.name !== "string"
        ) {
          throw new Error("Kai returned unsupported Workshop navigation.");
        }
        return { agentId: rawAgent.agent_id, name: rawAgent.name };
      });
      const participants = rawChannel.participants.map((rawParticipant) => {
        if (
          !isRecord(rawParticipant) ||
          typeof rawParticipant.principal_id !== "string" ||
          !PRINCIPAL_PATTERN.test(rawParticipant.principal_id) ||
          typeof rawParticipant.kind !== "string" ||
          typeof rawParticipant.display_name !== "string"
        ) {
          throw new Error("Kai returned unsupported Workshop navigation.");
        }
        return {
          displayName: rawParticipant.display_name,
          kind: rawParticipant.kind,
          principalId: rawParticipant.principal_id,
        };
      });
      return {
        agents,
        canSubmitCommands: rawChannel.can_submit_commands,
        channelId: rawChannel.channel_id,
        kind: rawChannel.kind as "direct" | "group" | "notification",
        name: rawChannel.name,
        participants,
        role: rawChannel.role,
      };
    });
    return {
      channels,
      name: rawWorkshop.name,
      role: rawWorkshop.role,
      workshopId: rawWorkshop.workshop_id,
    };
  });
  return {
    principal: {
      displayName: payload.principal.display_name,
      principalId: payload.principal.principal_id,
    },
    workshops,
  };
}

export async function submitCommand(
  session: WorkshopSession,
  clientMessageId: string,
  body: string,
): Promise<CommandSubmissionResult> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/commands`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body, client_message_id: clientMessageId }),
    },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Kai could not run this command."));
  }
  if (
    !isRecord(payload) ||
    payload.version !== 2 ||
    typeof payload.acceptance !== "string" ||
    typeof payload.message_id !== "string" ||
    typeof payload.run_id !== "string" ||
    !isRecord(payload.run)
  ) {
    throw new Error("Kai returned an unsupported command response.");
  }
  const run = parseRun(payload.run, session.channelId);
  if (!run || run.runId !== payload.run_id) {
    throw new Error("Kai returned an unsupported command response.");
  }
  return {
    acceptance: payload.acceptance,
    messageId: payload.message_id,
    run,
  };
}

export async function loadRun(
  session: WorkshopSession,
  runId: string,
): Promise<WorkshopRun> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/runs/${encodeURIComponent(runId)}`,
  );
  const payload = await responsePayload(response);
  const run = isRecord(payload) ? parseRun(payload.run, session.channelId) : null;
  if (!response.ok || payload === null) {
    throw new Error(safeErrorMessage(payload, "Could not inspect this run."));
  }
  if (!isRecord(payload) || payload.version !== 1 || !run || run.runId !== runId) {
    throw new Error("Kai returned an unsupported run response.");
  }
  return run;
}

const TRACE_KINDS = new Set<WorkshopRunTraceKind>(["tool_call", "tool_result", "truncated"]);

function parseTraceEntry(value: unknown): WorkshopRunTraceEntry | null {
  if (
    !isRecord(value) ||
    typeof value.seq !== "number" ||
    !Number.isSafeInteger(value.seq) ||
    typeof value.kind !== "string" ||
    !TRACE_KINDS.has(value.kind as WorkshopRunTraceKind) ||
    typeof value.summary !== "string" ||
    typeof value.detail !== "string" ||
    typeof value.is_diff !== "boolean" ||
    typeof value.is_error !== "boolean" ||
    typeof value.created_at !== "string" ||
    (value.tool_name !== null && typeof value.tool_name !== "string") ||
    (value.tool_use_id !== null && typeof value.tool_use_id !== "string")
  ) {
    return null;
  }
  return {
    createdAt: value.created_at,
    detail: value.detail,
    isDiff: value.is_diff,
    isError: value.is_error,
    kind: value.kind as WorkshopRunTraceKind,
    seq: value.seq,
    summary: value.summary,
    toolName: value.tool_name,
    toolUseId: value.tool_use_id,
  };
}

export async function loadRunTrace(
  session: WorkshopSession,
  runId: string,
  afterSeq: number,
): Promise<WorkshopRunTracePage> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/runs/${encodeURIComponent(runId)}/trace` +
      `?after_seq=${encodeURIComponent(String(afterSeq))}`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load this run's trace."));
  }
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.run_id !== runId ||
    !Array.isArray(payload.entries) ||
    typeof payload.has_more !== "boolean"
  ) {
    throw new Error("Kai returned an unsupported trace response.");
  }
  const entries: WorkshopRunTraceEntry[] = [];
  for (const value of payload.entries) {
    const entry = parseTraceEntry(value);
    if (!entry) {
      throw new Error("Kai returned an unsupported trace response.");
    }
    entries.push(entry);
  }
  return { entries, hasMore: payload.has_more };
}

export async function cancelRun(
  session: WorkshopSession,
  runId: string,
): Promise<WorkshopRun> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/runs/${encodeURIComponent(runId)}/cancel`,
    { method: "POST" },
  );
  const payload = await responsePayload(response);
  const run = isRecord(payload) ? parseRun(payload.run, session.channelId) : null;
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not stop this run."));
  }
  if (!isRecord(payload) || payload.version !== 1 || !run || run.runId !== runId) {
    throw new Error("Kai returned an unsupported cancellation response.");
  }
  return run;
}

function parseTimelinePage(payload: unknown, channelId: string): TimelineSnapshot {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.channel_id !== channelId ||
    !Array.isArray(payload.messages) ||
    !Number.isSafeInteger(payload.through_position)
  ) {
    throw new Error("Kai returned an unsupported timeline response.");
  }
  const messages: TimelineMessage[] = [];
  for (const rawMessage of payload.messages) {
    const message = parseMessage(rawMessage, channelId);
    if (!message) {
      throw new Error("Kai returned an unsupported timeline message.");
    }
    messages.push(message);
  }
  return {
    messages,
    throughPosition: payload.through_position as number,
    previousCursor:
      typeof payload.previous_cursor === "string" ? payload.previous_cursor : null,
  };
}

export async function loadTimeline(
  session: WorkshopSession,
  signal: AbortSignal,
): Promise<TimelineSnapshot> {
  // Tail-first: one bounded request for the newest window, so opening a
  // channel costs the same regardless of how long its history is.
  // Earlier history stays behind previousCursor and loads on demand.
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/timeline?tail=1&limit=100`,
    { signal },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load this channel."));
  }
  return parseTimelinePage(payload, session.channelId);
}

export async function loadEarlierTimeline(
  session: WorkshopSession,
  cursor: string,
  expectedThroughPosition: number,
  signal: AbortSignal,
): Promise<TimelineSnapshot> {
  const query = new URLSearchParams({ cursor, limit: "100" });
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/timeline?${query}`,
    { signal },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load earlier messages."));
  }
  const page = parseTimelinePage(payload, session.channelId);
  // The cursor pins the snapshot server-side, so a different bound here
  // means the pages cannot belong together; fail loudly over merging
  // history from two snapshots.
  if (page.throughPosition !== expectedThroughPosition) {
    throw new Error("The timeline snapshot changed while it was loading.");
  }
  return page;
}

export class EventStreamDecoder {
  private buffer = "";

  push(chunk: string): StreamEvent[] {
    this.buffer += chunk;
    const events: StreamEvent[] = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const event = this.parseBlock(this.buffer.slice(0, boundary));
      this.buffer = this.buffer.slice(boundary + 2);
      if (event) {
        events.push(event);
      }
      boundary = this.buffer.indexOf("\n\n");
    }
    return events;
  }

  private parseBlock(block: string): StreamEvent | null {
    let eventName: string | null = null;
    let eventId: string | null = null;
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) {
        continue;
      }
      const separator = line.indexOf(":");
      const field = separator === -1 ? line : line.slice(0, separator);
      let value = separator === -1 ? "" : line.slice(separator + 1);
      if (value.startsWith(" ")) {
        value = value.slice(1);
      }
      if (field === "event") {
        eventName = value;
      } else if (field === "id") {
        eventId = value;
      } else if (field === "data") {
        data.push(value);
      }
    }
    return eventName
      ? { data: data.join("\n"), eventId, eventName }
      : null;
  }
}

export async function streamTimeline(
  session: WorkshopSession,
  lastEventId: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const headers = new Headers({ "Last-Event-ID": lastEventId });
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/events`,
    { headers, signal },
  );
  if (response.status === 409) {
    throw new ResynchronizationRequired();
  }
  if (!response.ok || !response.body) {
    const payload = await responsePayload(response);
    throw new Error(
      safeErrorMessage(payload, "Live updates are unavailable."),
    );
  }

  handlers.onConnected();
  const reader = response.body.getReader();
  const textDecoder = new TextDecoder();
  const eventDecoder = new EventStreamDecoder();
  while (!signal.aborted) {
    const { done, value } = await reader.read();
    if (done) {
      return;
    }
    const events = eventDecoder.push(textDecoder.decode(value, { stream: true }));
    for (const event of events) {
      if (event.eventName === "run.preview.updated") {
        // Previews are ephemeral display state and deliberately carry no
        // SSE id, so they are handled before the resume-cursor guard and
        // never advance Last-Event-ID.
        let previewPayload: unknown;
        try {
          previewPayload = JSON.parse(event.data);
        } catch {
          continue;
        }
        if (
          !isRecord(previewPayload) ||
          previewPayload.version !== 1 ||
          previewPayload.channel_id !== session.channelId ||
          typeof previewPayload.run_id !== "string" ||
          typeof previewPayload.text !== "string" ||
          typeof previewPayload.sequence !== "number" ||
          !Number.isSafeInteger(previewPayload.sequence)
        ) {
          continue;
        }
        handlers.onRunPreview({
          runId: previewPayload.run_id,
          sequence: previewPayload.sequence,
          text: previewPayload.text,
        });
        continue;
      }
      if (event.eventName === "run.trace.updated") {
        // Trace doorbells deliberately carry no SSE id, so they are
        // handled before the resume-cursor guard and never advance
        // Last-Event-ID; the trace endpoint is the source of truth.
        let tracePayload: unknown;
        try {
          tracePayload = JSON.parse(event.data);
        } catch {
          continue;
        }
        if (
          !isRecord(tracePayload) ||
          tracePayload.version !== 1 ||
          tracePayload.channel_id !== session.channelId ||
          typeof tracePayload.run_id !== "string" ||
          typeof tracePayload.seq !== "number" ||
          !Number.isSafeInteger(tracePayload.seq)
        ) {
          continue;
        }
        handlers.onRunTrace({
          runId: tracePayload.run_id,
          seq: tracePayload.seq,
        });
        continue;
      }
      if (
        !event.eventId ||
        !/^\d+$/.test(event.eventId)
      ) {
        continue;
      }
      let payload: unknown;
      try {
        payload = JSON.parse(event.data);
      } catch {
        continue;
      }
      if (!isRecord(payload) || payload.version !== 1) {
        continue;
      }
      const eventPosition = Number(event.eventId);
      if (!Number.isSafeInteger(eventPosition) || payload.channel_id !== session.channelId) {
        continue;
      }
      if (event.eventName === "timeline.message.created") {
        const message = parseMessage(payload.message, session.channelId);
        if (!message || message.eventPosition !== eventPosition) {
          continue;
        }
        handlers.onMessage(message, event.eventId);
        continue;
      }
      if (event.eventName === "run.lifecycle.changed") {
        const run = parseRun(payload.run, session.channelId);
        const transition = payload.transition;
        if (
          !run ||
          payload.event_position !== eventPosition ||
          typeof payload.occurred_at !== "string" ||
          typeof transition !== "string" ||
          !RUN_TRANSITIONS.has(transition as WorkshopRunTransition)
        ) {
          continue;
        }
        handlers.onRunActivity(
          {
            eventPosition,
            occurredAt: payload.occurred_at,
            run,
            transition: transition as WorkshopRunTransition,
          },
          event.eventId,
        );
      }
    }
  }
}
