import type {
  CommandSubmissionResult,
  TimelineMessage,
  TimelineSnapshot,
  WorkshopRun,
  WorkshopRunActivity,
  WorkshopNavigation,
  WorkshopRunStatus,
  WorkshopRunTransition,
  WorkshopSession,
} from "./types";
import {
  AGENT_PATTERN,
  CHANNEL_PATTERN,
  PRINCIPAL_PATTERN,
  WORKSHOP_PATTERN,
} from "./types";

const MAX_TIMELINE_PAGES = 1000;

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
        !Array.isArray(rawChannel.agents)
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
      return {
        agents,
        canSubmitCommands: rawChannel.can_submit_commands,
        channelId: rawChannel.channel_id,
        kind: rawChannel.kind as "direct" | "group" | "notification",
        name: rawChannel.name,
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

export async function loadTimeline(
  session: WorkshopSession,
  signal: AbortSignal,
): Promise<TimelineSnapshot> {
  const messages: TimelineMessage[] = [];
  let cursor: string | null = null;
  let throughPosition: number | null = null;
  let pageCount = 0;

  do {
    const query = new URLSearchParams({ limit: "100" });
    if (cursor) {
      query.set("cursor", cursor);
    }
    const response = await authorizedFetch(
      session,
      `/v1/channels/${encodeURIComponent(session.channelId)}/timeline?${query}`,
      { signal },
    );
    const payload = await responsePayload(response);
    if (!response.ok) {
      throw new Error(
        safeErrorMessage(payload, "Could not load this channel."),
      );
    }
    if (
      !isRecord(payload) ||
      payload.version !== 1 ||
      payload.channel_id !== session.channelId ||
      !Array.isArray(payload.messages) ||
      !Number.isSafeInteger(payload.through_position)
    ) {
      throw new Error("Kai returned an unsupported timeline response.");
    }

    const pageThroughPosition = payload.through_position as number;
    if (throughPosition === null) {
      throughPosition = pageThroughPosition;
    } else if (throughPosition !== pageThroughPosition) {
      throw new Error("The timeline snapshot changed while it was loading.");
    }
    for (const rawMessage of payload.messages) {
      const message = parseMessage(rawMessage, session.channelId);
      if (!message) {
        throw new Error("Kai returned an unsupported timeline message.");
      }
      messages.push(message);
    }
    cursor =
      typeof payload.next_cursor === "string" ? payload.next_cursor : null;
    pageCount += 1;
    if (pageCount > MAX_TIMELINE_PAGES) {
      throw new Error("The timeline exceeded the client safety limit.");
    }
  } while (cursor);

  if (throughPosition === null) {
    throw new Error("Kai returned an unsupported timeline response.");
  }
  return { messages, throughPosition };
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
