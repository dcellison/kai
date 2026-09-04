import type {
  CommandSubmissionResult,
  TimelineMessage,
  TimelineSnapshot,
  ThreadTimelineSnapshot,
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
  WorkshopEditableCapability,
  WorkshopSettingsMutation,
  WorkshopModelCatalogue,
  WorkshopSettingsWorkspace,
  WorkshopRoutingEligibility,
  WorkshopRoutingPolicy,
  WorkshopRunRoutingDecision,
  WorkshopRoutingTaskClass,
  WorkshopWorkspaceConfig,
  WorkshopPreferenceDocument,
  WorkshopPreferenceHistory,
  WorkshopGitHubSettings,
  WorkshopGitHubSettingsChange,
  WorkshopNotificationPreferences,
  WorkshopNotificationPreferenceChange,
  WorkshopChannelNotificationPolicy,
  WorkshopChannelNotificationPolicyChange,
  WorkshopClientPreferences,
  WorkshopClientPreferenceChange,
  WorkshopAppearancePreferences,
  WorkshopHumanProfile,
  WorkshopHumanAvatar,
  WorkshopHumanAvatarDescriptor,
  WorkshopRuntimeSettingsChange,
  WorkshopWorkspaceSettingChange,
  WorkshopMemoryPage,
  WorkshopMemoryFilters,
  WorkshopMemoryListOptions,
  WorkshopMemoryDetail,
  WorkshopMemoryEditResult,
  WorkshopMemoryCreationResult,
  WorkshopMemoryMutationBatch,
  WorkshopMemoryRecord,
  WorkshopMemoryScope,
  WorkshopMemorySearch,
  WorkshopMemorySearchOptions,
  WorkshopMemorySourceContext,
  WorkshopMemorySourceMessage,
  WorkshopMemoryStats,
  WorkshopArtifactKind,
  WorkshopArtifactSummary,
  WorkshopMessageReaction,
  WorkshopReaction,
  WorkshopAgentCapability,
  WorkshopAgentChangeSignal,
  WorkshopAgentDefinition,
  WorkshopAgentEnablement,
  WorkshopAgentSummary,
  WorkshopHumanChannelMember,
  WorkshopHumanConversation,
  WorkshopHumanMembership,
  WorkshopHumanPeer,
  WorkshopHumanNotification,
  WorkshopHumanNotificationCounts,
  WorkshopHumanNotificationMutation,
  WorkshopHumanNotificationPage,
  WorkshopHumanNotificationSignal,
  WorkshopChannelReadPositionMutation,
  WorkshopChannelUnreadSignal,
  WorkshopChannelUnreadSnapshot,
  WorkshopChannelUnreadState,
  WorkshopThreadUnreadMutation,
  WorkshopThreadUnreadSignal,
  WorkshopThreadUnreadState,
  WorkshopFollowedThread,
  WorkshopFollowedThreadSnapshot,
  WorkshopPrincipalEventBatch,
  WorkshopReplyParticipant,
} from "./types";
import { HUMAN_NOTIFICATION_PATTERN, MESSAGE_PATTERN } from "./types";
import { isWorkshopThemeId } from "./theme";
import {
  AGENT_PATTERN,
  AGENT_DEFINITION_PATTERN,
  AGENT_ENABLEMENT_PATTERN,
  AGENT_REVISION_PATTERN,
  ARTIFACT_PATTERN,
  CHANNEL_PATTERN,
  HUMAN_HANDLE_PATTERN,
  PRINCIPAL_PATTERN,
  RUNTIME_PROFILE_PATTERN,
  WORKSHOP_PATTERN,
} from "./types";

export class AuthenticationError extends Error {}
export class ChannelAccessError extends Error {}
export class ChannelReadPositionConflictError extends Error {}
export class ThreadUnreadConflictError extends Error {}
export class ResynchronizationRequired extends Error {}
export class PreferenceRevisionConflictError extends Error {
  constructor(
    message: string,
    public readonly currentRevision: string,
  ) {
    super(message);
  }
}
export class SettingsRevisionConflictError extends Error {}
export class MemoryRevisionConflictError extends Error {
  constructor(
    message: string,
    public readonly currentRevision: string,
  ) {
    super(message);
  }
}

interface StreamEvent {
  data: string;
  eventId: string | null;
  eventName: string;
}

interface StreamHandlers {
  onConnected: () => void;
  onMessage: (message: TimelineMessage, eventId: string) => void;
  onReactions?: (
    messageId: string,
    reactions: WorkshopMessageReaction[],
    eventId: string,
  ) => void;
  onRunActivity: (activity: WorkshopRunActivity, eventId: string) => void;
  onRunPreview: (preview: WorkshopRunPreview) => void;
  onRunTrace: (signal: WorkshopRunTraceSignal) => void;
}

const REACTIONS = new Set<WorkshopReaction>([
  "thumbs_up",
  "thumbs_down",
  "heart",
  "laugh",
  "celebrate",
  "eyes",
  "check",
  "thinking",
  "surprised",
  "sad",
  "fire",
  "question",
]);

function parseReactions(value: unknown): WorkshopMessageReaction[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const reactions: WorkshopMessageReaction[] = [];
  for (const item of value) {
    if (!isRecord(item)) {
      return null;
    }
    const {
      count,
      reacted_by_viewer: reactedByViewer,
      reaction,
    } = item;
    if (
      typeof reaction !== "string" ||
      !REACTIONS.has(reaction as WorkshopReaction) ||
      !Number.isSafeInteger(count) ||
      (count as number) < 1 ||
      typeof reactedByViewer !== "boolean"
    ) {
      return null;
    }
    reactions.push({
      count: count as number,
      reactedByViewer,
      reaction: reaction as WorkshopReaction,
    });
  }
  return reactions;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function inactiveHumanAvatar(): WorkshopHumanAvatarDescriptor {
  return { active: false, stateVersion: 0, url: null };
}

function parseHumanAvatarDescriptor(
  value: unknown,
  principalId: string,
): WorkshopHumanAvatarDescriptor {
  if (!isRecord(value)) {
    return inactiveHumanAvatar();
  }
  const { active, state_version: stateVersion, url } = value;
  if (
    typeof active !== "boolean" ||
    !Number.isSafeInteger(stateVersion) ||
    (stateVersion as number) < 0 ||
    (active && (stateVersion as number) < 1) ||
    (active
      ? url !== `/v1/principals/${principalId}/avatar/${stateVersion}`
      : url !== null)
  ) {
    return inactiveHumanAvatar();
  }
  return {
    active,
    stateVersion: stateVersion as number,
    url: url as string | null,
  };
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

const EVENT_STREAM_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
let activeEventStreamId: string | null = null;

function createEventStreamId(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function eventStreamId(): string {
  if (
    activeEventStreamId !== null &&
    EVENT_STREAM_ID_PATTERN.test(activeEventStreamId)
  ) {
    return activeEventStreamId;
  }
  // getRandomValues remains available on plain-HTTP LAN origins, unlike
  // randomUUID(), which browsers restrict to secure contexts. Keep this
  // identity page-local: sessionStorage may be cloned into a duplicated tab,
  // which would make otherwise independent event streams replace each other.
  activeEventStreamId = createEventStreamId();
  return activeEventStreamId;
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
    author_avatar: authorAvatar,
    author_kind: authorKind,
    author_principal_id: authorPrincipalId,
    body,
    channel_id: messageChannelId,
    created_at: createdAt,
    event_position: eventPosition,
    latest_reply_at: latestReplyAt,
    message_id: messageId,
    reply_count: replyCount,
    reply_participant_count: replyParticipantCount,
    reply_participants: suppliedReplyParticipants,
    reply_to_message_id: replyToMessageId,
    thread_root_id: threadRootId,
    mentions: suppliedMentions,
    reactions: suppliedReactions,
    artifacts: suppliedArtifacts,
  } = value;
  const rawArtifacts = suppliedArtifacts ?? [];
  const rawReactions = suppliedReactions ?? [];
  if (
    typeof authorDisplayName !== "string" ||
    typeof authorKind !== "string" ||
    typeof authorPrincipalId !== "string" ||
    !PRINCIPAL_PATTERN.test(authorPrincipalId) ||
    typeof body !== "string" ||
    messageChannelId !== channelId ||
    typeof createdAt !== "string" ||
    !Number.isSafeInteger(eventPosition) ||
    typeof messageId !== "string" ||
    !MESSAGE_PATTERN.test(messageId) ||
    !Number.isSafeInteger(replyCount) ||
    (replyCount as number) < 0 ||
    !Number.isSafeInteger(replyParticipantCount) ||
    (replyParticipantCount as number) < 0 ||
    !Array.isArray(suppliedReplyParticipants) ||
    suppliedReplyParticipants.length > 3 ||
    (replyParticipantCount as number) < suppliedReplyParticipants.length ||
    (replyToMessageId !== null &&
      (typeof replyToMessageId !== "string" || !MESSAGE_PATTERN.test(replyToMessageId))) ||
    (threadRootId !== null &&
      (typeof threadRootId !== "string" || !MESSAGE_PATTERN.test(threadRootId))) ||
    (latestReplyAt !== null && typeof latestReplyAt !== "string") ||
    !Array.isArray(suppliedMentions) ||
    !Array.isArray(rawReactions) ||
    !Array.isArray(rawArtifacts)
  ) {
    return null;
  }
  const artifacts: WorkshopArtifactSummary[] = [];
  for (const rawArtifact of rawArtifacts) {
    if (!isRecord(rawArtifact)) {
      return null;
    }
    const {
      artifact_id: artifactId,
      byte_size: byteSize,
      content_sha256: contentSha256,
      created_at: artifactCreatedAt,
      kind,
      media_type: mediaType,
      original_filename: originalFilename,
    } = rawArtifact;
    if (
      typeof artifactId !== "string" ||
      !ARTIFACT_PATTERN.test(artifactId) ||
      !Number.isSafeInteger(byteSize) ||
      typeof contentSha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(contentSha256) ||
      typeof artifactCreatedAt !== "string" ||
      typeof kind !== "string" ||
      !["photo", "document", "voice"].includes(kind) ||
      typeof mediaType !== "string" ||
      (originalFilename !== null && typeof originalFilename !== "string")
    ) {
      return null;
    }
    artifacts.push({
      artifactId,
      byteSize: byteSize as number,
      contentSha256,
      createdAt: artifactCreatedAt,
      kind: kind as WorkshopArtifactKind,
      mediaType,
      originalFilename,
    });
  }
  const mentions = [];
  const bodyLength = Array.from(body).length;
  let previousEnd = 0;
  for (const rawMention of suppliedMentions) {
    if (!isRecord(rawMention)) {
      return null;
    }
    const { kind, length, principal_id: principalId, start } = rawMention;
    if (
      !["human", "agent"].includes(String(kind)) ||
      !Number.isSafeInteger(length) ||
      (length as number) <= 1 ||
      typeof principalId !== "string" ||
      !PRINCIPAL_PATTERN.test(principalId) ||
      !Number.isSafeInteger(start) ||
      (start as number) < previousEnd ||
      (start as number) + (length as number) > bodyLength
    ) {
      return null;
    }
    mentions.push({
      kind: kind as "human" | "agent",
      length: length as number,
      principalId,
      start: start as number,
    });
    previousEnd = (start as number) + (length as number);
  }
  const reactions = parseReactions(rawReactions);
  if (reactions === null) {
    return null;
  }
  const replyParticipants: WorkshopReplyParticipant[] = [];
  for (const rawParticipant of suppliedReplyParticipants) {
    if (!isRecord(rawParticipant)) {
      return null;
    }
    const {
      avatar,
      display_name: displayName,
      kind,
      principal_id: principalId,
    } = rawParticipant;
    if (
      typeof principalId !== "string" ||
      !PRINCIPAL_PATTERN.test(principalId) ||
      (kind !== "human" && kind !== "agent") ||
      typeof displayName !== "string" ||
      !displayName.trim()
    ) {
      return null;
    }
    replyParticipants.push({
      avatar: kind === "human"
        ? parseHumanAvatarDescriptor(avatar, principalId)
        : null,
      displayName,
      kind,
      principalId,
    });
  }
  return {
    artifacts,
    authorAvatar: authorKind === "human"
      ? parseHumanAvatarDescriptor(authorAvatar, authorPrincipalId)
      : null,
    authorDisplayName,
    authorKind,
    authorPrincipalId,
    body,
    channelId,
    createdAt,
    eventPosition: eventPosition as number,
    mentions,
    messageId,
    reactions,
    replyCount: replyCount as number,
    replyParticipantCount: replyParticipantCount as number,
    replyParticipants,
    replyToMessageId,
    latestReplyAt,
    threadRootId,
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
    (payload.principal.handle !== null &&
      (typeof payload.principal.handle !== "string" ||
        !HUMAN_HANDLE_PATTERN.test(payload.principal.handle))) ||
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
        (rawChannel.archived_at !== undefined &&
          rawChannel.archived_at !== null &&
          typeof rawChannel.archived_at !== "string") ||
        (rawChannel.lifecycle_event_position !== undefined &&
          rawChannel.lifecycle_event_position !== null &&
          (typeof rawChannel.lifecycle_event_position !== "number" ||
            !Number.isSafeInteger(rawChannel.lifecycle_event_position))) ||
        (rawChannel.direct_message_archived_at !== undefined &&
          rawChannel.direct_message_archived_at !== null &&
          typeof rawChannel.direct_message_archived_at !== "string") ||
        (rawChannel.direct_message_archive_event_position !== undefined &&
          rawChannel.direct_message_archive_event_position !== null &&
          (typeof rawChannel.direct_message_archive_event_position !== "number" ||
            !Number.isSafeInteger(
              rawChannel.direct_message_archive_event_position,
            ))) ||
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
          typeof rawAgent.principal_id !== "string" ||
          !PRINCIPAL_PATTERN.test(rawAgent.principal_id) ||
          typeof rawAgent.engaged !== "boolean" ||
          typeof rawAgent.available !== "boolean" ||
          !["draft", "active", "archived"].includes(
            String(rawAgent.lifecycle_state),
          ) ||
          (rawAgent.engaged_until !== null &&
            typeof rawAgent.engaged_until !== "string") ||
          !["private", "shared_channel"].includes(String(rawAgent.memory_scope)) ||
          (rawAgent.runtime_profile_id !== null &&
            (typeof rawAgent.runtime_profile_id !== "string" ||
              !RUNTIME_PROFILE_PATTERN.test(rawAgent.runtime_profile_id))) ||
          (rawAgent.sponsor_principal_id !== null &&
            (typeof rawAgent.sponsor_principal_id !== "string" ||
              !PRINCIPAL_PATTERN.test(rawAgent.sponsor_principal_id))) ||
          (rawAgent.sponsor_display_name !== null &&
            typeof rawAgent.sponsor_display_name !== "string") ||
          typeof rawAgent.name !== "string" ||
          typeof rawAgent.handle !== "string" ||
          !HUMAN_HANDLE_PATTERN.test(rawAgent.handle)
        ) {
          throw new Error("Kai returned unsupported Workshop navigation.");
        }
        return {
          agentId: rawAgent.agent_id,
          available: rawAgent.available,
          engaged: rawAgent.engaged,
          engagedUntil: rawAgent.engaged_until,
          handle: rawAgent.handle,
          lifecycleState: rawAgent.lifecycle_state as WorkshopAgentSummary["lifecycleState"],
          memoryScope: rawAgent.memory_scope as "private" | "shared_channel",
          name: rawAgent.name,
          principalId: rawAgent.principal_id,
          runtimeProfileId: rawAgent.runtime_profile_id,
          sponsorDisplayName: rawAgent.sponsor_display_name,
          sponsorPrincipalId: rawAgent.sponsor_principal_id,
        };
      });
      const participants = rawChannel.participants.map((rawParticipant) => {
        if (
          !isRecord(rawParticipant) ||
          typeof rawParticipant.principal_id !== "string" ||
          !PRINCIPAL_PATTERN.test(rawParticipant.principal_id) ||
          typeof rawParticipant.kind !== "string" ||
          typeof rawParticipant.display_name !== "string" ||
          (rawParticipant.handle !== null &&
            (typeof rawParticipant.handle !== "string" ||
              !HUMAN_HANDLE_PATTERN.test(rawParticipant.handle)))
        ) {
          throw new Error("Kai returned unsupported Workshop navigation.");
        }
        return {
          ...(rawParticipant.kind === "human"
            ? {
                avatar: parseHumanAvatarDescriptor(
                  rawParticipant.avatar,
                  rawParticipant.principal_id,
                ),
              }
            : {}),
          displayName: rawParticipant.display_name,
          handle: rawParticipant.handle,
          kind: rawParticipant.kind,
          principalId: rawParticipant.principal_id,
        };
      });
      return {
        agents,
        ...(rawChannel.archived_at === undefined
          ? {}
          : { archivedAt: rawChannel.archived_at }),
        canSubmitCommands: rawChannel.can_submit_commands,
        channelId: rawChannel.channel_id,
        ...(rawChannel.direct_message_archive_event_position === undefined
          ? {}
          : {
              directMessageArchiveEventPosition:
                rawChannel.direct_message_archive_event_position,
            }),
        ...(rawChannel.direct_message_archived_at === undefined
          ? {}
          : { directMessageArchivedAt: rawChannel.direct_message_archived_at }),
        kind: rawChannel.kind as "direct" | "group" | "notification",
        ...(rawChannel.lifecycle_event_position === undefined
          ? {}
          : { lifecycleEventPosition: rawChannel.lifecycle_event_position }),
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
      avatar: parseHumanAvatarDescriptor(
        payload.principal.avatar,
        payload.principal.principal_id,
      ),
      displayName: payload.principal.display_name,
      handle: payload.principal.handle,
      principalId: payload.principal.principal_id,
    },
    workshops,
  };
}

const AGENT_CAPABILITIES = new Set<WorkshopAgentCapability>([
  "agent_delegation",
  "image_input",
  "text_generation",
  "tool_activity",
  "workspace_execution",
]);

function parseAgentCapabilities(value: unknown): WorkshopAgentCapability[] | null {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }
  const capabilities: WorkshopAgentCapability[] = [];
  for (const item of value) {
    if (
      typeof item !== "string" ||
      !AGENT_CAPABILITIES.has(item as WorkshopAgentCapability) ||
      capabilities.includes(item as WorkshopAgentCapability)
    ) {
      return null;
    }
    capabilities.push(item as WorkshopAgentCapability);
  }
  return capabilities;
}

function parseAgentDefinition(value: unknown): WorkshopAgentDefinition | null {
  if (
    !isRecord(value) ||
    typeof value.definition_id !== "string" ||
    !AGENT_DEFINITION_PATTERN.test(value.definition_id) ||
    typeof value.agent_id !== "string" ||
    !AGENT_PATTERN.test(value.agent_id) ||
    typeof value.handle !== "string" ||
    typeof value.display_name !== "string" ||
    typeof value.description !== "string" ||
    !isRecord(value.presentation) ||
    Object.keys(value.presentation).some((key) => key !== "avatar") ||
    (value.presentation.avatar !== undefined &&
      typeof value.presentation.avatar !== "string") ||
    !["draft", "active", "archived"].includes(String(value.lifecycle_state)) ||
    (value.active_revision_id !== null &&
      (typeof value.active_revision_id !== "string" ||
        !AGENT_REVISION_PATTERN.test(value.active_revision_id))) ||
    !Number.isSafeInteger(value.state_version) ||
    (value.state_version as number) < 1 ||
    typeof value.created_at !== "string" ||
    (value.created_by_principal_id !== null &&
      (typeof value.created_by_principal_id !== "string" ||
        !PRINCIPAL_PATTERN.test(value.created_by_principal_id))) ||
    (value.owner_principal_id !== null &&
      (typeof value.owner_principal_id !== "string" ||
        !PRINCIPAL_PATTERN.test(value.owner_principal_id))) ||
    (value.owner_display_name !== null &&
      typeof value.owner_display_name !== "string") ||
    !Array.isArray(value.revisions)
  ) {
    return null;
  }
  const revisions = value.revisions.map((rawRevision) => {
    if (
      !isRecord(rawRevision) ||
      typeof rawRevision.revision_id !== "string" ||
      !AGENT_REVISION_PATTERN.test(rawRevision.revision_id) ||
      !Number.isSafeInteger(rawRevision.revision_number) ||
      (rawRevision.revision_number as number) < 1 ||
      typeof rawRevision.purpose !== "string" ||
      typeof rawRevision.instructions !== "string" ||
      typeof rawRevision.created_at !== "string" ||
      (rawRevision.created_by_principal_id !== null &&
        (typeof rawRevision.created_by_principal_id !== "string" ||
          !PRINCIPAL_PATTERN.test(rawRevision.created_by_principal_id))) ||
      !Number.isSafeInteger(rawRevision.event_position) ||
      (rawRevision.event_position as number) < 1
    ) {
      return null;
    }
    const capabilities = parseAgentCapabilities(rawRevision.capabilities);
    if (!capabilities) {
      return null;
    }
    return {
      capabilities,
      createdAt: rawRevision.created_at,
      createdByPrincipalId: rawRevision.created_by_principal_id,
      eventPosition: rawRevision.event_position as number,
      instructions: rawRevision.instructions,
      purpose: rawRevision.purpose,
      revisionId: rawRevision.revision_id,
      revisionNumber: rawRevision.revision_number as number,
    };
  });
  if (revisions.some((revision) => revision === null)) {
    return null;
  }
  return {
    activeRevisionId: value.active_revision_id,
    agentId: value.agent_id,
    createdAt: value.created_at,
    createdByPrincipalId: value.created_by_principal_id,
    definitionId: value.definition_id,
    description: value.description,
    displayName: value.display_name,
    handle: value.handle,
    lifecycleState: value.lifecycle_state as WorkshopAgentDefinition["lifecycleState"],
    ownerDisplayName: value.owner_display_name,
    ownerPrincipalId: value.owner_principal_id,
    presentation: value.presentation as { avatar?: string },
    revisions: revisions as WorkshopAgentDefinition["revisions"],
    stateVersion: value.state_version as number,
  };
}

function parseAgentEnablement(value: unknown): WorkshopAgentEnablement | null {
  if (
    !isRecord(value) ||
    (value.enablement_id !== null &&
      (typeof value.enablement_id !== "string" ||
        !AGENT_ENABLEMENT_PATTERN.test(value.enablement_id))) ||
    typeof value.definition_id !== "string" ||
    !AGENT_DEFINITION_PATTERN.test(value.definition_id) ||
    typeof value.agent_id !== "string" ||
    !AGENT_PATTERN.test(value.agent_id) ||
    typeof value.handle !== "string" ||
    typeof value.display_name !== "string" ||
    !["available", "enabled", "disabled"].includes(
      String(value.lifecycle_state),
    ) ||
    (value.direct_channel_id !== null &&
      (typeof value.direct_channel_id !== "string" ||
        !CHANNEL_PATTERN.test(value.direct_channel_id))) ||
    (value.runtime_profile_id !== null &&
      (typeof value.runtime_profile_id !== "string" ||
        !RUNTIME_PROFILE_PATTERN.test(value.runtime_profile_id))) ||
    (value.state_version !== null &&
      (!Number.isSafeInteger(value.state_version) ||
        (value.state_version as number) < 1)) ||
    !Array.isArray(value.eligible_runtimes) ||
    typeof value.can_manage !== "boolean" ||
    typeof value.conversation_started !== "boolean" ||
    (value.owner_principal_id !== null &&
      (typeof value.owner_principal_id !== "string" ||
        !PRINCIPAL_PATTERN.test(value.owner_principal_id))) ||
    (value.owner_runtime_profile_id !== null &&
      (typeof value.owner_runtime_profile_id !== "string" ||
        !RUNTIME_PROFILE_PATTERN.test(value.owner_runtime_profile_id)))
  ) {
    return null;
  }
  const eligibleRuntimes = value.eligible_runtimes.map((rawRuntime) => {
    if (
      !isRecord(rawRuntime) ||
      typeof rawRuntime.runtime_profile_id !== "string" ||
      !RUNTIME_PROFILE_PATTERN.test(rawRuntime.runtime_profile_id) ||
      typeof rawRuntime.display_name !== "string" ||
      !Array.isArray(rawRuntime.backend_options) ||
      rawRuntime.backend_options.some((option) => typeof option !== "string")
    ) {
      return null;
    }
    return {
      backendOptions: rawRuntime.backend_options as string[],
      displayName: rawRuntime.display_name,
      runtimeProfileId: rawRuntime.runtime_profile_id,
    };
  });
  if (eligibleRuntimes.some((runtime) => runtime === null)) {
    return null;
  }
  return {
    agentId: value.agent_id,
    definitionId: value.definition_id,
    directChannelId: value.direct_channel_id,
    displayName: value.display_name,
    eligibleRuntimes: eligibleRuntimes as WorkshopAgentEnablement["eligibleRuntimes"],
    enablementId: value.enablement_id,
    handle: value.handle,
    lifecycleState: value.lifecycle_state as WorkshopAgentEnablement["lifecycleState"],
    runtimeProfileId: value.runtime_profile_id,
    stateVersion: value.state_version as number | null,
    canManage: value.can_manage,
    conversationStarted: value.conversation_started,
    ownerPrincipalId: value.owner_principal_id,
    ownerRuntimeProfileId: value.owner_runtime_profile_id,
  };
}

async function agentMutation(
  token: string,
  path: string,
  body: Record<string, unknown>,
  fallback: string,
): Promise<WorkshopAgentDefinition> {
  const response = await authorizedFetch({ channelId: "", token }, path, {
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, fallback));
  }
  const agent = isRecord(payload) && payload.version === 1
    ? parseAgentDefinition(payload.agent)
    : null;
  if (!agent) {
    throw new Error("Kai returned an unsupported agent definition.");
  }
  return agent;
}

async function enablementMutation(
  token: string,
  path: string,
  body: Record<string, unknown>,
): Promise<WorkshopAgentEnablement> {
  const response = await authorizedFetch({ channelId: "", token }, path, {
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update this agent."));
  }
  const agent = isRecord(payload) && payload.version === 1
    ? parseAgentEnablement(payload.agent)
    : null;
  if (!agent) {
    throw new Error("Kai returned unsupported agent enablement state.");
  }
  return agent;
}

export async function loadAgentDefinitions(
  token: string,
): Promise<WorkshopAgentDefinition[]> {
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/agents",
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load agents."));
  }
  if (!isRecord(payload) || payload.version !== 1 || !Array.isArray(payload.agents)) {
    throw new Error("Kai returned unsupported agent definitions.");
  }
  const agents = payload.agents.map(parseAgentDefinition);
  if (agents.some((agent) => agent === null)) {
    throw new Error("Kai returned unsupported agent definitions.");
  }
  return agents as WorkshopAgentDefinition[];
}

export async function loadAgentEnablements(
  token: string,
): Promise<WorkshopAgentEnablement[]> {
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/agent-enablement",
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load agent access."));
  }
  if (!isRecord(payload) || payload.version !== 1 || !Array.isArray(payload.agents)) {
    throw new Error("Kai returned unsupported agent enablement state.");
  }
  const agents = payload.agents.map(parseAgentEnablement);
  if (agents.some((agent) => agent === null)) {
    throw new Error("Kai returned unsupported agent enablement state.");
  }
  return agents as WorkshopAgentEnablement[];
}

export interface AgentDefinitionDraftInput {
  avatar: string;
  capabilities: WorkshopAgentCapability[];
  description: string;
  displayName: string;
  handle: string;
  idempotencyKey: string;
  instructions: string;
  purpose: string;
}

export async function createAgentDefinition(
  token: string,
  input: AgentDefinitionDraftInput,
): Promise<WorkshopAgentDefinition> {
  return agentMutation(
    token,
    "/v1/client/agents",
    {
      capabilities: input.capabilities,
      description: input.description,
      display_name: input.displayName,
      handle: input.handle,
      idempotency_key: input.idempotencyKey,
      instructions: input.instructions,
      presentation: input.avatar.trim() ? { avatar: input.avatar.trim() } : {},
      purpose: input.purpose,
    },
    "Could not create this agent.",
  );
}

export async function addAgentRevision(
  token: string,
  definitionId: string,
  input: {
    capabilities: WorkshopAgentCapability[];
    expectedVersion: number;
    idempotencyKey: string;
    instructions: string;
    purpose: string;
  },
): Promise<WorkshopAgentDefinition> {
  return agentMutation(
    token,
    `/v1/client/agents/${encodeURIComponent(definitionId)}/revisions`,
    {
      capabilities: input.capabilities,
      expected_version: input.expectedVersion,
      idempotency_key: input.idempotencyKey,
      instructions: input.instructions,
      purpose: input.purpose,
    },
    "Could not save this agent revision.",
  );
}

export async function activateAgentRevision(
  token: string,
  definitionId: string,
  input: {
    expectedVersion: number;
    idempotencyKey: string;
    revisionId: string;
  },
): Promise<WorkshopAgentDefinition> {
  return agentMutation(
    token,
    `/v1/client/agents/${encodeURIComponent(definitionId)}/activate`,
    {
      expected_version: input.expectedVersion,
      idempotency_key: input.idempotencyKey,
      revision_id: input.revisionId,
    },
    "Could not activate this agent revision.",
  );
}

export async function archiveAgentDefinition(
  token: string,
  definitionId: string,
  input: { expectedVersion: number; idempotencyKey: string },
): Promise<WorkshopAgentDefinition> {
  return agentMutation(
    token,
    `/v1/client/agents/${encodeURIComponent(definitionId)}/archive`,
    {
      expected_version: input.expectedVersion,
      idempotency_key: input.idempotencyKey,
    },
    "Could not archive this agent.",
  );
}

export async function enableAgentDefinition(
  token: string,
  definitionId: string,
  input: {
    expectedVersion: number | null;
    idempotencyKey: string;
    runtimeProfileId: string;
  },
): Promise<WorkshopAgentEnablement> {
  return enablementMutation(
    token,
    `/v1/client/agents/${encodeURIComponent(definitionId)}/enable`,
    {
      ...(input.expectedVersion === null
        ? {}
        : { expected_version: input.expectedVersion }),
      idempotency_key: input.idempotencyKey,
      runtime_profile_id: input.runtimeProfileId,
    },
  );
}

export async function startAgentConversation(
  token: string,
  definitionId: string,
  input: { expectedVersion: number; idempotencyKey: string },
): Promise<WorkshopAgentEnablement> {
  return enablementMutation(
    token,
    `/v1/client/agents/${encodeURIComponent(definitionId)}/conversation`,
    {
      expected_version: input.expectedVersion,
      idempotency_key: input.idempotencyKey,
    },
  );
}

export async function createChannel(
  token: string,
  input: {
    agentIds: string[];
    name: string;
    originChannelId: string | null;
  },
): Promise<string> {
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/channels",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_ids: input.agentIds,
        name: input.name,
        ...(input.originChannelId
          ? { origin_channel_id: input.originChannelId }
          : {}),
      }),
    },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not create this channel."));
  }
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !isRecord(payload.channel) ||
    typeof payload.channel.channel_id !== "string" ||
    !CHANNEL_PATTERN.test(payload.channel.channel_id)
  ) {
    throw new Error("Kai returned an unsupported channel creation response.");
  }
  return payload.channel.channel_id;
}

function parseHumanPeer(value: unknown): WorkshopHumanPeer | null {
  if (
    !isRecord(value) ||
    typeof value.principal_id !== "string" ||
    !PRINCIPAL_PATTERN.test(value.principal_id) ||
    typeof value.display_name !== "string" ||
    typeof value.handle !== "string" ||
    !HUMAN_HANDLE_PATTERN.test(value.handle) ||
    (value.conversation_channel_id !== null &&
      (typeof value.conversation_channel_id !== "string" ||
        !CHANNEL_PATTERN.test(value.conversation_channel_id)))
  ) {
    return null;
  }
  return {
    avatar: parseHumanAvatarDescriptor(value.avatar, value.principal_id),
    conversationChannelId: value.conversation_channel_id,
    displayName: value.display_name,
    handle: value.handle,
    principalId: value.principal_id,
  };
}

export async function loadWorkshopHumans(
  token: string,
  workshopId: string,
): Promise<WorkshopHumanPeer[]> {
  if (!WORKSHOP_PATTERN.test(workshopId)) {
    throw new Error("Invalid Workshop identity.");
  }
  const response = await authorizedFetch(
    { channelId: "", token },
    `/v1/workshops/${encodeURIComponent(workshopId)}/humans`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load Workshop people."));
  }
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.workshop_id !== workshopId ||
    !Array.isArray(payload.humans)
  ) {
    throw new Error("Kai returned unsupported Workshop people.");
  }
  const humans = payload.humans.map(parseHumanPeer);
  if (humans.some((human) => human === null)) {
    throw new Error("Kai returned unsupported Workshop people.");
  }
  return humans as WorkshopHumanPeer[];
}

export async function startHumanConversation(
  token: string,
  workshopId: string,
  principalId: string,
): Promise<WorkshopHumanConversation> {
  if (!WORKSHOP_PATTERN.test(workshopId)) {
    throw new Error("Invalid Workshop identity.");
  }
  if (!PRINCIPAL_PATTERN.test(principalId)) {
    throw new Error("Invalid human identity.");
  }
  const response = await authorizedFetch(
    { channelId: "", token },
    `/v1/workshops/${encodeURIComponent(workshopId)}/humans/${encodeURIComponent(principalId)}/conversation`,
    { method: "POST" },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not start this conversation."));
  }
  const rawConversation = isRecord(payload) ? payload.conversation : null;
  const rawPeer = isRecord(rawConversation) && isRecord(rawConversation.peer)
    ? rawConversation.peer
    : null;
  const peer = rawPeer && isRecord(rawConversation)
    ? parseHumanPeer({
        ...rawPeer,
        conversation_channel_id: rawConversation.channel_id,
      })
    : null;
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    typeof payload.created !== "boolean" ||
    !isRecord(rawConversation) ||
    rawConversation.workshop_id !== workshopId ||
    typeof rawConversation.channel_id !== "string" ||
    !CHANNEL_PATTERN.test(rawConversation.channel_id) ||
    rawConversation.kind !== "direct" ||
    !peer ||
    peer.principalId !== principalId
  ) {
    throw new Error("Kai returned an unsupported human conversation response.");
  }
  return {
    channelId: rawConversation.channel_id,
    created: payload.created,
    peer,
    workshopId,
  };
}

async function mutateChannelLifecycle(
  token: string,
  channelId: string,
  operation: "archive" | "restore",
  clientOperationId: string,
): Promise<void> {
  if (!CHANNEL_PATTERN.test(channelId)) {
    throw new Error("Invalid channel identity.");
  }
  const response = await authorizedFetch(
    { channelId, token },
    `/v1/channels/${encodeURIComponent(channelId)}/${operation}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_operation_id: clientOperationId }),
    },
  );
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    !isRecord(payload.channel) ||
    payload.channel.channel_id !== channelId ||
    payload.channel.archived !== (operation === "archive") ||
    typeof payload.channel.changed !== "boolean" ||
    typeof payload.channel.occurred_at !== "string"
  ) {
    throw new Error(
      safeErrorMessage(
        payload,
        operation === "archive"
          ? "Could not archive this channel."
          : "Could not restore this channel.",
      ),
    );
  }
}

export async function archiveChannel(
  token: string,
  channelId: string,
  clientOperationId: string,
): Promise<void> {
  await mutateChannelLifecycle(token, channelId, "archive", clientOperationId);
}

export async function restoreChannel(
  token: string,
  channelId: string,
  clientOperationId: string,
): Promise<void> {
  await mutateChannelLifecycle(token, channelId, "restore", clientOperationId);
}

async function mutateDirectMessageArchive(
  token: string,
  channelId: string,
  operation: "archive" | "restore",
  clientOperationId: string,
): Promise<void> {
  if (!CHANNEL_PATTERN.test(channelId)) {
    throw new Error("Invalid channel identity.");
  }
  const response = await authorizedFetch(
    { channelId, token },
    `/v1/direct-messages/${encodeURIComponent(channelId)}/${operation}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_operation_id: clientOperationId }),
    },
  );
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    !isRecord(payload.direct_message) ||
    payload.direct_message.channel_id !== channelId ||
    payload.direct_message.archived !== (operation === "archive") ||
    typeof payload.direct_message.changed !== "boolean" ||
    typeof payload.direct_message.occurred_at !== "string"
  ) {
    throw new Error(
      safeErrorMessage(
        payload,
        operation === "archive"
          ? "Could not archive this direct message."
          : "Could not restore this direct message.",
      ),
    );
  }
}

export async function archiveDirectMessage(
  token: string,
  channelId: string,
  clientOperationId: string,
): Promise<void> {
  await mutateDirectMessageArchive(token, channelId, "archive", clientOperationId);
}

export async function restoreDirectMessage(
  token: string,
  channelId: string,
  clientOperationId: string,
): Promise<void> {
  await mutateDirectMessageArchive(token, channelId, "restore", clientOperationId);
}

function parseHumanChannelMember(value: unknown): WorkshopHumanChannelMember | null {
  if (
    !isRecord(value) ||
    typeof value.principal_id !== "string" ||
    !PRINCIPAL_PATTERN.test(value.principal_id) ||
    typeof value.display_name !== "string" ||
    typeof value.handle !== "string" ||
    !HUMAN_HANDLE_PATTERN.test(value.handle) ||
    !["owner", "participant", null].includes(value.role as string | null)
  ) {
    return null;
  }
  return {
    avatar: parseHumanAvatarDescriptor(value.avatar, value.principal_id),
    displayName: value.display_name,
    handle: value.handle,
    principalId: value.principal_id,
    role: value.role as "owner" | "participant" | null,
  };
}

export async function loadChannelMembers(
  session: WorkshopSession,
): Promise<WorkshopHumanMembership> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/members`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load channel members."));
  }
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.channel_id !== session.channelId ||
    typeof payload.workshop_id !== "string" ||
    !WORKSHOP_PATTERN.test(payload.workshop_id) ||
    typeof payload.archived !== "boolean" ||
    typeof payload.can_manage !== "boolean" ||
    !Number.isSafeInteger(payload.state_version) ||
    (payload.state_version as number) < 0 ||
    !Array.isArray(payload.members) ||
    !Array.isArray(payload.eligible_humans)
  ) {
    throw new Error("Kai returned unsupported channel membership state.");
  }
  const members = payload.members.map(parseHumanChannelMember);
  const eligibleHumans = payload.eligible_humans.map(parseHumanChannelMember);
  if (members.some((member) => member === null) || eligibleHumans.some((member) => member === null)) {
    throw new Error("Kai returned unsupported channel membership state.");
  }
  return {
    archived: payload.archived,
    canManage: payload.can_manage,
    channelId: payload.channel_id,
    eligibleHumans: eligibleHumans as WorkshopHumanChannelMember[],
    members: members as WorkshopHumanChannelMember[],
    stateVersion: payload.state_version as number,
    workshopId: payload.workshop_id,
  };
}

export async function changeChannelMember(
  session: WorkshopSession,
  principalId: string,
  operation: "add" | "remove",
  expectedStateVersion: number,
  clientOperationId: string,
): Promise<number> {
  if (!PRINCIPAL_PATTERN.test(principalId)) {
    throw new Error("Invalid human identity.");
  }
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/members/${encodeURIComponent(principalId)}/${operation}`,
    {
      body: JSON.stringify({
        client_operation_id: clientOperationId,
        expected_state_version: expectedStateVersion,
      }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    },
  );
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.operation !== operation ||
    typeof payload.changed !== "boolean" ||
    !Number.isSafeInteger(payload.state_version)
  ) {
    throw new Error(safeErrorMessage(payload, "Could not update channel members."));
  }
  return payload.state_version as number;
}

async function mutateChannelAgent(
  session: WorkshopSession,
  agentId: string,
  operation: "attach" | "detach",
  clientOperationId: string,
): Promise<void> {
  if (!AGENT_PATTERN.test(agentId)) {
    throw new Error("Invalid agent identity.");
  }
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/agents/${encodeURIComponent(agentId)}/${operation}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_operation_id: clientOperationId }),
    },
  );
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.operation !== operation ||
    typeof payload.changed !== "boolean"
  ) {
    throw new Error(
      safeErrorMessage(
        payload,
        operation === "attach"
          ? "Could not attach this agent."
          : "Could not detach this agent.",
      ),
    );
  }
}

export async function attachChannelAgent(
  session: WorkshopSession,
  agentId: string,
  clientOperationId: string,
): Promise<void> {
  await mutateChannelAgent(session, agentId, "attach", clientOperationId);
}

export async function detachChannelAgent(
  session: WorkshopSession,
  agentId: string,
  clientOperationId: string,
): Promise<void> {
  await mutateChannelAgent(session, agentId, "detach", clientOperationId);
}

export async function dismissChannelAgent(
  session: WorkshopSession,
  agentId: string,
  clientDismissalId: string,
  threadRootId: string | null = null,
): Promise<void> {
  if (!AGENT_PATTERN.test(agentId)) {
    throw new Error("Invalid agent identity.");
  }
  if (threadRootId !== null && !MESSAGE_PATTERN.test(threadRootId)) {
    throw new Error("Invalid thread root identity.");
  }
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/agents/${encodeURIComponent(agentId)}/dismiss`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        client_dismissal_id: clientDismissalId,
        ...(threadRootId ? { thread_root_id: threadRootId } : {}),
      }),
    },
  );
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.dismissed !== true ||
    typeof payload.replayed !== "boolean"
  ) {
    throw new Error(safeErrorMessage(payload, "Could not dismiss this agent."));
  }
}

export async function setMessageReaction(
  session: WorkshopSession,
  messageId: string,
  reaction: WorkshopReaction,
  active: boolean,
): Promise<WorkshopMessageReaction[]> {
  if (!MESSAGE_PATTERN.test(messageId) || !REACTIONS.has(reaction)) {
    throw new Error("Invalid message reaction.");
  }
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/messages/${encodeURIComponent(messageId)}/reactions`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reaction, active }),
    },
  );
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.message_id !== messageId
  ) {
    throw new Error(safeErrorMessage(payload, "Could not update this reaction."));
  }
  const reactions = parseReactions(payload.reactions);
  if (reactions === null) {
    throw new Error("Kai returned an unsupported reaction response.");
  }
  return reactions;
}

function parseEditableCapabilities(value: unknown): WorkshopEditableCapability[] {
  if (!Array.isArray(value)) {
    throw new Error("Kai returned unsupported settings capabilities.");
  }
  return value.map((raw) => {
    if (
      !isRecord(raw) ||
      typeof raw.field !== "string" ||
      !["runtime", "workspace"].includes(String(raw.scope)) ||
      !["authorized_workspace", "backend_id", "integer_seconds", "model_id", "text"].includes(
        String(raw.value_type),
      ) ||
      typeof raw.resettable !== "boolean" ||
      (raw.choices !== null &&
        (!Array.isArray(raw.choices) ||
          raw.choices.some((choice) => typeof choice !== "string"))) ||
      (raw.minimum !== null && !Number.isSafeInteger(raw.minimum)) ||
      (raw.maximum !== null && !Number.isSafeInteger(raw.maximum))
    ) {
      throw new Error("Kai returned unsupported settings capabilities.");
    }
    return {
      choices: raw.choices as string[] | null,
      field: raw.field,
      maximum: raw.maximum as number | null,
      minimum: raw.minimum as number | null,
      resettable: raw.resettable,
      scope: raw.scope as "runtime" | "workspace",
      valueType: raw.value_type as WorkshopEditableCapability["valueType"],
    };
  });
}

function parseSettingsMutation(value: unknown): WorkshopSettingsMutation | null {
  if (value === null) {
    return null;
  }
  if (
    !isRecord(value) ||
    typeof value.operation !== "string" ||
    typeof value.changed !== "boolean" ||
    !["deferred_until_next_run", "restarted", "unchanged"].includes(
      String(value.runtime_action),
    ) ||
    typeof value.provider_session_invalidated !== "boolean"
  ) {
    throw new Error("Kai returned an unsupported settings mutation result.");
  }
  return {
    changed: value.changed,
    operation: value.operation,
    providerSessionInvalidated: value.provider_session_invalidated,
    runtimeAction: value.runtime_action as WorkshopSettingsMutation["runtimeAction"],
  };
}

function parseSettingsWorkspace(
  payload: unknown,
  channelId: string,
): WorkshopSettingsWorkspace {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.channel_id !== channelId ||
    typeof payload.principal_id !== "string" ||
    !PRINCIPAL_PATTERN.test(payload.principal_id) ||
    typeof payload.runtime_profile_id !== "string" ||
    typeof payload.backend !== "string" ||
    typeof payload.backend_option_id !== "string" ||
    typeof payload.provider !== "string" ||
    !Array.isArray(payload.backend_options) ||
    typeof payload.workspace !== "string" ||
    typeof payload.revision !== "string" ||
    !isRecord(payload.model) ||
    typeof payload.model.value !== "string" ||
    typeof payload.model.source !== "string" ||
    typeof payload.model.default_value !== "string" ||
    !isRecord(payload.timeout_seconds) ||
    !Number.isSafeInteger(payload.timeout_seconds.value) ||
    typeof payload.timeout_seconds.source !== "string" ||
    !Number.isSafeInteger(payload.timeout_seconds.default_value) ||
    !Array.isArray(payload.workspaces) ||
    (payload.model_options !== null && !Array.isArray(payload.model_options))
  ) {
    throw new Error("Kai returned unsupported settings and workspace state.");
  }
  const workspaces = payload.workspaces.map((rawWorkspace) => {
    if (
      !isRecord(rawWorkspace) ||
      typeof rawWorkspace.path !== "string" ||
      typeof rawWorkspace.name !== "string" ||
      typeof rawWorkspace.current !== "boolean" ||
      typeof rawWorkspace.home !== "boolean"
    ) {
      throw new Error("Kai returned unsupported workspace state.");
    }
    return {
      current: rawWorkspace.current,
      home: rawWorkspace.home,
      name: rawWorkspace.name,
      path: rawWorkspace.path,
    };
  });
  const backendOptions = payload.backend_options.map((rawBackend) => {
    if (
      !isRecord(rawBackend) ||
      typeof rawBackend.option_id !== "string" ||
      typeof rawBackend.backend !== "string" ||
      typeof rawBackend.provider !== "string" ||
      typeof rawBackend.current !== "boolean"
    ) {
      throw new Error("Kai returned unsupported backend policy.");
    }
    return {
      backend: rawBackend.backend,
      current: rawBackend.current,
      optionId: rawBackend.option_id,
      provider: rawBackend.provider,
    };
  });
  const modelOptions = payload.model_options === null
    ? null
    : payload.model_options.map((rawModel) => {
        if (
          !isRecord(rawModel) ||
          typeof rawModel.model_id !== "string" ||
          typeof rawModel.display_name !== "string" ||
          !["available", "not_advertised", "unavailable", "unknown"].includes(
            String(rawModel.status),
          ) ||
          typeof rawModel.selectable !== "boolean" ||
          typeof rawModel.retained !== "boolean"
        ) {
          throw new Error("Kai returned unsupported model options.");
        }
        return {
          displayName: rawModel.display_name,
          modelId: rawModel.model_id,
          retained: rawModel.retained,
          selectable: rawModel.selectable,
          sources: [],
          status: rawModel.status as "available" | "not_advertised" | "unavailable" | "unknown",
        };
      });
  const rawCatalogue = payload.model_catalogue;
  if (
    rawCatalogue !== null &&
    (!isRecord(rawCatalogue) ||
      (rawCatalogue.status !== null && typeof rawCatalogue.status !== "string") ||
      typeof rawCatalogue.stale !== "boolean" ||
      typeof rawCatalogue.last_known_good !== "boolean" ||
      (rawCatalogue.last_attempt_at !== null && typeof rawCatalogue.last_attempt_at !== "string") ||
      (rawCatalogue.last_successful_refresh_at !== null &&
        typeof rawCatalogue.last_successful_refresh_at !== "string") ||
      (rawCatalogue.error_code !== null && typeof rawCatalogue.error_code !== "string") ||
      (rawCatalogue.error_detail !== null && typeof rawCatalogue.error_detail !== "string"))
  ) {
    throw new Error("Kai returned unsupported model catalogue status.");
  }
  return {
    backend: payload.backend,
    backendOptionId: payload.backend_option_id,
    backendOptions,
    channelId,
    model: {
      defaultValue: payload.model.default_value,
      source: payload.model.source,
      value: payload.model.value,
    },
    modelCatalogue: rawCatalogue === null ? null : {
      errorCode: rawCatalogue.error_code as string | null,
      errorDetail: rawCatalogue.error_detail as string | null,
      lastAttemptAt: rawCatalogue.last_attempt_at as string | null,
      lastKnownGood: rawCatalogue.last_known_good as boolean,
      lastSuccessfulRefreshAt: rawCatalogue.last_successful_refresh_at as string | null,
      stale: rawCatalogue.stale as boolean,
      status: rawCatalogue.status as string | null,
    },
    modelOptions,
    capabilities: parseEditableCapabilities(payload.capabilities),
    mutation: parseSettingsMutation(payload.mutation),
    principalId: payload.principal_id,
    provider: payload.provider,
    revision: payload.revision,
    runtimeProfileId: payload.runtime_profile_id,
    timeoutSeconds: {
      defaultValue: payload.timeout_seconds.default_value as number,
      source: payload.timeout_seconds.source,
      value: payload.timeout_seconds.value as number,
    },
    workspace: payload.workspace,
    workspaces,
  };
}

function parseRoutingEligibility(payload: unknown): WorkshopRoutingEligibility {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !["conversation", "coding", "vision"].includes(String(payload.task_class)) ||
    typeof payload.principal_id !== "string" ||
    !PRINCIPAL_PATTERN.test(payload.principal_id) ||
    typeof payload.channel_id !== "string" ||
    !CHANNEL_PATTERN.test(payload.channel_id) ||
    typeof payload.agent_id !== "string" ||
    !AGENT_PATTERN.test(payload.agent_id) ||
    typeof payload.runtime_profile_id !== "string" ||
    !RUNTIME_PROFILE_PATTERN.test(payload.runtime_profile_id) ||
    typeof payload.workspace !== "string" ||
    !Array.isArray(payload.required_capabilities) ||
    !payload.required_capabilities.every((item) => typeof item === "string") ||
    !Array.isArray(payload.candidates)
  ) {
    throw new Error("Kai returned unsupported routing eligibility.");
  }
  const candidates = payload.candidates.map((rawCandidate) => {
    if (
      !isRecord(rawCandidate) ||
      typeof rawCandidate.option_id !== "string" ||
      typeof rawCandidate.backend !== "string" ||
      typeof rawCandidate.provider !== "string" ||
      !Array.isArray(rawCandidate.allowed_services) ||
      !rawCandidate.allowed_services.every((item) => typeof item === "string") ||
      typeof rawCandidate.model_id !== "string" ||
      !["current_selection", "protected_default"].includes(String(rawCandidate.model_source)) ||
      typeof rawCandidate.selected !== "boolean" ||
      typeof rawCandidate.eligible !== "boolean" ||
      !Array.isArray(rawCandidate.capabilities) ||
      !Array.isArray(rawCandidate.reasons)
    ) {
      throw new Error("Kai returned unsupported routing candidate.");
    }
    const capabilities = rawCandidate.capabilities.map((rawCapability) => {
      if (
        !isRecord(rawCapability) ||
        !["image_input", "text_generation", "tool_activity", "workspace_execution"].includes(
          String(rawCapability.capability),
        ) ||
        !["supported", "unsupported", "unknown"].includes(String(rawCapability.support)) ||
        typeof rawCapability.evidence !== "string"
      ) {
        throw new Error("Kai returned unsupported routing capability evidence.");
      }
      return {
        capability: rawCapability.capability as WorkshopRoutingEligibility["candidates"][number]["capabilities"][number]["capability"],
        evidence: rawCapability.evidence,
        support: rawCapability.support as "supported" | "unsupported" | "unknown",
      };
    });
    const reasons = rawCandidate.reasons.map((rawReason) => {
      if (
        !isRecord(rawReason) ||
        typeof rawReason.code !== "string" ||
        typeof rawReason.detail !== "string"
      ) {
        throw new Error("Kai returned unsupported routing eligibility reason.");
      }
      return { code: rawReason.code, detail: rawReason.detail };
    });
    return {
      allowedServices: rawCandidate.allowed_services,
      backend: rawCandidate.backend,
      capabilities,
      eligible: rawCandidate.eligible,
      modelId: rawCandidate.model_id,
      modelSource: rawCandidate.model_source as "current_selection" | "protected_default",
      optionId: rawCandidate.option_id,
      provider: rawCandidate.provider,
      reasons,
      selected: rawCandidate.selected,
    };
  });
  return {
    agentId: payload.agent_id,
    candidates,
    channelId: payload.channel_id,
    principalId: payload.principal_id,
    requiredCapabilities: payload.required_capabilities,
    runtimeProfileId: payload.runtime_profile_id,
    taskClass: payload.task_class as WorkshopRoutingTaskClass,
    version: 1,
    workspace: payload.workspace,
  };
}

function parseRoutingPolicy(payload: unknown): WorkshopRoutingPolicy {
  if (
    !isRecord(payload) || payload.version !== 1 ||
    typeof payload.principal_id !== "string" || !PRINCIPAL_PATTERN.test(payload.principal_id) ||
    typeof payload.channel_id !== "string" || !CHANNEL_PATTERN.test(payload.channel_id) ||
    typeof payload.agent_id !== "string" || !AGENT_PATTERN.test(payload.agent_id) ||
    typeof payload.runtime_profile_id !== "string" ||
    !RUNTIME_PROFILE_PATTERN.test(payload.runtime_profile_id) ||
    !Array.isArray(payload.entries)
  ) {
    throw new Error("Kai returned an unsupported routing policy.");
  }
  const entries = payload.entries.map((rawEntry) => {
    if (
      !isRecord(rawEntry) ||
      !["conversation", "coding", "vision"].includes(String(rawEntry.task_class)) ||
      (rawEntry.backend_option_id !== null && typeof rawEntry.backend_option_id !== "string") ||
      !["selected", "fail_closed"].includes(String(rawEntry.fallback)) ||
      typeof rawEntry.revision !== "number" || !Number.isSafeInteger(rawEntry.revision) ||
      !Array.isArray(rawEntry.authorized_option_ids) ||
      !rawEntry.authorized_option_ids.every((item) => typeof item === "string") ||
      !Array.isArray(rawEntry.eligible_option_ids) ||
      !rawEntry.eligible_option_ids.every((item) => typeof item === "string")
    ) {
      throw new Error("Kai returned an unsupported routing policy entry.");
    }
    return {
      authorizedOptionIds: rawEntry.authorized_option_ids,
      backendOptionId: rawEntry.backend_option_id as string | null,
      eligibleOptionIds: rawEntry.eligible_option_ids,
      fallback: rawEntry.fallback as "selected" | "fail_closed",
      revision: rawEntry.revision,
      taskClass: rawEntry.task_class as WorkshopRoutingTaskClass,
    };
  });
  return {
    agentId: payload.agent_id,
    channelId: payload.channel_id,
    entries,
    principalId: payload.principal_id,
    runtimeProfileId: payload.runtime_profile_id,
    version: 1,
  };
}

function parseRoutingDecision(payload: unknown): WorkshopRunRoutingDecision | null {
  if (payload === null || payload === undefined) return null;
  if (
    !isRecord(payload) ||
    (payload.requested_task_class !== null &&
      !["conversation", "coding", "vision"].includes(String(payload.requested_task_class))) ||
    (payload.requested_backend_option_id !== null &&
      typeof payload.requested_backend_option_id !== "string") ||
    (payload.selected_backend_option_id !== null &&
      typeof payload.selected_backend_option_id !== "string") ||
    !["selected_default", "routed", "fallback_selected", "rejected"].includes(
      String(payload.disposition),
    ) ||
    typeof payload.reason_code !== "string" ||
    (payload.policy_revision !== null && typeof payload.policy_revision !== "number") ||
    typeof payload.backend !== "string" ||
    (payload.provider !== null && typeof payload.provider !== "string") ||
    typeof payload.model !== "string" ||
    typeof payload.evidence_version !== "number" ||
    typeof payload.decided_at !== "string"
  ) {
    throw new Error("Kai returned an unsupported routing decision.");
  }
  return {
    backend: payload.backend,
    decidedAt: payload.decided_at,
    disposition: payload.disposition as WorkshopRunRoutingDecision["disposition"],
    evidenceVersion: payload.evidence_version,
    model: payload.model,
    policyRevision: payload.policy_revision as number | null,
    provider: payload.provider as string | null,
    reasonCode: payload.reason_code,
    requestedBackendOptionId: payload.requested_backend_option_id as string | null,
    requestedTaskClass: payload.requested_task_class as WorkshopRoutingTaskClass | null,
    selectedBackendOptionId: payload.selected_backend_option_id as string | null,
  };
}

function parseModelCatalogue(payload: unknown): WorkshopModelCatalogue {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    typeof payload.runtime_profile_id !== "string" ||
    typeof payload.option_id !== "string" ||
    typeof payload.stale !== "boolean" ||
    typeof payload.last_known_good !== "boolean" ||
    !Array.isArray(payload.models) ||
    (payload.refresh !== null && !isRecord(payload.refresh))
  ) {
    throw new Error("Kai returned an unsupported model catalogue.");
  }
  const models = payload.models.map((rawModel) => {
    if (
      !isRecord(rawModel) ||
      typeof rawModel.model_id !== "string" ||
      typeof rawModel.display_name !== "string" ||
      !["available", "not_advertised", "unavailable", "unknown"].includes(
        String(rawModel.status),
      ) ||
      typeof rawModel.selectable !== "boolean" ||
      typeof rawModel.retained !== "boolean"
      || !Array.isArray(rawModel.sources)
      || rawModel.sources.some((source) => typeof source !== "string")
    ) {
      throw new Error("Kai returned unsupported model catalogue entries.");
    }
    return {
      displayName: rawModel.display_name,
      modelId: rawModel.model_id,
      retained: rawModel.retained,
      selectable: rawModel.selectable,
      sources: rawModel.sources as string[],
      status: rawModel.status as "available" | "not_advertised" | "unavailable" | "unknown",
    };
  });
  const refresh = payload.refresh;
  if (
    refresh !== null &&
    (!isRecord(refresh) ||
      typeof refresh.status !== "string" ||
      !Number.isSafeInteger(refresh.generation) ||
      typeof refresh.last_attempt_at !== "string" ||
      (refresh.last_successful_refresh_at !== null &&
        typeof refresh.last_successful_refresh_at !== "string") ||
      (refresh.expires_at !== null && typeof refresh.expires_at !== "string") ||
      (refresh.error_code !== null && typeof refresh.error_code !== "string") ||
      (refresh.error_detail !== null && typeof refresh.error_detail !== "string"))
  ) {
    throw new Error("Kai returned unsupported model refresh status.");
  }
  return {
    lastKnownGood: payload.last_known_good,
    models,
    optionId: payload.option_id,
    refresh: refresh === null ? null : {
      errorCode: refresh.error_code as string | null,
      errorDetail: refresh.error_detail as string | null,
      expiresAt: refresh.expires_at as string | null,
      generation: refresh.generation as number,
      lastAttemptAt: refresh.last_attempt_at as string,
      lastSuccessfulRefreshAt: refresh.last_successful_refresh_at as string | null,
      status: refresh.status as string,
    },
    runtimeProfileId: payload.runtime_profile_id,
    stale: payload.stale,
  };
}

function parseGitHubSettings(payload: unknown): WorkshopGitHubSettings {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    (payload.github_login !== null && typeof payload.github_login !== "string") ||
    typeof payload.repositories_resettable !== "boolean" ||
    !Array.isArray(payload.repositories) ||
    !isRecord(payload.pr_review) ||
    typeof payload.pr_review.enabled !== "boolean" ||
    typeof payload.pr_review.source !== "string" ||
    typeof payload.pr_review.resettable !== "boolean" ||
    !isRecord(payload.issue_triage) ||
    typeof payload.issue_triage.enabled !== "boolean" ||
    typeof payload.issue_triage.source !== "string" ||
    typeof payload.issue_triage.resettable !== "boolean" ||
    typeof payload.token_stored !== "boolean" ||
    typeof payload.revision !== "string" ||
    (payload.mutation !== null &&
      (!isRecord(payload.mutation) ||
        typeof payload.mutation.operation !== "string" ||
        typeof payload.mutation.changed !== "boolean"))
  ) {
    throw new Error("Kai returned unsupported GitHub settings.");
  }
  const repositories = payload.repositories.map((value) => {
    if (
      !isRecord(value) ||
      typeof value.repository !== "string" ||
      typeof value.source !== "string" ||
      typeof value.automation_authorized !== "boolean"
    ) {
      throw new Error("Kai returned unsupported GitHub repository settings.");
    }
    return {
      automationAuthorized: value.automation_authorized,
      repository: value.repository,
      source: value.source,
    };
  });
  const mutation = payload.mutation === null
    ? null
    : {
        changed: payload.mutation.changed as boolean,
        operation: payload.mutation.operation as string,
      };
  return {
    githubLogin: payload.github_login as string | null,
    issueTriage: {
      enabled: payload.issue_triage.enabled,
      resettable: payload.issue_triage.resettable,
      source: payload.issue_triage.source,
    },
    mutation,
    prReview: {
      enabled: payload.pr_review.enabled,
      resettable: payload.pr_review.resettable,
      source: payload.pr_review.source,
    },
    repositories,
    repositoriesResettable: payload.repositories_resettable as boolean,
    revision: payload.revision,
    tokenStored: payload.token_stored,
  };
}

function parseNotificationPreferences(
  payload: unknown,
): WorkshopNotificationPreferences {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !Array.isArray(payload.destinations) ||
    !Array.isArray(payload.preferences) ||
    typeof payload.revision !== "string" ||
    (payload.mutation !== null &&
      (!isRecord(payload.mutation) ||
        typeof payload.mutation.operation !== "string" ||
        typeof payload.mutation.changed !== "boolean"))
  ) {
    throw new Error("Kai returned unsupported notification preferences.");
  }
  const destinations = payload.destinations.map((value) => {
    if (
      !isRecord(value) ||
      typeof value.choice_id !== "string" ||
      typeof value.display_name !== "string" ||
      (value.kind !== "direct" && value.kind !== "notification") ||
      !Array.isArray(value.supported_classes) ||
      value.supported_classes.some((item) => item !== "github" && item !== "generic")
    ) {
      throw new Error("Kai returned an unsupported notification destination.");
    }
    return {
      choiceId: value.choice_id,
      displayName: value.display_name,
      kind: value.kind as "direct" | "notification",
      supportedClasses: value.supported_classes as ("generic" | "github")[],
    };
  });
  const preferences = payload.preferences.map((value) => {
    if (
      !isRecord(value) ||
      (value.integration_class !== "github" && value.integration_class !== "generic") ||
      typeof value.display_name !== "string" ||
      typeof value.destination_choice_id !== "string" ||
      typeof value.destination_name !== "string" ||
      (value.destination_kind !== "direct" && value.destination_kind !== "notification") ||
      typeof value.source !== "string" ||
      typeof value.editable !== "boolean" ||
      typeof value.resettable !== "boolean"
    ) {
      throw new Error("Kai returned an unsupported integration preference.");
    }
    return {
      destinationChoiceId: value.destination_choice_id,
      destinationKind: value.destination_kind as "direct" | "notification",
      destinationName: value.destination_name,
      displayName: value.display_name,
      editable: value.editable,
      integrationClass: value.integration_class as "generic" | "github",
      resettable: value.resettable,
      source: value.source,
    };
  });
  return {
    destinations,
    mutation: payload.mutation === null
      ? null
      : {
          changed: payload.mutation.changed as boolean,
          operation: payload.mutation.operation as string,
        },
    preferences,
    revision: payload.revision,
  };
}

function parseChannelNotificationPolicy(payload: unknown): WorkshopChannelNotificationPolicy {
  const validLevel = (value: unknown): value is "all" | "mentions_replies" | "muted" =>
    value === "all" || value === "mentions_replies" || value === "muted";
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !Array.isArray(payload.levels) ||
    payload.levels.some((level) => !validLevel(level)) ||
    !Array.isArray(payload.adapter_deliveries) ||
    !Array.isArray(payload.channels) ||
    typeof payload.muted_mentions_notify !== "boolean" ||
    !isRecord(payload.do_not_disturb) ||
    typeof payload.do_not_disturb.enabled !== "boolean" ||
    typeof payload.do_not_disturb.timezone !== "string" ||
    typeof payload.do_not_disturb.start !== "string" ||
    typeof payload.do_not_disturb.end !== "string" ||
    typeof payload.revision !== "string" ||
    (payload.mutation !== null &&
      (!isRecord(payload.mutation) ||
        typeof payload.mutation.operation !== "string" ||
        typeof payload.mutation.changed !== "boolean"))
  ) {
    throw new Error("Kai returned unsupported channel notification policy.");
  }
  const channels = payload.channels.map((value) => {
    if (
      !isRecord(value) ||
      typeof value.channel_id !== "string" ||
      typeof value.channel_name !== "string" ||
      !validLevel(value.level) ||
      typeof value.source !== "string"
    ) {
      throw new Error("Kai returned an unsupported channel notification setting.");
    }
    return {
      channelId: value.channel_id,
      channelName: value.channel_name,
      level: value.level,
      source: value.source,
    };
  });
  const adapterDeliveries = payload.adapter_deliveries.map((value) => {
    if (
      !isRecord(value) ||
      typeof value.transport !== "string" ||
      typeof value.display_name !== "string" ||
      typeof value.enabled !== "boolean" ||
      typeof value.source !== "string"
    ) {
      throw new Error("Kai returned an unsupported adapter notification setting.");
    }
    return {
      displayName: value.display_name,
      enabled: value.enabled,
      source: value.source,
      transport: value.transport,
    };
  });
  return {
    adapterDeliveries,
    channels,
    doNotDisturb: {
      enabled: payload.do_not_disturb.enabled,
      timezone: payload.do_not_disturb.timezone,
      start: payload.do_not_disturb.start,
      end: payload.do_not_disturb.end,
    },
    levels: payload.levels,
    mutedMentionsNotify: payload.muted_mentions_notify,
    mutation: payload.mutation === null ? null : {
      changed: payload.mutation.changed as boolean,
      operation: payload.mutation.operation as string,
    },
    revision: payload.revision,
  };
}

function parseClientPreferences(payload: unknown): WorkshopClientPreferences {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !isRecord(payload.voice_output) ||
    typeof payload.voice_output.available !== "boolean" ||
    (payload.voice_output.unavailable_reason !== null &&
      typeof payload.voice_output.unavailable_reason !== "string") ||
    !Array.isArray(payload.voice_output.modes) ||
    !Array.isArray(payload.voice_output.voices) ||
    !Array.isArray(payload.voice_output.bindings) ||
    typeof payload.revision !== "string" ||
    (payload.mutation !== null &&
      (!isRecord(payload.mutation) ||
        typeof payload.mutation.operation !== "string" ||
        typeof payload.mutation.changed !== "boolean"))
  ) {
    throw new Error("Kai returned unsupported client preferences.");
  }
  const modes = payload.voice_output.modes;
  if (modes.some((value) =>
    value !== "off" && value !== "text_and_voice" && value !== "voice_only")) {
    throw new Error("Kai returned an unsupported client voice mode.");
  }
  const voices = payload.voice_output.voices.map((value) => {
    if (!isRecord(value) || typeof value.value !== "string" || typeof value.display_name !== "string") {
      throw new Error("Kai returned an unsupported client voice.");
    }
    return { value: value.value, displayName: value.display_name };
  });
  const bindings = payload.voice_output.bindings.map((value) => {
    if (
      !isRecord(value) ||
      typeof value.choice_id !== "string" ||
      typeof value.client_name !== "string" ||
      (value.mode !== "off" && value.mode !== "text_and_voice" && value.mode !== "voice_only") ||
      typeof value.voice !== "string" ||
      typeof value.voice_name !== "string" ||
      typeof value.editable !== "boolean"
    ) {
      throw new Error("Kai returned unsupported client-binding preferences.");
    }
    return {
      choiceId: value.choice_id,
      clientName: value.client_name,
      mode: value.mode as "off" | "text_and_voice" | "voice_only",
      voice: value.voice,
      voiceName: value.voice_name,
      editable: value.editable,
    };
  });
  return {
    mutation: payload.mutation === null
      ? null
      : {
          changed: payload.mutation.changed as boolean,
          operation: payload.mutation.operation as string,
        },
    revision: payload.revision,
    voiceOutput: {
      available: payload.voice_output.available,
      unavailableReason: payload.voice_output.unavailable_reason,
      modes: modes as ("off" | "text_and_voice" | "voice_only")[],
      voices,
      bindings,
    },
  };
}

function parseAppearancePreferences(payload: unknown): WorkshopAppearancePreferences {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !isWorkshopThemeId(payload.theme_id) ||
    !Array.isArray(payload.themes) ||
    typeof payload.revision !== "string" ||
    (payload.mutation !== null &&
      (!isRecord(payload.mutation) ||
        typeof payload.mutation.operation !== "string" ||
        typeof payload.mutation.changed !== "boolean"))
  ) {
    throw new Error("Kai returned unsupported appearance preferences.");
  }
  const themes = payload.themes.map((value) => {
    if (
      !isRecord(value) ||
      !isWorkshopThemeId(value.theme_id) ||
      typeof value.display_name !== "string" ||
      (value.color_scheme !== "dark" && value.color_scheme !== "light")
    ) {
      throw new Error("Kai returned an unsupported Workshop theme.");
    }
    return {
      colorScheme: value.color_scheme as "dark" | "light",
      displayName: value.display_name,
      themeId: value.theme_id,
    };
  });
  return {
    mutation: payload.mutation === null
      ? null
      : {
          changed: payload.mutation.changed as boolean,
          operation: payload.mutation.operation as string,
        },
    revision: payload.revision,
    themeId: payload.theme_id,
    themes,
  };
}

function parseHumanProfile(payload: unknown): WorkshopHumanProfile {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    typeof payload.principal_id !== "string" ||
    !PRINCIPAL_PATTERN.test(payload.principal_id) ||
    typeof payload.display_name !== "string" ||
    typeof payload.handle !== "string" ||
    !HUMAN_HANDLE_PATTERN.test(payload.handle) ||
    !Number.isSafeInteger(payload.state_version) ||
    (payload.state_version as number) < 0 ||
    (payload.mutation !== null &&
      (!isRecord(payload.mutation) ||
        typeof payload.mutation.changed !== "boolean" ||
        typeof payload.mutation.replayed !== "boolean"))
  ) {
    throw new Error("Kai returned an unsupported human profile.");
  }
  return {
    avatar: parseHumanAvatarDescriptor(payload.avatar, payload.principal_id),
    principalId: payload.principal_id,
    displayName: payload.display_name,
    handle: payload.handle,
    stateVersion: payload.state_version as number,
    mutation: payload.mutation === null
      ? null
      : {
          changed: payload.mutation.changed as boolean,
          replayed: payload.mutation.replayed as boolean,
        },
  };
}

function parseHumanAvatar(payload: unknown): WorkshopHumanAvatar {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    typeof payload.principal_id !== "string" ||
    !PRINCIPAL_PATTERN.test(payload.principal_id) ||
    typeof payload.active !== "boolean" ||
    !Number.isSafeInteger(payload.state_version) ||
    (payload.state_version as number) < 0 ||
    (payload.mutation !== null &&
      (!isRecord(payload.mutation) ||
        typeof payload.mutation.changed !== "boolean" ||
        typeof payload.mutation.replayed !== "boolean"))
  ) {
    throw new Error("Kai returned an unsupported human avatar.");
  }
  const descriptor = parseHumanAvatarDescriptor(payload, payload.principal_id);
  if (descriptor.active !== payload.active || descriptor.stateVersion !== payload.state_version) {
    throw new Error("Kai returned an unsupported human avatar.");
  }
  const metadata = [payload.media_type, payload.byte_size, payload.width, payload.height, payload.sha256];
  if (payload.active) {
    if (
      payload.media_type !== "image/png" ||
      !Number.isSafeInteger(payload.byte_size) ||
      (payload.byte_size as number) < 1 ||
      !Number.isSafeInteger(payload.width) ||
      (payload.width as number) < 1 ||
      !Number.isSafeInteger(payload.height) ||
      (payload.height as number) < 1 ||
      typeof payload.sha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(payload.sha256)
    ) {
      throw new Error("Kai returned an unsupported human avatar.");
    }
  } else if (metadata.some((value) => value !== null)) {
    throw new Error("Kai returned an unsupported human avatar.");
  }
  return {
    ...descriptor,
    byteSize: payload.byte_size as number | null,
    height: payload.height as number | null,
    mediaType: payload.media_type as string | null,
    mutation: payload.mutation === null
      ? null
      : {
          changed: payload.mutation.changed as boolean,
          replayed: payload.mutation.replayed as boolean,
        },
    principalId: payload.principal_id,
    sha256: payload.sha256 as string | null,
    width: payload.width as number | null,
  };
}

function parseWorkspaceConfig(payload: unknown): WorkshopWorkspaceConfig {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    typeof payload.workspace !== "string" ||
    typeof payload.revision !== "string" ||
    !isRecord(payload.model) ||
    typeof payload.model.value !== "string" ||
    typeof payload.model.source !== "string" ||
    typeof payload.model.default_value !== "string" ||
    !isRecord(payload.timeout_seconds) ||
    !Number.isSafeInteger(payload.timeout_seconds.value) ||
    typeof payload.timeout_seconds.source !== "string" ||
    !Number.isSafeInteger(payload.timeout_seconds.default_value) ||
    !Array.isArray(payload.environment_keys) ||
    payload.environment_keys.some((item) => typeof item !== "string") ||
    (payload.prompt !== null && typeof payload.prompt !== "string") ||
    typeof payload.has_prompt !== "boolean" ||
    (payload.prompt_source !== null && typeof payload.prompt_source !== "string") ||
    !Array.isArray(payload.override_fields) ||
    payload.override_fields.some((item) => typeof item !== "string")
  ) {
    throw new Error("Kai returned unsupported workspace settings state.");
  }
  return {
    capabilities: parseEditableCapabilities(payload.capabilities),
    environmentKeys: payload.environment_keys as string[],
    hasPrompt: payload.has_prompt,
    model: {
      defaultValue: payload.model.default_value,
      source: payload.model.source,
      value: payload.model.value,
    },
    mutation: parseSettingsMutation(payload.mutation),
    overrideFields: payload.override_fields as string[],
    prompt: payload.prompt as string | null,
    promptSource: payload.prompt_source as string | null,
    revision: payload.revision,
    timeoutSeconds: {
      defaultValue: payload.timeout_seconds.default_value as number,
      source: payload.timeout_seconds.source,
      value: payload.timeout_seconds.value as number,
    },
    workspace: payload.workspace,
  };
}

function parsePreferenceDocument(payload: unknown): WorkshopPreferenceDocument {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !isRecord(payload.document) ||
    typeof payload.document.content !== "string" ||
    typeof payload.document.revision !== "string" ||
    (payload.document.updated_at !== null &&
      typeof payload.document.updated_at !== "string") ||
    !Number.isSafeInteger(payload.document.size_bytes) ||
    !Number.isSafeInteger(payload.document.max_bytes) ||
    typeof payload.document.editable !== "boolean"
  ) {
    throw new Error("Kai returned unsupported preference state.");
  }
  return {
    content: payload.document.content,
    editable: payload.document.editable,
    maxBytes: payload.document.max_bytes as number,
    revision: payload.document.revision,
    sizeBytes: payload.document.size_bytes as number,
    updatedAt: payload.document.updated_at as string | null,
  };
}

function parsePreferenceHistory(payload: unknown): WorkshopPreferenceHistory {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !Number.isSafeInteger(payload.limit) ||
    !Array.isArray(payload.revisions)
  ) {
    throw new Error("Kai returned unsupported preference history.");
  }
  const revisions = payload.revisions.map((item) => {
    if (
      !isRecord(item) ||
      typeof item.revision !== "string" ||
      typeof item.updated_at !== "string" ||
      !Number.isSafeInteger(item.size_bytes)
    ) {
      throw new Error("Kai returned unsupported preference history.");
    }
    return {
      revision: item.revision,
      sizeBytes: item.size_bytes as number,
      updatedAt: item.updated_at,
    };
  });
  return {
    limit: payload.limit as number,
    revisions,
  };
}

function parseCountMap(value: unknown): Record<string, number> | null {
  if (!isRecord(value)) {
    return null;
  }
  const parsed: Record<string, number> = {};
  for (const [key, count] of Object.entries(value)) {
    if (!Number.isSafeInteger(count) || (count as number) < 0) {
      return null;
    }
    parsed[key] = count as number;
  }
  return parsed;
}

function parseMemoryRecord(value: unknown): WorkshopMemoryRecord | null {
  if (!isRecord(value) || !isRecord(value.scope)) {
    return null;
  }
  const scope = parseMemoryScope(value.scope);
  if (
    typeof value.memory_id !== "string" ||
    typeof value.kind !== "string" ||
    !["fact", "episode"].includes(value.kind) ||
    typeof value.source !== "string" ||
    typeof value.memory_type !== "string" ||
    typeof value.preview !== "string" ||
    typeof value.revision !== "string" ||
    !Array.isArray(value.tags) ||
    !value.tags.every((tag) => typeof tag === "string") ||
    typeof value.speaker !== "string" ||
    typeof value.confidence !== "number" ||
    typeof value.created_at !== "string" ||
    typeof value.updated_at !== "string" ||
    scope === null
  ) {
    return null;
  }
  return {
    confidence: value.confidence,
    createdAt: value.created_at,
    kind: value.kind as "fact" | "episode",
    memoryId: value.memory_id,
    memoryType: value.memory_type,
    preview: value.preview,
    revision: value.revision,
    scope,
    source: value.source,
    speaker: value.speaker,
    tags: value.tags as string[],
    updatedAt: value.updated_at,
  };
}

function parseMemoryScope(value: unknown): WorkshopMemoryScope | null {
  if (!isRecord(value) ||
    typeof value.scope !== "string" ||
    !["global", "project", "task"].includes(value.scope) ||
    (value.project_id !== null && typeof value.project_id !== "string") ||
    typeof value.scope_confidence !== "number" ||
    typeof value.scope_source !== "string" ||
    typeof value.legacy_defaulted !== "boolean" ||
    typeof value.invalid_defaulted !== "boolean" ||
    typeof value.retrievable !== "boolean" ||
    (value.exclusion_reason !== null && typeof value.exclusion_reason !== "string")
  ) {
    return null;
  }
  return {
    exclusionReason: value.exclusion_reason,
    invalidDefaulted: value.invalid_defaulted,
    legacyDefaulted: value.legacy_defaulted,
    projectId: value.project_id,
    retrievable: value.retrievable,
    scope: value.scope as WorkshopMemoryScope["scope"],
    scopeConfidence: value.scope_confidence,
    scopeSource: value.scope_source,
  };
}

export async function loadMemoryStats(token: string): Promise<WorkshopMemoryStats> {
  const response = await authorizedFetch({ channelId: "", token }, "/v1/memory/stats");
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load memory statistics."));
  }
  if (!isRecord(payload) || payload.version !== 1 || !isRecord(payload.stats)) {
    throw new Error("Kai returned unsupported memory statistics.");
  }
  const stats = payload.stats;
  const byScope = parseCountMap(stats.by_scope);
  const bySource = parseCountMap(stats.by_source);
  const byType = parseCountMap(stats.by_type);
  const allowedProjects = Array.isArray(stats.allowed_projects)
    ? stats.allowed_projects.map((project) => (
      isRecord(project) && typeof project.project_id === "string" &&
      typeof project.display_name === "string"
        ? { projectId: project.project_id, displayName: project.display_name }
        : null
    ))
    : null;
  if (
    !Number.isSafeInteger(stats.total) ||
    !Number.isSafeInteger(stats.facts) ||
    !Number.isSafeInteger(stats.episodes) ||
    !byScope || !bySource || !byType || !allowedProjects ||
    allowedProjects.some((project) => project === null)
  ) {
    throw new Error("Kai returned unsupported memory statistics.");
  }
  return {
    allowedProjects: allowedProjects as WorkshopMemoryStats["allowedProjects"],
    byScope,
    bySource,
    byType,
    episodes: stats.episodes as number,
    facts: stats.facts as number,
    total: stats.total as number,
  };
}

function parseMemoryMutation(payload: unknown): WorkshopMemoryMutationBatch | null {
  if (!isRecord(payload) || payload.version !== 1 ||
    !["move_scope", "delete"].includes(String(payload.operation)) ||
    !Array.isArray(payload.results)
  ) {
    return null;
  }
  const results = payload.results.map((value) => {
    if (!isRecord(value) || typeof value.memory_id !== "string" ||
      !["succeeded", "not_found", "stale", "failed"].includes(String(value.outcome))
    ) {
      return null;
    }
    const priorScope = value.prior_scope === null ? null : parseMemoryScope(value.prior_scope);
    const newScope = value.new_scope === null ? null : parseMemoryScope(value.new_scope);
    if ((value.prior_scope !== null && priorScope === null) ||
      (value.new_scope !== null && newScope === null)) {
      return null;
    }
    return {
      memoryId: value.memory_id,
      newScope,
      outcome: value.outcome as WorkshopMemoryMutationBatch["results"][number]["outcome"],
      priorScope,
    };
  });
  if (results.some((result) => result === null)) return null;
  return {
    operation: payload.operation as WorkshopMemoryMutationBatch["operation"],
    results: results as WorkshopMemoryMutationBatch["results"],
  };
}

async function memoryMutationRequest(
  token: string,
  path: string,
  options: RequestInit,
): Promise<WorkshopMemoryMutationBatch> {
  const response = await authorizedFetch({ channelId: "", token }, path, options);
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update memories."));
  }
  const parsed = parseMemoryMutation(payload);
  if (!parsed) throw new Error("Kai returned an unsupported memory mutation result.");
  return parsed;
}

export async function moveMemoryScope(
  token: string,
  memoryId: string,
  target: { scope: "global" } | { scope: "project"; projectId: string },
): Promise<WorkshopMemoryMutationBatch> {
  return memoryMutationRequest(
    token,
    `/v1/memory/records/${encodeURIComponent(memoryId)}/scope`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(target.scope === "global"
        ? { scope: "global" }
        : { scope: "project", project_id: target.projectId }),
    },
  );
}

export async function moveMemoriesScope(
  token: string,
  memoryIds: string[],
  target: { scope: "global" } | { scope: "project"; projectId: string },
): Promise<WorkshopMemoryMutationBatch> {
  return memoryMutationRequest(token, "/v1/memory/actions/scope", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(target.scope === "global"
      ? { memory_ids: memoryIds, scope: "global" }
      : { memory_ids: memoryIds, scope: "project", project_id: target.projectId }),
  });
}

export async function deleteMemory(
  token: string,
  memoryId: string,
): Promise<WorkshopMemoryMutationBatch> {
  return memoryMutationRequest(
    token,
    `/v1/memory/records/${encodeURIComponent(memoryId)}`,
    { method: "DELETE" },
  );
}

export async function deleteMemories(
  token: string,
  memoryIds: string[],
): Promise<WorkshopMemoryMutationBatch> {
  return memoryMutationRequest(token, "/v1/memory/actions/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ memory_ids: memoryIds }),
  });
}

function memoryQueryParameters(
  options: WorkshopMemoryFilters & { limit?: number },
): URLSearchParams {
  const parameters = new URLSearchParams();
  if (options.kind !== undefined) parameters.set("kind", options.kind);
  if (options.source !== undefined) parameters.set("source", options.source);
  if (options.memoryType !== undefined) {
    parameters.set("memory_type", options.memoryType);
  }
  if (options.tag !== undefined) parameters.set("tag", options.tag);
  if (options.scope !== undefined) parameters.set("scope", options.scope);
  if (options.projectId !== undefined) {
    parameters.set("project_id", options.projectId);
  }
  if (options.limit !== undefined) parameters.set("limit", String(options.limit));
  return parameters;
}

export async function loadMemoryRecords(
  token: string,
  options: WorkshopMemoryListOptions = {},
): Promise<WorkshopMemoryPage> {
  const query = memoryQueryParameters(options);
  if (options.cursor !== undefined) query.set("cursor", options.cursor);
  if (options.order !== undefined) query.set("order", options.order);
  const suffix = query.size ? `?${query}` : "";
  const response = await authorizedFetch(
    { channelId: "", token },
    `/v1/memory/records${suffix}`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load memories."));
  }
  if (
    !isRecord(payload) || payload.version !== 1 ||
    !Array.isArray(payload.records) ||
    (payload.next_cursor !== null && typeof payload.next_cursor !== "string")
  ) {
    throw new Error("Kai returned an unsupported memory page.");
  }
  const records = payload.records.map(parseMemoryRecord);
  if (records.some((record) => record === null)) {
    throw new Error("Kai returned an unsupported memory record.");
  }
  return {
    nextCursor: payload.next_cursor,
    records: records as WorkshopMemoryRecord[],
  };
}

export async function searchMemories(
  token: string,
  query: string,
  options: WorkshopMemorySearchOptions = {},
): Promise<WorkshopMemorySearch> {
  const parameters = memoryQueryParameters(options);
  parameters.set("q", query);
  const response = await authorizedFetch(
    { channelId: "", token },
    `/v1/memory/search?${parameters}`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not search memories."));
  }
  if (
    !isRecord(payload) || payload.version !== 1 ||
    (payload.active_project_id !== null && typeof payload.active_project_id !== "string") ||
    typeof payload.reason !== "string" || !Array.isArray(payload.hits)
  ) {
    throw new Error("Kai returned unsupported memory search results.");
  }
  const hits = payload.hits.map((value) => {
    if (
      !isRecord(value) || typeof value.raw_score !== "number" ||
      typeof value.adjusted_score !== "number" ||
      typeof value.compact_recall !== "string"
    ) {
      return null;
    }
    const record = parseMemoryRecord(value.record);
    return record ? {
      adjustedScore: value.adjusted_score,
      compactRecall: value.compact_recall,
      rawScore: value.raw_score,
      record,
    } : null;
  });
  if (hits.some((hit) => hit === null)) {
    throw new Error("Kai returned unsupported memory search results.");
  }
  return {
    activeProjectId: payload.active_project_id,
    hits: hits as WorkshopMemorySearch["hits"],
    reason: payload.reason,
  };
}

function parseMemoryEpisode(value: unknown): WorkshopMemoryDetail["episode"] | undefined {
  if (value === null) return null;
  if (!isRecord(value)) return undefined;
  const requiredText = ["goal", "context", "approach", "outcome", "outcome_quality"];
  if (
    requiredText.some((field) => typeof value[field] !== "string") ||
    (value.lessons !== undefined && typeof value.lessons !== "string") ||
    !Array.isArray(value.tags) || !value.tags.every((item) => typeof item === "string") ||
    !Array.isArray(value.actors) || !value.actors.every((item) => typeof item === "string") ||
    !["success", "partial", "failure"].includes(String(value.outcome_quality))
  ) {
    return undefined;
  }
  return {
    actors: value.actors as string[],
    approach: value.approach as string,
    context: value.context as string,
    goal: value.goal as string,
    lessons: typeof value.lessons === "string" ? value.lessons : null,
    outcome: value.outcome as string,
    outcomeQuality: value.outcome_quality as "success" | "partial" | "failure",
    tags: value.tags as string[],
  };
}

function parseMemoryDetail(value: unknown): WorkshopMemoryDetail | null {
  if (!isRecord(value)) return null;
  const record = parseMemoryRecord(value);
  const episode = parseMemoryEpisode(value.episode);
  if (
    !record || episode === undefined || typeof value.content !== "string" ||
    typeof value.compact_recall !== "string" ||
    (value.confirmation_quote !== null && typeof value.confirmation_quote !== "string") ||
    (value.prompt_version !== null && typeof value.prompt_version !== "string")
  ) {
    return null;
  }
  return {
    ...record,
    compactRecall: value.compact_recall,
    confirmationQuote: value.confirmation_quote,
    content: value.content,
    episode,
    promptVersion: value.prompt_version,
  };
}

export async function loadMemoryDetail(
  token: string,
  memoryId: string,
): Promise<WorkshopMemoryDetail> {
  const response = await authorizedFetch(
    { channelId: "", token },
    `/v1/memory/records/${encodeURIComponent(memoryId)}`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load this memory."));
  }
  const record = isRecord(payload) && payload.version === 1
    ? parseMemoryDetail(payload.record)
    : null;
  if (!record) throw new Error("Kai returned an unsupported memory detail.");
  return record;
}

function mutationRequestId(): string {
  return typeof globalThis.crypto?.randomUUID === "function"
    ? globalThis.crypto.randomUUID()
    : `memory-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function memoryContentMutation(
  token: string,
  path: string,
  options: RequestInit,
): Promise<unknown> {
  const response = await authorizedFetch({ channelId: "", token }, path, options);
  const payload = await responsePayload(response);
  if (response.status === 409 && isRecord(payload) && isRecord(payload.error) &&
    typeof payload.error.current_revision === "string"
  ) {
    throw new MemoryRevisionConflictError(
      safeErrorMessage(payload, "This memory changed since it was opened."),
      payload.error.current_revision,
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not save this memory."));
  }
  return payload;
}

export async function createMemoryFact(
  token: string,
  input: {
    content: string;
    tags: string[];
    target: { scope: "global" } | { scope: "project"; projectId: string };
    requestId?: string;
  },
): Promise<WorkshopMemoryCreationResult> {
  const payload = await memoryContentMutation(token, "/v1/memory/records", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      kind: "fact",
      content: input.content,
      tags: input.tags,
      scope: input.target.scope,
      ...(input.target.scope === "project" ? { project_id: input.target.projectId } : {}),
      request_id: input.requestId ?? mutationRequestId(),
    }),
  });
  if (!isRecord(payload) || payload.version !== 1 || typeof payload.created !== "boolean") {
    throw new Error("Kai returned an unsupported memory creation result.");
  }
  const record = parseMemoryDetail(payload.record);
  if (!record) throw new Error("Kai returned an unsupported memory creation result.");
  return { created: payload.created, record };
}

export async function editMemory(
  token: string,
  input: {
    memoryId: string;
    revision: string;
    requestId?: string;
  } & (
    { kind: "fact"; content: string; tags: string[] } |
    { kind: "episode"; episode: NonNullable<WorkshopMemoryDetail["episode"]> }
  ),
): Promise<WorkshopMemoryEditResult> {
  const body = input.kind === "fact"
    ? {
        kind: "fact",
        revision: input.revision,
        request_id: input.requestId ?? mutationRequestId(),
        content: input.content,
        tags: input.tags,
      }
    : {
        kind: "episode",
        revision: input.revision,
        request_id: input.requestId ?? mutationRequestId(),
        episode: {
          goal: input.episode.goal,
          context: input.episode.context,
          approach: input.episode.approach,
          outcome: input.episode.outcome,
          outcome_quality: input.episode.outcomeQuality,
          ...(input.episode.lessons ? { lessons: input.episode.lessons } : {}),
          tags: input.episode.tags,
          actors: input.episode.actors,
        },
      };
  const payload = await memoryContentMutation(
    token,
    `/v1/memory/records/${encodeURIComponent(input.memoryId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  if (
    !isRecord(payload) || payload.version !== 1 ||
    !Array.isArray(payload.changed_fields) ||
    !payload.changed_fields.every((field) => typeof field === "string") ||
    typeof payload.idempotent_replay !== "boolean"
  ) {
    throw new Error("Kai returned an unsupported memory edit result.");
  }
  const record = parseMemoryDetail(payload.record);
  if (!record) throw new Error("Kai returned an unsupported memory edit result.");
  return {
    changedFields: payload.changed_fields as string[],
    idempotentReplay: payload.idempotent_replay,
    record,
  };
}

function parseMemorySourceMessage(value: unknown): WorkshopMemorySourceMessage | null {
  if (!isRecord(value)) {
    return null;
  }
  if (
    typeof value.message_id !== "string" ||
    typeof value.channel_id !== "string" ||
    !CHANNEL_PATTERN.test(value.channel_id) ||
    typeof value.author_principal_id !== "string" ||
    !PRINCIPAL_PATTERN.test(value.author_principal_id) ||
    typeof value.author_kind !== "string" ||
    typeof value.author_display_name !== "string" ||
    typeof value.body !== "string" ||
    typeof value.created_at !== "string"
  ) {
    return null;
  }
  return {
    authorDisplayName: value.author_display_name,
    authorKind: value.author_kind,
    authorPrincipalId: value.author_principal_id,
    body: value.body,
    channelId: value.channel_id,
    createdAt: value.created_at,
    messageId: value.message_id,
  };
}

export async function loadMemorySource(
  token: string,
  memoryId: string,
): Promise<WorkshopMemorySourceContext> {
  const response = await authorizedFetch(
    { channelId: "", token },
    `/v1/memory/records/${encodeURIComponent(memoryId)}/source`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load memory source context."));
  }
  if (!isRecord(payload) || payload.version !== 1 || !isRecord(payload.source_context)) {
    throw new Error("Kai returned unsupported memory source context.");
  }
  const context = payload.source_context;
  const source = context.source === null ? null : parseMemorySourceMessage(context.source);
  const result = context.result === null ? null : parseMemorySourceMessage(context.result);
  if (
    typeof context.status !== "string" ||
    !["available", "unavailable"].includes(context.status) ||
    (context.reason !== null && typeof context.reason !== "string") ||
    (context.run_id !== null && typeof context.run_id !== "string") ||
    (context.source !== null && source === null) ||
    (context.result !== null && result === null)
  ) {
    throw new Error("Kai returned unsupported memory source context.");
  }
  return {
    reason: context.reason,
    result,
    runId: context.run_id,
    source,
    status: context.status as "available" | "unavailable",
  };
}

export async function loadSettingsWorkspace(
  session: WorkshopSession,
): Promise<WorkshopSettingsWorkspace> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/settings`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(
      safeErrorMessage(payload, "Could not load settings and workspaces."),
    );
  }
  return parseSettingsWorkspace(payload, session.channelId);
}

export async function loadRoutingEligibility(
  session: WorkshopSession,
  taskClass: WorkshopRoutingTaskClass,
): Promise<WorkshopRoutingEligibility> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/routing-eligibility?task_class=${encodeURIComponent(taskClass)}`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load routing eligibility."));
  }
  return parseRoutingEligibility(payload);
}

export async function loadRoutingPolicy(
  session: WorkshopSession,
): Promise<WorkshopRoutingPolicy> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/routing-policy`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load routing policy."));
  }
  return parseRoutingPolicy(payload);
}

export async function updateRoutingPolicy(
  session: WorkshopSession,
  taskClass: WorkshopRoutingTaskClass,
  backendOptionId: string | null,
  fallback: "selected" | "fail_closed",
  expectedRevision: number,
): Promise<WorkshopRoutingPolicy> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/routing-policy`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_class: taskClass,
        backend_option_id: backendOptionId,
        fallback,
        expected_revision: expectedRevision,
      }),
    },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update routing policy."));
  }
  return parseRoutingPolicy(payload);
}

export async function loadModelCatalogue(
  session: WorkshopSession,
  optionId: string,
): Promise<WorkshopModelCatalogue> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/models?option_id=${encodeURIComponent(optionId)}`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load the model catalogue."));
  }
  return parseModelCatalogue(payload);
}

export async function refreshModelCatalogue(
  session: WorkshopSession,
  optionId: string,
): Promise<WorkshopModelCatalogue> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/models`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ option_id: optionId }),
    },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not refresh the model catalogue."));
  }
  return parseModelCatalogue(payload);
}

export async function refreshAllModelCatalogues(
  session: WorkshopSession,
): Promise<{ contexts: number; statuses: Record<string, number> }> {
  const response = await authorizedFetch(
    session,
    "/v1/settings/model-catalogue/refresh-all",
    { method: "POST" },
  );
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    !Number.isSafeInteger(payload.contexts) ||
    !isRecord(payload.statuses) ||
    Object.values(payload.statuses).some((count) => !Number.isSafeInteger(count))
  ) {
    throw new Error(safeErrorMessage(payload, "Could not refresh all model catalogues."));
  }
  return {
    contexts: payload.contexts as number,
    statuses: payload.statuses as Record<string, number>,
  };
}

export async function upsertOperatorModel(
  session: WorkshopSession,
  optionId: string,
  modelId: string,
  displayLabel: string,
): Promise<WorkshopModelCatalogue> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/models`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        option_id: optionId,
        model_id: modelId,
        display_label: displayLabel,
      }),
    },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not save the operator model."));
  }
  return parseModelCatalogue(payload);
}

export async function deactivateOperatorModel(
  session: WorkshopSession,
  optionId: string,
  modelId: string,
): Promise<WorkshopModelCatalogue> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/models`,
    {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ option_id: optionId, model_id: modelId }),
    },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not deactivate the operator model."));
  }
  return parseModelCatalogue(payload);
}

export async function loadGitHubSettings(
  session: WorkshopSession,
): Promise<WorkshopGitHubSettings> {
  const response = await authorizedFetch(session, "/v1/settings/github");
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load GitHub settings."));
  }
  return parseGitHubSettings(payload);
}

export async function updateGitHubSettings(
  session: WorkshopSession,
  revision: string,
  change: WorkshopGitHubSettingsChange,
): Promise<WorkshopGitHubSettings> {
  const operation = change.field === "repository"
    ? { repository: { name: change.name, subscribed: change.subscribed } }
    : change.field === "repository_reset"
      ? { reset_repositories: true }
    : change.field === "toggle"
      ? { toggle: { field: change.name, enabled: change.enabled } }
      : { token: change.token };
  const response = await authorizedFetch(session, "/v1/settings/github", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision, ...operation }),
  });
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "GitHub settings changed since they were loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update GitHub settings."));
  }
  return parseGitHubSettings(payload);
}

export async function loadNotificationPreferences(
  session: WorkshopSession,
): Promise<WorkshopNotificationPreferences> {
  const response = await authorizedFetch(session, "/v1/settings/notifications");
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load notification preferences."));
  }
  return parseNotificationPreferences(payload);
}

export async function updateNotificationPreference(
  session: WorkshopSession,
  revision: string,
  change: WorkshopNotificationPreferenceChange,
): Promise<WorkshopNotificationPreferences> {
  const operation = change.field === "destination"
    ? {
        destination_choice_id: change.choiceId,
        integration_class: change.integrationClass,
      }
    : { integration_class: change.integrationClass, reset: true };
  const response = await authorizedFetch(session, "/v1/settings/notifications", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision, ...operation }),
  });
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Notification preferences changed since they were loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update notification preferences."));
  }
  return parseNotificationPreferences(payload);
}

export async function loadChannelNotificationPolicy(
  session: WorkshopSession,
): Promise<WorkshopChannelNotificationPolicy> {
  const response = await authorizedFetch(session, "/v1/settings/channel-notifications");
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load channel notification policy."));
  }
  return parseChannelNotificationPolicy(payload);
}

export async function updateChannelNotificationPolicy(
  session: WorkshopSession,
  revision: string,
  change: WorkshopChannelNotificationPolicyChange,
): Promise<WorkshopChannelNotificationPolicy> {
  const operation = change.field === "channel"
    ? { channel: { channel_id: change.channelId, level: change.level } }
    : change.field === "muted_mentions_notify"
      ? { muted_mentions_notify: change.enabled }
      : change.field === "adapter_delivery"
        ? {
            adapter_delivery: {
              transport: change.transport,
              enabled: change.enabled,
            },
          }
      : {
          do_not_disturb: {
            enabled: change.enabled,
            timezone: change.timezone,
            start: change.start,
            end: change.end,
          },
        };
  const response = await authorizedFetch(session, "/v1/settings/channel-notifications", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision, ...operation }),
  });
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Channel notification policy changed since it was loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update channel notification policy."));
  }
  return parseChannelNotificationPolicy(payload);
}

export async function loadClientPreferences(
  session: WorkshopSession,
): Promise<WorkshopClientPreferences> {
  const response = await authorizedFetch(session, "/v1/settings/clients");
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load client preferences."));
  }
  return parseClientPreferences(payload);
}

export async function loadAppearancePreferences(
  session: Pick<WorkshopSession, "token">,
): Promise<WorkshopAppearancePreferences> {
  const response = await authorizedFetch(
    { channelId: "", token: session.token },
    "/v1/settings/appearance",
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load appearance preferences."));
  }
  return parseAppearancePreferences(payload);
}

export async function updateAppearancePreference(
  session: WorkshopSession,
  revision: string,
  themeId: string,
): Promise<WorkshopAppearancePreferences> {
  const response = await authorizedFetch(session, "/v1/settings/appearance", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision, theme_id: themeId }),
  });
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Appearance preferences changed since they were loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update appearance preferences."));
  }
  return parseAppearancePreferences(payload);
}

export async function loadHumanProfile(
  session: Pick<WorkshopSession, "token">,
): Promise<WorkshopHumanProfile> {
  const response = await authorizedFetch(
    { channelId: "", token: session.token },
    "/v1/settings/profile",
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load your profile."));
  }
  return parseHumanProfile(payload);
}

export async function loadHumanAvatar(
  session: Pick<WorkshopSession, "token">,
): Promise<WorkshopHumanAvatar> {
  const response = await authorizedFetch(
    { channelId: "", token: session.token },
    "/v1/settings/profile/avatar",
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load your avatar."));
  }
  return parseHumanAvatar(payload);
}

export async function uploadHumanAvatar(
  session: Pick<WorkshopSession, "token">,
  file: File,
  expectedStateVersion: number,
  clientOperationId: string,
): Promise<WorkshopHumanAvatar> {
  const body = new FormData();
  body.append("expected_state_version", String(expectedStateVersion));
  body.append("client_operation_id", clientOperationId);
  body.append("file", file, file.name);
  const response = await authorizedFetch(
    { channelId: "", token: session.token },
    "/v1/settings/profile/avatar",
    { method: "POST", body },
  );
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Your avatar changed since it was loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not save your avatar."));
  }
  return parseHumanAvatar(payload);
}

export async function clearHumanAvatar(
  session: Pick<WorkshopSession, "token">,
  expectedStateVersion: number,
  clientOperationId: string,
): Promise<WorkshopHumanAvatar> {
  const response = await authorizedFetch(
    { channelId: "", token: session.token },
    "/v1/settings/profile/avatar",
    {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_state_version: expectedStateVersion,
        client_operation_id: clientOperationId,
      }),
    },
  );
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Your avatar changed since it was loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not clear your avatar."));
  }
  return parseHumanAvatar(payload);
}

export async function updateHumanDisplayName(
  session: Pick<WorkshopSession, "token">,
  displayName: string,
  expectedStateVersion: number,
  clientOperationId: string,
): Promise<WorkshopHumanProfile> {
  const response = await authorizedFetch(
    { channelId: "", token: session.token },
    "/v1/settings/profile",
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        display_name: displayName,
        expected_state_version: expectedStateVersion,
        client_operation_id: clientOperationId,
      }),
    },
  );
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Your profile changed since it was loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update your profile."));
  }
  return parseHumanProfile(payload);
}

export async function updateClientPreference(
  session: WorkshopSession,
  revision: string,
  change: WorkshopClientPreferenceChange,
): Promise<WorkshopClientPreferences> {
  const operation = change.field === "mode"
    ? { mode: change.value }
    : { voice: change.value };
  const response = await authorizedFetch(session, "/v1/settings/clients", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      revision,
      binding_choice_id: change.bindingChoiceId,
      ...operation,
    }),
  });
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Client preferences changed since they were loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update client preferences."));
  }
  return parseClientPreferences(payload);
}

export async function updateRuntimeSettings(
  session: WorkshopSession,
  revision: string,
  change: WorkshopRuntimeSettingsChange,
): Promise<WorkshopSettingsWorkspace> {
  const operation = change.field === "model"
    ? { model: change.value }
    : change.field === "backend"
      ? { backend: change.value }
      : change.field === "timeout"
        ? { timeout_seconds: change.value }
        : { reset: change.value };
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/settings`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision, ...operation }),
    },
  );
  const payload = await responsePayload(response);
  if (response.status === 409) {
    if (
      isRecord(payload) &&
      isRecord(payload.error) &&
      payload.error.code === "runtime_busy"
    ) {
      throw new Error(
        safeErrorMessage(payload, "Finish or stop the active run before switching backends."),
      );
    }
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Settings changed since they were loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update runtime settings."));
  }
  return parseSettingsWorkspace(payload, session.channelId);
}

export async function loadWorkspaceConfig(
  session: WorkshopSession,
): Promise<WorkshopWorkspaceConfig> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/workspace-config`,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load workspace settings."));
  }
  return parseWorkspaceConfig(payload);
}

export async function updateWorkspaceConfig(
  session: WorkshopSession,
  revision: string,
  change: WorkshopWorkspaceSettingChange,
): Promise<WorkshopWorkspaceConfig> {
  const operation = change.field === "reset"
    ? { reset: change.value }
    : { field: change.field, value: change.value };
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/workspace-config`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision, ...operation }),
    },
  );
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Workspace settings changed since they were loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not update workspace settings."));
  }
  return parseWorkspaceConfig(payload);
}

export async function loadPreferenceDocument(
  session: WorkshopSession,
): Promise<WorkshopPreferenceDocument> {
  const response = await authorizedFetch(session, "/v1/preferences");
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load preferences."));
  }
  return parsePreferenceDocument(payload);
}

export async function savePreferenceDocument(
  session: WorkshopSession,
  content: string,
  revision: string,
): Promise<WorkshopPreferenceDocument> {
  const response = await authorizedFetch(session, "/v1/preferences", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, revision }),
  });
  const payload = await responsePayload(response);
  if (response.status === 409) {
    const currentRevision = isRecord(payload) && isRecord(payload.error) &&
      typeof payload.error.current_revision === "string"
      ? payload.error.current_revision
      : "";
    throw new PreferenceRevisionConflictError(
      safeErrorMessage(payload, "Preferences changed since they were opened."),
      currentRevision,
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not save preferences."));
  }
  return parsePreferenceDocument(payload);
}

export async function loadPreferenceHistory(
  session: WorkshopSession,
): Promise<WorkshopPreferenceHistory> {
  const response = await authorizedFetch(session, "/v1/preferences/revisions");
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load preference history."));
  }
  return parsePreferenceHistory(payload);
}

export async function restorePreferenceRevision(
  session: WorkshopSession,
  targetRevision: string,
  currentRevision: string,
): Promise<WorkshopPreferenceDocument> {
  const response = await authorizedFetch(
    session,
    `/v1/preferences/revisions/${encodeURIComponent(targetRevision)}/restore`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: currentRevision }),
    },
  );
  const payload = await responsePayload(response);
  if (response.status === 409) {
    const latestRevision = isRecord(payload) && isRecord(payload.error) &&
      typeof payload.error.current_revision === "string"
      ? payload.error.current_revision
      : "";
    throw new PreferenceRevisionConflictError(
      safeErrorMessage(payload, "Preferences changed before they could be restored."),
      latestRevision,
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not restore preferences."));
  }
  return parsePreferenceDocument(payload);
}

export async function switchWorkspace(
  session: WorkshopSession,
  path: string,
  revision: string,
): Promise<WorkshopSettingsWorkspace> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/workspace`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, revision }),
    },
  );
  const payload = await responsePayload(response);
  if (response.status === 409) {
    throw new SettingsRevisionConflictError(
      safeErrorMessage(payload, "Settings changed since they were loaded."),
    );
  }
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not switch workspace."));
  }
  return parseSettingsWorkspace(payload, session.channelId);
}

export async function submitCommand(
  session: WorkshopSession,
  clientMessageId: string,
  body: string,
  artifact: File | null = null,
  threadRootId: string | null = null,
): Promise<CommandSubmissionResult> {
  if (threadRootId !== null && !MESSAGE_PATTERN.test(threadRootId)) {
    throw new Error("Invalid thread identity.");
  }
  if (artifact && threadRootId) {
    throw new Error("Thread attachments are not available yet.");
  }
  const request = artifact
    ? (() => {
        const form = new FormData();
        form.append("client_message_id", clientMessageId);
        form.append("body", body);
        form.append("file", artifact, artifact.name);
        return { body: form, method: "POST" } satisfies RequestInit;
      })()
    : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          body,
          client_message_id: clientMessageId,
          ...(threadRootId ? { thread_root_id: threadRootId } : {}),
        }),
      };
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/commands`,
    request,
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Kai could not run this command."));
  }
  if (
    isRecord(payload) &&
    payload.version === 3 &&
    typeof payload.acceptance === "string" &&
    typeof payload.message_id === "string" &&
    Array.isArray(payload.runs)
  ) {
    const runs = payload.runs.map((item) => parseRun(item, session.channelId));
    if (runs.some((run) => run === null)) {
      throw new Error("Kai returned an unsupported command response.");
    }
    return {
      acceptance: payload.acceptance,
      messageId: payload.message_id,
      run: (runs as WorkshopRun[])[0] ?? null,
    };
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

export async function loadArtifactBlob(
  session: WorkshopSession,
  artifactId: string,
): Promise<Blob> {
  if (!ARTIFACT_PATTERN.test(artifactId)) {
    throw new Error("Invalid artifact identity.");
  }
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/artifacts/${encodeURIComponent(artifactId)}/content`,
  );
  if (!response.ok) {
    const payload = await responsePayload(response);
    throw new Error(safeErrorMessage(payload, "Could not load this attachment."));
  }
  return await response.blob();
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
  return {
    ...run,
    routingDecision: parseRoutingDecision(payload.routing_decision),
  };
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

function parseHumanNotification(value: unknown): WorkshopHumanNotification | null {
  if (!isRecord(value)) {
    return null;
  }
  const {
    channel_name: channelName,
    created_at: createdAt,
    created_event_position: createdEventPosition,
    kind,
    last_event_position: lastEventPosition,
    notification_id: notificationId,
    read,
    read_at: readAt,
    source_author_display_name: sourceAuthorDisplayName,
    source_author_avatar: sourceAuthorAvatar,
    source_author_principal_id: sourceAuthorPrincipalId,
    source_channel_id: sourceChannelId,
    source_message_id: sourceMessageId,
    source_thread_root_id: sourceThreadRootId,
    state_version: stateVersion,
  } = value;
  if (
    (channelName !== null && typeof channelName !== "string") ||
    typeof createdAt !== "string" ||
    !Number.isSafeInteger(createdEventPosition) ||
    (kind !== "mention" && kind !== "reply" && kind !== "message") ||
    !Number.isSafeInteger(lastEventPosition) ||
    typeof notificationId !== "string" ||
    !HUMAN_NOTIFICATION_PATTERN.test(notificationId) ||
    typeof read !== "boolean" ||
    (readAt !== null && typeof readAt !== "string") ||
    typeof sourceAuthorDisplayName !== "string" ||
    typeof sourceAuthorPrincipalId !== "string" ||
    !PRINCIPAL_PATTERN.test(sourceAuthorPrincipalId) ||
    typeof sourceChannelId !== "string" ||
    !CHANNEL_PATTERN.test(sourceChannelId) ||
    typeof sourceMessageId !== "string" ||
    !MESSAGE_PATTERN.test(sourceMessageId) ||
    (sourceThreadRootId !== null &&
      (typeof sourceThreadRootId !== "string" ||
        !MESSAGE_PATTERN.test(sourceThreadRootId))) ||
    !Number.isSafeInteger(stateVersion) ||
    (stateVersion as number) < 0
  ) {
    return null;
  }
  return {
    channelName,
    createdAt,
    createdEventPosition: createdEventPosition as number,
    kind,
    lastEventPosition: lastEventPosition as number,
    notificationId,
    read,
    readAt,
    sourceAuthorDisplayName,
    sourceAuthorAvatar: parseHumanAvatarDescriptor(
      sourceAuthorAvatar,
      sourceAuthorPrincipalId,
    ),
    sourceAuthorPrincipalId,
    sourceChannelId,
    sourceMessageId,
    sourceThreadRootId,
    stateVersion: stateVersion as number,
  };
}

function parseHumanNotificationCounts(value: unknown): WorkshopHumanNotificationCounts | null {
  if (
    !isRecord(value) ||
    !Number.isSafeInteger(value.total) ||
    !Number.isSafeInteger(value.unread) ||
    !Number.isSafeInteger(value.read) ||
    (value.total as number) < 0 ||
    (value.unread as number) < 0 ||
    (value.read as number) < 0 ||
    (value.total as number) !== (value.unread as number) + (value.read as number) ||
    !isRecord(value.unread_by_channel)
  ) {
    return null;
  }
  const unreadByChannel: Record<string, number> = {};
  let channelTotal = 0;
  for (const [channelId, count] of Object.entries(value.unread_by_channel)) {
    if (
      !CHANNEL_PATTERN.test(channelId) ||
      !Number.isSafeInteger(count) ||
      (count as number) < 1
    ) {
      return null;
    }
    unreadByChannel[channelId] = count as number;
    channelTotal += count as number;
  }
  if (channelTotal !== value.unread) {
    return null;
  }
  return {
    read: value.read as number,
    total: value.total as number,
    unread: value.unread as number,
    unreadByChannel,
  };
}

function parseHumanNotificationPage(payload: unknown): WorkshopHumanNotificationPage {
  if (
    !isRecord(payload) ||
    payload.version !== 1 ||
    !Array.isArray(payload.notifications) ||
    !Number.isSafeInteger(payload.through_position) ||
    (payload.through_position as number) < 0 ||
    (payload.next_cursor !== null && typeof payload.next_cursor !== "string")
  ) {
    throw new Error("Kai returned an unsupported Mentions inbox.");
  }
  const counts = parseHumanNotificationCounts(payload.counts);
  const notifications = payload.notifications.map(parseHumanNotification);
  if (!counts || notifications.some((notification) => notification === null)) {
    throw new Error("Kai returned an unsupported Mentions inbox.");
  }
  return {
    counts,
    nextCursor: payload.next_cursor,
    notifications: notifications as WorkshopHumanNotification[],
    throughPosition: payload.through_position as number,
  };
}

export async function loadHumanNotifications(
  token: string,
  options: { cursor?: string; limit?: number; unreadOnly?: boolean } = {},
  signal?: AbortSignal,
): Promise<WorkshopHumanNotificationPage> {
  const query = new URLSearchParams();
  if (options.cursor) query.set("cursor", options.cursor);
  query.set("limit", String(options.limit ?? 50));
  if (options.unreadOnly) query.set("unread", "1");
  const response = await authorizedFetch(
    { channelId: "", token },
    `/v1/client/notifications?${query}`,
    { signal },
  );
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw new Error(safeErrorMessage(payload, "Could not load mentions."));
  }
  return parseHumanNotificationPage(payload);
}

export async function loadHumanNotificationCounts(
  token: string,
  signal?: AbortSignal,
): Promise<WorkshopHumanNotificationCounts> {
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/notifications/counts",
    { signal },
  );
  const payload = await responsePayload(response);
  if (!response.ok || !isRecord(payload) || payload.version !== 1) {
    throw new Error(safeErrorMessage(payload, "Could not load mention counts."));
  }
  const counts = parseHumanNotificationCounts(payload);
  if (!counts) {
    throw new Error("Kai returned unsupported mention counts.");
  }
  return counts;
}

async function mutateHumanNotification(
  token: string,
  notificationId: string,
  expectedStateVersion: number,
  read: boolean,
  clientOperationId: string,
): Promise<WorkshopHumanNotificationMutation> {
  if (!HUMAN_NOTIFICATION_PATTERN.test(notificationId)) {
    throw new Error("Invalid notification identity.");
  }
  const response = await authorizedFetch(
    { channelId: "", token },
    `/v1/client/notifications/${encodeURIComponent(notificationId)}/${read ? "read" : "unread"}`,
    {
      body: JSON.stringify({
        client_operation_id: clientOperationId,
        expected_state_version: expectedStateVersion,
      }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    },
  );
  const payload = await responsePayload(response);
  const notification = isRecord(payload)
    ? parseHumanNotification(payload.notification)
    : null;
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    !notification ||
    typeof payload.changed !== "boolean" ||
    typeof payload.replayed !== "boolean"
  ) {
    throw new Error(safeErrorMessage(payload, "Could not update this mention."));
  }
  return { changed: payload.changed, notification, replayed: payload.replayed };
}

export function markHumanNotificationRead(
  token: string,
  notificationId: string,
  expectedStateVersion: number,
  clientOperationId: string,
): Promise<WorkshopHumanNotificationMutation> {
  return mutateHumanNotification(
    token,
    notificationId,
    expectedStateVersion,
    true,
    clientOperationId,
  );
}

export function markHumanNotificationUnread(
  token: string,
  notificationId: string,
  expectedStateVersion: number,
  clientOperationId: string,
): Promise<WorkshopHumanNotificationMutation> {
  return mutateHumanNotification(
    token,
    notificationId,
    expectedStateVersion,
    false,
    clientOperationId,
  );
}

export async function markHumanNotificationsRead(
  token: string,
  notifications: { notificationId: string; expectedStateVersion: number }[],
  clientOperationId: string,
): Promise<WorkshopHumanNotificationMutation[]> {
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/notifications/read",
    {
      body: JSON.stringify({
        client_operation_id: clientOperationId,
        notifications: notifications.map((notification) => ({
          expected_state_version: notification.expectedStateVersion,
          notification_id: notification.notificationId,
        })),
      }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    },
  );
  const payload = await responsePayload(response);
  if (!response.ok || !isRecord(payload) || payload.version !== 1 || !Array.isArray(payload.notifications)) {
    throw new Error(safeErrorMessage(payload, "Could not mark mentions read."));
  }
  return payload.notifications.map((raw) => {
    if (!isRecord(raw)) {
      throw new Error("Kai returned an unsupported mention mutation.");
    }
    const notification = parseHumanNotification(raw.notification);
    if (!notification || typeof raw.changed !== "boolean" || typeof raw.replayed !== "boolean") {
      throw new Error("Kai returned an unsupported mention mutation.");
    }
    return { changed: raw.changed, notification, replayed: raw.replayed };
  });
}

function parseChannelUnreadState(value: unknown): WorkshopChannelUnreadState | null {
  if (!isRecord(value)) return null;
  const {
    archived,
    channel_id: channelId,
    channel_kind: channelKind,
    channel_name: channelName,
    first_unread_event_position: firstUnreadEventPosition,
    first_unread_message_id: firstUnreadMessageId,
    last_event_position: lastEventPosition,
    membership_baseline_event_position: membershipBaselineEventPosition,
    read_through_event_position: readThroughEventPosition,
    read_through_message_id: readThroughMessageId,
    state_version: stateVersion,
    unread_count: unreadCount,
    unread_count_capped: unreadCountCapped,
    unread_reply_count: unreadReplyCount,
    unread_reply_count_capped: unreadReplyCountCapped,
    unread_thread_count: unreadThreadCount,
    first_unread_thread_root_id: firstUnreadThreadRootId,
    first_unread_thread_reply_id: firstUnreadThreadReplyId,
    first_unread_thread_event_position: firstUnreadThreadEventPosition,
  } = value;
  if (
    typeof archived !== "boolean" ||
    typeof channelId !== "string" ||
    !CHANNEL_PATTERN.test(channelId) ||
    (channelKind !== "direct" && channelKind !== "group" && channelKind !== "notification") ||
    (channelName !== null && typeof channelName !== "string") ||
    (firstUnreadEventPosition !== null &&
      (!Number.isSafeInteger(firstUnreadEventPosition) || (firstUnreadEventPosition as number) < 0)) ||
    (firstUnreadMessageId !== null &&
      (typeof firstUnreadMessageId !== "string" || !MESSAGE_PATTERN.test(firstUnreadMessageId))) ||
    !Number.isSafeInteger(lastEventPosition) ||
    (lastEventPosition as number) < 0 ||
    !Number.isSafeInteger(membershipBaselineEventPosition) ||
    (membershipBaselineEventPosition as number) < 0 ||
    !Number.isSafeInteger(readThroughEventPosition) ||
    (readThroughEventPosition as number) < 0 ||
    (readThroughMessageId !== null &&
      (typeof readThroughMessageId !== "string" || !MESSAGE_PATTERN.test(readThroughMessageId))) ||
    !Number.isSafeInteger(stateVersion) ||
    (stateVersion as number) < 0 ||
    !Number.isSafeInteger(unreadCount) ||
    (unreadCount as number) < 0 ||
    typeof unreadCountCapped !== "boolean" ||
    !Number.isSafeInteger(unreadReplyCount) ||
    (unreadReplyCount as number) < 0 ||
    typeof unreadReplyCountCapped !== "boolean" ||
    !Number.isSafeInteger(unreadThreadCount) ||
    (unreadThreadCount as number) < 0 ||
    (firstUnreadThreadRootId !== null &&
      (typeof firstUnreadThreadRootId !== "string" || !MESSAGE_PATTERN.test(firstUnreadThreadRootId))) ||
    (firstUnreadThreadReplyId !== null &&
      (typeof firstUnreadThreadReplyId !== "string" || !MESSAGE_PATTERN.test(firstUnreadThreadReplyId))) ||
    (firstUnreadThreadEventPosition !== null &&
      (!Number.isSafeInteger(firstUnreadThreadEventPosition) ||
        (firstUnreadThreadEventPosition as number) < 0)) ||
    ((firstUnreadThreadRootId === null) !== (firstUnreadThreadReplyId === null)) ||
    ((firstUnreadThreadReplyId === null) !== (firstUnreadThreadEventPosition === null)) ||
    ((unreadReplyCount as number) === 0) !== (firstUnreadThreadReplyId === null) ||
    ((firstUnreadMessageId === null) !== (firstUnreadEventPosition === null)) ||
    ((unreadCount as number) === 0) !== (firstUnreadMessageId === null)
  ) {
    return null;
  }
  return {
    archived,
    channelId,
    channelKind,
    channelName,
    firstUnreadEventPosition: firstUnreadEventPosition as number | null,
    firstUnreadMessageId,
    lastEventPosition: lastEventPosition as number,
    membershipBaselineEventPosition: membershipBaselineEventPosition as number,
    readThroughEventPosition: readThroughEventPosition as number,
    readThroughMessageId,
    stateVersion: stateVersion as number,
    unreadCount: unreadCount as number,
    unreadCountCapped,
    unreadReplyCount: unreadReplyCount as number,
    unreadReplyCountCapped,
    unreadThreadCount: unreadThreadCount as number,
    firstUnreadThreadRootId,
    firstUnreadThreadReplyId,
    firstUnreadThreadEventPosition: firstUnreadThreadEventPosition as number | null,
  };
}

function parseThreadUnreadState(value: unknown): WorkshopThreadUnreadState | null {
  if (!isRecord(value)) return null;
  const {
    channel_id: channelId,
    thread_root_id: threadRootId,
    followed,
    follow_baseline_event_position: followBaselineEventPosition,
    read_through_event_position: readThroughEventPosition,
    read_through_message_id: readThroughMessageId,
    state_version: stateVersion,
    last_event_position: lastEventPosition,
    unread_count: unreadCount,
    unread_count_capped: unreadCountCapped,
    first_unread_message_id: firstUnreadMessageId,
    first_unread_event_position: firstUnreadEventPosition,
  } = value;
  if (
    typeof channelId !== "string" || !CHANNEL_PATTERN.test(channelId) ||
    typeof threadRootId !== "string" || !MESSAGE_PATTERN.test(threadRootId) ||
    typeof followed !== "boolean" ||
    !Number.isSafeInteger(followBaselineEventPosition) || (followBaselineEventPosition as number) < 0 ||
    !Number.isSafeInteger(readThroughEventPosition) || (readThroughEventPosition as number) < 0 ||
    (readThroughMessageId !== null &&
      (typeof readThroughMessageId !== "string" || !MESSAGE_PATTERN.test(readThroughMessageId))) ||
    !Number.isSafeInteger(stateVersion) || (stateVersion as number) < 0 ||
    !Number.isSafeInteger(lastEventPosition) || (lastEventPosition as number) < 0 ||
    !Number.isSafeInteger(unreadCount) || (unreadCount as number) < 0 ||
    typeof unreadCountCapped !== "boolean" ||
    (firstUnreadMessageId !== null &&
      (typeof firstUnreadMessageId !== "string" || !MESSAGE_PATTERN.test(firstUnreadMessageId))) ||
    (firstUnreadEventPosition !== null &&
      (!Number.isSafeInteger(firstUnreadEventPosition) || (firstUnreadEventPosition as number) < 0)) ||
    ((firstUnreadMessageId === null) !== (firstUnreadEventPosition === null)) ||
    ((unreadCount as number) === 0) !== (firstUnreadMessageId === null)
  ) return null;
  return {
    channelId,
    threadRootId,
    followed,
    followBaselineEventPosition: followBaselineEventPosition as number,
    readThroughEventPosition: readThroughEventPosition as number,
    readThroughMessageId,
    stateVersion: stateVersion as number,
    lastEventPosition: lastEventPosition as number,
    unreadCount: unreadCount as number,
    unreadCountCapped,
    firstUnreadMessageId,
    firstUnreadEventPosition: firstUnreadEventPosition as number | null,
  };
}

function parseFollowedThread(value: unknown): WorkshopFollowedThread | null {
  if (!isRecord(value)) return null;
  const state = parseThreadUnreadState(value.state);
  const {
    channel_name: channelName,
    channel_archived: channelArchived,
    root_author_display_name: rootAuthorDisplayName,
    root_excerpt: rootExcerpt,
    root_created_at: rootCreatedAt,
    latest_reply_message_id: latestReplyMessageId,
    latest_reply_author_display_name: latestReplyAuthorDisplayName,
    latest_reply_excerpt: latestReplyExcerpt,
    latest_reply_created_at: latestReplyCreatedAt,
  } = value;
  if (
    !state || !state.followed ||
    (channelName !== null && typeof channelName !== "string") ||
    typeof channelArchived !== "boolean" ||
    typeof rootAuthorDisplayName !== "string" ||
    typeof rootExcerpt !== "string" || rootExcerpt.length > 280 ||
    typeof rootCreatedAt !== "string" ||
    (latestReplyMessageId !== null &&
      (typeof latestReplyMessageId !== "string" || !MESSAGE_PATTERN.test(latestReplyMessageId))) ||
    (latestReplyAuthorDisplayName !== null && typeof latestReplyAuthorDisplayName !== "string") ||
    (latestReplyExcerpt !== null &&
      (typeof latestReplyExcerpt !== "string" || latestReplyExcerpt.length > 280)) ||
    (latestReplyCreatedAt !== null && typeof latestReplyCreatedAt !== "string") ||
    (latestReplyMessageId === null) !== (latestReplyAuthorDisplayName === null) ||
    (latestReplyMessageId === null) !== (latestReplyExcerpt === null) ||
    (latestReplyMessageId === null) !== (latestReplyCreatedAt === null)
  ) return null;
  return {
    state,
    channelName,
    channelArchived,
    rootAuthorDisplayName,
    rootExcerpt,
    rootCreatedAt,
    latestReplyMessageId,
    latestReplyAuthorDisplayName,
    latestReplyExcerpt,
    latestReplyCreatedAt,
  };
}

export async function loadFollowedThreads(
  token: string,
  signal?: AbortSignal,
): Promise<WorkshopFollowedThreadSnapshot> {
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/followed-threads",
    { signal },
  );
  const payload = await responsePayload(response);
  const threads = isRecord(payload) && Array.isArray(payload.threads)
    ? payload.threads.map(parseFollowedThread)
    : [];
  if (
    !response.ok || !isRecord(payload) || payload.version !== 1 ||
    !Array.isArray(payload.threads) || threads.some((thread) => thread === null) ||
    !Number.isSafeInteger(payload.through_position) ||
    (payload.through_position as number) < 0
  ) {
    throw new Error(safeErrorMessage(payload, "Could not load followed threads."));
  }
  return {
    threads: threads as WorkshopFollowedThread[],
    throughPosition: payload.through_position as number,
  };
}

export async function loadThreadUnread(
  session: WorkshopSession,
  rootMessageId: string,
  signal?: AbortSignal,
): Promise<WorkshopThreadUnreadState> {
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/threads/${encodeURIComponent(rootMessageId)}/unread`,
    { signal },
  );
  const payload = await responsePayload(response);
  if (response.status === 404) throw new ChannelAccessError("This thread is no longer accessible.");
  const state = isRecord(payload) ? parseThreadUnreadState(payload.state) : null;
  if (!response.ok || !isRecord(payload) || payload.version !== 1 || !state) {
    throw new Error(safeErrorMessage(payload, "Could not load thread unread state."));
  }
  return state;
}

async function mutateThreadUnread(
  session: WorkshopSession,
  rootMessageId: string,
  operation: "follow" | "unfollow" | "read-position",
  expectedStateVersion: number,
  clientOperationId: string,
  messageId?: string,
): Promise<WorkshopThreadUnreadMutation> {
  const body: Record<string, unknown> = {
    client_operation_id: clientOperationId,
    expected_state_version: expectedStateVersion,
  };
  if (messageId !== undefined) body.message_id = messageId;
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/threads/${encodeURIComponent(rootMessageId)}/${operation}`,
    { body: JSON.stringify(body), headers: { "Content-Type": "application/json" }, method: "POST" },
  );
  const payload = await responsePayload(response);
  if (response.status === 404) throw new ChannelAccessError("This thread is no longer accessible.");
  if (response.status === 409) {
    throw new ThreadUnreadConflictError(safeErrorMessage(payload, "Thread state changed in another tab."));
  }
  const state = isRecord(payload) ? parseThreadUnreadState(payload.state) : null;
  if (!response.ok || !isRecord(payload) || payload.version !== 1 || !state ||
      typeof payload.replayed !== "boolean") {
    throw new Error(safeErrorMessage(payload, "Could not update thread unread state."));
  }
  return { replayed: payload.replayed, state };
}

export function setThreadFollowed(
  session: WorkshopSession,
  rootMessageId: string,
  followed: boolean,
  expectedStateVersion: number,
  clientOperationId: string,
): Promise<WorkshopThreadUnreadMutation> {
  return mutateThreadUnread(
    session,
    rootMessageId,
    followed ? "follow" : "unfollow",
    expectedStateVersion,
    clientOperationId,
  );
}

export function advanceThreadReadPosition(
  session: WorkshopSession,
  rootMessageId: string,
  messageId: string,
  expectedStateVersion: number,
  clientOperationId: string,
): Promise<WorkshopThreadUnreadMutation> {
  return mutateThreadUnread(
    session,
    rootMessageId,
    "read-position",
    expectedStateVersion,
    clientOperationId,
    messageId,
  );
}

export async function loadChannelUnread(
  token: string,
  signal?: AbortSignal,
): Promise<WorkshopChannelUnreadSnapshot> {
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/unread",
    { signal },
  );
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    !Array.isArray(payload.channels) ||
    !Number.isSafeInteger(payload.total_unread) ||
    (payload.total_unread as number) < 0 ||
    typeof payload.total_unread_capped !== "boolean" ||
    !Number.isSafeInteger(payload.through_position) ||
    (payload.through_position as number) < 0
  ) {
    throw new Error(safeErrorMessage(payload, "Could not load unread messages."));
  }
  const channels = payload.channels.map(parseChannelUnreadState);
  if (channels.some((state) => state === null)) {
    throw new Error("Kai returned unsupported unread state.");
  }
  return {
    channels: channels as WorkshopChannelUnreadState[],
    throughPosition: payload.through_position as number,
    totalUnread: payload.total_unread as number,
    totalUnreadCapped: payload.total_unread_capped,
  };
}

export async function advanceChannelReadPosition(
  session: WorkshopSession,
  messageId: string,
  expectedStateVersion: number,
  clientOperationId: string,
): Promise<WorkshopChannelReadPositionMutation> {
  if (!MESSAGE_PATTERN.test(messageId)) {
    throw new Error("Invalid message identity.");
  }
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/read-position`,
    {
      body: JSON.stringify({
        client_operation_id: clientOperationId,
        expected_state_version: expectedStateVersion,
        message_id: messageId,
      }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    },
  );
  const payload = await responsePayload(response);
  if (response.status === 404) {
    throw new ChannelAccessError("This session can no longer access that Workshop channel.");
  }
  if (response.status === 409) {
    throw new ChannelReadPositionConflictError(
      safeErrorMessage(payload, "Unread state changed in another tab."),
    );
  }
  const state = isRecord(payload) ? parseChannelUnreadState(payload.state) : null;
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    !state ||
    typeof payload.replayed !== "boolean"
  ) {
    throw new Error(safeErrorMessage(payload, "Could not update unread messages."));
  }
  return { replayed: payload.replayed, state };
}

export async function streamChannelUnread(
  token: string,
  lastEventId: string | null,
  handlers: {
    onChanged: (signal: WorkshopChannelUnreadSignal, eventId: string) => void;
    onConnected: () => void;
  },
  signal: AbortSignal,
): Promise<void> {
  const query = lastEventId === null
    ? ""
    : `?after_position=${encodeURIComponent(lastEventId)}`;
  const headers = new Headers();
  headers.set("X-Kai-Stream-ID", `${eventStreamId()}:unread`);
  const response = await authorizedFetch(
    { channelId: "", token },
    `/v1/client/unread/events${query}`,
    { headers, signal },
  );
  if (!response.ok || !response.body) {
    const payload = await responsePayload(response);
    throw new Error(safeErrorMessage(payload, "Live unread updates are unavailable."));
  }
  handlers.onConnected();
  const reader = response.body.getReader();
  const textDecoder = new TextDecoder();
  const eventDecoder = new EventStreamDecoder();
  while (!signal.aborted) {
    const { done, value } = await reader.read();
    if (done) return;
    for (const event of eventDecoder.push(textDecoder.decode(value, { stream: true }))) {
      if (
        event.eventName !== "channel_unread.changed" ||
        !event.eventId ||
        !/^\d+$/.test(event.eventId)
      ) continue;
      let payload: unknown;
      try {
        payload = JSON.parse(event.data);
      } catch {
        continue;
      }
      const eventPosition = Number(event.eventId);
      const state = isRecord(payload) ? parseChannelUnreadState(payload.state) : null;
      if (
        !isRecord(payload) ||
        payload.version !== 1 ||
        payload.event_position !== eventPosition ||
        !Number.isSafeInteger(eventPosition) ||
        !state
      ) continue;
      handlers.onChanged({ eventPosition, state }, event.eventId);
    }
  }
}

export async function loadChannelMessage(
  session: WorkshopSession,
  messageId: string,
  signal?: AbortSignal,
): Promise<TimelineMessage> {
  if (!MESSAGE_PATTERN.test(messageId)) {
    throw new Error("Invalid message identity.");
  }
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/messages/${encodeURIComponent(messageId)}`,
    { signal },
  );
  const payload = await responsePayload(response);
  const message = isRecord(payload) ? parseMessage(payload.message, session.channelId) : null;
  if (!response.ok || !isRecord(payload) || payload.version !== 1 || !message) {
    throw new Error(safeErrorMessage(payload, "Could not open the source message."));
  }
  return message;
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
    nextCursor: typeof payload.next_cursor === "string" ? payload.next_cursor : null,
    throughPosition: payload.through_position as number,
    previousCursor:
      typeof payload.previous_cursor === "string" ? payload.previous_cursor : null,
  };
}

export async function loadTimeline(
  session: WorkshopSession,
  signal: AbortSignal,
  startMessageId: string | null = null,
): Promise<TimelineSnapshot> {
  // Tail-first: one bounded request for the newest window, so opening a
  // channel costs the same regardless of how long its history is.
  // Earlier history stays behind previousCursor and loads on demand.
  const query = new URLSearchParams({ limit: "100" });
  if (startMessageId !== null) {
    if (!MESSAGE_PATTERN.test(startMessageId)) {
      throw new Error("Invalid unread message identity.");
    }
    query.set("start_message_id", startMessageId);
  } else {
    query.set("tail", "1");
  }
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/timeline?${query}`,
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

export async function loadThreadTimeline(
  session: WorkshopSession,
  rootMessageId: string,
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<ThreadTimelineSnapshot> {
  if (!MESSAGE_PATTERN.test(rootMessageId)) {
    throw new Error("Invalid thread identity.");
  }
  const query = new URLSearchParams({ limit: "100" });
  if (cursor) {
    query.set("cursor", cursor);
  }
  const response = await authorizedFetch(
    session,
    `/v1/channels/${encodeURIComponent(session.channelId)}/threads/${encodeURIComponent(rootMessageId)}?${query}`,
    { signal },
  );
  const payload = await responsePayload(response);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.version !== 1 ||
    payload.channel_id !== session.channelId ||
    payload.thread_root_id !== rootMessageId ||
    !Array.isArray(payload.messages) ||
    !Number.isSafeInteger(payload.through_position)
  ) {
    throw new Error(safeErrorMessage(payload, "Could not load this thread."));
  }
  const root = parseMessage(payload.root, session.channelId);
  const messages = payload.messages.map((raw) => parseMessage(raw, session.channelId));
  if (!root || messages.some((message) => message === null)) {
    throw new Error("Kai returned an unsupported thread response.");
  }
  return {
    root,
    messages: messages as TimelineMessage[],
    nextCursor: typeof payload.next_cursor === "string" ? payload.next_cursor : null,
    throughPosition: payload.through_position as number,
  };
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

export async function streamAgentChanges(
  token: string,
  lastEventId: string | null,
  handlers: {
    onChanged: (signal: WorkshopAgentChangeSignal, eventId: string) => void;
    onConnected: () => void;
  },
  signal: AbortSignal,
): Promise<void> {
  const headers = new Headers();
  if (lastEventId !== null) {
    headers.set("Last-Event-ID", lastEventId);
  }
  headers.set("X-Kai-Stream-ID", `${eventStreamId()}:agents`);
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/agents/events",
    { headers, signal },
  );
  if (response.status === 409) {
    throw new ResynchronizationRequired();
  }
  if (!response.ok || !response.body) {
    const payload = await responsePayload(response);
    throw new Error(
      safeErrorMessage(payload, "Live agent updates are unavailable."),
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
    for (const event of eventDecoder.push(textDecoder.decode(value, { stream: true }))) {
      if (
        (event.eventName !== "agent.definition.changed" &&
          event.eventName !== "agent.enablement.changed" &&
          event.eventName !== "workshop.navigation.changed") ||
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
      const eventPosition = Number(event.eventId);
      if (
        !isRecord(payload) ||
        payload.version !== 1 ||
        payload.event_position !== eventPosition ||
        !Number.isSafeInteger(eventPosition) ||
        typeof payload.event_type !== "string" ||
        (event.eventName === "workshop.navigation.changed"
          ? payload.definition_id !== null
          : typeof payload.definition_id !== "string" ||
            !AGENT_DEFINITION_PATTERN.test(payload.definition_id)) ||
        (payload.revision_id !== null &&
          (typeof payload.revision_id !== "string" ||
            !AGENT_REVISION_PATTERN.test(payload.revision_id))) ||
        typeof payload.occurred_at !== "string"
      ) {
        continue;
      }
      handlers.onChanged(
        {
          definitionId: payload.definition_id as string | null,
          eventPosition,
          eventType: payload.event_type,
          kind: event.eventName === "workshop.navigation.changed"
            ? "navigation"
            : event.eventName === "agent.definition.changed"
              ? "definition"
              : "enablement",
          occurredAt: payload.occurred_at,
          revisionId: payload.revision_id,
        },
        event.eventId,
      );
    }
  }
}

export async function streamHumanNotifications(
  token: string,
  lastEventId: string | null,
  handlers: {
    onChanged: (signal: WorkshopHumanNotificationSignal, eventId: string) => void;
    onConnected: () => void;
  },
  signal: AbortSignal,
): Promise<void> {
  const headers = new Headers();
  if (lastEventId !== null) {
    headers.set("Last-Event-ID", lastEventId);
  }
  headers.set("X-Kai-Stream-ID", `${eventStreamId()}:mentions`);
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/notifications/events",
    { headers, signal },
  );
  if (response.status === 409) {
    throw new ResynchronizationRequired();
  }
  if (!response.ok || !response.body) {
    const payload = await responsePayload(response);
    throw new Error(safeErrorMessage(payload, "Live mention updates are unavailable."));
  }
  handlers.onConnected();
  const reader = response.body.getReader();
  const textDecoder = new TextDecoder();
  const eventDecoder = new EventStreamDecoder();
  while (!signal.aborted) {
    const { done, value } = await reader.read();
    if (done) return;
    for (const event of eventDecoder.push(textDecoder.decode(value, { stream: true }))) {
      if (
        event.eventName !== "human_notification.changed" ||
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
      const eventPosition = Number(event.eventId);
      if (
        !isRecord(payload) ||
        payload.version !== 1 ||
        payload.event_position !== eventPosition ||
        !Number.isSafeInteger(eventPosition) ||
        ![
          "human_notification.created",
          "human_notification.read",
          "human_notification.unread",
        ].includes(String(payload.event_type))
      ) {
        continue;
      }
      const notification = parseHumanNotification(payload.notification);
      if (!notification) continue;
      handlers.onChanged(
        {
          eventPosition,
          notification,
          transition: payload.event_type as WorkshopHumanNotificationSignal["transition"],
        },
        event.eventId,
      );
    }
  }
}

export async function streamPrincipalEvents(
  token: string,
  lastEventId: string | null,
  handlers: {
    onBatch: (batch: WorkshopPrincipalEventBatch, eventId: string) => void;
    onConnected: () => void;
  },
  signal: AbortSignal,
): Promise<void> {
  const headers = new Headers();
  if (lastEventId !== null) {
    headers.set("Last-Event-ID", lastEventId);
  }
  headers.set("X-Kai-Stream-ID", `${eventStreamId()}:principal`);
  const response = await authorizedFetch(
    { channelId: "", token },
    "/v1/client/events",
    { headers, signal },
  );
  if (response.status === 409) {
    throw new ResynchronizationRequired();
  }
  if (!response.ok || !response.body) {
    const payload = await responsePayload(response);
    throw new Error(
      safeErrorMessage(payload, "Live Workshop updates are unavailable."),
    );
  }
  handlers.onConnected();
  const reader = response.body.getReader();
  const textDecoder = new TextDecoder();
  const eventDecoder = new EventStreamDecoder();
  while (!signal.aborted) {
    const { done, value } = await reader.read();
    if (done) return;
    for (const event of eventDecoder.push(textDecoder.decode(value, { stream: true }))) {
      if (
        event.eventName !== "workshop.principal.changed" ||
        !event.eventId ||
        !/^\d+$/.test(event.eventId)
      ) continue;
      let payload: unknown;
      try {
        payload = JSON.parse(event.data);
      } catch {
        continue;
      }
      const throughPosition = Number(event.eventId);
      if (
        !isRecord(payload) ||
        payload.version !== 1 ||
        payload.through_position !== throughPosition ||
        !Number.isSafeInteger(throughPosition) ||
        !Array.isArray(payload.changes)
      ) continue;
      const changes: WorkshopPrincipalEventBatch["changes"] = [];
      let valid = true;
      for (const rawChange of payload.changes) {
        if (
          !isRecord(rawChange) ||
          !Number.isSafeInteger(rawChange.event_position) ||
          (rawChange.event_position as number) < 0 ||
          (rawChange.event_position as number) > throughPosition ||
          !Array.isArray(rawChange.agent_changes) ||
          !Array.isArray(rawChange.notification_changes) ||
          !Array.isArray(rawChange.unread_changes) ||
          !Array.isArray(rawChange.thread_changes)
        ) {
          valid = false;
          break;
        }
        const eventPosition = rawChange.event_position as number;
        const agentChanges: WorkshopPrincipalEventBatch["changes"][number]["agentChanges"] = [];
        for (const item of rawChange.agent_changes) {
          if (
            !isRecord(item) ||
            ![
              "agent.definition.changed",
              "agent.enablement.changed",
              "workshop.navigation.changed",
            ].includes(String(item.kind)) ||
            typeof item.event_type !== "string" ||
            (item.kind === "workshop.navigation.changed"
              ? item.definition_id !== null
              : typeof item.definition_id !== "string" ||
                !AGENT_DEFINITION_PATTERN.test(item.definition_id)) ||
            (item.revision_id !== null &&
              (typeof item.revision_id !== "string" ||
                !AGENT_REVISION_PATTERN.test(item.revision_id))) ||
            typeof item.occurred_at !== "string"
          ) {
            valid = false;
            break;
          }
          agentChanges.push({
            definitionId: item.definition_id as string | null,
            eventPosition,
            eventType: item.event_type,
            kind: item.kind === "workshop.navigation.changed"
              ? "navigation"
              : item.kind === "agent.definition.changed"
                ? "definition"
                : "enablement",
            occurredAt: item.occurred_at,
            revisionId: item.revision_id as string | null,
          });
        }
        if (!valid) break;
        const notificationChanges: WorkshopPrincipalEventBatch["changes"][number]["notificationChanges"] = [];
        for (const item of rawChange.notification_changes) {
          const notification = isRecord(item)
            ? parseHumanNotification(item.notification)
            : null;
          if (
            !isRecord(item) ||
            !notification ||
            ![
              "human_notification.created",
              "human_notification.read",
              "human_notification.unread",
            ].includes(String(item.event_type))
          ) {
            valid = false;
            break;
          }
          notificationChanges.push({
            eventPosition,
            notification,
            transition: item.event_type as WorkshopHumanNotificationSignal["transition"],
          });
        }
        if (!valid) break;
        const unreadChanges: WorkshopPrincipalEventBatch["changes"][number]["unreadChanges"] = [];
        for (const item of rawChange.unread_changes) {
          const state = isRecord(item) ? parseChannelUnreadState(item.state) : null;
          if (!state) {
            valid = false;
            break;
          }
          unreadChanges.push({ eventPosition, state });
        }
        if (!valid) break;
        const threadChanges: WorkshopPrincipalEventBatch["changes"][number]["threadChanges"] = [];
        for (const item of rawChange.thread_changes) {
          const state = isRecord(item) ? parseThreadUnreadState(item.state) : null;
          if (
            !isRecord(item) ||
            !state ||
            ![
              "message.created",
              "thread.followed",
              "thread.unfollowed",
              "thread_read_position.advanced",
            ].includes(String(item.event_type))
          ) {
            valid = false;
            break;
          }
          threadChanges.push({
            eventPosition,
            state,
            transition: item.event_type as WorkshopThreadUnreadSignal["transition"],
          });
        }
        if (!valid) break;
        changes.push({ agentChanges, eventPosition, notificationChanges, threadChanges, unreadChanges });
      }
      if (!valid) continue;
      handlers.onBatch({ changes, throughPosition }, event.eventId);
    }
  }
}

export async function streamTimeline(
  session: WorkshopSession,
  lastEventId: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const headers = new Headers({ "Last-Event-ID": lastEventId });
  headers.set("X-Kai-Stream-ID", eventStreamId());
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
      if (event.eventName === "timeline.message.reactions_changed") {
        const messageId = payload.message_id;
        const reactions = parseReactions(payload.reactions);
        if (
          typeof messageId !== "string" ||
          !MESSAGE_PATTERN.test(messageId) ||
          reactions === null
        ) {
          continue;
        }
        handlers.onReactions?.(messageId, reactions, event.eventId);
        continue;
      }
      if (event.eventName === "run.lifecycle.changed") {
        const run = parseRun(payload.run, session.channelId);
        const routingDecision = parseRoutingDecision(payload.routing_decision);
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
            run: { ...run, routingDecision },
            transition: transition as WorkshopRunTransition,
          },
          event.eventId,
        );
      }
    }
  }
}
