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
  WorkshopEditableCapability,
  WorkshopSettingsMutation,
  WorkshopModelCatalogue,
  WorkshopSettingsWorkspace,
  WorkshopWorkspaceConfig,
  WorkshopPreferenceDocument,
  WorkshopPreferenceHistory,
  WorkshopGitHubSettings,
  WorkshopGitHubSettingsChange,
  WorkshopNotificationPreferences,
  WorkshopNotificationPreferenceChange,
  WorkshopClientPreferences,
  WorkshopClientPreferenceChange,
  WorkshopAppearancePreferences,
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
} from "./types";
import { isWorkshopThemeId } from "./theme";
import {
  AGENT_PATTERN,
  ARTIFACT_PATTERN,
  CHANNEL_PATTERN,
  PRINCIPAL_PATTERN,
  WORKSHOP_PATTERN,
} from "./types";

export class AuthenticationError extends Error {}
export class ChannelAccessError extends Error {}
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
    artifacts: suppliedArtifacts,
  } = value;
  const rawArtifacts = suppliedArtifacts ?? [];
  if (
    typeof authorDisplayName !== "string" ||
    typeof authorKind !== "string" ||
    typeof body !== "string" ||
    messageChannelId !== channelId ||
    typeof createdAt !== "string" ||
    !Number.isSafeInteger(eventPosition) ||
    typeof messageId !== "string" ||
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
  return {
    artifacts,
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
): Promise<CommandSubmissionResult> {
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
        body: JSON.stringify({ body, client_message_id: clientMessageId }),
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
