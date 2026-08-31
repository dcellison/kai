export const CHANNEL_PATTERN = /^chn_[0-9a-f]{32}$/;
export const AGENT_PATTERN = /^agt_[0-9a-f]{32}$/;
export const AGENT_DEFINITION_PATTERN = /^adf_[0-9a-f]{32}$/;
export const AGENT_REVISION_PATTERN = /^adr_[0-9a-f]{32}$/;
export const AGENT_ENABLEMENT_PATTERN = /^aen_[0-9a-f]{32}$/;
export const PRINCIPAL_PATTERN = /^prn_[0-9a-f]{32}$/;
export const RUNTIME_PROFILE_PATTERN = /^rtp_[0-9a-f]{32}$/;
export const WORKSHOP_PATTERN = /^wsp_[0-9a-f]{32}$/;
export const ARTIFACT_PATTERN = /^art_[0-9a-f]{32}$/;
export const MESSAGE_PATTERN = /^msg_[0-9a-f]{32}$/;
export const HUMAN_HANDLE_PATTERN = /^[a-z][a-z0-9_]{0,31}$/;

export interface WorkshopSession {
  channelId: string;
  token: string;
}

export interface WorkshopAgentSummary {
  agentId: string;
  available: boolean;
  engaged: boolean;
  engagedUntil: string | null;
  handle: string;
  memoryScope: "private" | "shared_channel";
  name: string;
  principalId: string;
  runtimeProfileId: string | null;
  sponsorDisplayName: string | null;
  sponsorPrincipalId: string | null;
}

export interface WorkshopParticipantSummary {
  displayName: string;
  handle: string | null;
  kind: string;
  principalId: string;
}

export type WorkshopChannelKind = "direct" | "group" | "notification";

export interface WorkshopChannelSummary {
  agents: WorkshopAgentSummary[];
  archivedAt?: string | null;
  canSubmitCommands: boolean;
  channelId: string;
  kind: WorkshopChannelKind;
  lifecycleEventPosition?: number | null;
  name: string | null;
  participants: WorkshopParticipantSummary[];
  role: string;
}

export interface WorkshopSummary {
  channels: WorkshopChannelSummary[];
  name: string;
  role: string;
  workshopId: string;
}

export interface WorkshopNavigation {
  principal: {
    displayName: string;
    handle: string | null;
    principalId: string;
  };
  workshops: WorkshopSummary[];
}

export type WorkshopAgentCapability =
  | "agent_delegation"
  | "image_input"
  | "text_generation"
  | "tool_activity"
  | "workspace_execution";

export type WorkshopAgentLifecycleState = "draft" | "active" | "archived";
export type WorkshopAgentEnablementState = "available" | "enabled" | "disabled";

export interface WorkshopAgentRevision {
  capabilities: WorkshopAgentCapability[];
  createdAt: string;
  createdByPrincipalId: string | null;
  eventPosition: number;
  instructions: string;
  purpose: string;
  revisionId: string;
  revisionNumber: number;
}

export interface WorkshopAgentDefinition {
  activeRevisionId: string | null;
  agentId: string;
  createdAt: string;
  createdByPrincipalId: string | null;
  definitionId: string;
  description: string;
  displayName: string;
  handle: string;
  lifecycleState: WorkshopAgentLifecycleState;
  presentation: { avatar?: string };
  revisions: WorkshopAgentRevision[];
  stateVersion: number;
}

export interface WorkshopAgentRuntimeOption {
  backendOptions: string[];
  displayName: string;
  runtimeProfileId: string;
}

export interface WorkshopAgentEnablement {
  agentId: string;
  definitionId: string;
  directChannelId: string | null;
  displayName: string;
  eligibleRuntimes: WorkshopAgentRuntimeOption[];
  enablementId: string | null;
  handle: string;
  lifecycleState: WorkshopAgentEnablementState;
  runtimeProfileId: string | null;
  stateVersion: number | null;
}

export interface WorkshopAgentChangeSignal {
  definitionId: string;
  eventPosition: number;
  eventType: string;
  kind: "definition" | "enablement";
  occurredAt: string;
  revisionId: string | null;
}

export interface WorkshopModelOption {
  displayName: string;
  modelId: string;
  retained: boolean;
  selectable: boolean;
  sources: string[];
  status: "available" | "not_advertised" | "unavailable" | "unknown";
}

export interface WorkshopModelCatalogue {
  lastKnownGood: boolean;
  models: WorkshopModelOption[];
  optionId: string;
  refresh: {
    errorCode: string | null;
    errorDetail: string | null;
    expiresAt: string | null;
    generation: number;
    lastAttemptAt: string;
    lastSuccessfulRefreshAt: string | null;
    status: string;
  } | null;
  runtimeProfileId: string;
  stale: boolean;
}

export interface WorkshopWorkspaceOption {
  current: boolean;
  home: boolean;
  name: string;
  path: string;
}

export interface WorkshopEditableCapability {
  choices: string[] | null;
  field: string;
  maximum: number | null;
  minimum: number | null;
  resettable: boolean;
  scope: "runtime" | "workspace";
  valueType: "authorized_workspace" | "backend_id" | "integer_seconds" | "model_id" | "text";
}

export interface WorkshopSettingsMutation {
  changed: boolean;
  operation: string;
  providerSessionInvalidated: boolean;
  runtimeAction: "deferred_until_next_run" | "restarted" | "unchanged";
}

export interface WorkshopSettingsWorkspace {
  backend: string;
  backendOptionId: string;
  backendOptions: { backend: string; current: boolean; optionId: string; provider: string }[];
  capabilities: WorkshopEditableCapability[];
  channelId: string;
  model: { defaultValue: string; source: string; value: string };
  modelCatalogue: {
    errorCode: string | null;
    errorDetail: string | null;
    lastAttemptAt: string | null;
    lastKnownGood: boolean;
    lastSuccessfulRefreshAt: string | null;
    stale: boolean;
    status: string | null;
  } | null;
  modelOptions: WorkshopModelOption[] | null;
  mutation: WorkshopSettingsMutation | null;
  principalId: string;
  provider: string;
  revision: string;
  runtimeProfileId: string;
  timeoutSeconds: { defaultValue: number; source: string; value: number };
  workspace: string;
  workspaces: WorkshopWorkspaceOption[];
}

export type WorkshopRoutingTaskClass = "conversation" | "coding" | "vision";

export interface WorkshopRoutingEligibility {
  agentId: string;
  candidates: {
    allowedServices: string[];
    backend: string;
    capabilities: {
      capability: "image_input" | "text_generation" | "tool_activity" | "workspace_execution";
      evidence: string;
      support: "supported" | "unsupported" | "unknown";
    }[];
    eligible: boolean;
    modelId: string;
    modelSource: "current_selection" | "protected_default";
    optionId: string;
    provider: string;
    reasons: { code: string; detail: string }[];
    selected: boolean;
  }[];
  channelId: string;
  principalId: string;
  requiredCapabilities: string[];
  runtimeProfileId: string;
  taskClass: WorkshopRoutingTaskClass;
  version: 1;
  workspace: string;
}

export interface WorkshopRoutingPolicyEntry {
  authorizedOptionIds: string[];
  backendOptionId: string | null;
  eligibleOptionIds: string[];
  fallback: "selected" | "fail_closed";
  revision: number;
  taskClass: WorkshopRoutingTaskClass;
}

export interface WorkshopRoutingPolicy {
  agentId: string;
  channelId: string;
  entries: WorkshopRoutingPolicyEntry[];
  principalId: string;
  runtimeProfileId: string;
  version: 1;
}

export interface WorkshopRunRoutingDecision {
  backend: string;
  decidedAt: string;
  disposition: "selected_default" | "routed" | "fallback_selected" | "rejected";
  evidenceVersion: number;
  model: string;
  policyRevision: number | null;
  provider: string | null;
  reasonCode: string;
  requestedBackendOptionId: string | null;
  requestedTaskClass: WorkshopRoutingTaskClass | null;
  selectedBackendOptionId: string | null;
}

export interface WorkshopWorkspaceConfig {
  capabilities: WorkshopEditableCapability[];
  environmentKeys: string[];
  hasPrompt: boolean;
  model: { defaultValue: string; source: string; value: string };
  mutation: WorkshopSettingsMutation | null;
  overrideFields: string[];
  prompt: string | null;
  promptSource: string | null;
  revision: string;
  timeoutSeconds: { defaultValue: number; source: string; value: number };
  workspace: string;
}

export interface WorkshopPreferenceDocument {
  content: string;
  editable: boolean;
  maxBytes: number;
  revision: string;
  sizeBytes: number;
  updatedAt: string | null;
}

export interface WorkshopPreferenceRevision {
  revision: string;
  sizeBytes: number;
  updatedAt: string;
}

export interface WorkshopPreferenceHistory {
  limit: number;
  revisions: WorkshopPreferenceRevision[];
}

export interface WorkshopGitHubSettings {
  githubLogin: string | null;
  issueTriage: { enabled: boolean; resettable: boolean; source: string };
  mutation: { changed: boolean; operation: string } | null;
  prReview: { enabled: boolean; resettable: boolean; source: string };
  repositories: {
    automationAuthorized: boolean;
    repository: string;
    source: string;
  }[];
  repositoriesResettable: boolean;
  revision: string;
  tokenStored: boolean;
}

export type WorkshopGitHubSettingsChange =
  | { field: "repository"; name: string; subscribed: boolean }
  | { field: "repository_reset" }
  | { field: "toggle"; name: "issue_triage" | "pr_review"; enabled: boolean | null }
  | { field: "token"; token: string | null };

export interface WorkshopNotificationPreferences {
  destinations: {
    choiceId: string;
    displayName: string;
    kind: "direct" | "notification";
    supportedClasses: ("generic" | "github")[];
  }[];
  mutation: { changed: boolean; operation: string } | null;
  preferences: {
    destinationChoiceId: string;
    destinationKind: "direct" | "notification";
    destinationName: string;
    displayName: string;
    editable: boolean;
    integrationClass: "generic" | "github";
    resettable: boolean;
    source: string;
  }[];
  revision: string;
}

export type WorkshopNotificationPreferenceChange =
  | {
      field: "destination";
      integrationClass: "generic" | "github";
      choiceId: string;
    }
  | {
      field: "reset";
      integrationClass: "generic" | "github";
    };

export interface WorkshopClientPreferences {
  mutation: { changed: boolean; operation: string } | null;
  revision: string;
  voiceOutput: {
    available: boolean;
    unavailableReason: string | null;
    modes: ("off" | "text_and_voice" | "voice_only")[];
    voices: { value: string; displayName: string }[];
    bindings: {
      choiceId: string;
      clientName: string;
      mode: "off" | "text_and_voice" | "voice_only";
      voice: string;
      voiceName: string;
      editable: boolean;
    }[];
  };
}

export interface WorkshopAppearancePreferences {
  mutation: { changed: boolean; operation: string } | null;
  revision: string;
  themeId: string;
  themes: {
    colorScheme: "dark" | "light";
    displayName: string;
    themeId: string;
  }[];
}

export type WorkshopClientPreferenceChange =
  | {
      field: "mode";
      bindingChoiceId: string;
      value: "off" | "text_and_voice" | "voice_only";
    }
  | {
      field: "voice";
      bindingChoiceId: string;
      value: string;
    };

export type WorkshopRuntimeSettingsChange =
  | { field: "backend"; value: string }
  | { field: "model"; value: string }
  | { field: "timeout"; value: number }
  | { field: "reset"; value: "all" | "model" | "timeout" };

export type WorkshopWorkspaceSettingChange =
  | { field: "model" | "prompt" | "timeout"; value: string }
  | { field: "reset"; value: "all" | "model" | "prompt" | "timeout" };

export interface WorkshopMemoryScope {
  exclusionReason: string | null;
  invalidDefaulted: boolean;
  legacyDefaulted: boolean;
  projectId: string | null;
  retrievable: boolean;
  scope: "global" | "project" | "task";
  scopeConfidence: number;
  scopeSource: string;
}

export interface WorkshopMemoryRecord {
  confidence: number;
  createdAt: string;
  kind: "fact" | "episode";
  memoryId: string;
  memoryType: string;
  preview: string;
  revision: string;
  scope: WorkshopMemoryScope;
  source: string;
  speaker: string;
  tags: string[];
  updatedAt: string;
}

export interface WorkshopMemoryEpisodeFields {
  actors: string[];
  approach: string;
  context: string;
  goal: string;
  lessons: string | null;
  outcome: string;
  outcomeQuality: "success" | "partial" | "failure";
  tags: string[];
}

export interface WorkshopMemoryPage {
  nextCursor: string | null;
  records: WorkshopMemoryRecord[];
}

export interface WorkshopMemoryFilters {
  kind?: "fact" | "episode";
  memoryType?: string;
  projectId?: string;
  scope?: "global" | "project" | "task";
  source?: string;
  tag?: string;
}

export interface WorkshopMemoryListOptions extends WorkshopMemoryFilters {
  cursor?: string;
  limit?: number;
  order?: "newest" | "oldest";
}

export interface WorkshopMemorySearchOptions extends WorkshopMemoryFilters {
  limit?: number;
}

export interface WorkshopMemoryDetail extends WorkshopMemoryRecord {
  compactRecall: string;
  confirmationQuote: string | null;
  content: string;
  episode: WorkshopMemoryEpisodeFields | null;
  promptVersion: string | null;
}

export interface WorkshopMemoryEditResult {
  changedFields: string[];
  idempotentReplay: boolean;
  record: WorkshopMemoryDetail;
}

export interface WorkshopMemoryCreationResult {
  created: boolean;
  record: WorkshopMemoryDetail;
}

export interface WorkshopMemorySourceMessage {
  authorDisplayName: string;
  authorKind: string;
  authorPrincipalId: string;
  body: string;
  channelId: string;
  createdAt: string;
  messageId: string;
}

export interface WorkshopMemorySourceContext {
  reason: string | null;
  result: WorkshopMemorySourceMessage | null;
  runId: string | null;
  source: WorkshopMemorySourceMessage | null;
  status: "available" | "unavailable";
}

export interface WorkshopMemoryStats {
  allowedProjects: WorkshopMemoryProjectOption[];
  byScope: Record<string, number>;
  bySource: Record<string, number>;
  byType: Record<string, number>;
  episodes: number;
  facts: number;
  total: number;
}

export interface WorkshopMemoryProjectOption {
  displayName: string;
  projectId: string;
}

export type WorkshopMemoryMutationOutcome =
  | "succeeded"
  | "not_found"
  | "stale"
  | "failed";

export interface WorkshopMemoryMutationResult {
  memoryId: string;
  newScope: WorkshopMemoryScope | null;
  outcome: WorkshopMemoryMutationOutcome;
  priorScope: WorkshopMemoryScope | null;
}

export interface WorkshopMemoryMutationBatch {
  operation: "move_scope" | "delete";
  results: WorkshopMemoryMutationResult[];
}

export interface WorkshopMemorySearchHit {
  adjustedScore: number;
  compactRecall: string;
  rawScore: number;
  record: WorkshopMemoryRecord;
}

export interface WorkshopMemorySearch {
  activeProjectId: string | null;
  hits: WorkshopMemorySearchHit[];
  reason: string;
}

export interface TimelineMessage {
  artifacts: WorkshopArtifactSummary[];
  authorDisplayName: string;
  authorKind: string;
  authorPrincipalId: string;
  body: string;
  channelId: string;
  createdAt: string;
  eventPosition: number;
  mentions: WorkshopMessageMention[];
  messageId: string;
  reactions: WorkshopMessageReaction[];
  replyCount: number;
  replyToMessageId: string | null;
  latestReplyAt: string | null;
  threadRootId: string | null;
}

export type WorkshopReaction =
  | "thumbs_up"
  | "heart"
  | "laugh"
  | "celebrate"
  | "eyes"
  | "check";

export interface WorkshopMessageReaction {
  count: number;
  reactedByViewer: boolean;
  reaction: WorkshopReaction;
}

export interface WorkshopMessageMention {
  kind: "human" | "agent";
  length: number;
  principalId: string;
  start: number;
}

export type WorkshopArtifactKind = "photo" | "document" | "voice";

export interface WorkshopArtifactSummary {
  artifactId: string;
  byteSize: number;
  contentSha256: string;
  createdAt: string;
  kind: WorkshopArtifactKind;
  mediaType: string;
  originalFilename: string | null;
}

export interface TimelineSnapshot {
  messages: TimelineMessage[];
  throughPosition: number;
  // Walks history older than this page; null when the page reaches the
  // start of the channel.
  previousCursor: string | null;
}

export interface ThreadTimelineSnapshot {
  messages: TimelineMessage[];
  nextCursor: string | null;
  root: TimelineMessage;
  throughPosition: number;
}

export interface CommandSubmissionResult {
  acceptance: string;
  messageId: string;
  run: WorkshopRun | null;
}

export type WorkshopRunStatus =
  | "accepted"
  | "started"
  | "completed"
  | "failed"
  | "cancelled";

export interface WorkshopRun {
  acceptedAt: string;
  cancellationRequestedAt: string | null;
  channelId: string;
  resultMessageId: string | null;
  runId: string;
  startedAt: string | null;
  status: WorkshopRunStatus;
  terminalAt: string | null;
  terminalCode: string | null;
  routingDecision?: WorkshopRunRoutingDecision | null;
}

export type WorkshopRunTransition =
  | "run.accepted"
  | "run.started"
  | "run.cancellation_requested"
  | "run.completed"
  | "run.failed"
  | "run.cancelled";

export interface WorkshopRunPreview {
  runId: string;
  sequence: number;
  text: string;
}

export interface WorkshopRunActivity {
  eventPosition: number;
  occurredAt: string;
  run: WorkshopRun;
  transition: WorkshopRunTransition;
}

export type ConnectionTone = "connected" | "connecting" | "disconnected";

export interface ConnectionState {
  label: string;
  tone: ConnectionTone;
}

export type WorkshopRunTraceKind = "tool_call" | "tool_result" | "truncated";

export interface WorkshopRunTraceEntry {
  createdAt: string;
  detail: string;
  isDiff: boolean;
  isError: boolean;
  kind: WorkshopRunTraceKind;
  seq: number;
  summary: string;
  toolName: string | null;
  toolUseId: string | null;
}

export interface WorkshopRunTracePage {
  entries: WorkshopRunTraceEntry[];
  hasMore: boolean;
}

export interface WorkshopRunTraceSignal {
  runId: string;
  seq: number;
}
