export const CHANNEL_PATTERN = /^chn_[0-9a-f]{32}$/;
export const AGENT_PATTERN = /^agt_[0-9a-f]{32}$/;
export const PRINCIPAL_PATTERN = /^prn_[0-9a-f]{32}$/;
export const WORKSHOP_PATTERN = /^wsp_[0-9a-f]{32}$/;
export const ARTIFACT_PATTERN = /^art_[0-9a-f]{32}$/;

export interface WorkshopSession {
  channelId: string;
  token: string;
}

export interface WorkshopAgentSummary {
  agentId: string;
  name: string;
}

export interface WorkshopParticipantSummary {
  displayName: string;
  kind: string;
  principalId: string;
}

export type WorkshopChannelKind = "direct" | "group" | "notification";

export interface WorkshopChannelSummary {
  agents: WorkshopAgentSummary[];
  canSubmitCommands: boolean;
  channelId: string;
  kind: WorkshopChannelKind;
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
    principalId: string;
  };
  workshops: WorkshopSummary[];
}

export interface WorkshopModelOption {
  displayName: string;
  modelId: string;
}

export interface WorkshopWorkspaceOption {
  current: boolean;
  home: boolean;
  name: string;
  path: string;
}

export interface WorkshopSettingsWorkspace {
  backend: string;
  channelId: string;
  model: { source: string; value: string };
  modelOptions: WorkshopModelOption[] | null;
  principalId: string;
  provider: string;
  runtimeProfileId: string;
  timeoutSeconds: { source: string; value: number };
  workspace: string;
  workspaces: WorkshopWorkspaceOption[];
}

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
  scope: WorkshopMemoryScope;
  source: string;
  speaker: string;
  tags: string[];
  updatedAt: string;
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
  episode: Record<string, string> | null;
  promptVersion: string | null;
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
  body: string;
  channelId: string;
  createdAt: string;
  eventPosition: number;
  messageId: string;
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

export interface CommandSubmissionResult {
  acceptance: string;
  messageId: string;
  run: WorkshopRun;
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
