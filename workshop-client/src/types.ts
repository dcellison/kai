export const CHANNEL_PATTERN = /^chn_[0-9a-f]{32}$/;
export const AGENT_PATTERN = /^agt_[0-9a-f]{32}$/;
export const PRINCIPAL_PATTERN = /^prn_[0-9a-f]{32}$/;
export const WORKSHOP_PATTERN = /^wsp_[0-9a-f]{32}$/;

export interface WorkshopSession {
  channelId: string;
  token: string;
}

export interface WorkshopAgentSummary {
  agentId: string;
  name: string;
}

export type WorkshopChannelKind = "direct" | "group" | "notification";

export interface WorkshopChannelSummary {
  agents: WorkshopAgentSummary[];
  canSubmitCommands: boolean;
  channelId: string;
  kind: WorkshopChannelKind;
  name: string | null;
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

export interface TimelineMessage {
  authorDisplayName: string;
  authorKind: string;
  body: string;
  channelId: string;
  createdAt: string;
  eventPosition: number;
  messageId: string;
}

export interface TimelineSnapshot {
  messages: TimelineMessage[];
  throughPosition: number;
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
