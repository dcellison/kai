export const CHANNEL_PATTERN = /^chn_[0-9a-f]{32}$/;

export interface WorkshopSession {
  channelId: string;
  token: string;
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

export type ConnectionTone = "connected" | "connecting" | "disconnected";

export interface ConnectionState {
  label: string;
  tone: ConnectionTone;
}
