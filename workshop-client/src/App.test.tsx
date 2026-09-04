import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  archiveChannel,
  archiveDirectMessage,
  advanceChannelReadPosition,
  advanceThreadReadPosition,
  attachChannelAgent,
  AuthenticationError,
  cancelRun,
  changeChannelMember,
  ChannelAccessError,
  createChannel,
  detachChannelAgent,
  dismissChannelAgent,
  enableAgentDefinition,
  loadAppearancePreferences,
  loadAgentDefinitions,
  loadAgentEnablements,
  loadEarlierTimeline,
  loadArtifactBlob,
  loadChannelMembers,
  loadChannelMessage,
  loadChannelUnread,
  loadHumanNotificationCounts,
  loadHumanNotifications,
  loadWorkshopHumans,
  loadFollowedThreads,
  loadNavigation,
  loadNotificationPreferences,
  loadMemoryDetail,
  loadMemoryRecords,
  loadMemorySource,
  loadMemoryStats,
  loadPreferenceDocument,
  loadPreferenceHistory,
  loadRun,
  loadRunTrace,
  loadSettingsWorkspace,
  loadTimeline,
  loadThreadTimeline,
  loadThreadUnread,
  loadWorkspaceConfig,
  redeemEnrollment,
  restoreChannel,
  restoreDirectMessage,
  markHumanNotificationRead,
  markHumanNotificationUnread,
  markHumanNotificationsRead,
  streamTimeline,
  streamPrincipalEvents,
  setMessageReaction,
  setThreadFollowed,
  startAgentConversation,
  startHumanConversation,
  submitCommand,
  switchWorkspace,
} from "./api";
import type {
  WorkshopMemoryRecord,
  WorkshopAgentDefinition,
  WorkshopAgentEnablement,
  TimelineMessage,
  TimelineSnapshot,
  WorkshopNavigation,
  WorkshopRun,
  WorkshopSettingsWorkspace,
  WorkshopHumanNotification,
  WorkshopChannelUnreadState,
  WorkshopThreadUnreadState,
  WorkshopFollowedThread,
} from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    attachChannelAgent: vi.fn(),
    archiveChannel: vi.fn(),
    archiveDirectMessage: vi.fn(),
    advanceChannelReadPosition: vi.fn(),
    advanceThreadReadPosition: vi.fn(),
    cancelRun: vi.fn(),
    changeChannelMember: vi.fn(),
    createChannel: vi.fn(),
    detachChannelAgent: vi.fn(),
    dismissChannelAgent: vi.fn(),
    enableAgentDefinition: vi.fn(),
    loadEarlierTimeline: vi.fn(),
    loadArtifactBlob: vi.fn(),
    loadChannelMembers: vi.fn(),
    loadChannelMessage: vi.fn(),
    loadChannelUnread: vi.fn(),
    loadHumanNotificationCounts: vi.fn(),
    loadHumanNotifications: vi.fn(),
    loadWorkshopHumans: vi.fn(),
    loadFollowedThreads: vi.fn(),
    loadAppearancePreferences: vi.fn(),
    loadAgentDefinitions: vi.fn(),
    loadAgentEnablements: vi.fn(),
    loadNavigation: vi.fn(),
    loadNotificationPreferences: vi.fn(),
    loadMemoryDetail: vi.fn(),
    loadMemoryRecords: vi.fn(),
    loadMemorySource: vi.fn(),
    loadMemoryStats: vi.fn(),
    loadPreferenceDocument: vi.fn(),
    loadPreferenceHistory: vi.fn(),
    loadTimeline: vi.fn(),
    loadThreadTimeline: vi.fn(),
    loadThreadUnread: vi.fn(),
    loadRun: vi.fn(),
    loadRunTrace: vi.fn(),
    loadSettingsWorkspace: vi.fn(),
    loadWorkspaceConfig: vi.fn(),
    redeemEnrollment: vi.fn(),
    restoreChannel: vi.fn(),
    restoreDirectMessage: vi.fn(),
    markHumanNotificationRead: vi.fn(),
    markHumanNotificationUnread: vi.fn(),
    markHumanNotificationsRead: vi.fn(),
    streamTimeline: vi.fn(),
    streamPrincipalEvents: vi.fn(),
    setMessageReaction: vi.fn(),
    setThreadFollowed: vi.fn(),
    startAgentConversation: vi.fn(),
    startHumanConversation: vi.fn(),
    submitCommand: vi.fn(),
    switchWorkspace: vi.fn(),
  };
});

const channelId = "chn_d3dfdfd7df9151ba8a1742b92403faa5";
const notificationChannelId = "chn_11111111111111111111111111111111";
const secondChannelId = "chn_22222222222222222222222222222222";
const humanDirectChannelId = "chn_33333333333333333333333333333333";
const definitionId = "adf_00000000000000000000000000000001";
const revisionId = "adr_00000000000000000000000000000001";
const runtimeProfileId = "rtp_00000000000000000000000000000001";
const navigation: WorkshopNavigation = {
  principal: {
    displayName: "Daniel",
    handle: "daniel",
    principalId: "prn_00000000000000000000000000000001",
  },
  workshops: [
    {
      channels: [
        {
          agents: [
            {
              agentId: "agt_00000000000000000000000000000001",
              available: true,
              engaged: false,
              engagedUntil: null,
              handle: "kai",
              lifecycleState: "active",
              memoryScope: "private",
              name: "Kai",
              principalId: "prn_00000000000000000000000000000002",
              runtimeProfileId,
              sponsorDisplayName: "Daniel",
              sponsorPrincipalId: "prn_00000000000000000000000000000001",
            },
          ],
          canSubmitCommands: true,
          channelId,
          kind: "direct",
          name: "Conversation",
          participants: [
            {
              displayName: "Kai",
              handle: "kai",
              kind: "agent",
              principalId: "prn_00000000000000000000000000000002",
            },
          ],
          role: "owner",
        },
        {
          agents: [
            {
              agentId: "agt_00000000000000000000000000000001",
              available: true,
              engaged: false,
              engagedUntil: null,
              handle: "kai",
              lifecycleState: "active",
              memoryScope: "private",
              name: "Kai",
              principalId: "prn_00000000000000000000000000000002",
              runtimeProfileId,
              sponsorDisplayName: "Daniel",
              sponsorPrincipalId: "prn_00000000000000000000000000000001",
            },
          ],
          canSubmitCommands: false,
          channelId: notificationChannelId,
          kind: "notification",
          name: "GitHub notifications",
          participants: [
            {
              displayName: "Kai",
              handle: "kai",
              kind: "agent",
              principalId: "prn_00000000000000000000000000000002",
            },
          ],
          role: "participant",
        },
        {
          agents: [],
          canSubmitCommands: true,
          channelId: humanDirectChannelId,
          kind: "direct",
          name: "Direct",
          participants: [
            {
              displayName: "Scott",
              handle: "scott",
              kind: "human",
              principalId: "prn_00000000000000000000000000000003",
            },
          ],
          role: "owner",
        },
      ],
      name: "Kai Workshop",
      role: "admin",
      workshopId: "wsp_00000000000000000000000000000001",
    },
  ],
};

function navigationWithGroup({
  engaged = false,
  name = "Wake policy qualification",
  role = "participant",
}: {
  engaged?: boolean;
  name?: string;
  role?: string;
} = {}): WorkshopNavigation {
  const direct = navigation.workshops[0].channels[0];
  return {
    ...navigation,
    workshops: [
      {
        ...navigation.workshops[0],
        channels: [
          ...navigation.workshops[0].channels,
          {
            ...direct,
            agents: direct.agents.map((agent) => ({
              ...agent,
              engaged,
              engagedUntil: engaged ? "2099-08-28T12:00:00Z" : null,
              memoryScope: "shared_channel",
            })),
            channelId: secondChannelId,
            kind: "group",
            name,
            role,
          },
        ],
      },
    ],
  };
}
const historyMessage: TimelineMessage = {
  artifacts: [],
  authorDisplayName: "Kai",
  authorKind: "agent",
  authorPrincipalId: "prn_00000000000000000000000000000002",
  body: "Canonical history is ready.",
  channelId,
  createdAt: "2026-08-13T09:00:00Z",
  eventPosition: 25,
  mentions: [],
  messageId: "msg_00000000000000000000000000000025",
  reactions: [],
  replyCount: 0,
  replyParticipantCount: 0,
  replyParticipants: [],
  replyToMessageId: null,
  latestReplyAt: null,
  threadRootId: null,
};

function unreadState(
  channel: string,
  overrides: Partial<WorkshopChannelUnreadState> = {},
): WorkshopChannelUnreadState {
  return {
    archived: false,
    channelId: channel,
    channelKind: "group",
    channelName: "Wake policy qualification",
    firstUnreadEventPosition: 25,
    firstUnreadMessageId: historyMessage.messageId,
    lastEventPosition: 25,
    membershipBaselineEventPosition: 20,
    readThroughEventPosition: 20,
    readThroughMessageId: null,
    stateVersion: 0,
    unreadCount: 1,
    unreadCountCapped: false,
    unreadReplyCount: 0,
    unreadReplyCountCapped: false,
    unreadThreadCount: 0,
    firstUnreadThreadRootId: null,
    firstUnreadThreadReplyId: null,
    firstUnreadThreadEventPosition: null,
    ...overrides,
  };
}

function threadUnreadState(
  rootMessageId: string,
  overrides: Partial<WorkshopThreadUnreadState> = {},
): WorkshopThreadUnreadState {
  return {
    channelId: secondChannelId,
    firstUnreadEventPosition: null,
    firstUnreadMessageId: null,
    followBaselineEventPosition: 25,
    followed: true,
    lastEventPosition: 25,
    readThroughEventPosition: 25,
    readThroughMessageId: null,
    stateVersion: 0,
    threadRootId: rootMessageId,
    unreadCount: 0,
    unreadCountCapped: false,
    ...overrides,
  };
}

const followedThread: WorkshopFollowedThread = {
  state: threadUnreadState(historyMessage.messageId, {
    channelId: secondChannelId,
    firstUnreadEventPosition: 32,
    firstUnreadMessageId: "msg_00000000000000000000000000000032",
    lastEventPosition: 32,
    readThroughEventPosition: 31,
    stateVersion: 2,
    unreadCount: 1,
  }),
  channelName: "Wake policy qualification",
  channelArchived: false,
  rootAuthorDisplayName: "Daniel",
  rootExcerpt: "Please review the qualification output.",
  rootCreatedAt: "2026-09-01T12:00:00Z",
  latestReplyMessageId: "msg_00000000000000000000000000000032",
  latestReplyAuthorDisplayName: "Scott",
  latestReplyExcerpt: "The qualification passed.",
  latestReplyCreatedAt: "2026-09-01T12:05:00Z",
};

const agentDefinition: WorkshopAgentDefinition = {
  activeRevisionId: revisionId,
  agentId: "agt_00000000000000000000000000000001",
  createdAt: "2026-08-29T10:00:00Z",
  createdByPrincipalId: "prn_00000000000000000000000000000001",
  definitionId,
  description: "Kai's base agent definition.",
  displayName: "Kai",
  handle: "kai",
  lifecycleState: "active",
  ownerDisplayName: "Daniel",
  ownerPrincipalId: "prn_00000000000000000000000000000001",
  presentation: { avatar: "K" },
  revisions: [
    {
      capabilities: ["text_generation", "tool_activity"],
      createdAt: "2026-08-29T10:00:00Z",
      createdByPrincipalId: "prn_00000000000000000000000000000001",
      eventPosition: 10,
      instructions: "Be useful.",
      purpose: "General assistance",
      revisionId,
      revisionNumber: 1,
    },
  ],
  stateVersion: 2,
};

const agentEnablement: WorkshopAgentEnablement = {
  agentId: agentDefinition.agentId,
  definitionId,
  directChannelId: channelId,
  displayName: "Kai",
  eligibleRuntimes: [
    {
      backendOptions: ["claude:anthropic"],
      displayName: "Daniel's runtime",
      runtimeProfileId,
    },
  ],
  enablementId: "aen_00000000000000000000000000000001",
  handle: "kai",
  lifecycleState: "enabled",
  runtimeProfileId,
  stateVersion: 3,
  canManage: true,
  conversationStarted: false,
  ownerPrincipalId: "prn_00000000000000000000000000000001",
  ownerRuntimeProfileId: runtimeProfileId,
};
const memoryRecord: WorkshopMemoryRecord = {
  confidence: 1,
  createdAt: "2026-08-24T10:00:00Z",
  kind: "fact",
  memoryId: "memory-1",
  memoryType: "fact",
  preview: "Workshop memory navigation works.",
  revision: "mr1_test",
  scope: {
    exclusionReason: null,
    invalidDefaulted: false,
    legacyDefaulted: false,
    projectId: null,
    retrievable: true,
    scope: "global",
    scopeConfidence: 1,
    scopeSource: "operator",
  },
  source: "extracted",
  speaker: "user",
  tags: [],
  updatedAt: "2026-08-24T10:00:00Z",
};

const completedRun: WorkshopRun = {
  acceptedAt: "2026-08-13T09:00:00Z",
  cancellationRequestedAt: null,
  channelId,
  resultMessageId: "msg_00000000000000000000000000000031",
  runId: "run_00000000000000000000000000000030",
  startedAt: "2026-08-13T09:00:01Z",
  status: "completed",
  terminalAt: "2026-08-13T09:00:02Z",
  terminalCode: null,
};
const settingsWorkspace: WorkshopSettingsWorkspace = {
  backend: "codex",
  backendOptionId: "codex:openai",
  backendOptions: [
    { optionId: "claude:anthropic", backend: "claude", provider: "anthropic", current: false },
    { optionId: "codex:openai", backend: "codex", provider: "openai", current: true },
  ],
  capabilities: [
    {
      choices: ["claude", "codex"],
      field: "backend",
      maximum: null,
      minimum: null,
      resettable: false,
      scope: "runtime",
      valueType: "backend_id",
    },
    {
      choices: ["gpt-5.6-sol"],
      field: "model",
      maximum: null,
      minimum: null,
      resettable: true,
      scope: "runtime",
      valueType: "model_id",
    },
  ],
  channelId,
  model: {
    defaultValue: "gpt-5.6-sol",
    source: "runtime policy",
    value: "gpt-5.6-sol",
  },
  modelCatalogue: {
    errorCode: null,
    errorDetail: null,
    lastAttemptAt: "2026-08-28T10:00:00Z",
    lastKnownGood: false,
    lastSuccessfulRefreshAt: "2026-08-28T10:00:00Z",
    stale: false,
    status: "succeeded",
  },
  modelOptions: [
    {
      displayName: "GPT-5.6 Sol",
      modelId: "gpt-5.6-sol",
      retained: true,
      selectable: true,
      sources: ["curated"],
      status: "available",
    },
  ],
  mutation: null,
  principalId: "prn_00000000000000000000000000000001",
  provider: "openai",
  revision: "sws_current",
  runtimeProfileId: "rtp_00000000000000000000000000000001",
  timeoutSeconds: { defaultValue: 120, source: "runtime policy", value: 120 },
  workspace: "/Users/kai/Projects/kai",
  workspaces: [
    {
      current: true,
      home: false,
      name: "kai",
      path: "/Users/kai/Projects/kai",
    },
    {
      current: false,
      home: true,
      name: "Home",
      path: "/var/lib/kai/home/principal",
    },
  ],
};

type StreamHandlers = Parameters<typeof streamTimeline>[2];
type PrincipalEventStreamHandlers = Parameters<typeof streamPrincipalEvents>[2];

function observeMessagesAsVisible(): void {
  class TestIntersectionObserver {
    readonly root = null;
    readonly rootMargin = "-15% 0px -35% 0px";
    readonly thresholds = [0];

    constructor(private readonly callback: IntersectionObserverCallback) {}

    disconnect(): void {}

    observe(target: Element): void {
      const bounds = target.getBoundingClientRect();
      this.callback([{
        boundingClientRect: bounds,
        intersectionRatio: 1,
        intersectionRect: bounds,
        isIntersecting: true,
        rootBounds: null,
        target,
        time: 0,
      }], this as unknown as IntersectionObserver);
    }

    takeRecords(): IntersectionObserverEntry[] {
      return [];
    }

    unobserve(): void {}
  }

  vi.stubGlobal("IntersectionObserver", TestIntersectionObserver);
}

function observeMessagesAtViewportEdge(): void {
  class TestIntersectionObserver {
    readonly root = null;
    readonly rootMargin: string;
    readonly thresholds: number[];

    constructor(
      private readonly callback: IntersectionObserverCallback,
      options?: IntersectionObserverInit,
    ) {
      this.rootMargin = options?.rootMargin ?? "0px";
      const threshold = options?.threshold ?? 0;
      this.thresholds = Array.isArray(threshold) ? threshold : [threshold];
    }

    disconnect(): void {}

    observe(target: Element): void {
      const bounds = target.getBoundingClientRect();
      const isIntersecting = this.rootMargin === "0px";
      this.callback([{
        boundingClientRect: bounds,
        intersectionRatio: isIntersecting ? 1 : 0,
        intersectionRect: bounds,
        isIntersecting,
        rootBounds: null,
        target,
        time: 0,
      }], this as unknown as IntersectionObserver);
    }

    takeRecords(): IntersectionObserverEntry[] {
      return [];
    }

    unobserve(): void {}
  }

  vi.stubGlobal("IntersectionObserver", TestIntersectionObserver);
}

describe("Workshop React client", () => {
  let handlers: StreamHandlers | null;
  let principalEventHandlers: PrincipalEventStreamHandlers | null;
  let failStream: ((reason: Error) => void) | null;

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    sessionStorage.clear();
    window.history.replaceState(null, "", "/workshop/");
    handlers = null;
    principalEventHandlers = null;
    failStream = null;
    vi.mocked(redeemEnrollment).mockResolvedValue("redeemed-session-token");
    vi.mocked(attachChannelAgent).mockResolvedValue(undefined);
    vi.mocked(archiveChannel).mockResolvedValue(undefined);
    vi.mocked(archiveDirectMessage).mockResolvedValue(undefined);
    vi.mocked(createChannel).mockResolvedValue(secondChannelId);
    vi.mocked(detachChannelAgent).mockResolvedValue(undefined);
    vi.mocked(dismissChannelAgent).mockResolvedValue(undefined);
    vi.mocked(restoreChannel).mockResolvedValue(undefined);
    vi.mocked(restoreDirectMessage).mockResolvedValue(undefined);
    vi.mocked(setMessageReaction).mockResolvedValue([]);
    vi.mocked(loadNavigation).mockResolvedValue(navigation);
    vi.mocked(loadAppearancePreferences).mockResolvedValue({
      mutation: null,
      revision: "apr_current",
      themeId: "atom-one-dark",
      themes: [
        {
          colorScheme: "dark",
          displayName: "Atom One Dark",
          themeId: "atom-one-dark",
        },
      ],
    });
    vi.mocked(loadAgentDefinitions).mockResolvedValue([agentDefinition]);
    vi.mocked(loadAgentEnablements).mockResolvedValue([agentEnablement]);
    vi.mocked(enableAgentDefinition).mockResolvedValue(agentEnablement);
    vi.mocked(startAgentConversation).mockResolvedValue({
      ...agentEnablement,
      conversationStarted: true,
      stateVersion: 4,
    });
    vi.mocked(loadWorkshopHumans).mockResolvedValue([
      {
        conversationChannelId: humanDirectChannelId,
        displayName: "Scott",
        handle: "scott",
        principalId: "prn_00000000000000000000000000000003",
      },
    ]);
    vi.mocked(startHumanConversation).mockResolvedValue({
      channelId: humanDirectChannelId,
      created: false,
      peer: {
        conversationChannelId: humanDirectChannelId,
        displayName: "Scott",
        handle: "scott",
        principalId: "prn_00000000000000000000000000000003",
      },
      workshopId: "wsp_00000000000000000000000000000001",
    });
    vi.mocked(loadNotificationPreferences).mockResolvedValue({
      destinations: [
        {
          choiceId: "ndst_notifications",
          displayName: "GitHub notifications",
          kind: "notification",
          supportedClasses: ["github"],
        },
      ],
      mutation: null,
      preferences: [
        {
          destinationChoiceId: "ndst_notifications",
          destinationKind: "notification",
          destinationName: "GitHub notifications",
          displayName: "GitHub",
          editable: true,
          integrationClass: "github",
          resettable: false,
          source: "protected policy",
        },
      ],
      revision: "nps_current",
    });
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [historyMessage],
      throughPosition: 25,
      previousCursor: null,
    });
    vi.mocked(loadChannelMessage).mockResolvedValue(historyMessage);
    vi.mocked(loadHumanNotificationCounts).mockResolvedValue({
      read: 0,
      total: 0,
      unread: 0,
      unreadByChannel: {},
    });
    vi.mocked(loadHumanNotifications).mockResolvedValue({
      counts: { read: 0, total: 0, unread: 0, unreadByChannel: {} },
      nextCursor: null,
      notifications: [],
      throughPosition: 25,
    });
    vi.mocked(loadFollowedThreads).mockResolvedValue({
      threads: [],
      throughPosition: 25,
    });
    vi.mocked(loadChannelUnread).mockResolvedValue({
      channels: [],
      throughPosition: 25,
      totalUnread: 0,
      totalUnreadCapped: false,
    });
    vi.mocked(loadThreadUnread).mockImplementation(async (_session, rootMessageId) => (
      threadUnreadState(rootMessageId)
    ));
    vi.mocked(advanceThreadReadPosition).mockImplementation(
      async (_session, rootMessageId) => ({
        replayed: false,
        state: threadUnreadState(rootMessageId),
      }),
    );
    vi.mocked(setThreadFollowed).mockImplementation(
      async (_session, rootMessageId, followed) => ({
        replayed: false,
        state: threadUnreadState(rootMessageId, { followed }),
      }),
    );
    vi.mocked(markHumanNotificationsRead).mockResolvedValue([]);
    vi.mocked(loadRun).mockResolvedValue(completedRun);
    vi.mocked(loadRunTrace).mockResolvedValue({ entries: [], hasMore: false });
    vi.mocked(loadSettingsWorkspace).mockResolvedValue(settingsWorkspace);
    vi.mocked(loadPreferenceDocument).mockResolvedValue({
      content: "# Preferences\n\nBe concise.\n",
      editable: true,
      maxBytes: 65536,
      revision: "pref_current",
      sizeBytes: 27,
      updatedAt: "2026-08-26T10:00:00Z",
    });
    vi.mocked(loadPreferenceHistory).mockResolvedValue({
      limit: 20,
      revisions: [],
    });
    vi.mocked(loadWorkspaceConfig).mockResolvedValue({
      capabilities: [],
      environmentKeys: ["PROTECTED_KEY"],
      hasPrompt: false,
      model: settingsWorkspace.model,
      mutation: null,
      overrideFields: [],
      prompt: null,
      promptSource: null,
      revision: "sws_workspace",
      timeoutSeconds: settingsWorkspace.timeoutSeconds,
      workspace: settingsWorkspace.workspace,
    });
    vi.mocked(loadMemoryStats).mockResolvedValue({
      allowedProjects: [],
      byScope: { global: 1 },
      bySource: { extracted: 1 },
      byType: { fact: 1 },
      episodes: 0,
      facts: 1,
      total: 1,
    });
    vi.mocked(loadMemoryRecords).mockResolvedValue({
      nextCursor: null,
      records: [memoryRecord],
    });
    vi.mocked(loadMemoryDetail).mockImplementation(async (_token, memoryId) => ({
      ...memoryRecord,
      memoryId,
      compactRecall: "{\"record_type\":\"memory\"}",
      confirmationQuote: null,
      content: "Workshop memory navigation works.",
      episode: null,
      promptVersion: "v1",
    }));
    vi.mocked(loadMemorySource).mockResolvedValue({
      reason: "legacy_source",
      result: null,
      runId: null,
      source: null,
      status: "unavailable",
    });
    vi.mocked(switchWorkspace).mockResolvedValue({
      ...settingsWorkspace,
      workspace: "/var/lib/kai/home/principal",
      workspaces: settingsWorkspace.workspaces.map((workspace) => ({
        ...workspace,
        current: workspace.home,
      })),
    });
    vi.mocked(loadArtifactBlob).mockResolvedValue(
      new Blob(["artifact"], { type: "text/plain" }),
    );
    vi.mocked(cancelRun).mockResolvedValue({
      ...completedRun,
      status: "cancelled",
      terminalCode: "requested_by_human",
    });
    vi.mocked(loadChannelMembers).mockResolvedValue({
      archived: false,
      canManage: true,
      channelId: secondChannelId,
      eligibleHumans: [
        {
          displayName: "Scott",
          handle: "scott",
          principalId: "prn_00000000000000000000000000000003",
          role: null,
        },
      ],
      members: [
        {
          displayName: "Daniel",
          handle: "daniel",
          principalId: "prn_00000000000000000000000000000001",
          role: "owner",
        },
      ],
      stateVersion: 0,
      workshopId: "wsp_00000000000000000000000000000001",
    });
    vi.mocked(changeChannelMember).mockResolvedValue(201);
    vi.mocked(submitCommand).mockResolvedValue({
      acceptance: "newly_accepted",
      messageId: "msg_00000000000000000000000000000030",
      run: completedRun,
    });
    vi.mocked(streamTimeline).mockImplementation(
      async (_session, _position, streamHandlers, signal) => {
        handlers = streamHandlers;
        streamHandlers.onConnected();
        await new Promise<void>((resolve, reject) => {
          failStream = reject;
          signal.addEventListener("abort", () => resolve(), { once: true });
        });
      },
    );
    vi.mocked(streamPrincipalEvents).mockImplementation(
      async (_token, _position, streamHandlers, signal) => {
        principalEventHandlers = streamHandlers;
        streamHandlers.onConnected();
        await new Promise<void>((resolve) => {
          signal.addEventListener("abort", () => resolve(), { once: true });
        });
      },
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "visible",
    });
  });

  it("enrolls, renders canonical history safely, and appends live messages", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    expect(screen.getByRole("heading", { name: "People and agents, working in the same room." })).toBeVisible();
    await user.type(screen.getByLabelText("Enrollment token"), "one-time-token");
    await user.click(screen.getByRole("button", { name: "Open Workshop" }));

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(screen.queryByLabelText("Task route")).toBeNull();
    expect(screen.queryByLabelText("Workshop switcher")).toBeNull();
    expect(screen.queryByText("Canonical Workshop command")).toBeNull();
    const channelContext = screen.getByLabelText("Channel context");
    expect(within(channelContext).queryByText("Canonical identity")).toBeNull();
    expect(within(channelContext).getByText(
      "You can read and send messages in this channel. Mention an agent to direct a request to it.",
    )).toBeVisible();
    expect(
      Array.from(channelContext.querySelectorAll(".section-number"), (section) =>
        section.textContent,
      ),
    ).toEqual(["01", "02", "03", "04"]);
    const navigationPanel = screen.getByLabelText("Workshop navigation");
    expect(navigationPanel).toBeVisible();
    expect(navigationPanel.querySelector(".sidebar-title")).toHaveTextContent(
      /^Kai Workshop$/,
    );
    expect(
      navigationPanel.querySelector(".sidebar-title .overline"),
    ).toBeVisible();
    expect(navigationPanel.querySelector(".sidebar-header")).not.toHaveTextContent(
      "admin",
    );
    expect(screen.getByText("Workshop administrator")).toBeVisible();
    expect((await screen.findAllByText("Live")).length).toBeGreaterThanOrEqual(1);
    expect(redeemEnrollment).toHaveBeenCalledWith(
      "one-time-token",
      "Workshop browser",
    );
    expect(loadNavigation).toHaveBeenCalledWith("redeemed-session-token");
    expect(localStorage.getItem("kai.workshop.client-credential.v1")).toContain(
      "redeemed-session-token",
    );
    expect(sessionStorage.getItem("kai.workshop.active-channel.v1")).toContain(
      channelId,
    );
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toBeNull();

    const liveMessage: TimelineMessage = {
      ...historyMessage,
      authorDisplayName: "Daniel",
      authorKind: "human",
      body: '<img src=x onerror="alert(1)">',
      eventPosition: 30,
      messageId: "msg_00000000000000000000000000000030",
    };
    act(() => handlers?.onMessage(liveMessage, "30"));

    expect(await screen.findByText(liveMessage.body)).toBeVisible();
    expect(container.querySelector("img")).toBeNull();
  });

  it("opens a new tab from the browser-scoped credential", async () => {
    localStorage.setItem(
      "kai.workshop.client-credential.v1",
      JSON.stringify({ token: "browser-session" }),
    );

    render(<App />);

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(loadNavigation).toHaveBeenCalledWith("browser-session");
    expect(loadTimeline).toHaveBeenCalledWith(
      { channelId, token: "browser-session" },
      expect.anything(),
    );
    expect(sessionStorage.getItem("kai.workshop.active-channel.v1")).toContain(
      channelId,
    );
    expect(redeemEnrollment).not.toHaveBeenCalled();
  });

  it("migrates the legacy tab credential without re-enrollment", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "legacy-session" }),
    );

    render(<App />);

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(localStorage.getItem("kai.workshop.client-credential.v1")).toContain(
      "legacy-session",
    );
    expect(sessionStorage.getItem("kai.workshop.active-channel.v1")).toContain(
      channelId,
    );
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toBeNull();
    expect(redeemEnrollment).not.toHaveBeenCalled();
  });

  it("forgets the browser credential across open tabs", async () => {
    localStorage.setItem(
      "kai.workshop.client-credential.v1",
      JSON.stringify({ token: "browser-session" }),
    );
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    localStorage.removeItem("kai.workshop.client-credential.v1");
    act(() =>
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "kai.workshop.client-credential.v1",
          newValue: null,
          oldValue: JSON.stringify({ token: "browser-session" }),
          storageArea: localStorage,
        }),
      ),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Workshop session was forgotten in another tab.",
    );
    expect(screen.getByLabelText("Enrollment token")).toBeVisible();
    expect(sessionStorage.getItem("kai.workshop.active-channel.v1")).toBeNull();
  });

  it("restores a credential-free Memory deep link and returns to conversations", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    window.history.replaceState(
      null,
      "",
      "/workshop/?view=memory&memory=memory-1",
    );

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Memory", level: 1 })).toBeVisible();
    expect(
      (await screen.findAllByText("Workshop memory navigation works.")).length,
    ).toBeGreaterThanOrEqual(2);
    expect(loadMemoryDetail).toHaveBeenCalledWith("session-secret", "memory-1");
    expect(window.location.search).toBe("?view=memory&memory=memory-1");
    expect(window.location.href).not.toContain("session-secret");

    await user.click(screen.getByRole("button", { name: "Back to conversation" }));
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(window.location.search).toBe("");

    await user.click(screen.getByRole("button", { name: "Memory" }));
    expect(await screen.findByRole("heading", { name: "Memory", level: 1 })).toBeVisible();
    expect(window.location.search).toContain("view=memory");
  });

  it("shows principal-private mentions and opens the exact source thread", async () => {
    const user = userEvent.setup();
    observeMessagesAsVisible();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    const sourceRoot: TimelineMessage = {
      ...historyMessage,
      body: "Thread root",
      channelId: secondChannelId,
      messageId: "msg_00000000000000000000000000000040",
    };
    const sourceReply: TimelineMessage = {
      ...historyMessage,
      body: "@daniel please review this reply",
      channelId: secondChannelId,
      eventPosition: 41,
      messageId: "msg_00000000000000000000000000000041",
      replyToMessageId: sourceRoot.messageId,
      threadRootId: sourceRoot.messageId,
    };
    const mention: WorkshopHumanNotification = {
      channelName: "Wake policy qualification",
      createdAt: "2026-08-31T15:00:00Z",
      createdEventPosition: 41,
      kind: "mention",
      lastEventPosition: 42,
      notificationId: "ntf_00000000000000000000000000000001",
      read: false,
      readAt: null,
      sourceAuthorDisplayName: "Scott",
      sourceAuthorPrincipalId: "prn_00000000000000000000000000000003",
      sourceChannelId: secondChannelId,
      sourceMessageId: sourceReply.messageId,
      sourceThreadRootId: sourceRoot.messageId,
      stateVersion: 0,
    };
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadHumanNotifications).mockResolvedValue({
      counts: {
        read: 0,
        total: 1,
        unread: 1,
        unreadByChannel: { [secondChannelId]: 1 },
      },
      nextCursor: null,
      notifications: [mention],
      throughPosition: 42,
    });
    vi.mocked(loadHumanNotificationCounts).mockResolvedValue({
      read: 1,
      total: 1,
      unread: 0,
      unreadByChannel: {},
    });
    vi.mocked(markHumanNotificationRead).mockResolvedValue({
      changed: true,
      notification: { ...mention, read: true, readAt: "2026-08-31T15:01:00Z", stateVersion: 1 },
      replayed: false,
    });
    vi.mocked(loadChannelMessage).mockImplementation(async (_session, messageId) => {
      if (messageId === sourceReply.messageId) return sourceReply;
      if (messageId === sourceRoot.messageId) return sourceRoot;
      throw new Error("Source message not found");
    });
    vi.mocked(loadTimeline).mockImplementation(async (activeSession) => ({
      messages: activeSession.channelId === secondChannelId ? [] : [historyMessage],
      previousCursor: null,
      throughPosition: 41,
    }));
    vi.mocked(loadThreadTimeline).mockResolvedValue({
      messages: [],
      nextCursor: null,
      root: sourceRoot,
      throughPosition: 41,
    });

    render(<App />);

    const mentionsButton = await screen.findByRole("button", {
      name: "Mentions, 1 unread",
    });
    expect(mentionsButton).toBeVisible();
    expect(screen.getByRole("button", { name: /Wake policy qualification/ })).toHaveTextContent("1");
    await user.click(mentionsButton);
    expect(await screen.findByRole("heading", { name: "Mentions" })).toBeVisible();
    expect(within(mentionsButton).getByLabelText("Open")).toHaveClass("live-pip");
    const sourceButton = screen.getByRole("button", { name: /Scott mentioned you/ });
    expect(sourceButton).toBeVisible();
    await user.click(sourceButton);

    await waitFor(() => expect(loadChannelMessage).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "session-secret" },
      sourceReply.messageId,
      expect.any(AbortSignal),
    ));
    await waitFor(() => expect(loadChannelMessage).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "session-secret" },
      sourceRoot.messageId,
      expect.any(AbortSignal),
    ));
    expect(markHumanNotificationRead).toHaveBeenCalledWith(
      "session-secret",
      mention.notificationId,
      0,
      expect.stringMatching(/^mention-viewed-/),
    );
    expect(window.location.search).toContain(`message=${sourceReply.messageId}`);
    expect(window.location.search).toContain(`thread=${sourceRoot.messageId}`);
    const context = await screen.findByLabelText("Channel context");
    const focused = await within(context).findByText(sourceReply.body);
    expect(focused.closest("li")).toHaveClass("focused-message");
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Mentions" })).toBeVisible();
      expect(screen.getByRole("button", { name: "Wake policy qualification" })).not.toHaveTextContent("1");
    });
  });

  it("marks an unread mention read when its source message enters the active timeline", async () => {
    observeMessagesAsVisible();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    const mention: WorkshopHumanNotification = {
      channelName: "Conversation",
      createdAt: "2026-08-31T15:00:00Z",
      createdEventPosition: historyMessage.eventPosition,
      kind: "mention",
      lastEventPosition: historyMessage.eventPosition,
      notificationId: "ntf_00000000000000000000000000000002",
      read: false,
      readAt: null,
      sourceAuthorDisplayName: "Kai",
      sourceAuthorPrincipalId: historyMessage.authorPrincipalId,
      sourceChannelId: channelId,
      sourceMessageId: historyMessage.messageId,
      sourceThreadRootId: null,
      stateVersion: 0,
    };
    vi.mocked(loadHumanNotifications).mockResolvedValue({
      counts: {
        read: 0,
        total: 1,
        unread: 1,
        unreadByChannel: { [channelId]: 1 },
      },
      nextCursor: null,
      notifications: [mention],
      throughPosition: historyMessage.eventPosition,
    });
    vi.mocked(loadHumanNotificationCounts).mockResolvedValue({
      read: 1,
      total: 1,
      unread: 0,
      unreadByChannel: {},
    });
    vi.mocked(markHumanNotificationRead).mockResolvedValue({
      changed: true,
      notification: {
        ...mention,
        read: true,
        readAt: "2026-08-31T15:01:00Z",
        stateVersion: 1,
      },
      replayed: false,
    });

    render(<App />);

    expect(await screen.findByText(historyMessage.body)).toBeVisible();
    await waitFor(() => expect(markHumanNotificationRead).toHaveBeenCalledWith(
      "session-secret",
      mention.notificationId,
      0,
      expect.stringMatching(/^mention-viewed-/),
    ));
    await waitFor(() => expect(
      screen.getByRole("button", { name: "Mentions" }),
    ).toBeVisible());
  });

  it("lists followed threads and opens one at its unread reply", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    const root: TimelineMessage = {
      ...historyMessage,
      body: followedThread.rootExcerpt,
      channelId: secondChannelId,
      messageId: followedThread.state.threadRootId,
      replyCount: 1,
    };
    const reply: TimelineMessage = {
      ...historyMessage,
      authorDisplayName: "Scott",
      authorPrincipalId: "prn_00000000000000000000000000000003",
      body: followedThread.latestReplyExcerpt ?? "",
      channelId: secondChannelId,
      eventPosition: 32,
      messageId: followedThread.latestReplyMessageId ?? "",
      replyToMessageId: root.messageId,
      threadRootId: root.messageId,
    };
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadFollowedThreads).mockResolvedValue({
      threads: [followedThread],
      throughPosition: 32,
    });
    vi.mocked(loadTimeline).mockImplementation(async (activeSession) => ({
      messages: activeSession.channelId === secondChannelId ? [] : [historyMessage],
      previousCursor: null,
      throughPosition: 32,
    }));
    vi.mocked(loadChannelMessage).mockImplementation(async (_session, messageId) => {
      if (messageId === root.messageId) return root;
      if (messageId === reply.messageId) return reply;
      throw new Error("Message not found");
    });
    vi.mocked(loadThreadTimeline).mockResolvedValue({
      messages: [reply],
      nextCursor: null,
      root,
      throughPosition: 32,
    });

    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Following, 1 unread" }));
    expect(await screen.findByRole("heading", { name: "Following" })).toBeVisible();
    expect(screen.getByText("Please review the qualification output.")).toBeVisible();
    expect(screen.getByText("1 unread")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /Please review the qualification output/ }));

    await waitFor(() => expect(loadChannelMessage).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "session-secret" },
      reply.messageId,
      expect.any(AbortSignal),
    ));
    expect(await screen.findByText("Thread in Wake policy qualification")).toBeVisible();
    expect(window.location.search).toContain(`message=${reply.messageId}`);
    expect(window.location.search).toContain(`thread=${root.messageId}`);
  });

  it("unfollows a thread from the Following workspace", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadFollowedThreads).mockResolvedValue({
      threads: [followedThread],
      throughPosition: 32,
    });

    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Following, 1 unread" }));
    const unfollow = screen.getByRole("button", { name: "Unfollow thread by Daniel" });
    expect(unfollow).toHaveAttribute("aria-pressed", "true");
    expect(unfollow.querySelector("svg")).toHaveAttribute("fill", "currentColor");
    await user.click(unfollow);

    expect(setThreadFollowed).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "session-secret" },
      followedThread.state.threadRootId,
      false,
      followedThread.state.stateVersion,
      expect.stringMatching(/^following-unfollow-/),
    );
    expect(await screen.findByText("No followed threads.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Following" })).toBeVisible();
  });

  it("orders channels before direct messages in Workshop navigation", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());

    render(<App />);

    const navigationPanel = await screen.findByLabelText("Workshop navigation");
    expect(
      Array.from(navigationPanel.querySelectorAll(".nav-heading"), (heading) =>
        heading.textContent?.trim(),
      ),
    ).toEqual([
      "Workspace",
      "Channels",
      "Direct messages",
      "Notifications",
      "Agents",
    ]);
    for (const name of ["Memory", "Mentions", "Following"]) {
      expect(
        within(navigationPanel).getByRole("button", { name }).querySelector(".workspace-nav-icon"),
      ).toBeInTheDocument();
    }
    expect(
      within(navigationPanel).getByRole("button", { name: "Following" }).querySelector("svg"),
    ).toHaveAttribute("fill", "none");
  });

  it("refreshes Following from principal thread events", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadFollowedThreads)
      .mockReset()
      .mockResolvedValueOnce({ threads: [], throughPosition: 31 })
      .mockResolvedValue({ threads: [followedThread], throughPosition: 32 });

    render(<App />);

    expect(await screen.findByRole("button", { name: "Following" })).toBeVisible();
    await waitFor(() => expect(principalEventHandlers).not.toBeNull());
    const callsBeforeEvent = vi.mocked(loadFollowedThreads).mock.calls.length;
    act(() => principalEventHandlers?.onBatch({
      changes: [{
        agentChanges: [],
        eventPosition: 32,
        notificationChanges: [],
        threadChanges: [{
          eventPosition: 32,
          state: followedThread.state,
          transition: "message.created",
        }],
        unreadChanges: [],
      }],
      throughPosition: 32,
    }, "32"));

    expect(await screen.findByRole("button", { name: "Following, 1 unread" })).toBeVisible();
    expect(vi.mocked(loadFollowedThreads).mock.calls.length).toBeGreaterThan(callsBeforeEvent);
  });

  it("marks a live mention read when its source is already visible at the timeline edge", async () => {
    observeMessagesAtViewportEdge();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    const mention: WorkshopHumanNotification = {
      channelName: "Conversation",
      createdAt: "2026-08-31T15:00:00Z",
      createdEventPosition: historyMessage.eventPosition,
      kind: "mention",
      lastEventPosition: historyMessage.eventPosition,
      notificationId: "ntf_00000000000000000000000000000004",
      read: false,
      readAt: null,
      sourceAuthorDisplayName: "Kai",
      sourceAuthorPrincipalId: historyMessage.authorPrincipalId,
      sourceChannelId: channelId,
      sourceMessageId: historyMessage.messageId,
      sourceThreadRootId: null,
      stateVersion: 0,
    };
    vi.mocked(loadHumanNotificationCounts).mockResolvedValue({
      read: 1,
      total: 1,
      unread: 0,
      unreadByChannel: {},
    });
    vi.mocked(markHumanNotificationRead).mockResolvedValue({
      changed: true,
      notification: {
        ...mention,
        read: true,
        readAt: "2026-08-31T15:01:00Z",
        stateVersion: 1,
      },
      replayed: false,
    });

    render(<App />);

    expect(await screen.findByText(historyMessage.body)).toBeVisible();
    await waitFor(() => expect(streamPrincipalEvents).toHaveBeenCalled());
    act(() => {
      principalEventHandlers?.onBatch({
        changes: [{
          agentChanges: [],
          threadChanges: [],
          eventPosition: historyMessage.eventPosition,
          notificationChanges: [{
            eventPosition: historyMessage.eventPosition,
            notification: mention,
            transition: "human_notification.created",
          }],
          unreadChanges: [],
        }],
        throughPosition: historyMessage.eventPosition,
      },
        String(historyMessage.eventPosition),
      );
    });

    await waitFor(() => expect(markHumanNotificationRead).toHaveBeenCalledWith(
      "session-secret",
      mention.notificationId,
      0,
      expect.stringMatching(/^mention-viewed-/),
    ));
    await waitFor(() => expect(
      screen.getByRole("button", { name: "Mentions" }),
    ).toBeVisible());
  });

  it("keeps an unread mention when its source message cannot be displayed", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    const mention: WorkshopHumanNotification = {
      channelName: "Wake policy qualification",
      createdAt: "2026-08-31T15:00:00Z",
      createdEventPosition: 41,
      kind: "mention",
      lastEventPosition: 41,
      notificationId: "ntf_00000000000000000000000000000003",
      read: false,
      readAt: null,
      sourceAuthorDisplayName: "Scott",
      sourceAuthorPrincipalId: "prn_00000000000000000000000000000003",
      sourceChannelId: secondChannelId,
      sourceMessageId: "msg_00000000000000000000000000000043",
      sourceThreadRootId: null,
      stateVersion: 0,
    };
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadHumanNotifications).mockResolvedValue({
      counts: {
        read: 0,
        total: 1,
        unread: 1,
        unreadByChannel: { [secondChannelId]: 1 },
      },
      nextCursor: null,
      notifications: [mention],
      throughPosition: 41,
    });
    vi.mocked(loadChannelMessage).mockRejectedValue(new Error("Source message unavailable"));
    vi.mocked(loadTimeline).mockImplementation(async (activeSession) => ({
      messages: activeSession.channelId === secondChannelId ? [] : [historyMessage],
      previousCursor: null,
      throughPosition: 41,
    }));

    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Mentions, 1 unread" }));
    await user.click(screen.getByRole("button", { name: /Scott mentioned you/ }));

    expect(await screen.findByText(/Source message unavailable/)).toBeVisible();
    expect(markHumanNotificationRead).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Mentions, 1 unread" })).toBeVisible();
  });

  it("updates mention and channel badges from the live notification stream without duplicates", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "session-secret" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadHumanNotificationCounts).mockResolvedValue({
      read: 0,
      total: 1,
      unread: 1,
      unreadByChannel: { [secondChannelId]: 1 },
    });
    render(<App />);
    await waitFor(() => expect(streamPrincipalEvents).toHaveBeenCalled());
    const mention: WorkshopHumanNotification = {
      channelName: "Wake policy qualification",
      createdAt: "2026-08-31T15:00:00Z",
      createdEventPosition: 41,
      kind: "mention",
      lastEventPosition: 41,
      notificationId: "ntf_00000000000000000000000000000002",
      read: false,
      readAt: null,
      sourceAuthorDisplayName: "Scott",
      sourceAuthorPrincipalId: "prn_00000000000000000000000000000003",
      sourceChannelId: secondChannelId,
      sourceMessageId: "msg_00000000000000000000000000000041",
      sourceThreadRootId: null,
      stateVersion: 0,
    };
    act(() => {
      const batch = {
        changes: [{
          agentChanges: [],
          threadChanges: [],
          eventPosition: 41,
          notificationChanges: [{
            eventPosition: 41,
            notification: mention,
            transition: "human_notification.created" as const,
          }],
          unreadChanges: [],
        }],
        throughPosition: 41,
      };
      principalEventHandlers?.onBatch(batch, "41");
    });

    expect(await screen.findByRole("button", { name: "Mentions, 1 unread" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Mentions, 1 unread" }));
    expect(screen.getAllByRole("button", { name: /Scott mentioned you/ })).toHaveLength(1);
  });

  it("opens the personal Settings workspace from the profile menu", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );

    render(<App />);

    const profile = await screen.findByRole("button", { name: "Daniel profile" });
    await user.click(profile);
    const menu = screen.getByRole("menu", { name: "Profile menu" });
    expect(menu).toBeVisible();
    await user.click(screen.getByRole("menuitem", { name: /Settings/ }));

    expect(await screen.findByRole("heading", { name: "Settings", level: 1 })).toBeVisible();
    expect(screen.getByText("Workshop administrator")).toBeVisible();
    expect(screen.queryByLabelText("Backend")).toBeNull();
    expect(screen.queryByRole("heading", { name: "Runtime settings" })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Workspace settings" })).toBeNull();
    const settingsNavigation = screen.getByRole("navigation", {
      name: "Settings sections",
    });
    expect(
      within(settingsNavigation).queryByRole("button", { name: "Runtime settings" }),
    ).toBeNull();
    expect(
      within(settingsNavigation).queryByRole("button", { name: "Workspace settings" }),
    ).toBeNull();
    expect(screen.queryByText("PROTECTED_KEY")).not.toBeInTheDocument();
    expect(screen.queryByText(settingsWorkspace.principalId)).not.toBeInTheDocument();
    expect(screen.queryByText(settingsWorkspace.runtimeProfileId)).not.toBeInTheDocument();
    expect(screen.queryByText(settingsWorkspace.workspace)).not.toBeInTheDocument();
    expect(window.location.search).toBe("?view=settings");

    const backToConversation = screen.getByRole("button", {
      name: "Back to conversation",
    });
    expect(backToConversation).toHaveClass("panel-icon-button");
    expect(backToConversation).toHaveTextContent("←");
    expect(backToConversation.querySelector("span")).toHaveAttribute(
      "aria-hidden",
      "true",
    );
    await user.click(backToConversation);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(window.location.search).toBe("");
  });

  it("keeps session forgetting in the profile menu and confirms it", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );

    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Forget session" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Daniel profile" }));
    await user.click(screen.getByRole("menuitem", { name: /Forget session/ }));
    const confirmation = screen.getByRole("dialog", { name: "Continue?" });
    expect(confirmation).toHaveTextContent(
      "Forget this browser session? You will need to enroll again.",
    );
    await user.click(
      within(confirmation).getByRole("button", { name: "Continue" }),
    );

    expect(await screen.findByLabelText("Enrollment token")).toBeVisible();
  });

  it("closes the profile menu with Escape", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );

    render(<App />);
    await user.click(await screen.findByRole("button", { name: "Daniel profile" }));
    expect(screen.getByRole("menu", { name: "Profile menu" })).toBeVisible();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu", { name: "Profile menu" })).toBeNull();
  });

  it("protects unsaved preferences before leaving Settings", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);

    render(<App />);
    await user.click(await screen.findByRole("button", { name: "Daniel profile" }));
    await user.click(screen.getByRole("menuitem", { name: /Settings/ }));
    const editor = await screen.findByLabelText("Preference Markdown");
    await user.type(editor, "Keep this draft.");
    await user.click(screen.getByRole("button", { name: "Scott" }));

    const cancellation = screen.getByRole("dialog", { name: "Continue?" });
    expect(cancellation).toHaveTextContent("Discard unsaved preference changes?");
    expect(confirm).not.toHaveBeenCalled();
    await user.click(within(cancellation).getByRole("button", { name: "Cancel" }));
    expect(screen.getByRole("heading", { name: "Settings", level: 1 })).toBeVisible();
    expect(window.location.search).toBe("?view=settings");

    await user.click(screen.getByRole("button", { name: "Scott" }));
    await user.click(within(
      screen.getByRole("dialog", { name: "Continue?" }),
    ).getByRole("button", { name: "Continue" }));
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(window.location.search).toBe("");
  });

  it("shows canonical runtime settings and switches an authorized workspace", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );

    render(<App />);

    expect(await screen.findByText("gpt-5.6-sol")).toBeVisible();
    const selector = screen.getByLabelText("Workspace");
    expect(selector).toHaveValue("/Users/kai/Projects/kai");
    expect(screen.getByRole("option", { name: "Home" })).toHaveValue(
      "/var/lib/kai/home/principal",
    );
    expect(
      screen.queryByRole("option", { name: "Home (home)" }),
    ).not.toBeInTheDocument();
    await user.selectOptions(selector, "/var/lib/kai/home/principal");

    expect(switchWorkspace).toHaveBeenCalledWith(
      { channelId, token: "existing-session" },
      "/var/lib/kai/home/principal",
      "sws_current",
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Workspace")).toHaveValue(
        "/var/lib/kai/home/principal",
      ),
    );
  });

  it("fetches the run trace incrementally on trace doorbells", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const traceEntry = (seq: number) => ({
      createdAt: "2026-08-13T09:00:00+00:00",
      detail: "",
      isDiff: false,
      isError: false,
      kind: "tool_call" as const,
      seq,
      summary: `step ${seq}`,
      toolName: "Bash",
      toolUseId: `toolu_${seq}`,
    });
    vi.mocked(loadRunTrace)
      .mockResolvedValueOnce({ entries: [traceEntry(1), traceEntry(2)], hasMore: false })
      .mockResolvedValueOnce({ entries: [traceEntry(3)], hasMore: false });
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    const startedRun: WorkshopRun = {
      ...completedRun,
      resultMessageId: null,
      status: "started",
      terminalAt: null,
    };
    act(() =>
      handlers?.onRunActivity(
        {
          eventPosition: 30,
          occurredAt: "2026-08-13T09:00:01Z",
          run: startedRun,
          transition: "run.started",
        },
        "30",
      ),
    );

    expect(await screen.findByText("step 2")).toBeVisible();

    act(() => handlers?.onRunTrace({ runId: startedRun.runId, seq: 3 }));
    expect(await screen.findByText("step 3")).toBeVisible();
    expect(
      vi.mocked(loadRunTrace).mock.calls.map((call) => [call[1], call[2]]),
    ).toEqual([
      [startedRun.runId, 0],
      [startedRun.runId, 2],
    ]);

    // A doorbell at or below the held position fetches nothing more.
    act(() => handlers?.onRunTrace({ runId: startedRun.runId, seq: 3 }));
    expect(vi.mocked(loadRunTrace)).toHaveBeenCalledTimes(2);
  });

  it("does not duplicate rows when a doorbell races the initial drain", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const page = {
      entries: [
        {
          createdAt: "2026-08-13T09:00:00+00:00",
          detail: "",
          isDiff: false,
          isError: false,
          kind: "tool_call" as const,
          seq: 1,
          summary: "step 1",
          toolName: "Bash",
          toolUseId: "toolu_1",
        },
      ],
      hasMore: false,
    };
    let resolveInitial: ((value: typeof page) => void) | null = null;
    vi.mocked(loadRunTrace)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveInitial = resolve;
          }),
      )
      .mockResolvedValueOnce(page);
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    const startedRun: WorkshopRun = {
      ...completedRun,
      resultMessageId: null,
      status: "started",
      terminalAt: null,
    };
    act(() =>
      handlers?.onRunActivity(
        {
          eventPosition: 30,
          occurredAt: "2026-08-13T09:00:01Z",
          run: startedRun,
          transition: "run.started",
        },
        "30",
      ),
    );
    await waitFor(() => expect(loadRunTrace).toHaveBeenCalledTimes(1));

    // The stale doorbell every fresh connection receives lands while the
    // initial drain's fetch is still in flight; both drains resolve with
    // the same page.
    act(() => handlers?.onRunTrace({ runId: startedRun.runId, seq: 1 }));
    await waitFor(() => expect(loadRunTrace).toHaveBeenCalledTimes(2));
    act(() => resolveInitial?.(page));

    expect(await screen.findByText("step 1")).toBeVisible();
    expect(screen.getAllByText("step 1")).toHaveLength(1);
  });

  it("streams a growing run preview and replaces it with the canonical answer", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Enrollment token"), "one-time-token");
    await user.click(screen.getByRole("button", { name: "Open Workshop" }));
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    act(() =>
      handlers?.onRunPreview({
        runId: "run_00000000000000000000000000000030",
        sequence: 1,
        text: "First sentence.",
      }),
    );
    expect(await screen.findByText("First sentence.")).toBeVisible();
    expect(screen.getByText("writing")).toBeVisible();

    act(() =>
      handlers?.onRunPreview({
        runId: "run_00000000000000000000000000000030",
        sequence: 2,
        text: "First sentence. Second sentence.",
      }),
    );
    expect(await screen.findByText("First sentence. Second sentence.")).toBeVisible();

    // A stale lower-sequence event must not roll the bubble backwards.
    act(() =>
      handlers?.onRunPreview({
        runId: "run_00000000000000000000000000000030",
        sequence: 1,
        text: "First sentence.",
      }),
    );
    expect(screen.getByText("First sentence. Second sentence.")).toBeVisible();

    const canonicalAnswer: TimelineMessage = {
      ...historyMessage,
      body: "First sentence. Second sentence. Final answer.",
      eventPosition: 31,
      messageId: "msg_00000000000000000000000000000031",
    };
    act(() => handlers?.onMessage(canonicalAnswer, "31"));

    expect(await screen.findByText(canonicalAnswer.body)).toBeVisible();
    expect(screen.queryByText("writing")).toBeNull();
    expect(screen.queryByText("First sentence. Second sentence.")).toBeNull();
  });

  it("loads earlier history on demand and preserves the reader's position", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [historyMessage],
      throughPosition: 25,
      previousCursor: "earlier-page",
    });
    let resolveEarlier: ((value: TimelineSnapshot) => void) | null = null;
    vi.mocked(loadEarlierTimeline).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveEarlier = resolve;
        }),
    );
    render(<App />);

    const timeline = await screen.findByLabelText("Conversation timeline");
    let scrollHeight = 1000;
    Object.defineProperties(timeline, {
      clientHeight: { configurable: true, get: () => 300 },
      scrollHeight: { configurable: true, get: () => scrollHeight },
      scrollTop: { configurable: true, value: 0, writable: true },
    });
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    // The reader has scrolled near the top, where the control lives.
    timeline.scrollTop = 40;
    fireEvent.scroll(timeline);
    await user.click(screen.getByRole("button", { name: "Load earlier messages" }));
    expect(vi.mocked(loadEarlierTimeline)).toHaveBeenCalledWith(
      { channelId, token: "existing-session" },
      "earlier-page",
      25,
      expect.anything(),
    );

    // The prepended page grows the content above the viewport by 600px;
    // the viewport must shift by exactly that amount to stay put.
    scrollHeight = 1600;
    const earlierMessage: TimelineMessage = {
      ...historyMessage,
      body: "Older history.",
      eventPosition: 5,
      messageId: "msg_00000000000000000000000000000005",
    };
    act(() => {
      resolveEarlier?.({
        messages: [earlierMessage],
        throughPosition: 25,
        previousCursor: null,
      });
    });

    expect(await screen.findByText("Older history.")).toBeVisible();
    const bodies = screen.getAllByRole("listitem").map((item) => item.textContent ?? "");
    expect(bodies.findIndex((text) => text.includes("Older history."))).toBeLessThan(
      bodies.findIndex((text) => text.includes("Canonical history is ready.")),
    );
    await waitFor(() => expect(timeline.scrollTop).toBe(640));
    // The final page reached the start of the channel; the control goes away.
    expect(screen.queryByRole("button", { name: "Load earlier messages" })).toBeNull();
  });

  it("opens at the canonical first unread boundary and advances only visible messages", async () => {
    const visible = new Map<Element, IntersectionObserverCallback>();
    class ControlledIntersectionObserver {
      readonly root: Element | Document | null;
      readonly rootMargin = "0px";
      readonly thresholds = [0];

      constructor(
        private readonly callback: IntersectionObserverCallback,
        options?: IntersectionObserverInit,
      ) {
        this.root = options?.root ?? null;
      }

      disconnect(): void {
        for (const [target, callback] of visible) {
          if (callback === this.callback) visible.delete(target);
        }
      }

      observe(target: Element): void {
        visible.set(target, this.callback);
      }

      takeRecords(): IntersectionObserverEntry[] { return []; }
      unobserve(target: Element): void { visible.delete(target); }
    }
    vi.stubGlobal("IntersectionObserver", ControlledIntersectionObserver);
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    const firstUnread = { ...historyMessage, channelId: secondChannelId };
    const secondUnread: TimelineMessage = {
      ...firstUnread,
      body: "Still below the viewport.",
      eventPosition: 26,
      messageId: "msg_00000000000000000000000000000026",
    };
    const initial = unreadState(secondChannelId, { unreadCount: 2 });
    vi.mocked(loadChannelUnread).mockResolvedValue({
      channels: [initial],
      throughPosition: 26,
      totalUnread: 2,
      totalUnreadCapped: false,
    });
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [firstUnread, secondUnread],
      nextCursor: null,
      previousCursor: "before-unread",
      throughPosition: 26,
    });
    vi.mocked(advanceChannelReadPosition).mockResolvedValue({
      replayed: false,
      state: unreadState(secondChannelId, {
        firstUnreadEventPosition: 26,
        firstUnreadMessageId: secondUnread.messageId,
        lastEventPosition: 27,
        readThroughEventPosition: 25,
        readThroughMessageId: firstUnread.messageId,
        stateVersion: 1,
      }),
    });

    render(<App />);

    expect(await screen.findByLabelText("First unread message")).toHaveTextContent("New messages");
    expect(loadTimeline).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      expect.anything(),
      firstUnread.messageId,
    );
    expect(screen.getByRole("button", {
      name: "Wake policy qualification, 2 unread messages",
    })).toHaveClass("unread");
    const firstRow = document.querySelector(`[data-message-id="${firstUnread.messageId}"]`);
    const secondRow = document.querySelector(`[data-message-id="${secondUnread.messageId}"]`);
    expect(firstRow).not.toBeNull();
    expect(secondRow).not.toBeNull();
    const reveal = (target: Element): void => {
      const bounds = target.getBoundingClientRect();
      visible.get(target)?.([{
        boundingClientRect: bounds,
        intersectionRatio: 1,
        intersectionRect: bounds,
        isIntersecting: true,
        rootBounds: null,
        target,
        time: 0,
      }], {} as IntersectionObserver);
    };
    act(() => reveal(firstRow as Element));
    expect(advanceChannelReadPosition).not.toHaveBeenCalled();
    fireEvent.pointerDown(screen.getByLabelText("Conversation timeline"));
    act(() => reveal(firstRow as Element));
    await waitFor(() => expect(advanceChannelReadPosition).toHaveBeenCalledTimes(1));
    expect(advanceChannelReadPosition).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      firstUnread.messageId,
      0,
      expect.stringMatching(/^channel-read-/),
    );

    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "hidden",
    });
    act(() => reveal(secondRow as Element));
    expect(advanceChannelReadPosition).toHaveBeenCalledTimes(1);
  });

  it("updates ordinary unread navigation from canonical live and cross-tab changes", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());

    render(<App />);

    await screen.findByText("Canonical history is ready.");
    await waitFor(() => expect(principalEventHandlers).not.toBeNull());
    const unread = unreadState(secondChannelId, {
      firstUnreadEventPosition: 30,
      firstUnreadMessageId: "msg_00000000000000000000000000000030",
      lastEventPosition: 30,
      unreadCount: 1,
    });
    act(() => principalEventHandlers?.onBatch({
      changes: [{
        agentChanges: [],
        threadChanges: [],
        eventPosition: 30,
        notificationChanges: [],
        unreadChanges: [{ eventPosition: 30, state: unread }],
      }],
      throughPosition: 30,
    }, "30"));

    expect(screen.getByRole("button", {
      name: "Wake policy qualification, 1 unread message",
    })).toHaveClass("unread");

    // Replaying a position after reconnect is harmless.
    act(() => principalEventHandlers?.onBatch({
      changes: [{
        agentChanges: [],
        threadChanges: [],
        eventPosition: 30,
        notificationChanges: [],
        unreadChanges: [{ eventPosition: 30, state: unread }],
      }],
      throughPosition: 30,
    }, "30"));
    expect(screen.getAllByRole("button", {
      name: "Wake policy qualification, 1 unread message",
    })).toHaveLength(1);

    // A read-position mutation from another tab is authoritative here too.
    act(() => principalEventHandlers?.onBatch({
      changes: [{
        agentChanges: [],
        threadChanges: [],
        eventPosition: 31,
        notificationChanges: [],
        unreadChanges: [{
          eventPosition: 31,
          state: unreadState(secondChannelId, {
        firstUnreadEventPosition: null,
        firstUnreadMessageId: null,
        lastEventPosition: 31,
        readThroughEventPosition: 30,
        readThroughMessageId: unread.firstUnreadMessageId,
        stateVersion: 1,
        unreadCount: 0,
          }),
        }],
      }],
      throughPosition: 31,
    }, "31"));
    expect(screen.getByRole("button", {
      name: "Wake policy qualification",
    })).not.toHaveClass("unread");
  });

  it("opens at the latest message and preserves deliberate scroll position", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    let resolveTimeline: ((value: {
      messages: TimelineMessage[];
      throughPosition: number;
      previousCursor: string | null;
    }) => void) | null = null;
    vi.mocked(loadTimeline).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveTimeline = resolve;
        }),
    );
    render(<App />);

    const timeline = await screen.findByLabelText("Conversation timeline");
    let scrollHeight = 1000;
    Object.defineProperties(timeline, {
      clientHeight: { configurable: true, get: () => 300 },
      scrollHeight: { configurable: true, get: () => scrollHeight },
      scrollTop: { configurable: true, value: 0, writable: true },
    });
    await waitFor(() => expect(resolveTimeline).not.toBeNull());
    act(() => {
      resolveTimeline?.({ messages: [historyMessage], throughPosition: 25, previousCursor: null });
    });

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    await waitFor(() => expect(timeline.scrollTop).toBe(1000));
    timeline.scrollTop = 100;
    fireEvent.scroll(timeline);

    scrollHeight = 1100;
    const firstNewMessage: TimelineMessage = {
      ...historyMessage,
      body: "First unread message",
      eventPosition: 30,
      messageId: "msg_00000000000000000000000000000030",
    };
    act(() => handlers?.onMessage(firstNewMessage, "30"));

    expect(await screen.findByText("First unread message")).toBeVisible();
    expect(timeline.scrollTop).toBe(100);
    await user.click(screen.getByRole("button", { name: "1 new message" }));
    expect(timeline.scrollTop).toBe(1100);
    expect(screen.queryByRole("button", { name: "1 new message" })).toBeNull();

    scrollHeight = 1200;
    const secondNewMessage: TimelineMessage = {
      ...historyMessage,
      body: "Followed message",
      eventPosition: 31,
      messageId: "msg_00000000000000000000000000000031",
    };
    act(() => handlers?.onMessage(secondNewMessage, "31"));

    expect(await screen.findByText("Followed message")).toBeVisible();
    await waitFor(() => expect(timeline.scrollTop).toBe(1200));
    expect(
      screen.queryByRole("button", { name: /new messages?/ }),
    ).toBeNull();

    scrollHeight = 1300;
    await user.type(screen.getByLabelText("Message Kai"), "Show activity");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByLabelText("Agent run activity")).toBeVisible();
    await waitFor(() => expect(timeline.scrollTop).toBe(1300));

    timeline.scrollTop = 100;
    fireEvent.scroll(timeline);
    scrollHeight = 1400;
    act(() =>
      handlers?.onRunActivity(
        {
          eventPosition: 32,
          occurredAt: "2026-08-13T09:00:03Z",
          run: {
            ...completedRun,
            resultMessageId: null,
            status: "started",
            terminalAt: null,
          },
          transition: "run.started",
        },
        "32",
      ),
    );
    expect(await screen.findByText("The agent is working on this request.")).toBeVisible();
    expect(timeline.scrollTop).toBe(100);
  });

  it("offers jump-to-latest while reading history with no new arrivals", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    let resolveTimeline: ((value: TimelineSnapshot) => void) | null = null;
    vi.mocked(loadTimeline).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveTimeline = resolve;
        }),
    );
    render(<App />);

    const timeline = await screen.findByLabelText("Conversation timeline");
    let scrollHeight = 1000;
    Object.defineProperties(timeline, {
      clientHeight: { configurable: true, get: () => 300 },
      scrollHeight: { configurable: true, get: () => scrollHeight },
      scrollTop: { configurable: true, value: 0, writable: true },
    });
    await waitFor(() => expect(resolveTimeline).not.toBeNull());
    act(() => {
      resolveTimeline?.({ messages: [historyMessage], throughPosition: 25, previousCursor: null });
    });
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    await waitFor(() => expect(timeline.scrollTop).toBe(1000));

    // At the bottom: no button of either form.
    expect(screen.queryByRole("button", { name: /Jump to latest|new message/ })).toBeNull();

    // Within the follow distance (96px of the bottom): still none.
    timeline.scrollTop = 650;
    fireEvent.scroll(timeline);
    expect(screen.queryByRole("button", { name: /Jump to latest|new message/ })).toBeNull();

    // Past the follow distance: the neutral button appears with no
    // new messages required.
    timeline.scrollTop = 100;
    fireEvent.scroll(timeline);
    const jumpButton = screen.getByRole("button", { name: "Jump to latest messages" });
    expect(jumpButton).toHaveTextContent("Jump to latest");

    // A live arrival while away: the count label takes precedence.
    scrollHeight = 1100;
    const arrival: TimelineMessage = {
      ...historyMessage,
      body: "Arrived while reading back",
      eventPosition: 30,
      messageId: "msg_00000000000000000000000000000030",
    };
    act(() => handlers?.onMessage(arrival, "30"));
    expect(await screen.findByRole("button", { name: "1 new message" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Jump to latest messages" })).toBeNull();

    // Clicking returns to the bottom, persists follow, and hides the
    // button entirely.
    await user.click(screen.getByRole("button", { name: "1 new message" }));
    expect(timeline.scrollTop).toBe(1100);
    expect(screen.queryByRole("button", { name: /Jump to latest|new message/ })).toBeNull();
    const viewports: unknown = JSON.parse(
      sessionStorage.getItem("kai.workshop.timeline-viewports.v1") ?? "{}",
    );
    expect((viewports as Record<string, { follow: boolean }>)[channelId]?.follow).toBe(true);
  });

  it("shows jump-to-latest on mount for a restored away-from-bottom viewport", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    sessionStorage.setItem(
      "kai.workshop.timeline-viewports.v1",
      JSON.stringify({ [channelId]: { follow: false, scrollTop: 100 } }),
    );
    let resolveTimeline: ((value: TimelineSnapshot) => void) | null = null;
    vi.mocked(loadTimeline).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveTimeline = resolve;
        }),
    );
    render(<App />);

    const timeline = await screen.findByLabelText("Conversation timeline");
    Object.defineProperties(timeline, {
      clientHeight: { configurable: true, get: () => 300 },
      scrollHeight: { configurable: true, get: () => 1000 },
      scrollTop: { configurable: true, value: 0, writable: true },
    });
    await waitFor(() => expect(resolveTimeline).not.toBeNull());
    act(() => {
      resolveTimeline?.({ messages: [historyMessage], throughPosition: 25, previousCursor: null });
    });
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    // The restored deliberate position is honored, and the button is
    // there from the start rather than waiting for a scroll event.
    await waitFor(() => expect(timeline.scrollTop).toBe(100));
    expect(
      await screen.findByRole("button", { name: "Jump to latest messages" }),
    ).toBeVisible();
  });

  it("hides jump-to-latest when a restored viewport lands at the bottom", async () => {
    // A viewport persisted with earlier pages loaded can restore into
    // a latest-page window too short to put the position away from the
    // bottom; the button must derive from the clamped geometry, not
    // the stored follow flag.
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    sessionStorage.setItem(
      "kai.workshop.timeline-viewports.v1",
      JSON.stringify({ [channelId]: { follow: false, scrollTop: 900 } }),
    );
    let resolveTimeline: ((value: TimelineSnapshot) => void) | null = null;
    vi.mocked(loadTimeline).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveTimeline = resolve;
        }),
    );
    render(<App />);

    const timeline = await screen.findByLabelText("Conversation timeline");
    Object.defineProperties(timeline, {
      clientHeight: { configurable: true, get: () => 300 },
      scrollHeight: { configurable: true, get: () => 250 },
      scrollTop: { configurable: true, value: 0, writable: true },
    });
    await waitFor(() => expect(resolveTimeline).not.toBeNull());
    act(() => {
      resolveTimeline?.({ messages: [historyMessage], throughPosition: 25, previousCursor: null });
    });
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    expect(
      screen.queryByRole("button", { name: "Jump to latest messages" }),
    ).toBeNull();
  });

  it("returns to enrollment and clears the tab session after revocation", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    act(() => failStream?.(new AuthenticationError("Session revoked.")));

    expect(await screen.findByRole("alert")).toHaveTextContent("Session revoked.");
    expect(localStorage.getItem("kai.workshop.client-credential.v1")).toBeNull();
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toBeNull();
    expect(sessionStorage.getItem("kai.workshop.active-channel.v1")).toBeNull();
    expect(screen.getByLabelText("Enrollment token")).toBeVisible();
  });

  it("preserves an enrolled session while refreshing changed channel access", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadTimeline).mockRejectedValueOnce(
      new ChannelAccessError("Channel access changed."),
    );
    vi.mocked(loadNavigation)
      .mockResolvedValueOnce(navigation)
      .mockResolvedValueOnce({
        ...navigation,
        workshops: [
          {
            ...navigation.workshops[0],
            channels: [
              {
                ...navigation.workshops[0].channels[0],
                channelId: secondChannelId,
                kind: "group",
                name: "Replacement channel",
              },
            ],
          },
        ],
      });
    render(<App />);

    expect(
      (await screen.findAllByRole("heading", { name: /Replacement channel/ })).length,
    ).toBeGreaterThan(0);
    expect(screen.queryByLabelText("Enrollment token")).toBeNull();
    expect(localStorage.getItem("kai.workshop.client-credential.v1")).toContain(
      "existing-session",
    );
    expect(sessionStorage.getItem("kai.workshop.active-channel.v1")).toContain(
      secondChannelId,
    );
  });

  it("switches authorized channels without re-enrollment and isolates drafts", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadTimeline).mockImplementation(async (selectedSession) => ({
      messages: [
        {
          ...historyMessage,
          body: `History for ${selectedSession.channelId}`,
          channelId: selectedSession.channelId,
        },
      ],
      throughPosition: 25,
      previousCursor: null,
    }));
    render(<App />);

    expect(await screen.findByText(`History for ${channelId}`)).toBeVisible();
    await user.type(screen.getByLabelText("Message Kai"), "Keep this draft");
    await user.click(
      screen.getByRole("button", { name: /GitHub notifications/ }),
    );

    expect(
      await screen.findByText(`History for ${notificationChannelId}`),
    ).toBeVisible();
    expect(screen.getByText("GitHub")).toBeVisible();
    expect(screen.getByText("Durable notification feed")).toBeVisible();
    expect(
      screen.getByText("Active delivery: GitHub → GitHub notifications"),
    ).toBeVisible();
    expect(document.querySelector(".notification-row")).not.toBeNull();
    expect(screen.queryByLabelText("Message Kai")).toBeNull();
    expect(screen.getByText(/outbound-only/)).toBeVisible();
    expect(
      screen.getByText(
        "You can read this outbound channel, but you cannot send messages here.",
      ),
    ).toBeVisible();
    expect(sessionStorage.getItem("kai.workshop.active-channel.v1")).toContain(
      notificationChannelId,
    );

    await user.click(screen.getByRole("button", { name: "Kai" }));
    expect(await screen.findByText(`History for ${channelId}`)).toBeVisible();
    expect(screen.getByLabelText("Message Kai")).toHaveValue("Keep this draft");
    expect(redeemEnrollment).not.toHaveBeenCalled();
  });

  it("uses the principal's direct runtime settings from a group channel", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const groupChannel = {
      ...navigation.workshops[0].channels[0],
      channelId: secondChannelId,
      kind: "group" as const,
      name: "Wake policy qualification",
    };
    vi.mocked(loadNavigation).mockResolvedValue({
      ...navigation,
      workshops: [
        {
          ...navigation.workshops[0],
          channels: [...navigation.workshops[0].channels, groupChannel],
        },
      ],
    });
    vi.mocked(loadTimeline).mockImplementation(async (selectedSession) => ({
      messages: [
        {
          ...historyMessage,
          body: `History for ${selectedSession.channelId}`,
          channelId: selectedSession.channelId,
        },
      ],
      throughPosition: 25,
      previousCursor: null,
    }));
    render(<App />);
    expect(await screen.findByText(`History for ${channelId}`)).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Wake policy qualification" }),
    );

    expect(
      await screen.findByText(`History for ${secondChannelId}`),
    ).toBeVisible();
    expect(await screen.findByText("gpt-5.6-sol")).toBeVisible();
    expect(loadSettingsWorkspace).toHaveBeenLastCalledWith({
      channelId,
      token: "existing-session",
    });
    await user.selectOptions(
      screen.getByLabelText("Workspace"),
      "/var/lib/kai/home/principal",
    );
    expect(switchWorkspace).toHaveBeenCalledWith(
      { channelId, token: "existing-session" },
      "/var/lib/kai/home/principal",
      "sws_current",
    );
    await waitFor(() => expect(loadNavigation).toHaveBeenCalledTimes(1));
    expect((await screen.findAllByText("Live")).length).toBeGreaterThanOrEqual(1);

    await user.click(screen.getByRole("button", { name: "Kai" }));
    expect(await screen.findByText(`History for ${channelId}`)).toBeVisible();
  });

  it("creates a channel from the sidebar and opens it", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation)
      .mockResolvedValueOnce(navigation)
      .mockResolvedValueOnce(
        navigationWithGroup({ name: "Release planning" }),
      );
    vi.mocked(loadTimeline).mockImplementation(async (selectedSession) => ({
      messages: [
        {
          ...historyMessage,
          body: `History for ${selectedSession.channelId}`,
          channelId: selectedSession.channelId,
        },
      ],
      throughPosition: 25,
      previousCursor: null,
    }));

    render(<App />);
    expect(await screen.findByText(`History for ${channelId}`)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Start channel" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Create channel" }));
    expect(screen.queryByText(/Start from/)).toBeNull();
    await user.type(screen.getByLabelText("Channel name"), "Release planning");
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create channel",
      }),
    );

    await waitFor(() =>
      expect(createChannel).toHaveBeenCalledWith("existing-session", {
        agentIds: ["agt_00000000000000000000000000000001"],
        name: "Release planning",
        originChannelId: null,
      }),
    );
    expect(
      await screen.findByText(`History for ${secondChannelId}`),
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Release planning" }),
    ).toBeVisible();
  });

  it("opens channel creation immediately from auxiliary Workshop views", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );

    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Memory" }));
    expect(
      await screen.findByRole("heading", { name: "Memory", level: 1 }),
    ).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Create channel" }));
    let creationDialog = screen.getByRole("dialog", { name: "Create channel" });
    expect(creationDialog).toBeVisible();
    await user.click(
      within(creationDialog).getByRole("button", { name: "Cancel" }),
    );
    expect(screen.queryByRole("dialog", { name: "Create channel" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Manage Kai" }));
    expect(
      await screen.findByRole("heading", { name: "Agents", level: 1 }),
    ).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Create channel" }));
    creationDialog = screen.getByRole("dialog", { name: "Create channel" });
    expect(creationDialog).toBeVisible();
  });

  it("archives a group channel read-only and restores it from the archive", async () => {
    const user = userEvent.setup();
    const active = navigationWithGroup({
      name: "Lifecycle qualification",
      role: "owner",
    });
    const archived = {
      ...active,
      workshops: active.workshops.map((workshop) => ({
        ...workshop,
        channels: workshop.channels.map((candidate) =>
          candidate.channelId === secondChannelId
            ? {
                ...candidate,
                archivedAt: "2026-08-30T15:00:00Z",
                canSubmitCommands: false,
                lifecycleEventPosition: 501,
              }
            : candidate,
        ),
      })),
    };
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation)
      .mockResolvedValueOnce(active)
      .mockResolvedValueOnce(archived)
      .mockResolvedValueOnce(active);
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [{ ...historyMessage, channelId: secondChannelId }],
      throughPosition: 25,
      previousCursor: null,
    });

    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Archive channel" }));
    const confirmation = screen.getByRole("dialog", { name: "Continue?" });
    await user.click(
      within(confirmation).getByRole("button", { name: "Continue" }),
    );

    await waitFor(() => expect(archiveChannel).toHaveBeenCalledOnce());
    expect(vi.mocked(archiveChannel).mock.calls[0]?.slice(0, 2)).toEqual([
      "existing-session",
      secondChannelId,
    ]);
    expect(
      await screen.findByText(/preserved and read-only/),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Lifecycle qualification" }),
    ).toBeNull();
    expect(screen.queryByLabelText("Message Lifecycle qualification")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Archived channels" }));
    const archiveDialog = screen.getByRole("dialog", { name: "Archive" });
    expect(archiveDialog).toHaveTextContent("Lifecycle qualification");
    await user.click(
      within(archiveDialog).getByRole("button", {
        name: "Restore channel Lifecycle qualification",
      }),
    );

    await waitFor(() => expect(restoreChannel).toHaveBeenCalledOnce());
    expect(vi.mocked(restoreChannel).mock.calls[0]?.slice(0, 2)).toEqual([
      "existing-session",
      secondChannelId,
    ]);
    expect(
      await screen.findByRole("button", { name: "Lifecycle qualification" }),
    ).toBeVisible();
    expect(screen.getByLabelText("Message Lifecycle qualification")).toBeVisible();
  });

  it("archives a direct message personally and restores it from the archive", async () => {
    const user = userEvent.setup();
    const archived: WorkshopNavigation = {
      ...navigation,
      workshops: navigation.workshops.map((workshop) => ({
        ...workshop,
        channels: workshop.channels.map((candidate) =>
          candidate.channelId === humanDirectChannelId
            ? {
                ...candidate,
                canSubmitCommands: false,
                directMessageArchiveEventPosition: 601,
                directMessageArchivedAt: "2026-09-03T15:00:00Z",
              }
            : candidate,
        ),
      })),
    };
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({
        channelId: humanDirectChannelId,
        token: "existing-session",
      }),
    );
    vi.mocked(loadNavigation)
      .mockResolvedValueOnce(navigation)
      .mockResolvedValueOnce(archived)
      .mockResolvedValueOnce(navigation);
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [{ ...historyMessage, channelId: humanDirectChannelId }],
      throughPosition: 25,
      previousCursor: null,
    });

    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Archive direct message" }));
    await user.click(
      within(screen.getByRole("dialog", { name: "Continue?" })).getByRole(
        "button",
        { name: "Continue" },
      ),
    );

    await waitFor(() => expect(archiveDirectMessage).toHaveBeenCalledOnce());
    expect(vi.mocked(archiveDirectMessage).mock.calls[0]?.slice(0, 2)).toEqual([
      "existing-session",
      humanDirectChannelId,
    ]);
    expect(screen.queryByRole("button", { name: "Scott" })).toBeNull();
    expect(
      (await screen.findAllByText(/direct message is archived for you/i)).length,
    ).toBeGreaterThan(0);

    const archivedDirectMessages = screen.getByRole("button", {
      name: "Archived direct messages",
    });
    expect(archivedDirectMessages).toHaveClass("nav-tool-button");
    await user.click(archivedDirectMessages);
    const archiveDialog = screen.getByRole("dialog", { name: "Archive" });
    expect(archiveDialog).toHaveTextContent("Scott");
    await user.click(
      within(archiveDialog).getByRole("button", {
        name: "Restore direct message Scott",
      }),
    );

    await waitFor(() => expect(restoreDirectMessage).toHaveBeenCalledOnce());
    expect(vi.mocked(restoreDirectMessage).mock.calls[0]?.slice(0, 2)).toEqual([
      "existing-session",
      humanDirectChannelId,
    ]);
    expect(await screen.findByRole("button", { name: "Scott" })).toBeVisible();
    expect(screen.getByLabelText("Message Scott")).toBeVisible();
  });

  it("inserts member mentions as plain text and submits them canonically", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [{ ...historyMessage, channelId: secondChannelId }],
      throughPosition: 25,
      previousCursor: null,
    });

    render(<App />);
    const composer = await screen.findByLabelText(
      "Message Wake policy qualification",
    );
    await user.type(composer, "@ka");
    await user.click(screen.getByRole("option", { name: "@kai — Kai — agent" }));
    expect(composer).toHaveValue("@kai ");
    await user.type(composer, "reply plainly{Enter}");

    await waitFor(() => expect(submitCommand).toHaveBeenCalledOnce());
    expect(vi.mocked(submitCommand).mock.calls[0]?.[2]).toBe(
      "@kai reply plainly",
    );
  });

  it("shows a human handle with secondary display name only when the human is a channel member", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    const groupNavigation = navigationWithGroup();
    const channels = groupNavigation.workshops[0].channels;
    const group = channels[channels.length - 1];
    group.participants = [
      ...group.participants,
      {
        displayName: "Scott Ellison",
        handle: "scott",
        kind: "human",
        principalId: "prn_55555555555555555555555555555555",
      },
    ];
    vi.mocked(loadNavigation).mockResolvedValue(groupNavigation);
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [{ ...historyMessage, channelId: secondChannelId }],
      throughPosition: 25,
      previousCursor: null,
    });

    render(<App />);
    const composer = await screen.findByLabelText(
      "Message Wake policy qualification",
    );
    await user.type(composer, "@sc");
    await user.click(
      screen.getByRole("option", { name: "@scott — Scott Ellison — human" }),
    );
    expect(composer).toHaveValue("@scott ");
    expect(screen.queryByRole("option", { name: /Daniel/ })).toBeNull();
    const channelPeople = screen.getByRole("heading", { name: "People" }).closest("section");
    expect(channelPeople).not.toBeNull();
    expect(channelPeople?.querySelectorAll(".context-person-avatar")).toHaveLength(2);
  });

  it("lets a channel owner add an eligible human member", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(
      navigationWithGroup({ role: "owner" }),
    );
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [{ ...historyMessage, channelId: secondChannelId }],
      throughPosition: 25,
      previousCursor: null,
    });

    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: "Manage channel members" }),
    );
    const dialog = await screen.findByRole("dialog", { name: "People" });
    const daniel = within(dialog).getByRole("checkbox", { name: /Daniel/ });
    const scott = within(dialog).getByRole("checkbox", { name: /Scott/ });
    expect(daniel).toBeChecked();
    expect(daniel).toBeDisabled();
    expect(scott).not.toBeChecked();

    await user.click(scott);
    await user.click(within(dialog).getByRole("button", { name: "Save" }));

    await waitFor(() => expect(changeChannelMember).toHaveBeenCalledOnce());
    expect(changeChannelMember).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      "prn_00000000000000000000000000000003",
      "add",
      0,
      expect.stringMatching(/^browser-/),
    );
    expect(loadNavigation).toHaveBeenCalledTimes(2);
  });

  it("opens a group-message thread and submits replies against its canonical root", async () => {
    const user = userEvent.setup();
    observeMessagesAsVisible();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    const root = {
      ...historyMessage,
      channelId: secondChannelId,
      replyCount: 1,
      replyParticipantCount: 2,
      replyParticipants: [
        {
          avatar: { active: false, stateVersion: 0, url: null },
          displayName: "Scott",
          kind: "human" as const,
          principalId: "prn_00000000000000000000000000000003",
        },
        {
          avatar: null,
          displayName: "Kai",
          kind: "agent" as const,
          principalId: "prn_00000000000000000000000000000002",
        },
      ],
      latestReplyAt: "2026-08-13T09:01:00Z",
    };
    const reply = {
      ...historyMessage,
      body: "Existing thread reply",
      channelId: secondChannelId,
      eventPosition: 26,
      messageId: "msg_00000000000000000000000000000026",
      replyToMessageId: root.messageId,
      threadRootId: root.messageId,
    };
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [root],
      throughPosition: 26,
      previousCursor: null,
    });
    vi.mocked(loadChannelUnread).mockResolvedValue({
      channels: [unreadState(secondChannelId, {
        firstUnreadEventPosition: null,
        firstUnreadMessageId: null,
        firstUnreadThreadEventPosition: reply.eventPosition,
        firstUnreadThreadReplyId: reply.messageId,
        firstUnreadThreadRootId: root.messageId,
        unreadCount: 0,
        unreadReplyCount: 1,
        unreadThreadCount: 1,
      })],
      throughPosition: 26,
      totalUnread: 1,
      totalUnreadCapped: false,
    });
    vi.mocked(loadThreadTimeline).mockResolvedValue({
      root,
      messages: [reply],
      nextCursor: null,
      throughPosition: 26,
    });
    vi.mocked(loadThreadUnread).mockResolvedValue(threadUnreadState(root.messageId, {
      firstUnreadEventPosition: reply.eventPosition,
      firstUnreadMessageId: reply.messageId,
      lastEventPosition: reply.eventPosition,
      unreadCount: 1,
    }));

    render(<App />);
    const threadButton = await screen.findByRole("button", {
      name: "Open thread with 1 reply, including unread replies",
    });
    expect(threadButton).toHaveTextContent("1 reply");
    expect(within(threadButton).getByText("View thread")).toHaveClass(
      "thread-summary-label",
    );
    expect(threadButton).toHaveTextContent("New replies");
    expect(threadButton.querySelectorAll(".thread-participant-avatar")).toHaveLength(2);
    expect(threadButton.querySelector(".thread-summary-chevron")).not.toBeNull();
    await user.click(threadButton);
    const context = screen.getByLabelText("Channel context");
    expect(await within(context).findByText("Existing thread reply")).toBeVisible();
    const followThread = within(context).getByRole("button", {
      name: "Unfollow thread",
    });
    expect(followThread).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(followThread).toHaveTextContent("");
    expect(followThread.querySelector("svg")).not.toBeNull();
    expect(followThread).toHaveClass("panel-icon-button", "followed");
    expect(followThread.nextElementSibling).toBe(
      within(context).getByRole("button", { name: "Close thread" }),
    );
    expect(within(context).getByRole("separator", { name: "First unread reply" })).toBeVisible();
    await waitFor(() => expect(advanceThreadReadPosition).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      root.messageId,
      reply.messageId,
      0,
      expect.stringMatching(/^browser-/),
    ));
    await user.click(followThread);
    await waitFor(() => expect(setThreadFollowed).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      root.messageId,
      false,
      expect.any(Number),
      expect.stringMatching(/^browser-/),
    ));
    expect(within(context).getByRole("button", { name: "Follow thread" })).not.toHaveClass(
      "followed",
    );
    const closeThread = within(context).getByRole("button", { name: "Close thread" });
    expect(closeThread).toHaveClass("panel-icon-button");
    expect(closeThread).toHaveTextContent("×");
    expect(closeThread.querySelector("span")).toHaveAttribute("aria-hidden", "true");
    const composer = within(context).getByLabelText("Reply in Wake policy qualification");
    expect(composer).toHaveAttribute("rows", "1");
    expect(composer).toHaveAttribute("placeholder", "Reply…");
    await user.type(composer, "@Kai continue here");
    const sendReply = within(context).getByRole("button", { name: "Send reply" });
    expect(sendReply).toHaveTextContent("");
    expect(sendReply.querySelector("svg")).not.toBeNull();
    await user.click(sendReply);

    await waitFor(() => expect(submitCommand).toHaveBeenCalledOnce());
    expect(submitCommand).toHaveBeenCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      expect.stringMatching(/^browser-/),
      "@Kai continue here",
      null,
      root.messageId,
    );
    await user.click(closeThread);
    expect(within(context).queryByText("Existing thread reply")).toBeNull();
  });

  it("shows only a reply icon for a message without replies", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [{ ...historyMessage, channelId: secondChannelId }],
      throughPosition: 25,
      previousCursor: null,
    });

    render(<App />);
    const replyButton = await screen.findByRole("button", {
      name: "Reply to message",
    });
    expect(replyButton).toHaveTextContent("");
    expect(replyButton.querySelector("svg")).not.toBeNull();
  });

  it("offers monochrome hover actions and lays coloured reactions above replies", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(navigationWithGroup());
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [{
        ...historyMessage,
        channelId: secondChannelId,
        replyCount: 2,
      }],
      throughPosition: 25,
      previousCursor: null,
    });
    vi.mocked(setMessageReaction).mockResolvedValueOnce([
      { count: 1, reactedByViewer: true, reaction: "fire" },
    ]).mockResolvedValueOnce([]);

    render(<App />);
    const actions = await screen.findByRole("group", {
      name: "Actions for message from Kai",
    });
    const reactionAction = within(actions).getByRole("button", {
      name: "Add reaction",
    });
    const replyAction = within(actions).getByRole("button", {
      name: "Reply to message",
    });
    expect(reactionAction.querySelector("svg")).not.toBeNull();
    expect(replyAction.querySelector("svg")).not.toBeNull();

    await user.click(reactionAction);
    expect(screen.getAllByRole("menuitemcheckbox")).toHaveLength(12);
    await user.click(screen.getByRole("menuitemcheckbox", {
      name: "Add Fire reaction",
    }));
    expect(setMessageReaction).toHaveBeenLastCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      historyMessage.messageId,
      "fire",
      true,
    );

    const reactionChip = await screen.findByRole("button", {
      name: "Fire: 1. Remove your reaction",
    });
    const engagement = screen.getByRole("group", { name: "Message engagement" });
    expect(engagement).toHaveClass("message-engagement");
    const threadButton = within(engagement).getByRole("button", {
      name: "Open thread with 2 replies",
    });
    expect(threadButton).toBeVisible();
    const reactions = within(engagement).getByRole("group", {
      name: "Message reactions",
    });
    expect(within(reactions).getByRole("button", {
      name: "Fire: 1. Remove your reaction",
    })).toBe(reactionChip);
    expect(reactions.nextElementSibling).toBe(threadButton);
    await user.click(reactionChip);
    expect(setMessageReaction).toHaveBeenLastCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      historyMessage.messageId,
      "fire",
      false,
    );
  });

  it("shows and dismisses authoritative agent engagement", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(
      navigationWithGroup({ engaged: true }),
    );
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [{ ...historyMessage, channelId: secondChannelId }],
      throughPosition: 25,
      previousCursor: null,
    });

    render(<App />);
    expect(await screen.findByText("Awake in this channel")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Dismiss" }));

    await waitFor(() => expect(dismissChannelAgent).toHaveBeenCalledOnce());
    expect(vi.mocked(dismissChannelAgent).mock.calls[0]?.[1]).toBe(
      "agt_00000000000000000000000000000001",
    );
    expect(await screen.findByText("Available · not engaged")).toBeVisible();
  });

  it("lets a channel owner manage explicit sponsored agent attachments", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    vi.mocked(loadNavigation).mockResolvedValue(
      navigationWithGroup({ role: "owner" }),
    );
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [{ ...historyMessage, channelId: secondChannelId }],
      throughPosition: 25,
      previousCursor: null,
    });

    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: "Manage channel agents" }),
    );
    const dialog = await screen.findByRole("dialog", { name: "Agents" });
    expect(within(dialog).getByText(/sponsored by Daniel/i)).toBeVisible();
    const kai = within(dialog).getByRole("checkbox", { name: /Kai/ });
    expect(kai).toBeChecked();
    await user.click(kai);
    await user.click(within(dialog).getByRole("button", { name: "Save" }));

    await waitFor(() => expect(detachChannelAgent).toHaveBeenCalledOnce());
    expect(vi.mocked(detachChannelAgent).mock.calls[0]?.slice(0, 2)).toEqual([
      { channelId: secondChannelId, token: "existing-session" },
      "agt_00000000000000000000000000000001",
    ]);
    expect(attachChannelAgent).not.toHaveBeenCalled();
  });

  it("groups direct messages and presents human conversations without agent controls", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadTimeline).mockImplementation(async (selectedSession) => ({
      messages: [
        {
          ...historyMessage,
          channelId: selectedSession.channelId,
        },
      ],
      throughPosition: 25,
      previousCursor: null,
    }));
    render(<App />);

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(screen.getByText("Direct messages")).toBeVisible();
    expect(screen.getByRole("button", { name: "Kai" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Scott" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Welcome to Kai" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Scott" }));

    expect(
      (await screen.findAllByRole("heading", { name: "@ Scott" })).length,
    ).toBeGreaterThan(0);
    expect(screen.getByRole("textbox", { name: "Message Scott" })).toBeEnabled();
    expect(screen.getByRole("heading", { name: "Conversation with Scott" })).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Welcome to Scott" })).toBeNull();
    expect(screen.getByText("Messages here are private to you and Scott.")).toBeVisible();
    const directPeople = screen.getByRole("heading", { name: "People" }).closest("section");
    expect(directPeople).not.toBeNull();
    expect(directPeople?.querySelectorAll(".context-person-avatar")).toHaveLength(2);
    expect(await screen.findByRole("button", { name: "Reply to message" })).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Runtime and workspace" })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Run inspector" })).toBeNull();
  });

  it("discovers a human from Direct messages and reuses the canonical conversation", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadTimeline).mockImplementation(async (selectedSession) => ({
      messages: [{ ...historyMessage, channelId: selectedSession.channelId }],
      throughPosition: 25,
      previousCursor: null,
    }));

    render(<App />);
    await screen.findByText("Canonical history is ready.");
    await user.click(screen.getByRole("button", { name: "Start direct message" }));

    const dialog = await screen.findByRole("dialog", { name: "Start a conversation" });
    expect(within(dialog).getByText("@scott")).toBeVisible();
    await user.click(
      within(dialog).getByRole("button", {
        name: "Scott, @scott, conversation started",
      }),
    );

    await waitFor(() => expect(startHumanConversation).toHaveBeenCalledWith(
      "existing-session",
      "wsp_00000000000000000000000000000001",
      "prn_00000000000000000000000000000003",
    ));
    expect(screen.queryByRole("dialog", { name: "Start a conversation" })).toBeNull();
    expect(await screen.findByRole("textbox", { name: "Message Scott" })).toBeEnabled();
    expect(window.location.search).toBe("");
  });

  it("clears a failed people picker when it is closed and reopened", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadWorkshopHumans)
      .mockRejectedValueOnce(new Error("People are temporarily unavailable."))
      .mockResolvedValueOnce([
        {
          conversationChannelId: null,
          displayName: "Scott",
          handle: "scott",
          principalId: "prn_00000000000000000000000000000003",
        },
      ]);

    render(<App />);
    await screen.findByText("Canonical history is ready.");
    await user.click(screen.getByRole("button", { name: "Start direct message" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "People are temporarily unavailable.",
    );
    await user.click(screen.getByRole("button", { name: "Close people picker" }));
    expect(screen.queryByRole("dialog", { name: "Start a conversation" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Start direct message" }));
    expect(
      await screen.findByRole("button", { name: "Scott, @scott" }),
    ).toBeVisible();
    expect(screen.queryByText("People are temporarily unavailable.")).toBeNull();
  });

  it("explains that an archived agent conversation is permanently read-only", async () => {
    const user = userEvent.setup();
    const archivedDefinition: WorkshopAgentDefinition = {
      ...agentDefinition,
      lifecycleState: "archived",
    };
    const archivedNavigation: WorkshopNavigation = {
      ...navigation,
      workshops: navigation.workshops.map((workshop) => ({
        ...workshop,
        channels: workshop.channels.map((channel) =>
          channel.channelId === channelId
            ? {
                ...channel,
                agents: channel.agents.map((agent) => ({
                  ...agent,
                  lifecycleState: "archived",
                })),
                canSubmitCommands: false,
              }
            : channel,
        ),
      })),
    };
    vi.mocked(loadNavigation).mockResolvedValue(archivedNavigation);
    vi.mocked(loadAgentDefinitions).mockResolvedValue([archivedDefinition]);
    vi.mocked(loadAgentEnablements).mockResolvedValue([]);
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );

    render(<App />);

    expect(
      await screen.findByText(
        "This agent has been archived. This conversation is read-only.",
      ),
    ).toBeVisible();
    expect(
      screen.queryByText(
        "Sending messages from Workshop is not available for this conversation yet.",
      ),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: "Kai" })).toBeNull();

    await user.click(
      screen.getByRole("button", { name: "Your drafts and archived agents" }),
    );
    const archive = await screen.findByRole("dialog", { name: "Drafts and archive" });
    expect(within(archive).getByText("Kai")).toBeVisible();
    const openArchivedConversation = within(archive).getByRole("button", {
      name: "Open archived conversation with Kai",
    });
    expect(openArchivedConversation).toBeVisible();
    expect(
      within(archive).getByRole("button", { name: "View archived agent Kai" }),
    ).toBeVisible();
    await user.click(openArchivedConversation);
    expect(screen.queryByRole("dialog", { name: "Drafts and archive" })).toBeNull();
    expect(
      await screen.findByText(
        "This agent has been archived. This conversation is read-only.",
      ),
    ).toBeVisible();
  });

  it("gives a nonowner access to an archived conversation only through the agent archive", async () => {
    const archivedNavigation: WorkshopNavigation = {
      ...navigation,
      principal: {
        displayName: "Scott",
        handle: "scott",
        principalId: "prn_99999999999999999999999999999999",
      },
      workshops: navigation.workshops.map((workshop) => ({
        ...workshop,
        channels: workshop.channels.map((channel) =>
          channel.channelId === channelId
            ? {
                ...channel,
                agents: channel.agents.map((agent) => ({
                  ...agent,
                  lifecycleState: "archived",
                  sponsorDisplayName: "Daniel",
                })),
                canSubmitCommands: false,
              }
            : channel,
        ),
      })),
    };
    vi.mocked(loadNavigation).mockResolvedValue(archivedNavigation);
    vi.mocked(loadAgentDefinitions).mockResolvedValue([]);
    vi.mocked(loadAgentEnablements).mockResolvedValue([]);
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );

    render(<App />);

    await screen.findByText("Canonical history is ready.");
    expect(screen.queryByRole("button", { name: "Kai" })).toBeNull();
    await userEvent.setup().click(
      screen.getByRole("button", { name: "Your drafts and archived agents" }),
    );
    const archive = await screen.findByRole("dialog", { name: "Drafts and archive" });
    expect(
      within(archive).getByRole("button", {
        name: "Open archived conversation with Kai",
      }),
    ).toBeVisible();
    expect(
      within(archive).queryByRole("button", { name: "View archived agent Kai" }),
    ).toBeNull();
  });

  it("opens agent management from the sidebar and starts a direct conversation", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);

    await screen.findByText("Canonical history is ready.");
    expect(screen.queryByRole("button", { name: "Browse agents" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Create agent" }));
    expect(
      await screen.findByRole("heading", { name: "Create agent", level: 2 }),
    ).toBeVisible();
    expect(window.location.search).toBe("?view=agents&new=1");
    const creationDialog = screen.getByRole("dialog", { name: "Create agent" });
    const closeCreation = within(creationDialog).getByRole("button", {
      name: "Close agent creation",
    });
    expect(closeCreation).toHaveClass("panel-icon-button");
    expect(closeCreation).toHaveAttribute("title", "Close agent creation");
    expect(closeCreation.querySelector("span")).toHaveAttribute("aria-hidden", "true");
    expect(within(creationDialog).queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(
      within(creationDialog).getByLabelText(/Display name/).previousElementSibling,
    ).toHaveClass("agent-field-hint-placeholder");
    await user.click(closeCreation);
    expect(screen.queryByRole("dialog", { name: "Create agent" })).toBeNull();
    expect(window.location.search).toBe("?view=agents");
    const manageKai = screen.getByRole("button", { name: "Manage Kai" });
    await user.click(manageKai);

    expect(await screen.findByRole("heading", { name: "Agents", level: 1 })).toBeVisible();
    expect(within(manageKai).getByLabelText("Open")).toHaveClass("live-pip");
    expect(screen.getByRole("heading", { name: "Kai", level: 2 })).toBeVisible();
    expect(screen.queryByText("Owner runtime")).toBeNull();
    expect(screen.queryByText("Runtime active")).toBeNull();
    expect(screen.queryByLabelText("Authorized runtime")).toBeNull();
    expect(screen.getByText("You own and manage this agent.")).toBeVisible();
    const agentWorkspace = within(screen.getByLabelText("Agents workspace"));
    const createAgent = agentWorkspace.getByRole("button", {
      name: "Create agent",
    });
    const closeAgents = agentWorkspace.getByRole("button", {
      name: "Close agents",
    });
    expect(createAgent).toBeVisible();
    expect(closeAgents).toBeVisible();
    expect(createAgent).toHaveClass("panel-icon-button");
    expect(closeAgents).toHaveClass("panel-icon-button");
    expect(createAgent.parentElement).toBe(closeAgents.parentElement);
    expect(createAgent.querySelector("span")).toHaveAttribute("aria-hidden", "true");
    expect(closeAgents.querySelector("span")).toHaveAttribute("aria-hidden", "true");
    expect(streamPrincipalEvents).toHaveBeenCalledTimes(1);

    const startConversation = screen.getByRole("button", {
      name: "Start conversation with Kai",
    });
    expect(startConversation).toHaveClass("agent-conversation-button");
    expect(startConversation.querySelector("svg")).toBeInTheDocument();
    await user.click(startConversation);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(window.location.search).toBe("");
  });

  it("uses the sole eligible execution profile without exposing a redundant selector", async () => {
    const user = userEvent.setup();
    vi.mocked(loadAgentEnablements).mockResolvedValue([
      {
        ...agentEnablement,
        directChannelId: null,
        enablementId: null,
        lifecycleState: "available",
        runtimeProfileId: null,
        stateVersion: null,
      },
    ]);
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);

    await screen.findByText("Canonical history is ready.");
    await user.click(screen.getByRole("button", { name: "Manage Kai" }));

    expect(
      await screen.findByText(
        "Your authorized execution profile will be used automatically.",
      ),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Enable agent" })).toBeVisible();
    expect(screen.queryByText("Owner runtime")).toBeNull();
    expect(screen.queryByText("Advanced execution profile")).toBeNull();
    expect(screen.queryByLabelText("Execution profile")).toBeNull();
  });

  it("keeps multiple eligible execution profiles behind an advanced disclosure", async () => {
    const user = userEvent.setup();
    vi.mocked(loadAgentEnablements).mockResolvedValue([
      {
        ...agentEnablement,
        directChannelId: null,
        eligibleRuntimes: [
          ...agentEnablement.eligibleRuntimes,
          {
            backendOptions: ["codex:openai"],
            displayName: "Daniel's alternate runtime",
            runtimeProfileId: "rtp_22222222222222222222222222222222",
          },
        ],
        enablementId: null,
        lifecycleState: "available",
        runtimeProfileId: null,
        stateVersion: null,
      },
    ]);
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);

    await screen.findByText("Canonical history is ready.");
    await user.click(screen.getByRole("button", { name: "Manage Kai" }));

    const disclosure = await screen.findByText("Advanced execution profile");
    expect(disclosure).toBeVisible();
    await user.click(disclosure);
    expect(screen.getByLabelText("Execution profile")).toHaveValue(runtimeProfileId);
    expect(screen.getByRole("option", { name: "Daniel's runtime" })).toBeVisible();
    expect(
      screen.getByRole("option", { name: "Daniel's alternate runtime" }),
    ).toBeVisible();
    expect(screen.queryByRole("option", { name: /claude|codex/i })).toBeNull();
  });

  it("offers principal-owned agent creation to Workshop members", async () => {
    const user = userEvent.setup();
    vi.mocked(loadNavigation).mockResolvedValue({
      ...navigation,
      workshops: navigation.workshops.map((workshop) => ({
        ...workshop,
        role: "member",
      })),
    });
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);

    await screen.findByText("Canonical history is ready.");
    const sidebar = within(screen.getByLabelText("Workshop navigation"));
    const createAgent = sidebar.getByRole("button", { name: "Create agent" });
    expect(createAgent).toBeVisible();

    await user.click(createAgent);
    const creationDialog = await screen.findByRole("dialog", { name: "Create agent" });
    await user.type(within(creationDialog).getByLabelText(/Stable handle/), "draft_agent");
    await user.click(within(creationDialog).getByRole("button", {
      name: "Close agent creation",
    }));
    const confirmation = screen.getByRole("dialog", { name: "Continue?" });
    expect(confirmation).toHaveTextContent("Discard your unsaved agent changes?");
    await user.click(within(confirmation).getByRole("button", { name: "Cancel" }));
    expect(screen.getByRole("dialog", { name: "Create agent" })).toBeVisible();
    await user.click(within(creationDialog).getByRole("button", {
      name: "Close agent creation",
    }));
    await user.click(within(screen.getByRole("dialog", { name: "Continue?" })).getByRole(
      "button",
      { name: "Continue" },
    ));
    expect(screen.queryByRole("dialog", { name: "Create agent" })).toBeNull();
    expect(window.location.search).toBe("?view=agents");
  });

  it("keeps active agents in the sidebar and owner drafts in the inactive browser", async () => {
    const user = userEvent.setup();
    vi.mocked(loadNavigation).mockResolvedValue({
      ...navigation,
      workshops: navigation.workshops.map((workshop) => ({
        ...workshop,
        role: "member",
      })),
    });
    const qualificationDefinition: WorkshopAgentDefinition = {
      ...agentDefinition,
      activeRevisionId: "adr_55555555555555555555555555555555",
      agentId: "agt_55555555555555555555555555555555",
      definitionId: "adf_55555555555555555555555555555555",
      displayName: "Qualification agent",
      handle: "qualification_agent",
      ownerDisplayName: "Scott",
      ownerPrincipalId: "prn_99999999999999999999999999999999",
      revisions: agentDefinition.revisions.map((revision) => ({
        ...revision,
        revisionId: "adr_55555555555555555555555555555555",
      })),
    };
    const qualificationEnablement: WorkshopAgentEnablement = {
      ...agentEnablement,
      agentId: qualificationDefinition.agentId,
      canManage: false,
      definitionId: qualificationDefinition.definitionId,
      directChannelId: null,
      displayName: qualificationDefinition.displayName,
      eligibleRuntimes: [],
      enablementId: null,
      handle: qualificationDefinition.handle,
      lifecycleState: "available",
      runtimeProfileId: null,
      stateVersion: null,
    };
    const archivedDefinition: WorkshopAgentDefinition = {
      ...qualificationDefinition,
      activeRevisionId: "adr_66666666666666666666666666666666",
      agentId: "agt_66666666666666666666666666666666",
      definitionId: "adf_66666666666666666666666666666666",
      displayName: "Archived specialist",
      handle: "archived_specialist",
      lifecycleState: "archived",
      ownerDisplayName: "Daniel",
      ownerPrincipalId: navigation.principal.principalId,
      revisions: qualificationDefinition.revisions.map((revision) => ({
        ...revision,
        revisionId: "adr_66666666666666666666666666666666",
      })),
    };
    const draftDefinition: WorkshopAgentDefinition = {
      ...qualificationDefinition,
      activeRevisionId: null,
      agentId: "agt_77777777777777777777777777777777",
      definitionId: "adf_77777777777777777777777777777777",
      displayName: "Daniel draft",
      handle: "daniel_draft",
      lifecycleState: "draft",
      ownerDisplayName: "Daniel",
      ownerPrincipalId: navigation.principal.principalId,
      revisions: qualificationDefinition.revisions.map((revision) => ({
        ...revision,
        revisionId: "adr_77777777777777777777777777777777",
      })),
    };
    vi.mocked(loadAgentDefinitions).mockResolvedValue([
      agentDefinition,
      qualificationDefinition,
      archivedDefinition,
      draftDefinition,
    ]);
    vi.mocked(loadAgentEnablements).mockResolvedValue([
      { ...agentEnablement, canManage: false },
      qualificationEnablement,
    ]);
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);

    await screen.findByText("Canonical history is ready.");
    const sidebar = within(screen.getByLabelText("Workshop navigation"));
    expect(sidebar.getByRole("button", { name: "Kai" })).toBeVisible();
    expect(sidebar.queryByRole("button", { name: "Qualification agent" })).toBeNull();
    expect(sidebar.getByRole("button", { name: "Manage Kai" })).toBeVisible();
    expect(
      sidebar.getByRole("button", { name: "Manage Qualification agent" }),
    ).toBeVisible();
    expect(sidebar.queryByText("conversation started")).toBeNull();
    expect(sidebar.queryByText("available")).toBeNull();
    expect(
      sidebar.queryByRole("button", { name: "Manage Archived specialist" }),
    ).toBeNull();

    await user.click(
      sidebar.getByRole("button", { name: "Manage Qualification agent" }),
    );
    expect(
      await screen.findByRole("heading", { name: "Qualification agent", level: 2 }),
    ).toBeVisible();
    expect(screen.queryByLabelText("Agent catalogue")).toBeNull();
    expect(screen.getByText("Unavailable to this Workshop account")).toBeVisible();

    await user.click(
      sidebar.getByRole("button", { name: "Your drafts and archived agents" }),
    );
    let archive = await screen.findByRole("dialog", { name: "Drafts and archive" });
    expect(within(archive).getByText("Archived specialist")).toBeVisible();
    expect(within(archive).getByText("@archived_specialist · archived")).toBeVisible();
    expect(within(archive).getByText("Daniel draft")).toBeVisible();
    expect(within(archive).getByText("@daniel_draft · draft")).toBeVisible();
    await user.click(
      within(archive).getByRole("button", {
        name: "View draft agent Daniel draft",
      }),
    );
    expect(
      await screen.findByRole("heading", { name: "Daniel draft", level: 2 }),
    ).toBeVisible();
    expect(screen.getByText("You own and manage this agent.")).toBeVisible();
    expect(screen.getByText("Owner controls")).toBeVisible();
    expect(screen.getByRole("button", { name: "Activate" })).toBeVisible();

    await user.click(
      sidebar.getByRole("button", { name: "Your drafts and archived agents" }),
    );
    archive = await screen.findByRole("dialog", { name: "Drafts and archive" });
    await user.click(
      within(archive).getByRole("button", {
        name: "View archived agent Archived specialist",
      }),
    );
    expect(
      await screen.findByRole("heading", { name: "Archived specialist", level: 2 }),
    ).toBeVisible();
    expect(screen.getByText(/This definition is archived/)).toBeVisible();
  });

  it("starts a nonowner conversation without exposing enablement plumbing", async () => {
    const user = userEvent.setup();
    const accessRuntimeProfileId = "rtp_22222222222222222222222222222222";
    const available: WorkshopAgentEnablement = {
      ...agentEnablement,
      canManage: false,
      directChannelId: null,
      eligibleRuntimes: [
        {
          backendOptions: ["codex:openai"],
          displayName: "Scott's runtime",
          runtimeProfileId: accessRuntimeProfileId,
        },
      ],
      enablementId: null,
      lifecycleState: "available",
      ownerPrincipalId: agentDefinition.ownerPrincipalId,
      ownerRuntimeProfileId: runtimeProfileId,
      runtimeProfileId: null,
      stateVersion: null,
    };
    const enabled: WorkshopAgentEnablement = {
      ...available,
      directChannelId: channelId,
      enablementId: "aen_22222222222222222222222222222222",
      lifecycleState: "enabled",
      runtimeProfileId: accessRuntimeProfileId,
      stateVersion: 1,
    };
    const started = {
      ...enabled,
      conversationStarted: true,
      stateVersion: 2,
    };
    vi.mocked(loadNavigation).mockResolvedValue({
      ...navigation,
      principal: {
        displayName: "Scott",
        handle: "scott",
        principalId: "prn_99999999999999999999999999999999",
      },
    });
    let conversationProvisioned = false;
    vi.mocked(loadAgentEnablements).mockImplementation(async () => [
      conversationProvisioned ? started : available,
    ]);
    vi.mocked(enableAgentDefinition).mockImplementation(async () => {
      conversationProvisioned = true;
      return enabled;
    });
    vi.mocked(startAgentConversation).mockResolvedValue(started);
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);

    await screen.findByText("Canonical history is ready.");
    await user.click(screen.getByRole("button", { name: "Manage Kai" }));

    expect(screen.queryByText("Enable conversation")).toBeNull();
    expect(screen.queryByText("Conversation access")).toBeNull();
    await user.click(
      await screen.findByRole("button", { name: "Start conversation with Kai" }),
    );

    await waitFor(() => expect(enableAgentDefinition).toHaveBeenCalledOnce());
    expect(enableAgentDefinition).toHaveBeenCalledWith(
      "existing-session",
      definitionId,
      expect.objectContaining({
        expectedVersion: null,
        runtimeProfileId: accessRuntimeProfileId,
      }),
    );
    expect(startAgentConversation).toHaveBeenCalledWith(
      "existing-session",
      definitionId,
      expect.objectContaining({ expectedVersion: 1 }),
    );
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect(window.location.search).toBe("");
  });

  it("shows one shared agent definition without owner controls to another principal", async () => {
    const user = userEvent.setup();
    const danielArchivedDefinition: WorkshopAgentDefinition = {
      ...agentDefinition,
      activeRevisionId: null,
      agentId: "agt_88888888888888888888888888888888",
      definitionId: "adf_88888888888888888888888888888888",
      displayName: "Daniel archived agent",
      handle: "daniel_archived_agent",
      lifecycleState: "archived",
    };
    vi.mocked(loadNavigation).mockResolvedValue({
      ...navigation,
      principal: {
        displayName: "Scott",
        handle: "scott",
        principalId: "prn_99999999999999999999999999999999",
      },
      workshops: navigation.workshops.map((workshop) => ({
        ...workshop,
        role: "member",
      })),
    });
    vi.mocked(loadAgentEnablements).mockResolvedValue([
      {
        ...agentEnablement,
        canManage: false,
        directChannelId: secondChannelId,
        eligibleRuntimes: [],
        ownerPrincipalId: agentDefinition.ownerPrincipalId,
        ownerRuntimeProfileId: runtimeProfileId,
        runtimeProfileId: "rtp_22222222222222222222222222222222",
      },
    ]);
    vi.mocked(loadAgentDefinitions).mockResolvedValue([
      agentDefinition,
      danielArchivedDefinition,
    ]);
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);

    await screen.findByText("Canonical history is ready.");
    await user.click(screen.getByRole("button", { name: "Manage Kai" }));

    expect(await screen.findByText("Owned and managed by Daniel.")).toBeVisible();
    expect(screen.queryByText("Conversation access")).toBeNull();
    expect(
      screen.getByRole("button", { name: "Start conversation with Kai" }),
    ).toBeVisible();
    expect(screen.queryByText("Owner controls")).toBeNull();
    expect(screen.queryByText("Runtime active")).toBeNull();
    expect(screen.queryByLabelText("Authorized runtime")).toBeNull();
    expect(screen.queryByRole("button", { name: "New revision" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Disable" })).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Your drafts and archived agents" }),
    ).toBeNull();
  });

  it("opens runtime settings for the exact enabled agent lane", async () => {
    const user = userEvent.setup();
    const qualificationChannelId = "chn_44444444444444444444444444444444";
    const qualificationDefinition: WorkshopAgentDefinition = {
      ...agentDefinition,
      agentId: "agt_44444444444444444444444444444444",
      definitionId: "adf_44444444444444444444444444444444",
      displayName: "Qualification agent",
      handle: "qualification",
      revisions: agentDefinition.revisions.map((revision) => ({
        ...revision,
        revisionId: "adr_44444444444444444444444444444444",
      })),
      activeRevisionId: "adr_44444444444444444444444444444444",
    };
    const qualificationEnablement: WorkshopAgentEnablement = {
      ...agentEnablement,
      agentId: qualificationDefinition.agentId,
      definitionId: qualificationDefinition.definitionId,
      directChannelId: qualificationChannelId,
      displayName: qualificationDefinition.displayName,
      enablementId: "aen_44444444444444444444444444444444",
      handle: qualificationDefinition.handle,
    };
    const qualificationChannel: WorkshopNavigation["workshops"][number]["channels"][number] = {
      ...navigation.workshops[0].channels[0],
      agents: [
        {
          agentId: qualificationDefinition.agentId,
          available: true,
          engaged: false,
          engagedUntil: null,
          handle: qualificationDefinition.handle,
          lifecycleState: "active",
          memoryScope: "private",
          name: qualificationDefinition.displayName,
          principalId: "prn_44444444444444444444444444444444",
          runtimeProfileId: qualificationEnablement.runtimeProfileId,
          sponsorDisplayName: navigation.principal.displayName,
          sponsorPrincipalId: navigation.principal.principalId,
        },
      ],
      channelId: qualificationChannelId,
      name: "Qualification agent",
      participants: [
        {
          displayName: qualificationDefinition.displayName,
          handle: qualificationDefinition.handle,
          kind: "agent",
          principalId: "prn_44444444444444444444444444444444",
        },
      ],
    };
    vi.mocked(loadNavigation).mockResolvedValue({
      ...navigation,
      workshops: [
        {
          ...navigation.workshops[0],
          channels: [
            ...navigation.workshops[0].channels,
            qualificationChannel,
          ],
        },
      ],
    });
    vi.mocked(loadAgentDefinitions).mockResolvedValue([
      agentDefinition,
      qualificationDefinition,
    ]);
    vi.mocked(loadAgentEnablements).mockResolvedValue([
      agentEnablement,
      qualificationEnablement,
    ]);
    vi.mocked(loadSettingsWorkspace).mockImplementation(async (selectedSession) => ({
      ...settingsWorkspace,
      channelId: selectedSession.channelId,
    }));
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );

    const { unmount } = render(<App />);
    await screen.findByText("Canonical history is ready.");
    await user.click(
      screen.getByRole("button", { name: "Manage Qualification agent" }),
    );
    expect(
      await screen.findByRole("heading", { name: "Qualification agent", level: 2 }),
    ).toBeVisible();
    expect(
      await screen.findByRole("heading", {
        name: "Runtime settings",
        level: 2,
      }),
    ).toBeVisible();
    expect(screen.queryByText("Personal preferences")).toBeNull();
    expect(screen.queryByRole("button", { name: "Back to agent" })).toBeNull();
    expect(screen.queryByText("Your runtime")).toBeNull();
    expect(screen.queryByRole("heading", { name: "Runtime and workspace" })).toBeNull();
    expect(screen.getByText(/These controls apply only to Qualification agent/)).toBeVisible();
    await waitFor(() => expect(loadSettingsWorkspace).toHaveBeenCalledWith({
      channelId: qualificationChannelId,
      token: "existing-session",
    }));
    expect(window.location.search).toBe(
      `?view=agents&agent=${qualificationDefinition.definitionId}`,
    );

    expect(screen.queryByRole("button", { name: "Runtime settings" })).toBeNull();

    unmount();
    window.history.replaceState(
      null,
      "",
      `?view=agents&agent=${qualificationDefinition.definitionId}` +
        "&section=runtime",
    );
    vi.mocked(loadSettingsWorkspace).mockClear();
    render(<App />);
    expect(
      await screen.findByRole("heading", {
        name: "Runtime settings",
        level: 2,
      }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Qualification agent", level: 2 }),
    ).toBeVisible();
    await waitFor(() => expect(loadSettingsWorkspace).toHaveBeenCalledWith({
      channelId: qualificationChannelId,
      token: "existing-session",
    }));

    unmount();
    window.history.replaceState(
      null,
      "",
      `?view=settings&runtime=${qualificationChannelId}` +
        `&agent=${qualificationDefinition.definitionId}`,
    );
    vi.mocked(loadSettingsWorkspace).mockClear();
    render(<App />);
    expect(
      await screen.findByRole("heading", {
        name: "Runtime settings",
        level: 2,
      }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Qualification agent", level: 2 }),
    ).toBeVisible();
    await waitFor(() => expect(loadSettingsWorkspace).toHaveBeenCalledWith({
      channelId: qualificationChannelId,
      token: "existing-session",
    }));
  });

  it("collapses the navigation to labeled icons and restores its layout", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    const navigationPanel = screen.getByLabelText("Workshop navigation");
    await user.click(screen.getByRole("button", { name: "Collapse navigation" }));

    expect(navigationPanel).toHaveClass("collapsed");
    expect(screen.getByRole("button", { name: "Kai" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Scott" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Create channel" })).toBeNull();
    expect(sessionStorage.getItem("kai.workshop.sidebar-layout.v4")).toContain(
      '"collapsed":true',
    );

    await user.click(screen.getByRole("button", { name: "Expand navigation" }));
    expect(navigationPanel).not.toHaveClass("collapsed");
    expect(screen.getByRole("button", { name: "Create channel" })).toBeVisible();
  });

  it("resizes navigation with an accessible separator", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const { container } = render(<App />);

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    const resizeHandle = screen.getByRole("separator", { name: "Resize navigation" });
    expect(container.querySelector(".workshop-app")).toHaveStyle(
      "--channel-sidebar-width: 264px",
    );
    fireEvent.keyDown(resizeHandle, { key: "ArrowRight" });

    expect(container.querySelector(".workshop-app")).toHaveStyle(
      "--channel-sidebar-width: 288px",
    );
    expect(resizeHandle).toHaveAttribute("aria-valuenow", "288");
  });

  it("resizes the context pane with an accessible separator", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const { container } = render(<App />);

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    const resizeHandle = screen.getByRole("separator", { name: "Resize channel context" });
    expect(container.querySelector(".workshop-app")).toHaveStyle(
      "--context-pane-width: 360px",
    );
    // The pane sits on the right, so ArrowLeft moves the separator left
    // and widens it.
    fireEvent.keyDown(resizeHandle, { key: "ArrowLeft" });

    expect(container.querySelector(".workshop-app")).toHaveStyle(
      "--context-pane-width: 384px",
    );
    expect(resizeHandle).toHaveAttribute("aria-valuenow", "384");
    expect(
      JSON.parse(sessionStorage.getItem("kai.workshop.context-layout.v4") ?? "null"),
    ).toEqual({ width: 384 });
  });

  it("submits over LAN HTTP and reuses the command identity on retry", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const getRandomValues = vi.fn((array: Uint8Array): Uint8Array => {
      array.fill(0x2a);
      return array;
    });
    vi.stubGlobal("crypto", { getRandomValues });
    vi.mocked(submitCommand)
      .mockRejectedValueOnce(new Error("Backend temporarily unavailable."))
      .mockResolvedValueOnce({
        acceptance: "ready_replay",
        messageId: "msg_00000000000000000000000000000030",
        run: completedRun,
      });
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    await user.type(screen.getByLabelText("Message Kai"), "Hello from Workshop");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Backend temporarily unavailable.",
    );
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(submitCommand).toHaveBeenCalledTimes(2));
    expect(vi.mocked(submitCommand).mock.calls[0]).toEqual([
      { channelId, token: "existing-session" },
      "browser-2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a",
      "Hello from Workshop",
      null,
      null,
    ]);
    expect(vi.mocked(submitCommand).mock.calls[1]).toEqual(
      vi.mocked(submitCommand).mock.calls[0],
    );
    expect(getRandomValues).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText("Message Kai")).toHaveValue("");
  });

  it("submits a file-only command and clears the selected attachment on success", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const getRandomValues = vi.fn((array: Uint8Array): Uint8Array => {
      array.fill(0x3b);
      return array;
    });
    vi.stubGlobal("crypto", { getRandomValues });
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    const artifact = new File(["artifact body"], "notes.txt", {
      type: "text/plain",
    });
    const attachButton = screen.getByRole("button", { name: "Attach" });
    const sendButton = screen.getByRole("button", { name: "Send" });
    expect(attachButton).toBeEnabled();
    expect(attachButton).toHaveClass("attach-button");
    expect(attachButton).toHaveTextContent("");
    expect(attachButton.querySelector("svg")).not.toBeNull();
    expect(sendButton).toBeDisabled();
    expect(sendButton).toHaveClass("send-button");
    expect(sendButton).toHaveTextContent("");
    expect(sendButton.querySelector("svg")).not.toBeNull();

    await user.upload(screen.getByLabelText("Attach a file"), artifact);
    expect(screen.getByText("notes.txt")).toBeVisible();
    expect(sendButton).toBeEnabled();
    await user.click(sendButton);

    await waitFor(() => expect(submitCommand).toHaveBeenCalledOnce());
    expect(submitCommand).toHaveBeenCalledWith(
      { channelId, token: "existing-session" },
      "browser-3b3b3b3b3b3b3b3b3b3b3b3b3b3b3b3b",
      "",
      artifact,
      null,
    );
    expect(screen.queryByText("notes.txt")).toBeNull();
  });

  it("renders canonical artifact metadata from the timeline", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadTimeline).mockResolvedValueOnce({
      messages: [
        {
          ...historyMessage,
          artifacts: [
            {
              artifactId: "art_00000000000000000000000000000001",
              byteSize: 1200,
              contentSha256: "a".repeat(64),
              createdAt: "2026-08-13T09:00:00Z",
              kind: "document",
              mediaType: "text/plain",
              originalFilename: "workshop-notes.txt",
            },
          ],
        },
      ],
      throughPosition: 25,
      previousCursor: null,
    });

    render(<App />);

    expect(await screen.findByText("workshop-notes.txt")).toBeVisible();
    expect(screen.getByText("2 KB")).toBeVisible();
    expect(screen.getByRole("button", { name: "Download" })).toBeVisible();
  });

  it("sends the draft on Enter and keeps Shift+Enter as a newline", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    // Enter on an empty composer must not submit; the send path rejects
    // blank drafts regardless of how submission is triggered.
    const composer = screen.getByLabelText("Message Kai");
    await user.click(composer);
    await user.keyboard("{Enter}");
    expect(submitCommand).not.toHaveBeenCalled();

    // Shift+Enter stays a plain newline inside the draft.
    await user.type(composer, "first line{Shift>}{Enter}{/Shift}second line");
    expect(composer).toHaveValue("first line\nsecond line");
    expect(submitCommand).not.toHaveBeenCalled();

    // Enter during an IME composition, and the WebKit variant that reports
    // the composition-confirming Enter with the legacy 229 keyCode after
    // composition has ended, must not send the draft.
    fireEvent.keyDown(composer, { key: "Enter", isComposing: true });
    fireEvent.keyDown(composer, { key: "Enter", keyCode: 229 });
    expect(submitCommand).not.toHaveBeenCalled();

    // A bare Enter sends the full multi-line draft and clears the composer.
    await user.keyboard("{Enter}");
    await waitFor(() => expect(submitCommand).toHaveBeenCalledTimes(1));
    expect(vi.mocked(submitCommand).mock.calls[0]?.[2]).toBe(
      "first line\nsecond line",
    );
    expect(composer).toHaveValue("");
  });

  it("accepts work asynchronously and exposes an exact run Stop control", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const acceptedRun: WorkshopRun = {
      ...completedRun,
      resultMessageId: null,
      startedAt: null,
      status: "accepted",
      terminalAt: null,
    };
    vi.mocked(submitCommand).mockResolvedValueOnce({
      acceptance: "newly_accepted",
      messageId: "msg_00000000000000000000000000000030",
      run: acceptedRun,
    });
    vi.mocked(cancelRun).mockResolvedValueOnce({
      ...acceptedRun,
      cancellationRequestedAt: "2026-08-13T09:00:01Z",
      status: "cancelled",
      terminalAt: "2026-08-13T09:00:01Z",
      terminalCode: "requested_by_human",
    });
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    await user.type(screen.getByLabelText("Message Kai"), "Take your time");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("accepted")).toBeVisible();
    expect(screen.getByLabelText("Message Kai")).toHaveValue("");
    await user.click(screen.getByRole("button", { name: "Stop" }));
    expect(await screen.findByText("cancelled")).toBeVisible();
    expect(cancelRun).toHaveBeenCalledWith(
      { channelId, token: "existing-session" },
      acceptedRun.runId,
    );
  });

  it("updates run activity from the live stream without polling run state", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const acceptedRun: WorkshopRun = {
      ...completedRun,
      resultMessageId: null,
      startedAt: null,
      status: "accepted",
      terminalAt: null,
    };
    vi.mocked(submitCommand).mockResolvedValueOnce({
      acceptance: "newly_accepted",
      messageId: "msg_00000000000000000000000000000030",
      run: acceptedRun,
    });
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    await user.type(screen.getByLabelText("Message Kai"), "Inspect the event stream");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Queued for the configured agent.")).toBeVisible();

    const startedRun: WorkshopRun = {
      ...acceptedRun,
      routingDecision: {
        backend: "opencode",
        decidedAt: "2026-08-13T09:00:00Z",
        disposition: "routed",
        evidenceVersion: 1,
        model: "deepseek-chat",
        policyRevision: 1,
        provider: "deepseek",
        reasonCode: "configured_route_eligible",
        requestedBackendOptionId: "opencode:deepseek",
        requestedTaskClass: "coding",
        selectedBackendOptionId: "opencode:deepseek",
      },
      startedAt: "2026-08-13T09:00:01Z",
      status: "started",
    };
    act(() =>
      handlers?.onRunActivity(
        {
          eventPosition: 31,
          occurredAt: "2026-08-13T09:00:01Z",
          run: startedRun,
          transition: "run.started",
        },
        "31",
      ),
    );
    expect(await screen.findByText("The agent is working on this request.")).toBeVisible();
    expect(screen.queryByText(/Route: routed/)).toBeNull();

    const completedRoutedRun: WorkshopRun = {
      ...completedRun,
      routingDecision: startedRun.routingDecision,
    };
    act(() =>
      handlers?.onRunActivity(
        {
          eventPosition: 32,
          occurredAt: "2026-08-13T09:00:02Z",
          run: completedRoutedRun,
          transition: "run.completed",
        },
        "32",
      ),
    );
    expect(await screen.findByText("The agent completed this request.")).toBeVisible();
    expect(screen.queryByText(/Route: routed/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
    expect(loadRun).not.toHaveBeenCalled();
  });

  it("keeps a streamed terminal state when it arrives before command acceptance", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const acceptedRun: WorkshopRun = {
      ...completedRun,
      resultMessageId: null,
      startedAt: null,
      status: "accepted",
      terminalAt: null,
    };
    let resolveSubmission:
      | ((result: Awaited<ReturnType<typeof submitCommand>>) => void)
      | null = null;
    vi.mocked(submitCommand).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveSubmission = resolve;
        }),
    );
    render(<App />);
    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();

    await user.type(screen.getByLabelText("Message Kai"), "Finish very quickly");
    const submitting = user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(submitCommand).toHaveBeenCalledOnce());
    const sendingButton = screen.getByRole("button", { name: "Sending…" });
    expect(sendingButton).toBeDisabled();
    expect(sendingButton).toHaveAttribute("aria-busy", "true");
    act(() =>
      handlers?.onRunActivity(
        {
          eventPosition: 32,
          occurredAt: "2026-08-13T09:00:02Z",
          run: completedRun,
          transition: "run.completed",
        },
        "32",
      ),
    );
    act(() =>
      resolveSubmission?.({
        acceptance: "newly_accepted",
        messageId: "msg_00000000000000000000000000000030",
        run: acceptedRun,
      }),
    );
    await submitting;

    expect(await screen.findByText("The agent completed this request.")).toBeVisible();
    expect(screen.queryByText("Queued for the configured agent.")).toBeNull();
  });
});
