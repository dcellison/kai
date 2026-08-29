import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  AuthenticationError,
  cancelRun,
  ChannelAccessError,
  createChannel,
  dismissChannelAgent,
  loadAppearancePreferences,
  loadEarlierTimeline,
  loadArtifactBlob,
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
  loadWorkspaceConfig,
  redeemEnrollment,
  streamTimeline,
  setMessageReaction,
  submitCommand,
  switchWorkspace,
} from "./api";
import type {
  WorkshopMemoryRecord,
  TimelineMessage,
  TimelineSnapshot,
  WorkshopNavigation,
  WorkshopRun,
  WorkshopSettingsWorkspace,
} from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    cancelRun: vi.fn(),
    createChannel: vi.fn(),
    dismissChannelAgent: vi.fn(),
    loadEarlierTimeline: vi.fn(),
    loadArtifactBlob: vi.fn(),
    loadAppearancePreferences: vi.fn(),
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
    loadRun: vi.fn(),
    loadRunTrace: vi.fn(),
    loadSettingsWorkspace: vi.fn(),
    loadWorkspaceConfig: vi.fn(),
    redeemEnrollment: vi.fn(),
    streamTimeline: vi.fn(),
    setMessageReaction: vi.fn(),
    submitCommand: vi.fn(),
    switchWorkspace: vi.fn(),
  };
});

const channelId = "chn_d3dfdfd7df9151ba8a1742b92403faa5";
const notificationChannelId = "chn_11111111111111111111111111111111";
const secondChannelId = "chn_22222222222222222222222222222222";
const humanDirectChannelId = "chn_33333333333333333333333333333333";
const navigation: WorkshopNavigation = {
  principal: {
    displayName: "Daniel",
    principalId: "prn_00000000000000000000000000000001",
  },
  workshops: [
    {
      channels: [
        {
          agents: [
            {
              agentId: "agt_00000000000000000000000000000001",
              engaged: false,
              engagedUntil: null,
              name: "Kai",
              principalId: "prn_00000000000000000000000000000002",
            },
          ],
          canSubmitCommands: true,
          channelId,
          kind: "direct",
          name: "Conversation",
          participants: [
            {
              displayName: "Kai",
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
              engaged: false,
              engagedUntil: null,
              name: "Kai",
              principalId: "prn_00000000000000000000000000000002",
            },
          ],
          canSubmitCommands: false,
          channelId: notificationChannelId,
          kind: "notification",
          name: "GitHub notifications",
          participants: [
            {
              displayName: "Kai",
              kind: "agent",
              principalId: "prn_00000000000000000000000000000002",
            },
          ],
          role: "participant",
        },
        {
          agents: [],
          canSubmitCommands: false,
          channelId: humanDirectChannelId,
          kind: "direct",
          name: "Direct",
          participants: [
            {
              displayName: "Scott",
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
}: {
  engaged?: boolean;
  name?: string;
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
            })),
            channelId: secondChannelId,
            kind: "group",
            name,
            role: "participant",
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
  replyToMessageId: null,
  latestReplyAt: null,
  threadRootId: null,
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

describe("Workshop React client", () => {
  let handlers: StreamHandlers | null;
  let failStream: ((reason: Error) => void) | null;

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    sessionStorage.clear();
    window.history.replaceState(null, "", "/workshop/");
    handlers = null;
    failStream = null;
    vi.mocked(redeemEnrollment).mockResolvedValue("redeemed-session-token");
    vi.mocked(createChannel).mockResolvedValue(secondChannelId);
    vi.mocked(dismissChannelAgent).mockResolvedValue(undefined);
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
  });

  afterEach(() => {
    vi.unstubAllGlobals();
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
    expect(screen.getByLabelText("Backend")).toHaveValue("codex:openai");
    expect(screen.getByRole("option", { name: "codex · openai" })).toBeVisible();
    expect(screen.queryByText("PROTECTED_KEY")).not.toBeInTheDocument();
    expect(screen.queryByText(settingsWorkspace.principalId)).not.toBeInTheDocument();
    expect(screen.queryByText(settingsWorkspace.runtimeProfileId)).not.toBeInTheDocument();
    expect(screen.queryByText(settingsWorkspace.workspace)).not.toBeInTheDocument();
    expect(window.location.search).toBe("?view=settings");

    await user.click(screen.getByRole("button", { name: "Back to conversation" }));
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
    await user.click(screen.getByRole("option", { name: "@Kai — agent" }));
    expect(composer).toHaveValue("@Kai ");
    await user.type(composer, "reply plainly{Enter}");

    await waitFor(() => expect(submitCommand).toHaveBeenCalledOnce());
    expect(vi.mocked(submitCommand).mock.calls[0]?.[2]).toBe(
      "@Kai reply plainly",
    );
  });

  it("opens a group-message thread and submits replies against its canonical root", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId: secondChannelId, token: "existing-session" }),
    );
    const root = {
      ...historyMessage,
      channelId: secondChannelId,
      replyCount: 1,
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
    vi.mocked(loadThreadTimeline).mockResolvedValue({
      root,
      messages: [reply],
      nextCursor: null,
      throughPosition: 26,
    });

    render(<App />);
    const threadButton = await screen.findByRole("button", {
      name: "Open thread with 1 reply",
    });
    expect(threadButton).toHaveTextContent("1 reply");
    expect(threadButton.querySelector("svg")).toBeNull();
    await user.click(threadButton);
    const context = screen.getByLabelText("Channel context");
    expect(await within(context).findByText("Existing thread reply")).toBeVisible();
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

  it("offers monochrome hover actions and toggles a canonical reaction", async () => {
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
    vi.mocked(setMessageReaction).mockResolvedValueOnce([
      { count: 1, reactedByViewer: true, reaction: "eyes" },
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
    await user.click(screen.getByRole("menuitemcheckbox", {
      name: "Add Eyes reaction",
    }));
    expect(setMessageReaction).toHaveBeenLastCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      historyMessage.messageId,
      "eyes",
      true,
    );

    const reactionChip = await screen.findByRole("button", {
      name: "Eyes: 1. Remove your reaction",
    });
    await user.click(reactionChip);
    expect(setMessageReaction).toHaveBeenLastCalledWith(
      { channelId: secondChannelId, token: "existing-session" },
      historyMessage.messageId,
      "eyes",
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
    expect(await screen.findByText("Not engaged")).toBeVisible();
  });

  it("groups direct messages and names agent and human conversations by participant", async () => {
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

    await user.click(screen.getByRole("button", { name: "Scott" }));

    expect(
      (await screen.findAllByRole("heading", { name: "@ Scott" })).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getByText(
        "Sending messages from Workshop is not available for this conversation yet.",
      ),
    ).toBeVisible();
    expect(screen.queryByText("Agents")).toBeNull();
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
