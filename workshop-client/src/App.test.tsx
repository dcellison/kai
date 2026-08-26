import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  AuthenticationError,
  cancelRun,
  ChannelAccessError,
  loadEarlierTimeline,
  loadArtifactBlob,
  loadNavigation,
  loadMemoryDetail,
  loadMemoryRecords,
  loadMemorySource,
  loadMemoryStats,
  loadRun,
  loadRunTrace,
  loadSettingsWorkspace,
  loadTimeline,
  redeemEnrollment,
  streamTimeline,
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
    loadEarlierTimeline: vi.fn(),
    loadArtifactBlob: vi.fn(),
    loadNavigation: vi.fn(),
    loadMemoryDetail: vi.fn(),
    loadMemoryRecords: vi.fn(),
    loadMemorySource: vi.fn(),
    loadMemoryStats: vi.fn(),
    loadTimeline: vi.fn(),
    loadRun: vi.fn(),
    loadRunTrace: vi.fn(),
    loadSettingsWorkspace: vi.fn(),
    redeemEnrollment: vi.fn(),
    streamTimeline: vi.fn(),
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
          agents: [{ agentId: "agt_00000000000000000000000000000001", name: "Kai" }],
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
          agents: [{ agentId: "agt_00000000000000000000000000000001", name: "Kai" }],
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
const historyMessage: TimelineMessage = {
  artifacts: [],
  authorDisplayName: "Kai",
  authorKind: "agent",
  body: "Canonical history is ready.",
  channelId,
  createdAt: "2026-08-13T09:00:00Z",
  eventPosition: 25,
  messageId: "msg_00000000000000000000000000000025",
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
  channelId,
  model: { source: "runtime policy", value: "gpt-5.6-sol" },
  modelOptions: [
    { displayName: "GPT-5.6 Sol", modelId: "gpt-5.6-sol" },
  ],
  principalId: "prn_00000000000000000000000000000001",
  provider: "openai",
  runtimeProfileId: "rtp_00000000000000000000000000000001",
  timeoutSeconds: { source: "runtime policy", value: 120 },
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
    sessionStorage.clear();
    window.history.replaceState(null, "", "/workshop/");
    handlers = null;
    failStream = null;
    vi.mocked(redeemEnrollment).mockResolvedValue("redeemed-session-token");
    vi.mocked(loadNavigation).mockResolvedValue(navigation);
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [historyMessage],
      throughPosition: 25,
      previousCursor: null,
    });
    vi.mocked(loadRun).mockResolvedValue(completedRun);
    vi.mocked(loadRunTrace).mockResolvedValue({ entries: [], hasMore: false });
    vi.mocked(loadSettingsWorkspace).mockResolvedValue(settingsWorkspace);
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
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toContain(
      "redeemed-session-token",
    );

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
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toBeNull();
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
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toContain(
      "existing-session",
    );
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toContain(
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
    expect(document.querySelector(".notification-row")).not.toBeNull();
    expect(screen.queryByLabelText("Message Kai")).toBeNull();
    expect(screen.getByText(/outbound-only/)).toBeVisible();
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toContain(
      notificationChannelId,
    );

    await user.click(screen.getByRole("button", { name: "Kai" }));
    expect(await screen.findByText(`History for ${channelId}`)).toBeVisible();
    expect(screen.getByLabelText("Message Kai")).toHaveValue("Keep this draft");
    expect(redeemEnrollment).not.toHaveBeenCalled();
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
    expect(sessionStorage.getItem("kai.workshop.sidebar-layout.v1")).toContain(
      '"collapsed":true',
    );

    await user.click(screen.getByRole("button", { name: "Expand navigation" }));
    expect(navigationPanel).not.toHaveClass("collapsed");
  });

  it("resizes navigation with an accessible separator", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const { container } = render(<App />);

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    const resizeHandle = screen.getByRole("separator", { name: "Resize navigation" });
    fireEvent.keyDown(resizeHandle, { key: "ArrowRight" });

    expect(container.querySelector(".workshop-app")).toHaveStyle(
      "--channel-sidebar-width: 280px",
    );
    expect(resizeHandle).toHaveAttribute("aria-valuenow", "280");
  });

  it("resizes the context pane with an accessible separator", async () => {
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    const { container } = render(<App />);

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    const resizeHandle = screen.getByRole("separator", { name: "Resize channel context" });
    // The pane sits on the right, so ArrowLeft moves the separator left
    // and widens it.
    fireEvent.keyDown(resizeHandle, { key: "ArrowLeft" });

    expect(container.querySelector(".workshop-app")).toHaveStyle(
      "--context-pane-width: 320px",
    );
    expect(resizeHandle).toHaveAttribute("aria-valuenow", "320");
    expect(
      JSON.parse(sessionStorage.getItem("kai.workshop.context-layout.v1") ?? "null"),
    ).toEqual({ width: 320 });
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
    expect(sendButton).toBeDisabled();
    expect(sendButton).toHaveClass("send-button");

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
    expect(await screen.findByText("The agent completed this request.")).toBeVisible();
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
