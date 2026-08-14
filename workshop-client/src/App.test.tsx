import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  AuthenticationError,
  cancelRun,
  ChannelAccessError,
  loadNavigation,
  loadRun,
  loadTimeline,
  redeemEnrollment,
  streamTimeline,
  submitCommand,
} from "./api";
import type {
  TimelineMessage,
  WorkshopNavigation,
  WorkshopRun,
} from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    cancelRun: vi.fn(),
    loadNavigation: vi.fn(),
    loadTimeline: vi.fn(),
    loadRun: vi.fn(),
    redeemEnrollment: vi.fn(),
    streamTimeline: vi.fn(),
    submitCommand: vi.fn(),
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
  authorDisplayName: "Kai",
  authorKind: "agent",
  body: "Canonical history is ready.",
  channelId,
  createdAt: "2026-08-13T09:00:00Z",
  eventPosition: 25,
  messageId: "msg_00000000000000000000000000000025",
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

type StreamHandlers = Parameters<typeof streamTimeline>[2];

describe("Workshop React client", () => {
  let handlers: StreamHandlers | null;
  let failStream: ((reason: Error) => void) | null;

  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
    handlers = null;
    failStream = null;
    vi.mocked(redeemEnrollment).mockResolvedValue("redeemed-session-token");
    vi.mocked(loadNavigation).mockResolvedValue(navigation);
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [historyMessage],
      throughPosition: 25,
    });
    vi.mocked(loadRun).mockResolvedValue(completedRun);
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

  it("opens at the latest message and preserves deliberate scroll position", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    let resolveTimeline: ((value: {
      messages: TimelineMessage[];
      throughPosition: number;
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
      resolveTimeline?.({ messages: [historyMessage], throughPosition: 25 });
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
    ]);
    expect(vi.mocked(submitCommand).mock.calls[1]).toEqual(
      vi.mocked(submitCommand).mock.calls[0],
    );
    expect(getRandomValues).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText("Message Kai")).toHaveValue("");
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
