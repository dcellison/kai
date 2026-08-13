import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  AuthenticationError,
  ChannelAccessError,
  loadTimeline,
  redeemEnrollment,
  streamTimeline,
  submitCommand,
} from "./api";
import type { TimelineMessage } from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    loadTimeline: vi.fn(),
    redeemEnrollment: vi.fn(),
    streamTimeline: vi.fn(),
    submitCommand: vi.fn(),
  };
});

const channelId = "chn_d3dfdfd7df9151ba8a1742b92403faa5";
const historyMessage: TimelineMessage = {
  authorDisplayName: "Kai",
  authorKind: "agent",
  body: "Canonical history is ready.",
  channelId,
  createdAt: "2026-08-13T09:00:00Z",
  eventPosition: 25,
  messageId: "msg_00000000000000000000000000000025",
};

type StreamHandlers = Parameters<typeof streamTimeline>[2];

describe("Workshop React client", () => {
  let handlers: StreamHandlers | null;
  let failStream: ((reason: Error) => void) | null;

  beforeEach(() => {
    sessionStorage.clear();
    handlers = null;
    failStream = null;
    vi.mocked(redeemEnrollment).mockResolvedValue("redeemed-session-token");
    vi.mocked(loadTimeline).mockResolvedValue({
      messages: [historyMessage],
      throughPosition: 25,
    });
    vi.mocked(submitCommand).mockResolvedValue({
      acceptance: "newly_accepted",
      execution: "completed",
      messageId: "msg_00000000000000000000000000000030",
      runId: "run_00000000000000000000000000000030",
      runStatus: "completed",
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
    await user.type(screen.getByLabelText("Channel ID"), channelId);
    await user.click(screen.getByRole("button", { name: "Open channel" }));

    expect(await screen.findByText("Canonical history is ready.")).toBeVisible();
    expect((await screen.findAllByText("Live")).length).toBeGreaterThanOrEqual(1);
    expect(redeemEnrollment).toHaveBeenCalledWith(
      "one-time-token",
      "Workshop browser",
    );
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

  it("preserves an enrolled session while correcting channel access", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem(
      "kai.workshop.read-session.v1",
      JSON.stringify({ channelId, token: "existing-session" }),
    );
    vi.mocked(loadTimeline).mockRejectedValueOnce(
      new ChannelAccessError("Channel access changed."),
    );
    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Channel access changed.",
    );
    expect(screen.queryByLabelText("Enrollment token")).toBeNull();
    expect(screen.getByLabelText("Channel ID")).toHaveValue(channelId);
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toContain(
      "existing-session",
    );

    await user.click(screen.getByRole("button", { name: "Forget session" }));
    await waitFor(() =>
      expect(screen.getByLabelText("Enrollment token")).toBeVisible(),
    );
    expect(sessionStorage.getItem("kai.workshop.read-session.v1")).toBeNull();
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
        execution: "completed",
        messageId: "msg_00000000000000000000000000000030",
        runId: "run_00000000000000000000000000000030",
        runStatus: "completed",
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
});
