import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  EventStreamDecoder,
  cancelRun,
  loadRun,
  loadTimeline,
  redeemEnrollment,
  submitCommand,
  streamTimeline,
} from "./api";
import type { WorkshopSession } from "./types";

const channelId = "chn_d3dfdfd7df9151ba8a1742b92403faa5";
const session: WorkshopSession = { channelId, token: "session-secret" };

function message(position: number, body = `Message ${position}`): Record<string, unknown> {
  return {
    author_display_name: position % 2 ? "Daniel" : "Kai",
    author_kind: position % 2 ? "human" : "agent",
    body,
    channel_id: channelId,
    created_at: "2026-08-13T09:00:00Z",
    event_position: position,
    message_id: `msg_${position.toString().padStart(32, "0")}`,
  };
}

function run(status = "accepted"): Record<string, unknown> {
  return {
    accepted_at: "2026-08-13T09:00:00Z",
    cancellation_requested_at: null,
    channel_id: channelId,
    result_message_id: null,
    run_id: "run_00000000000000000000000000000001",
    started_at: null,
    status,
    terminal_at: null,
    terminal_code: null,
  };
}

describe("Workshop client API", () => {
  beforeEach(() => vi.unstubAllGlobals());

  it("redeems an enrollment grant without putting credentials in the URL", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        version: 1,
        session: { token: "redeemed-session" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(redeemEnrollment("one-time-token", "Daniel's Mini")).resolves.toBe(
      "redeemed-session",
    );
    expect(fetchMock).toHaveBeenCalledOnce();
    const [path, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/v1/client/enrollment/redeem");
    expect(options.method).toBe("POST");
    expect(path).not.toContain("one-time-token");
    expect(JSON.parse(options.body as string)).toEqual({
      device_display_name: "Daniel's Mini",
      enrollment_token: "one-time-token",
    });
  });

  it("loads every page from one stable canonical snapshot", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        Response.json({
          version: 1,
          channel_id: channelId,
          messages: [message(10)],
          next_cursor: "next-page",
          through_position: 20,
        }),
      )
      .mockResolvedValueOnce(
        Response.json({
          version: 1,
          channel_id: channelId,
          messages: [message(20)],
          next_cursor: null,
          through_position: 20,
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const snapshot = await loadTimeline(session, new AbortController().signal);

    expect(snapshot.throughPosition).toBe(20);
    expect(snapshot.messages.map((item) => item.eventPosition)).toEqual([10, 20]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const firstRequest = fetchMock.mock.calls[0] as [string, RequestInit];
    const secondRequest = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(new Headers(firstRequest[1].headers).get("Authorization")).toBe(
      "Bearer session-secret",
    );
    expect(secondRequest[0]).toContain("cursor=next-page");
  });

  it("submits only the opaque id and body under bearer authority", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        version: 2,
        acceptance: "newly_accepted",
        message_id: "msg_00000000000000000000000000000001",
        run_id: "run_00000000000000000000000000000001",
        run: run(),
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      submitCommand(session, "browser-command-1", "Hello from Workshop"),
    ).resolves.toEqual({
      acceptance: "newly_accepted",
      messageId: "msg_00000000000000000000000000000001",
      run: {
        acceptedAt: "2026-08-13T09:00:00Z",
        cancellationRequestedAt: null,
        channelId,
        resultMessageId: null,
        runId: "run_00000000000000000000000000000001",
        startedAt: null,
        status: "accepted",
        terminalAt: null,
        terminalCode: null,
      },
    });
    const [path, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe(`/v1/channels/${channelId}/commands`);
    expect(new Headers(options.headers).get("Authorization")).toBe(
      "Bearer session-secret",
    );
    expect(JSON.parse(options.body as string)).toEqual({
      body: "Hello from Workshop",
      client_message_id: "browser-command-1",
    });
  });

  it("inspects and cancels a run through channel-scoped routes", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ version: 1, run: run("started") }))
      .mockResolvedValueOnce(Response.json({ version: 1, run: run("cancelled") }));
    vi.stubGlobal("fetch", fetchMock);
    const runId = "run_00000000000000000000000000000001";

    await expect(loadRun(session, runId)).resolves.toMatchObject({
      runId,
      status: "started",
    });
    await expect(cancelRun(session, runId)).resolves.toMatchObject({
      runId,
      status: "cancelled",
    });

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      `/v1/channels/${channelId}/runs/${runId}`,
    );
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      `/v1/channels/${channelId}/runs/${runId}/cancel`,
    );
    expect((fetchMock.mock.calls[1]?.[1] as RequestInit).method).toBe("POST");
  });

  it("decodes fragmented event-stream blocks", () => {
    const decoder = new EventStreamDecoder();
    expect(decoder.push("id: 42\nevent: timeline.")).toEqual([]);
    expect(decoder.push("message.created\ndata: {\"version\":1}\n\n")).toEqual([
      {
        data: '{"version":1}',
        eventId: "42",
        eventName: "timeline.message.created",
      },
    ]);
  });

  it("resumes live messages with authorization and Last-Event-ID", async () => {
    const rawMessage = message(31, "Live update");
    const event = [
      "id: 31",
      "event: timeline.message.created",
      `data: ${JSON.stringify({ version: 1, channel_id: channelId, message: rawMessage })}`,
      "",
      "",
    ].join("\n");
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(event));
        controller.close();
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(new Response(stream, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const onConnected = vi.fn();
    const onMessage = vi.fn();
    const onRunActivity = vi.fn();

    await streamTimeline(
      session,
      "30",
      { onConnected, onMessage, onRunActivity },
      new AbortController().signal,
    );

    expect(onConnected).toHaveBeenCalledOnce();
    expect(onMessage).toHaveBeenCalledWith(
      expect.objectContaining({ body: "Live update", eventPosition: 31 }),
      "31",
    );
    const request = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = new Headers(request[1].headers);
    expect(headers.get("Authorization")).toBe("Bearer session-secret");
    expect(headers.get("Last-Event-ID")).toBe("30");
  });

  it("receives authoritative run lifecycle activity on the same stream", async () => {
    const rawRun = {
      ...run("started"),
      started_at: "2026-08-13T09:00:01Z",
    };
    const event = [
      "id: 32",
      "event: run.lifecycle.changed",
      `data: ${JSON.stringify({
        version: 1,
        channel_id: channelId,
        event_position: 32,
        occurred_at: "2026-08-13T09:00:01Z",
        transition: "run.started",
        run: rawRun,
      })}`,
      "",
      "",
    ].join("\n");
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(event));
        controller.close();
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(stream, { status: 200 })),
    );
    const onRunActivity = vi.fn();

    await streamTimeline(
      session,
      "31",
      { onConnected: vi.fn(), onMessage: vi.fn(), onRunActivity },
      new AbortController().signal,
    );

    expect(onRunActivity).toHaveBeenCalledWith(
      {
        eventPosition: 32,
        occurredAt: "2026-08-13T09:00:01Z",
        transition: "run.started",
        run: expect.objectContaining({ status: "started" }),
      },
      "32",
    );
  });
});
