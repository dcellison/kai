import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  AuthenticationError,
  loadMemoryDetail,
  loadMemoryRecords,
  loadMemorySource,
  loadMemoryStats,
  searchMemories,
} from "./api";
import { MemoryExplorer } from "./MemoryExplorer";
import type {
  WorkshopMemoryDetail,
  WorkshopMemoryRecord,
  WorkshopMemorySourceContext,
} from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    loadMemoryDetail: vi.fn(),
    loadMemoryRecords: vi.fn(),
    loadMemorySource: vi.fn(),
    loadMemoryStats: vi.fn(),
    searchMemories: vi.fn(),
  };
});

function record(
  memoryId: string,
  preview: string,
  kind: "fact" | "episode" = "fact",
): WorkshopMemoryRecord {
  return {
    confidence: 1,
    createdAt: "2026-08-24T10:00:00Z",
    kind,
    memoryId,
    memoryType: kind,
    preview,
    scope: {
      exclusionReason: null,
      invalidDefaulted: false,
      legacyDefaulted: false,
      projectId: kind === "episode" ? "kai" : null,
      retrievable: true,
      scope: kind === "episode" ? "project" : "global",
      scopeConfidence: 1,
      scopeSource: "operator",
    },
    source: kind === "episode" ? "episode" : "extracted",
    speaker: kind === "episode" ? "episode_summary" : "user",
    tags: kind === "episode" ? ["deployment"] : ["preference"],
    updatedAt: kind === "episode"
      ? "2026-08-24T11:00:00Z"
      : "2026-08-24T10:00:00Z",
  };
}

function detail(item: WorkshopMemoryRecord): WorkshopMemoryDetail {
  return {
    ...item,
    compactRecall: `{"record_type":"memory","memory_id":"${item.memoryId}"}`,
    confirmationQuote: null,
    content: `${item.preview}\n\n<script>window.bad = true</script>`,
    episode: item.kind === "episode"
      ? { goal: "Deploy Kai", outcome: "Succeeded" }
      : null,
    promptVersion: "v1",
  };
}

const source: WorkshopMemorySourceContext = {
  reason: null,
  result: {
    authorDisplayName: "Kai",
    authorKind: "agent",
    authorPrincipalId: "prn_00000000000000000000000000000002",
    body: "Deployment completed.",
    channelId: "chn_00000000000000000000000000000001",
    createdAt: "2026-08-24T10:01:00Z",
    messageId: "msg_00000000000000000000000000000002",
  },
  runId: "run_00000000000000000000000000000001",
  source: {
    authorDisplayName: "Daniel",
    authorKind: "human",
    authorPrincipalId: "prn_00000000000000000000000000000001",
    body: "Please deploy Kai.",
    channelId: "chn_00000000000000000000000000000001",
    createdAt: "2026-08-24T10:00:00Z",
    messageId: "msg_00000000000000000000000000000001",
  },
  status: "available",
};

describe("Workshop Memory explorer", () => {
  const fact = record("memory-1", "Daniel prefers concise output.");
  const episode = record("memory-2", "Kai deployment episode", "episode");

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadMemoryStats).mockResolvedValue({
      byScope: { global: 1, "project:kai": 1 },
      bySource: { episode: 1, extracted: 1 },
      byType: { episode: 1, fact: 1 },
      episodes: 1,
      facts: 1,
      total: 2,
    });
    vi.mocked(loadMemoryRecords).mockResolvedValue({
      nextCursor: null,
      records: [episode, fact],
    });
    vi.mocked(loadMemoryDetail).mockImplementation(async (_token, memoryId) =>
      detail(memoryId === episode.memoryId ? episode : fact)
    );
    vi.mocked(loadMemorySource).mockResolvedValue(source);
    vi.mocked(searchMemories).mockResolvedValue({
      activeProjectId: "kai",
      hits: [{ adjustedScore: 0.8, compactRecall: "{}", rawScore: 0.9, record: fact }],
      reason: "ok",
    });
  });

  it("browses rich detail, source context, and the distinct compact recall safely", async () => {
    const onSelectMemory = vi.fn();
    const { container } = render(
      <MemoryExplorer
        initialMemoryId={null}
        onAuthenticationFailure={vi.fn()}
        onClose={vi.fn()}
        onForget={vi.fn()}
        onSelectMemory={onSelectMemory}
        token="session-secret"
      />,
    );

    expect(await screen.findByRole("heading", { name: "Memory", level: 1 })).toBeVisible();
    expect(screen.getByText("2")).toBeVisible();
    expect(await screen.findByText("Kai deployment episode")).toBeVisible();
    expect(await screen.findByText("Episode structure")).toBeVisible();
    expect(screen.getByText("Deploy Kai")).toBeVisible();
    expect(screen.getByText("Agent recall preview")).toBeVisible();
    expect(screen.getByText("Please deploy Kai.")).toBeVisible();
    expect(container.querySelector("script")).toBeNull();
    expect(screen.getByText("<script>window.bad = true</script>")).toBeVisible();
    expect(onSelectMemory).toHaveBeenCalledWith("memory-2");
  });

  it("combines filters, server-backed browse order, search, and paging", async () => {
    const user = userEvent.setup();
    vi.mocked(loadMemoryRecords)
      .mockResolvedValueOnce({ nextCursor: "page-2", records: [episode] })
      .mockResolvedValueOnce({ nextCursor: null, records: [fact] })
      .mockResolvedValue({ nextCursor: "page-2", records: [episode] });
    render(
      <MemoryExplorer
        initialMemoryId={null}
        onAuthenticationFailure={vi.fn()}
        onClose={vi.fn()}
        onForget={vi.fn()}
        onSelectMemory={vi.fn()}
        token="session-secret"
      />,
    );

    expect(await screen.findByText("Kai deployment episode")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Load more memories" }));
    expect(await screen.findByText("Daniel prefers concise output.")).toBeVisible();
    expect(loadMemoryRecords).toHaveBeenCalledWith(
      "session-secret",
      expect.objectContaining({ cursor: "page-2", order: "newest" }),
    );

    await user.selectOptions(screen.getByLabelText("Kind"), "episode");
    await user.selectOptions(screen.getByLabelText("Project"), "kai");
    await user.type(screen.getByLabelText("Tag"), "deployment");
    await user.click(screen.getByRole("button", { name: "Apply filters" }));
    await waitFor(() => expect(loadMemoryRecords).toHaveBeenLastCalledWith(
      "session-secret",
      expect.objectContaining({
        kind: "episode",
        projectId: "kai",
        scope: "project",
        tag: "deployment",
      }),
    ));

    await user.selectOptions(screen.getByLabelText("Sort"), "oldest");
    await waitFor(() => expect(loadMemoryRecords).toHaveBeenLastCalledWith(
      "session-secret",
      expect.objectContaining({ order: "oldest" }),
    ));

    await user.type(screen.getByLabelText("Search memories"), "concise output");
    await user.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(searchMemories).toHaveBeenCalledWith(
      "session-secret",
      "concise output",
      expect.objectContaining({ kind: "episode", projectId: "kai", limit: 50 }),
    ));
  });

  it("supports keyboard selection and explains unavailable canonical sources", async () => {
    const user = userEvent.setup();
    vi.mocked(loadMemorySource).mockResolvedValue({
      reason: "legacy_source",
      result: null,
      runId: null,
      source: null,
      status: "unavailable",
    });
    render(
      <MemoryExplorer
        initialMemoryId={episode.memoryId}
        onAuthenticationFailure={vi.fn()}
        onClose={vi.fn()}
        onForget={vi.fn()}
        onSelectMemory={vi.fn()}
        token="session-secret"
      />,
    );

    const first = await screen.findByRole("option", { name: /Kai deployment episode/ });
    first.focus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("option", { name: /Daniel prefers concise output/ })).toHaveFocus();
    expect(await screen.findByText("This memory predates canonical source links.")).toBeVisible();
  });

  it("reports loading failures with retry and expires the existing session boundary", async () => {
    const user = userEvent.setup();
    const onAuthenticationFailure = vi.fn();
    vi.mocked(loadMemoryRecords)
      .mockRejectedValueOnce(new Error("Memory service unavailable"))
      .mockRejectedValueOnce(new AuthenticationError("Session expired"));
    render(
      <MemoryExplorer
        initialMemoryId={null}
        onAuthenticationFailure={onAuthenticationFailure}
        onClose={vi.fn()}
        onForget={vi.fn()}
        onSelectMemory={vi.fn()}
        token="session-secret"
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("Memory service unavailable");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(onAuthenticationFailure).toHaveBeenCalledWith("Session expired"));
  });
});
