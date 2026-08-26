import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  AuthenticationError,
  MemoryRevisionConflictError,
  createMemoryFact,
  deleteMemories,
  deleteMemory,
  editMemory,
  loadMemoryDetail,
  loadMemoryRecords,
  loadMemorySource,
  loadMemoryStats,
  moveMemoriesScope,
  moveMemoryScope,
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
    moveMemoryScope: vi.fn(),
    moveMemoriesScope: vi.fn(),
    deleteMemory: vi.fn(),
    deleteMemories: vi.fn(),
    createMemoryFact: vi.fn(),
    editMemory: vi.fn(),
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
    revision: `mr1_${memoryId}`,
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
      ? {
          actors: ["Daniel", "Kai"],
          approach: "Install and verify.",
          context: "A production qualification.",
          goal: "Deploy Kai",
          lessons: null,
          outcome: "Succeeded",
          outcomeQuality: "success",
          tags: ["deployment"],
        }
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
      allowedProjects: [{ displayName: "Kai", projectId: "kai" }],
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
    vi.mocked(moveMemoryScope).mockResolvedValue({
      operation: "move_scope",
      results: [{ memoryId: "memory-1", outcome: "succeeded", priorScope: null, newScope: null }],
    });
    vi.mocked(moveMemoriesScope).mockResolvedValue({
      operation: "move_scope",
      results: [
        { memoryId: "memory-1", outcome: "succeeded", priorScope: null, newScope: null },
        { memoryId: "memory-2", outcome: "failed", priorScope: null, newScope: null },
      ],
    });
    vi.mocked(deleteMemory).mockResolvedValue({
      operation: "delete",
      results: [{ memoryId: "memory-1", outcome: "succeeded", priorScope: null, newScope: null }],
    });
    vi.mocked(deleteMemories).mockResolvedValue({ operation: "delete", results: [] });
    vi.mocked(createMemoryFact).mockResolvedValue({
      created: true,
      record: detail(fact),
    });
    vi.mocked(editMemory).mockResolvedValue({
      changedFields: ["content", "tags"],
      idempotentReplay: false,
      record: detail(fact),
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

  it("requires confirmation, supports cancellation, and reports partial bulk outcomes", async () => {
    const user = userEvent.setup();
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

    await screen.findByText("Kai deployment episode");
    await user.click(screen.getByRole("checkbox", { name: /Select Kai deployment episode/ }));
    await user.click(screen.getByRole("checkbox", { name: /Select Daniel prefers concise output/ }));
    await user.selectOptions(screen.getByLabelText("Move selected memories to"), "project:kai");
    await user.click(screen.getByRole("button", { name: "Move selected…" }));
    const dialog = screen.getByRole("dialog", { name: "Move 2 memories?" });
    expect(dialog).toHaveTextContent("project kai");
    expect(dialog).toHaveTextContent("Daniel prefers concise output.");

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(moveMemoriesScope).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Move selected…" }));
    await user.click(screen.getByRole("button", { name: "Confirm move" }));
    await waitFor(() => expect(moveMemoriesScope).toHaveBeenCalledWith(
      "session-secret",
      expect.arrayContaining(["memory-1", "memory-2"]),
      { scope: "project", projectId: "kai" },
    ));
    expect(await screen.findByRole("status")).toHaveTextContent("1 succeeded, 1 failed");
    expect(screen.getByText("1 selected")).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Forget memory…" }));
    expect(screen.getByRole("dialog", { name: "Forget 1 memory?" })).toHaveTextContent(
      "permanently removed",
    );
  });

  it("creates an explicit fact with typed content, tags, and project scope", async () => {
    const user = userEvent.setup();
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

    await screen.findByText("Kai deployment episode");
    await user.click(screen.getByRole("button", { name: "Add fact…" }));
    const dialog = screen.getByRole("dialog", { name: "Create fact" });
    await user.type(screen.getByLabelText(/Content/), "A deliberately explicit fact.");
    await user.type(screen.getByLabelText(/Tags/), "qualification, explicit");
    await user.selectOptions(screen.getByLabelText("Recall scope"), "project:kai");
    await user.click(screen.getByRole("button", { name: "Create memory" }));

    await waitFor(() => expect(createMemoryFact).toHaveBeenCalledWith(
      "session-secret",
      expect.objectContaining({
        content: "A deliberately explicit fact.",
        tags: ["qualification", "explicit"],
        target: { scope: "project", projectId: "kai" },
      }),
    ));
    expect(await screen.findByRole("status")).toHaveTextContent("Explicit memory created.");
  });

  it("edits a fact explicitly and requires confirmation before discarding dirty fields", async () => {
    const user = userEvent.setup();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(
      <MemoryExplorer
        initialMemoryId={fact.memoryId}
        onAuthenticationFailure={vi.fn()}
        onClose={vi.fn()}
        onForget={vi.fn()}
        onSelectMemory={vi.fn()}
        token="session-secret"
      />,
    );

    await screen.findByText("Manage memory");
    await user.click(screen.getByRole("button", { name: "Edit memory…" }));
    const content = screen.getByLabelText(/Content/);
    await user.clear(content);
    await user.type(content, "Corrected semantic wording.");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(confirm).toHaveBeenCalledOnce();
    expect(screen.getByRole("dialog", { name: "Correct fact" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Save correction" }));
    await waitFor(() => expect(editMemory).toHaveBeenCalledWith(
      "session-secret",
      expect.objectContaining({
        kind: "fact",
        memoryId: fact.memoryId,
        revision: fact.revision,
        content: "Corrected semantic wording.",
      }),
    ));
    expect(screen.queryByRole("dialog", { name: "Correct fact" })).not.toBeInTheDocument();
  });

  it("keeps conflicting episode edits open and offers a latest-revision reload", async () => {
    const user = userEvent.setup();
    vi.mocked(editMemory).mockRejectedValue(
      new MemoryRevisionConflictError("Memory changed since it was opened", "mr1_current"),
    );
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

    await screen.findByText("Episode structure");
    await user.click(screen.getByRole("button", { name: "Edit memory…" }));
    await user.clear(screen.getByLabelText(/Goal/));
    await user.type(screen.getByLabelText(/Goal/), "Corrected deployment goal");
    await user.click(screen.getByRole("button", { name: "Save correction" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("changed after you opened it");
    expect(screen.getByRole("button", { name: "Reload latest" })).toBeVisible();
  });
});
