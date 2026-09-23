import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  MemoryConflictChangedError,
  loadForgottenMemories,
  loadMemoryConflict,
  loadMemoryConflicts,
  resolveMemoryConflict,
  restoreForgottenMemory,
} from "./api";
import { FactReview } from "./FactReview";
import type { WorkshopMemoryLifecycle, WorkshopMemoryLifecycleRevision } from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    loadForgottenMemories: vi.fn(),
    loadMemoryConflict: vi.fn(),
    loadMemoryConflicts: vi.fn(),
    resolveMemoryConflict: vi.fn(),
    restoreForgottenMemory: vi.fn(),
  };
});

function revision(
  revisionId: string,
  content: string,
  storedAt: string,
  overrides: Partial<WorkshopMemoryLifecycleRevision> = {},
): WorkshopMemoryLifecycleRevision {
  return {
    admissionAuthority: "provenance_verified",
    confidence: 0.95,
    content,
    model: "claude-sonnet-5",
    provider: "anthropic",
    reason: "Extracted from conversation",
    revisionId,
    state: "unresolved_conflict",
    storedAt,
    ...overrides,
  };
}

function lifecycle(revisions: WorkshopMemoryLifecycleRevision[]): WorkshopMemoryLifecycle {
  return {
    authority: "canonical",
    createdAt: "2026-09-22T10:00:00Z",
    currentRevisionId: null,
    currentState: "unresolved_conflict",
    events: [],
    followups: [],
    identity: "mcl_1",
    kind: "fact",
    revisions,
    runtimeProfileId: "rtp_1",
    scope: { key: null, kind: "global" },
  } as WorkshopMemoryLifecycle;
}

// Listed newest first by the server; the detail must still read oldest first.
const newer = revision("mfr_new", "The operator prefers light themes.", "2026-09-23T10:00:00Z", { confidence: 0.6 });
const older = revision("mfr_old", "The operator prefers dark themes.", "2026-09-22T10:00:00Z");

const conflictList = {
  total: 1,
  items: [{
    claimId: "mcl_1",
    openedAt: "2026-09-23T10:00:00Z",
    revisions: [
      { preview: "The operator prefers dark themes.", revisionId: "mfr_old", storedAt: older.storedAt },
      { preview: "The operator prefers light themes.", revisionId: "mfr_new", storedAt: newer.storedAt },
    ],
    scope: { key: null, kind: "global" },
  }],
};

const forgottenList = {
  total: 1,
  items: [{
    changedAt: "2026-09-21T10:00:00Z",
    claimId: "mcl_2",
    preview: "Deploys run on Fridays.",
    reason: "Retired during reconciliation.",
    revisionId: "mfr_gone",
    scope: { key: "kai", kind: "project" },
    state: "retracted" as const,
  }],
};

const empty = { total: 0, items: [] };

function renderReview(overrides: { onChanged?: () => void; onOpenMemory?: (id: string) => void } = {}) {
  return render(
    <FactReview
      onAuthenticationFailure={vi.fn()}
      onBack={vi.fn()}
      onChanged={overrides.onChanged ?? vi.fn()}
      onOpenMemory={overrides.onOpenMemory ?? vi.fn()}
      token="session-secret"
    />,
  );
}

describe("Workshop fact review", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadMemoryConflicts).mockResolvedValue(conflictList);
    vi.mocked(loadForgottenMemories).mockResolvedValue(forgottenList);
    vi.mocked(loadMemoryConflict).mockResolvedValue(lifecycle([newer, older]));
  });

  it("compares both versions oldest first and keeps the chosen one with a note", async () => {
    const onChanged = vi.fn();
    const onOpenMemory = vi.fn();
    vi.mocked(resolveMemoryConflict).mockResolvedValue({
      activeRevisionId: "mfr_new", claimId: "mcl_1", memoryId: "memory-9", replayed: false,
    });
    const user = userEvent.setup();
    renderReview({ onChanged, onOpenMemory });

    expect(await screen.findByRole("tab", { name: /Conflicts 1/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: /Forgotten 1/ })).toBeInTheDocument();
    await user.click(await screen.findByRole("option", { name: /dark themes/ }));

    const sides = await screen.findAllByRole("article", { name: "Competing version" });
    expect(sides.map((side) => side.textContent)).toEqual([
      expect.stringContaining("dark themes"),
      expect.stringContaining("light themes"),
    ]);
    expect(within(sides[1]).getByText("60%")).toBeInTheDocument();
    expect(within(sides[1]).getByText("anthropic · claude-sonnet-5")).toBeInTheDocument();

    await user.click(within(sides[1]).getByRole("button", { name: "Keep this version" }));
    const confirmation = screen.getByRole("region", { name: "Confirm decision" });
    expect(confirmation).toHaveTextContent("kept in history as superseded");
    expect(within(confirmation).queryByRole("note")).not.toBeInTheDocument();
    await user.type(within(confirmation).getByLabelText("Note (optional)"), "Newer is right.");
    vi.mocked(loadMemoryConflicts).mockResolvedValue(empty);
    await user.click(within(confirmation).getByRole("button", { name: "Keep this version" }));

    expect(resolveMemoryConflict).toHaveBeenCalledWith("session-secret", "mcl_1", {
      expectedRevisionIds: ["mfr_old", "mfr_new"],
      keepRevisionId: "mfr_new",
      note: "Newer is right.",
    });
    expect(await screen.findByRole("status")).toHaveTextContent("back in recall");
    expect(onChanged).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("No open conflicts")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Open in memory" }));
    expect(onOpenMemory).toHaveBeenCalledWith("memory-9");
  });

  it("warns before keeping a quarantined version", async () => {
    vi.mocked(loadMemoryConflict).mockResolvedValue(
      lifecycle([older, { ...newer, admissionAuthority: "quarantined" }]),
    );
    const user = userEvent.setup();
    renderReview();

    await user.click(await screen.findByRole("option", { name: /dark themes/ }));
    const sides = await screen.findAllByRole("article", { name: "Competing version" });
    await user.click(within(sides[1]).getByRole("button", { name: "Keep this version" }));

    expect(screen.getByRole("note")).toHaveTextContent("stay out of recall until its admission is reviewed");
  });

  it("reloads a conflict that changed since it was opened", async () => {
    vi.mocked(resolveMemoryConflict).mockRejectedValue(
      new MemoryConflictChangedError("This conflict changed since you opened it"),
    );
    const user = userEvent.setup();
    renderReview();

    await user.click(await screen.findByRole("option", { name: /dark themes/ }));
    const sides = await screen.findAllByRole("article", { name: "Competing version" });
    await user.click(within(sides[0]).getByRole("button", { name: "Keep this version" }));
    await user.click(within(screen.getByRole("region", { name: "Confirm decision" }))
      .getByRole("button", { name: "Keep this version" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("changed since you opened it");
    await waitFor(() => expect(loadMemoryConflict).toHaveBeenCalledTimes(2));
    expect(loadMemoryConflicts).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("region", { name: "Confirm decision" })).not.toBeInTheDocument();
  });

  it("restores a forgotten fact after confirming it becomes current with no end date", async () => {
    const onChanged = vi.fn();
    vi.mocked(restoreForgottenMemory).mockResolvedValue({
      activeRevisionId: "mfr_back", claimId: "mcl_2", memoryId: null, replayed: false,
    });
    const user = userEvent.setup();
    renderReview({ onChanged });

    await user.click(await screen.findByRole("tab", { name: /Forgotten/ }));
    await user.click(screen.getByRole("option", { name: /Deploys run on Fridays/ }));
    expect(screen.getByText("Retired during reconciliation.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Restore" }));
    const confirmation = screen.getByRole("region", { name: "Confirm decision" });
    expect(confirmation).toHaveTextContent("current from now, with no end date");
    vi.mocked(loadForgottenMemories).mockResolvedValue(empty);
    await user.click(within(confirmation).getByRole("button", { name: "Restore" }));

    expect(restoreForgottenMemory).toHaveBeenCalledWith("session-secret", "mcl_2", {
      note: "",
      revisionId: "mfr_gone",
    });
    expect(await screen.findByText("No forgotten facts")).toBeInTheDocument();
    expect(onChanged).toHaveBeenCalledTimes(1);
    // No projected row yet, so there is nothing to open.
    expect(screen.queryByRole("button", { name: "Open in memory" })).not.toBeInTheDocument();
  });
});
