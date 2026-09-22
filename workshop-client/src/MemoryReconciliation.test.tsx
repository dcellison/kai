import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  applyMemoryTriage,
  approveSafeMemoryTriage,
  loadMemoryTriage,
  loadMemoryTriageGroups,
  previewSafeMemoryTriage,
  recommendMemoryTriage,
  saveMemoryTriageDecision,
} from "./api";
import { ConfirmationProvider } from "./ConfirmationDialog";
import { MemoryReconciliation } from "./MemoryReconciliation";
import type { WorkshopMemoryTriageGroup, WorkshopMemoryTriageSummary } from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    applyMemoryTriage: vi.fn(),
    approveSafeMemoryTriage: vi.fn(),
    loadMemoryTriage: vi.fn(),
    loadMemoryTriageGroups: vi.fn(),
    previewSafeMemoryTriage: vi.fn(),
    recommendMemoryTriage: vi.fn(),
    saveMemoryTriageDecision: vi.fn(),
  };
});

const summary: WorkshopMemoryTriageSummary = {
  appliedAt: null,
  auditId: "mra_test",
  deterministicGroups: 2,
  pendingDeterministicGroups: 2,
  dispositionCounts: { approve: 0, defer: 0, pending: 3, reject: 0 },
  exceptionGroups: 1,
  groupCount: 3,
  memoryCount: 437,
  planId: "mtp_test",
  recommendedGroups: 0,
  resolutionCounts: { adopt: 312, consolidate: 84, obsolete: 29, needs_review: 12 },
  reviewVersion: 0,
  status: "open",
};

const group: WorkshopMemoryTriageGroup = {
  bulkEligible: false,
  classification: "contradiction",
  decision: {
    action: { kind: "manual_edit_required" },
    disposition: "pending",
    operatorNote: "",
    recommendation: {},
    stateVersion: 0,
  },
  deterministic: false,
  evidence: [{
    assertedAt: null,
    backend: null,
    confidence: 0.8,
    createdAt: "2026-09-21T18:00:00Z",
    kind: "fact",
    memoryId: "mem_test",
    migrationClassification: "legacy_incomplete",
    migrationGaps: ["source receipt"],
    model: null,
    observedAt: null,
    occurredFrom: null,
    occurredUntil: null,
    projectId: null,
    promptVersion: null,
    provider: null,
    schemaVersion: null,
    scope: "global",
    source: "extracted",
    text: "An inaccurate legacy fact",
    updatedAt: "2026-09-21T18:00:00Z",
    validFrom: null,
    validUntil: null,
  }],
  groupId: "mtg_test",
  priorReviewEvidence: [],
  proposedAction: { kind: "manual_edit_required" },
  rationale: "The evidence conflicts.",
  resolution: "needs_review",
  stateSha256: "a".repeat(64),
};

const safeGroups: WorkshopMemoryTriageGroup[] = ["safe-1", "safe-2"].map((groupId, index) => ({
  ...group,
  bulkEligible: true,
  classification: "stable_non_conflicting",
  decision: {
    action: { kind: "adopt_as_current" },
    disposition: "pending",
    operatorNote: "",
    recommendation: {},
    stateVersion: 0,
  },
  deterministic: true,
  evidence: [{ ...group.evidence[0], memoryId: `safe-memory-${index}`, text: `Stable legacy fact ${index + 1}` }],
  groupId,
  proposedAction: { kind: "adopt_as_current", migration_classification: "legacy_incomplete" },
  rationale: "No deterministic warning was found; explicit operator approval is still required.",
  resolution: "adopt",
}));

describe("Memory reconciliation triage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadMemoryTriage).mockResolvedValue(summary);
    vi.mocked(loadMemoryTriageGroups).mockImplementation(async (_token, _planId, options = {}) => ({
      triage: summary,
      groups: options.exceptionsOnly ? [group] : safeGroups,
      nextOffset: null,
    }));
    vi.mocked(previewSafeMemoryTriage).mockResolvedValue({
      groupCount: 2,
      groupIds: ["safe-1", "safe-2"],
      memoryCount: 425,
      planId: summary.planId,
      previewSha256: "b".repeat(64),
      resolutionCounts: { adopt: 312, consolidate: 84, obsolete: 29 },
      reviewVersion: 0,
    });
    vi.mocked(recommendMemoryTriage).mockResolvedValue({ recommended: 1, remaining: 0 });
  });

  it("shows aggregate outcomes and only exceptional groups", async () => {
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    expect((await screen.findAllByText("An inaccurate legacy fact"))[0]).toBeVisible();
    expect(screen.getByText(/partitions 437 legacy memories into 3 non-overlapping groups/i)).toBeVisible();
    expect(screen.getByRole("button", { name: "Preview safe groups" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Analyze exceptions" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Apply plan" })).toBeDisabled();
  });

  it("saves one corrected exception instead of exposing hundreds of raw candidates", async () => {
    const user = userEvent.setup();
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );
    const text = await screen.findByLabelText("Current fact");
    await user.clear(text);
    await user.type(text, "The corrected current fact");
    await user.selectOptions(screen.getByLabelText("Disposition"), "approve");
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryTriageDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryTriageDecision).mock.calls[0]?.[3]).toMatchObject({
      disposition: "approve",
      action: { kind: "adopt_corrected", replacement: { content: "The corrected current fact" } },
    });
    expect(approveSafeMemoryTriage).not.toHaveBeenCalled();
    expect(recommendMemoryTriage).not.toHaveBeenCalled();
    expect(applyMemoryTriage).not.toHaveBeenCalled();
  });

  it("shows complete deterministic evidence before final bulk confirmation", async () => {
    const user = userEvent.setup();
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    await user.click(await screen.findByRole("button", { name: "Preview safe groups" }));
    expect(await screen.findByText("Complete deterministic preview")).toBeVisible();
    expect(previewSafeMemoryTriage).toHaveBeenCalledWith(
      "secret",
      summary.planId,
      0,
      ["safe-1", "safe-2"],
    );
    expect(screen.getByText(/explicit approval supplies separate retrieval admission authority/i)).toBeVisible();
    expect(screen.getByText(/425 adopt as current truth/i)).toBeVisible();
    expect(screen.getByText("Stable non conflicting · Stable legacy fact 1 · 1 memory · Adopt as current truth")).toBeVisible();
    expect(screen.getByText("Stable non conflicting · Stable legacy fact 2 · 1 memory · Adopt as current truth")).toBeVisible();
    expect(screen.getAllByText(/Stable non conflicting/)).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: "Approve current-truth adoption" }));
    const dialog = screen.getByRole("dialog", { name: "Continue?" });
    expect(dialog).toHaveTextContent("original provenance will remain incomplete");
    expect(dialog).toHaveTextContent("separate retrieval admission authority");
    await user.click(within(dialog).getByRole("button", { name: "Continue" }));

    await waitFor(() => expect(approveSafeMemoryTriage).toHaveBeenCalledTimes(1));
  });

  it("reports operator admission separately from lifecycle outcomes", async () => {
    const user = userEvent.setup();
    const readySummary = {
      ...summary,
      dispositionCounts: { approve: 3, defer: 0, pending: 0, reject: 0 },
      pendingDeterministicGroups: 0,
    };
    vi.mocked(loadMemoryTriage).mockResolvedValue(readySummary);
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({
      triage: readySummary,
      groups: [],
      nextOffset: null,
    });
    vi.mocked(applyMemoryTriage).mockResolvedValue({
      adopted: 312,
      consolidated: 84,
      deferred: 0,
      failed: 0,
      obsolete: 29,
      operator_admitted: 425,
      rejected: 0,
      still_unresolved: 0,
    });

    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    await user.click(await screen.findByRole("button", { name: "Apply plan" }));
    await user.click(within(screen.getByRole("dialog", { name: "Continue?" })).getByRole("button", { name: "Continue" }));

    expect(await screen.findByText(/425 operator admitted/i)).toBeVisible();
    expect(screen.getByText(/312 adopted by lifecycle outcome/i)).toBeVisible();
  });

  it("makes completed exception analysis explicit", async () => {
    const analyzedSummary = { ...summary, recommendedGroups: 1 };
    vi.mocked(loadMemoryTriage).mockResolvedValue(analyzedSummary);
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({
      triage: analyzedSummary,
      groups: [group],
      nextOffset: null,
    });

    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    expect(await screen.findByRole("button", { name: "Exceptions analyzed (1)" })).toBeDisabled();
  });
});
