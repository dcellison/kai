import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  applyMemoryReconciliation,
  bulkMemoryReconciliationDecision,
  loadMemoryReconciliation,
  loadMemoryReconciliationCandidates,
  saveMemoryReconciliationDecision,
} from "./api";
import { ConfirmationProvider } from "./ConfirmationDialog";
import { MemoryReconciliation } from "./MemoryReconciliation";
import type {
  WorkshopMemoryReconciliationCandidate,
  WorkshopMemoryReconciliationSummary,
} from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    applyMemoryReconciliation: vi.fn(),
    bulkMemoryReconciliationDecision: vi.fn(),
    loadMemoryReconciliation: vi.fn(),
    loadMemoryReconciliationCandidates: vi.fn(),
    saveMemoryReconciliationDecision: vi.fn(),
  };
});

const summary: WorkshopMemoryReconciliationSummary = {
  actionCounts: { manual_edit_required: 1 },
  appliedAt: null,
  auditId: "mra_test",
  candidateCount: 1,
  categoryCounts: { malformed_provenance: 1 },
  corpusCount: 1,
  dispositionCounts: { approve: 0, defer: 0, pending: 1, reject: 0 },
  generatedAt: "2026-09-21T18:00:00Z",
  gapCounts: { "source receipt": 1 },
  kindCounts: { fact: 1 },
  reviewVersion: 0,
  runtimeProfileId: "rtp_test",
  status: "open",
  uncertaintyCounts: { high: 1 },
};

const candidate: WorkshopMemoryReconciliationCandidate = {
  candidateId: "mrc_test",
  category: "malformed_provenance",
  decision: {
    action: { kind: "manual_edit_required" },
    disposition: "pending",
    operatorNote: "",
    stateVersion: 0,
  },
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
  proposedAction: { kind: "manual_edit_required" },
  rationale: "Canonical provenance is missing.",
  stateSha256: "a".repeat(64),
  uncertainty: "high",
};

describe("Memory reconciliation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadMemoryReconciliation).mockResolvedValue(summary);
    vi.mocked(loadMemoryReconciliationCandidates).mockResolvedValue({
      audit: summary,
      candidates: [candidate],
      nextOffset: null,
    });
    vi.mocked(saveMemoryReconciliationDecision).mockResolvedValue({ reviewVersion: 1, stateVersion: 1 });
  });

  it("explains legacy uncertainty and never offers bulk approval", async () => {
    render(
      <ConfirmationProvider>
        <MemoryReconciliation
          allowedProjects={[]}
          onAuthenticationFailure={vi.fn()}
          onBack={vi.fn()}
          token="session-secret"
        />
      </ConfirmationProvider>,
    );

    expect((await screen.findAllByText("An inaccurate legacy fact"))[0]).toBeVisible();
    expect(screen.getAllByText(/does not (?:establish|mean).*memory is false/i)[0]).toBeVisible();
    expect(screen.queryByRole("button", { name: /approve selected/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /apply reviewed batch/i })).toBeDisabled();
  });

  it("saves corrected replacement fields instead of exposing action JSON", async () => {
    const user = userEvent.setup();
    render(
      <ConfirmationProvider>
        <MemoryReconciliation
          allowedProjects={[]}
          onAuthenticationFailure={vi.fn()}
          onBack={vi.fn()}
          token="session-secret"
        />
      </ConfirmationProvider>,
    );
    const text = await screen.findByLabelText("Fact text");
    await user.clear(text);
    await user.type(text, "The corrected current fact");
    await user.selectOptions(screen.getByLabelText("Disposition"), "approve");
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryReconciliationDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryReconciliationDecision).mock.calls[0]?.[3]).toMatchObject({
      disposition: "approve",
      action: {
        kind: "adopt_corrected",
        replacement: { content: "The corrected current fact" },
      },
    });
    expect(bulkMemoryReconciliationDecision).not.toHaveBeenCalled();
    expect(applyMemoryReconciliation).not.toHaveBeenCalled();
  });
});
