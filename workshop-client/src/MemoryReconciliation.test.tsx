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
  missingFields: [],
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

const priorDeferredGroup: WorkshopMemoryTriageGroup = {
  ...group,
  classification: "prior_review",
  evidence: [{
    ...group.evidence[0],
    text: "Workshop and Telegram should preserve workflow continuity",
  }],
  groupId: "prior-deferred",
  priorReviewEvidence: [{
    candidateId: "raw-candidate",
    disposition: "defer",
    memoryIds: ["mem_test"],
    operatorNote: "Qualification probe",
    stateVersion: 1,
  }],
  proposedAction: { kind: "adopt_as_current" },
  rationale: "An earlier raw-audit decision remains evidence.",
};

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

  it("labels a saved decision by its outcome, so obsolete is not shown as approved", async () => {
    const decided = (groupId: string, text: string, kind: string): WorkshopMemoryTriageGroup => ({
      ...group,
      decision: { ...group.decision, action: { kind }, disposition: "approve", stateVersion: 1 },
      evidence: [{ ...group.evidence[0], memoryId: `${groupId}-memory`, text }],
      groupId,
    });
    vi.mocked(loadMemoryTriageGroups).mockImplementation(async (_token, _planId, options = {}) => ({
      triage: summary,
      groups: options.exceptionsOnly
        ? [decided("retired", "An outdated fact", "expire_all"), decided("kept", "A current fact", "adopt_as_current")]
        : safeGroups,
      nextOffset: null,
    }));
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    const list = await screen.findByRole("listbox", { name: "Memory triage exceptions" });
    const retired = within(list).getByRole("option", { name: /An outdated fact/ });
    const kept = within(list).getByRole("option", { name: /A current fact/ });
    expect(retired).toHaveTextContent("obsolete");
    expect(retired).not.toHaveTextContent("approve");
    expect(within(retired).getByText("obsolete")).toHaveClass("memory-review-state", "obsolete");
    expect(kept).toHaveTextContent("approved");
  });

  it("offers only reject or defer for an episode missing required fields", async () => {
    const incomplete: WorkshopMemoryTriageGroup = {
      ...group,
      classification: "incomplete_episode",
      evidence: [{ ...group.evidence[0], kind: "episode", memoryId: "episode-1", text: "Deployed the fix" }],
      groupId: "incomplete-1",
      missingFields: ["actors", "outcome_quality"],
      proposedAction: { kind: "manual_edit_required" },
    };
    vi.mocked(loadMemoryTriageGroups).mockImplementation(async (_token, _planId, options = {}) => ({
      triage: summary,
      groups: options.exceptionsOnly ? [incomplete] : safeGroups,
      nextOffset: null,
    }));
    const user = userEvent.setup();
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    const decision = await screen.findByLabelText("Decision");
    expect(screen.getByText(/saved without the people involved or an outcome/)).toBeInTheDocument();
    expect(within(decision).queryByRole("option", { name: /approve|retain|adopt/i })).not.toBeInTheDocument();
    expect(within(decision).getAllByRole("option").map((option) => option.getAttribute("value")))
      .toEqual(["reject", "defer"]);
    expect(decision).toHaveValue("reject");
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryTriageDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryTriageDecision).mock.calls[0]?.[3]).toMatchObject({ disposition: "reject" });
  });

  it("saves one corrected exception instead of exposing hundreds of raw candidates", async () => {
    const user = userEvent.setup();
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );
    const text = await screen.findByLabelText("Current wording");
    await user.clear(text);
    await user.type(text, "The corrected current fact");
    await user.selectOptions(screen.getByLabelText("Decision"), "approve");
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

  it("turns an ordinary adopt recommendation into unchanged current-truth approval", async () => {
    const user = userEvent.setup();
    const recommended = {
      ...group,
      classification: "time_sensitive",
      decision: {
        ...group.decision,
        recommendation: { outcome: "adopt", confidence: 0.91, rationale: "The preference still applies." },
      },
    };
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({ triage: summary, groups: [recommended], nextOffset: null });
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    const decision = await screen.findByLabelText("Decision");
    expect(decision).toHaveValue("approve");
    expect(within(decision).getByRole("option", { name: "Approve as current truth" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryTriageDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryTriageDecision).mock.calls[0]?.[3]).toMatchObject({
      disposition: "approve",
      action: { kind: "adopt_as_current" },
    });
  });

  it("consolidates every row in a multi-fact recommendation without artificial editing", async () => {
    const user = userEvent.setup();
    const consolidated = {
      ...group,
      decision: {
        ...group.decision,
        recommendation: { outcome: "consolidate", confidence: 0.84, rationale: "Keep one canonical statement." },
      },
      evidence: [
        group.evidence[0],
        { ...group.evidence[0], memoryId: "mem_second", text: "A similar legacy fact" },
      ],
    };
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({ triage: summary, groups: [consolidated], nextOffset: null });
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    expect(await screen.findByLabelText("Decision")).toHaveValue("approve");
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryTriageDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryTriageDecision).mock.calls[0]?.[3]).toMatchObject({
      disposition: "approve",
      action: { kind: "adopt_corrected", replacement: { content: "An inaccurate legacy fact" } },
    });
  });

  it("keeps a valid displayed scope selected while explicitly confirming incomplete provenance", async () => {
    const user = userEvent.setup();
    const uncertain = {
      ...group,
      decision: {
        ...group.decision,
        recommendation: { outcome: "adopt", confidence: 0.76, rationale: "Keep after choosing scope." },
      },
      evidence: [{ ...group.evidence[0], migrationGaps: ["scope", "source receipt"] }],
    };
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({ triage: summary, groups: [uncertain], nextOffset: null });
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    const decision = await screen.findByLabelText("Decision");
    expect(decision).toHaveValue("approve");
    expect(screen.getByLabelText("Scope")).toHaveValue("global");
    expect(screen.getByRole("button", { name: "Save decision" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryTriageDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryTriageDecision).mock.calls[0]?.[3]).toMatchObject({
      disposition: "approve",
      action: { kind: "adopt_corrected", replacement: { scope_kind: "global" } },
    });
  });

  it("explains genuinely missing scope without disabling the approval decision", async () => {
    const missing = {
      ...group,
      decision: {
        ...group.decision,
        recommendation: { outcome: "adopt", confidence: 0.76, rationale: "Keep after choosing scope." },
      },
      evidence: [{ ...group.evidence[0], migrationGaps: ["scope"], scope: "" }],
    };
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({ triage: summary, groups: [missing], nextOffset: null });
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    const decision = await screen.findByLabelText("Decision");
    expect(within(decision).getByRole("option", { name: "Approve as current truth" })).toBeEnabled();
    expect(screen.getByText("Choose a scope before approval.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Save decision" })).toBeDisabled();
  });

  it("keeps an unauthorized project visible and explains why it cannot be approved", async () => {
    const projectScoped = {
      ...group,
      decision: {
        ...group.decision,
        recommendation: { outcome: "adopt", confidence: 0.8, rationale: "The project fact still applies." },
      },
      evidence: [{ ...group.evidence[0], projectId: "project-private", scope: "project" }],
    };
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({
      triage: summary,
      groups: [projectScoped],
      nextOffset: null,
    });
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    expect(await screen.findByLabelText("Scope")).toHaveValue("project");
    expect(screen.getByLabelText("Project")).toHaveValue("project-private");
    expect(screen.getByText("The selected project is not currently authorized for this runtime.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Save decision" })).toBeDisabled();
  });

  it("restores a saved corrected approval without reverting to the legacy evidence", async () => {
    const saved = {
      ...group,
      decision: {
        ...group.decision,
        action: {
          kind: "adopt_corrected",
          replacement: {
            content: "The operator-confirmed current fact",
            scope_kind: "project",
            scope_key: "project-kai",
          },
        },
        disposition: "approve" as const,
      },
    };
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({ triage: summary, groups: [saved], nextOffset: null });
    render(
      <ConfirmationProvider>
        <MemoryReconciliation
          allowedProjects={[{ displayName: "Kai", projectId: "project-kai" }]}
          onAuthenticationFailure={vi.fn()}
          onBack={vi.fn()}
          token="secret"
        />
      </ConfirmationProvider>,
    );

    expect(await screen.findByLabelText("Decision")).toHaveValue("approve");
    expect(screen.getByLabelText("Current wording")).toHaveValue("The operator-confirmed current fact");
    expect(screen.getByLabelText("Scope")).toHaveValue("project");
    expect(screen.getByLabelText("Project")).toHaveValue("project-kai");
    expect(screen.getByRole("button", { name: "Save decision" })).toBeEnabled();
  });

  it.each(["reject", "defer"] as const)("maps %s directly without a lifecycle prerequisite", async (choice) => {
    const user = userEvent.setup();
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    await user.selectOptions(await screen.findByLabelText("Decision"), choice);
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryTriageDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryTriageDecision).mock.calls[0]?.[3]).toMatchObject({ disposition: choice });
  });

  it("turns an obsolete recommendation into an explicit retirement action", async () => {
    const user = userEvent.setup();
    const obsolete = {
      ...group,
      decision: {
        ...group.decision,
        recommendation: { outcome: "obsolete", confidence: 0.95, rationale: "This no longer applies." },
      },
    };
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({ triage: summary, groups: [obsolete], nextOffset: null });
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    expect(await screen.findByLabelText("Decision")).toHaveValue("obsolete");
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryTriageDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryTriageDecision).mock.calls[0]?.[3]).toMatchObject({
      disposition: "approve",
      action: { kind: "expire_all" },
    });
  });

  it("turns an episode recommendation into explicit immutable-history retention", async () => {
    const user = userEvent.setup();
    const episode = {
      ...group,
      decision: {
        ...group.decision,
        recommendation: { outcome: "adopt", confidence: 0.88, rationale: "Retain this useful history." },
      },
      evidence: [{ ...group.evidence[0], kind: "episode" as const, memoryId: "episode_test" }],
    };
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({ triage: summary, groups: [episode], nextOffset: null });
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    expect(await screen.findByLabelText("Decision")).toHaveValue("approve");
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryTriageDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryTriageDecision).mock.calls[0]?.[3]).toMatchObject({
      disposition: "approve",
      action: { kind: "record_episode_chain" },
    });
  });

  it("shows stable references and navigates legacy cross-group recommendations", async () => {
    const relatedId = "mtg_a37a772857d24bb5556da8997ff2381e";
    const source = {
      ...group,
      groupId: "mtg_975558625271cc5d41a185bb6533d986",
      decision: {
        ...group.decision,
        recommendation: {
          outcome: "consolidate",
          confidence: 0.8,
          rationale: `Duplicate of another fact; consolidate with ${relatedId}.`,
        },
      },
    };
    const related = {
      ...group,
      evidence: [{ ...group.evidence[0], memoryId: "mem_related", text: "The related canonical fact" }],
      groupId: relatedId,
    };
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({
      triage: summary,
      groups: [source, related],
      nextOffset: null,
    });
    const user = userEvent.setup();
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    const linked = await screen.findByRole("button", { name: "Group a37a7728" });
    expect(screen.getByText(/Group 97555862/)).toBeVisible();
    expect(screen.getByLabelText("Decision")).toHaveValue("approve");
    expect(within(screen.getByLabelText("Decision")).getByRole("option", { name: "Approve as current truth" })).toBeEnabled();
    await user.click(linked);
    expect(await screen.findByLabelText("Current wording")).toHaveValue("The related canonical fact");
  });

  it("allows an eligible previously deferred fact to be approved unchanged", async () => {
    const user = userEvent.setup();
    vi.mocked(loadMemoryTriageGroups).mockResolvedValue({
      triage: summary,
      groups: [priorDeferredGroup],
      nextOffset: null,
    });
    render(
      <ConfirmationProvider>
        <MemoryReconciliation allowedProjects={[]} onAuthenticationFailure={vi.fn()} onBack={vi.fn()} token="secret" />
      </ConfirmationProvider>,
    );

    const decision = await screen.findByLabelText("Decision");
    expect(decision).toHaveValue("approve");
    expect(within(decision).getByRole("option", { name: "Approve as current truth" })).toBeEnabled();
    expect(screen.getByText(/earlier decision remains audit evidence/i)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() => expect(saveMemoryTriageDecision).toHaveBeenCalled());
    expect(vi.mocked(saveMemoryTriageDecision).mock.calls[0]?.[3]).toMatchObject({
      disposition: "approve",
      action: { kind: "adopt_as_current" },
    });
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
