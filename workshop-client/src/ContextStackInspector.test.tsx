import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ContextStackInspector } from "./ContextStackInspector";
import type { WorkshopContextSource } from "./types";

const api = vi.hoisted(() => ({
  loadPreferenceDocument: vi.fn(),
  loadPrincipalPolicyDocument: vi.fn(),
  loadRunContextManifests: vi.fn(),
  savePreferenceDocument: vi.fn(),
  savePrincipalPolicyDocument: vi.fn(),
}));

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  ...api,
}));

function source(
  kind: string,
  title: string,
  editTarget: WorkshopContextSource["inspection"]["editTarget"] = null,
): WorkshopContextSource {
  return {
    authorityClass: "principal",
    authorizationOperations: null,
    budgetBytes: 1024,
    deliveryRole: "session_context",
    deliveryShape: "inline_verified_document",
    historyBoundary: null,
    inspection: {
      changeEffect: editTarget ? "provider_session_refresh" : null,
      currentRevision: "a".repeat(64),
      description: `${title} description`,
      editable: editTarget !== null,
      editTarget,
      editTargetId: null,
      freshness: "current",
      preview: `${title} rendered content`,
      previewReason: null,
      previewState: "exact",
      sourceReference: `${title} canonical source`,
      title,
    },
    kind,
    nativeInstructionSources: [],
    ownerId: "prn_" + "1".repeat(32),
    ownerKind: "principal",
    reason: "live_provider_session",
    refreshClass: "provider_session",
    renderedBytes: 24,
    revision: "a".repeat(64),
    scope: "principal",
    state: "retained",
    trustClass: "principal_policy",
  };
}

describe("ContextStackInspector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const sources = [
      source("host_policy", "Host policy"),
      source("principal_policy", "Principal policy", "principal_policy"),
      source("agent_definition", "Agent definition"),
      source("workspace_policy", "Workspace policy"),
      source("personal_preferences", "Personal preferences", "personal_preferences"),
      source("file_memory", "File memory"),
      source("semantic_recall", "Semantic recall"),
      source("canonical_conversation", "Conversation"),
      source("capability_guidance", "Capability guidance"),
      source("attempt_authority", "Attempt authority"),
      source("current_input", "Current input"),
      source("provider_native", "Provider-owned context"),
    ];
    api.loadRunContextManifests.mockResolvedValue({
      channelId: "chn_" + "1".repeat(32),
      runId: "run_" + "2".repeat(32),
      manifests: [{
        agentId: "agt_" + "3".repeat(32),
        attemptId: "rat_" + "4".repeat(32),
        backend: "codex",
        channelId: "chn_" + "1".repeat(32),
        createdAt: "2026-09-13T12:00:00Z",
        manifestSha256: "5".repeat(64),
        model: "gpt-5.6-sol",
        provider: "openai",
        providerSessionRevision: "6".repeat(64),
        runId: "run_" + "2".repeat(32),
        runtimeProfileId: "rtp_" + "7".repeat(32),
        sources,
        workspaceDigest: "8".repeat(64),
        workspaceKind: "foreign",
      }],
    });
    api.loadPrincipalPolicyDocument.mockResolvedValue({
      content: "# Principal Policy\n\nBe concise.",
      editable: true,
      maxBytes: 131072,
      revision: "a".repeat(64),
      sizeBytes: 32,
    });
    api.savePrincipalPolicyDocument.mockResolvedValue({
      contextInvalidation: { applied: 1, pending: 0, state: "applied" },
      content: "# Principal Policy\n\nBe precise.",
      editable: true,
      maxBytes: 131072,
      revision: "b".repeat(64),
      sizeBytes: 32,
    });
  });

  it("shows the complete canonical source vocabulary and exact previews", async () => {
    render(
      <ContextStackInspector
        session={{ channelId: "chn_" + "1".repeat(32), token: "token" }}
        runId={"run_" + "2".repeat(32)}
        onNavigate={vi.fn()}
      />,
    );

    expect(await screen.findByText("12 canonical sources · 0 changed since this run")).toBeVisible();
    expect(screen.getByText("Provider-owned context")).toBeVisible();
    fireEvent.click(screen.getAllByText("Rendered preview")[0]);
    expect(screen.getByText("Host policy rendered content")).toBeVisible();
  });

  it("edits owned principal policy through its revision-checked authority", async () => {
    render(
      <ContextStackInspector
        session={{ channelId: "chn_" + "1".repeat(32), token: "token" }}
        runId={"run_" + "2".repeat(32)}
        onNavigate={vi.fn()}
      />,
    );

    const editButton = await screen.findByRole("button", { name: "Edit Principal policy" });
    const sourceCard = editButton.closest("article");
    expect(editButton.querySelector("svg")).toHaveAttribute("fill", "none");
    expect(editButton.querySelector("svg")).toHaveAttribute("stroke", "currentColor");
    fireEvent.click(editButton);
    const editor = await screen.findByRole("textbox", { name: "Edit Principal policy" });
    expect(editor).toHaveFocus();
    expect(sourceCard?.querySelector(".context-source-facts")).toBeNull();
    expect(screen.queryByRole("button", { name: "Edit Principal policy" })).toBeNull();
    fireEvent.change(editor, { target: { value: "# Principal Policy\n\nBe precise." } });
    fireEvent.click(screen.getByRole("button", { name: "Save Principal policy" }));

    await waitFor(() => expect(api.savePrincipalPolicyDocument).toHaveBeenCalledWith(
      expect.objectContaining({ token: "token" }),
      "# Principal Policy\n\nBe precise.",
      "a".repeat(64),
    ));
    expect(await screen.findByText("Saved. Context refreshed in 1 lane.")).toBeVisible();
  });
});
