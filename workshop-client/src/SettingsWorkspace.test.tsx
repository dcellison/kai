import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  loadPreferenceDocument,
  loadPreferenceHistory,
  loadSettingsWorkspace,
  loadWorkspaceConfig,
  PreferenceRevisionConflictError,
  restorePreferenceRevision,
  savePreferenceDocument,
  switchWorkspace,
  updateRuntimeSettings,
  updateWorkspaceConfig,
} from "./api";
import { SettingsWorkspace } from "./SettingsWorkspace";
import type {
  WorkshopPreferenceDocument,
  WorkshopSettingsWorkspace,
  WorkshopWorkspaceConfig,
} from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    loadPreferenceDocument: vi.fn(),
    loadPreferenceHistory: vi.fn(),
    loadSettingsWorkspace: vi.fn(),
    loadWorkspaceConfig: vi.fn(),
    restorePreferenceRevision: vi.fn(),
    savePreferenceDocument: vi.fn(),
    switchWorkspace: vi.fn(),
    updateRuntimeSettings: vi.fn(),
    updateWorkspaceConfig: vi.fn(),
  };
});

const session = {
  channelId: "chn_d3dfdfd7df9151ba8a1742b92403faa5",
  token: "session-secret",
};

const preference: WorkshopPreferenceDocument = {
  content: "# Preferences\n\nBe concise.\n",
  editable: true,
  maxBytes: 65536,
  revision: "pref_current",
  sizeBytes: 27,
  updatedAt: "2026-08-26T10:00:00Z",
};

const runtime: WorkshopSettingsWorkspace = {
  backend: "claude",
  backendOptionId: "claude:anthropic",
  backendOptions: [
    { optionId: "claude:anthropic", backend: "claude", provider: "anthropic", current: true },
    { optionId: "codex:openai", backend: "codex", provider: "openai", current: false },
  ],
  capabilities: [
    {
      choices: ["claude:anthropic", "codex:openai"],
      field: "backend",
      maximum: null,
      minimum: null,
      resettable: false,
      scope: "runtime",
      valueType: "backend_id",
    },
    {
      choices: ["claude-sonnet-4-6", "claude-opus-4-1"],
      field: "model",
      maximum: null,
      minimum: null,
      resettable: true,
      scope: "runtime",
      valueType: "model_id",
    },
    {
      choices: null,
      field: "timeout",
      maximum: 1800,
      minimum: 1,
      resettable: true,
      scope: "runtime",
      valueType: "integer_seconds",
    },
  ],
  channelId: session.channelId,
  model: {
    defaultValue: "claude-sonnet-4-6",
    source: "runtime policy",
    value: "claude-sonnet-4-6",
  },
  modelOptions: [
    { displayName: "Claude Sonnet 4.6", modelId: "claude-sonnet-4-6" },
    { displayName: "Claude Opus 4.1", modelId: "claude-opus-4-1" },
  ],
  mutation: null,
  principalId: "prn_00000000000000000000000000000001",
  provider: "anthropic",
  revision: "sws_current",
  runtimeProfileId: "rtp_00000000000000000000000000000001",
  timeoutSeconds: {
    defaultValue: 1800,
    source: "runtime policy",
    value: 1800,
  },
  workspace: "/srv/kai",
  workspaces: [
    { current: true, home: false, name: "Kai", path: "/srv/kai" },
    { current: false, home: true, name: "Home", path: "/srv/home" },
  ],
};

const workspaceConfig: WorkshopWorkspaceConfig = {
  capabilities: [
    {
      choices: ["claude-sonnet-4-6", "claude-opus-4-1"],
      field: "model",
      maximum: null,
      minimum: null,
      resettable: true,
      scope: "workspace",
      valueType: "model_id",
    },
    {
      choices: null,
      field: "timeout",
      maximum: 1800,
      minimum: 1,
      resettable: true,
      scope: "workspace",
      valueType: "integer_seconds",
    },
    {
      choices: null,
      field: "prompt",
      maximum: 4000,
      minimum: null,
      resettable: true,
      scope: "workspace",
      valueType: "text",
    },
  ],
  environmentKeys: ["PROTECTED_KEY"],
  hasPrompt: false,
  model: runtime.model,
  mutation: null,
  overrideFields: [],
  prompt: null,
  promptSource: null,
  revision: "sws_workspace",
  timeoutSeconds: runtime.timeoutSeconds,
  workspace: runtime.workspace,
};

function renderSettings(onDirtyChange = vi.fn(), runActive = false): void {
  render(
    <SettingsWorkspace
      onAuthenticationFailure={vi.fn()}
      onChannelAccessFailure={vi.fn()}
      onClose={vi.fn()}
      onDirtyChange={onDirtyChange}
      principalName="Daniel"
      roleLabel="Workshop administrator"
      runtimeLabel="Kai"
      runActive={runActive}
      session={session}
    />,
  );
}

describe("Settings workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadPreferenceDocument).mockResolvedValue(preference);
    vi.mocked(loadPreferenceHistory).mockResolvedValue({
      limit: 20,
      revisions: [
        {
          revision: "pref_previous",
          sizeBytes: 18,
          updatedAt: "2026-08-25T10:00:00Z",
        },
      ],
    });
    vi.mocked(loadSettingsWorkspace).mockResolvedValue(runtime);
    vi.mocked(loadWorkspaceConfig).mockResolvedValue(workspaceConfig);
    vi.mocked(savePreferenceDocument).mockResolvedValue({
      ...preference,
      content: "# Preferences\n\nUse examples.\n",
      revision: "pref_saved",
    });
    vi.mocked(restorePreferenceRevision).mockResolvedValue({
      ...preference,
      revision: "pref_restored",
    });
    vi.mocked(updateRuntimeSettings).mockResolvedValue(runtime);
    vi.mocked(updateWorkspaceConfig).mockResolvedValue(workspaceConfig);
    vi.mocked(switchWorkspace).mockResolvedValue({
      ...runtime,
      revision: "sws_home",
      workspace: "/srv/home",
      workspaces: runtime.workspaces.map((item) => ({
        ...item,
        current: item.home,
      })),
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows private preferences and policy-bounded settings without protected details", async () => {
    renderSettings();

    expect(await screen.findByLabelText("Preference Markdown")).toHaveValue(
      preference.content,
    );
    expect(screen.getByLabelText("Backend")).toHaveValue("claude:anthropic");
    expect(screen.getByRole("option", { name: "claude · anthropic" })).toBeVisible();
    expect(screen.getByRole("option", { name: "Kai" })).toBeVisible();
    expect(screen.getByRole("option", { name: "Home" })).toBeVisible();
    expect(screen.getByRole("option", { name: "Kai" })).toHaveValue("0");
    expect(screen.getByRole("option", { name: "Home" })).toHaveValue("1");
    expect(screen.queryByText("PROTECTED_KEY")).not.toBeInTheDocument();
    expect(screen.queryByText(runtime.principalId)).not.toBeInTheDocument();
    expect(screen.queryByText(runtime.runtimeProfileId)).not.toBeInTheDocument();
    expect(screen.queryByText("/srv/kai")).not.toBeInTheDocument();
    expect(document.body.innerHTML).not.toContain("/srv/");
  });

  it("saves preference Markdown and reports dirty state", async () => {
    const user = userEvent.setup();
    const onDirtyChange = vi.fn();
    renderSettings(onDirtyChange);
    const editor = await screen.findByLabelText("Preference Markdown");

    await user.clear(editor);
    await user.type(editor, "# Preferences\n\nUse examples.\n");
    await waitFor(() => expect(onDirtyChange).toHaveBeenLastCalledWith(true));
    await user.click(screen.getByRole("button", { name: "Save preferences" }));

    expect(savePreferenceDocument).toHaveBeenCalledWith(
      session,
      "# Preferences\n\nUse examples.\n",
      "pref_current",
    );
    expect(await screen.findByText(/Preferences saved/)).toBeVisible();
    await waitFor(() => expect(onDirtyChange).toHaveBeenLastCalledWith(false));
  });

  it("preserves a draft and offers an explicit conflict comparison", async () => {
    const user = userEvent.setup();
    const latest = {
      ...preference,
      content: "# Preferences\n\nLatest elsewhere.\n",
      revision: "pref_latest",
    };
    vi.mocked(loadPreferenceDocument)
      .mockResolvedValueOnce(preference)
      .mockResolvedValueOnce(latest);
    vi.mocked(savePreferenceDocument)
      .mockRejectedValueOnce(
        new PreferenceRevisionConflictError("Preferences changed", "pref_latest"),
      )
      .mockResolvedValueOnce({
        ...preference,
        content: "My reviewed draft",
        revision: "pref_reapplied",
      });
    renderSettings();
    const editor = await screen.findByLabelText("Preference Markdown");
    await user.clear(editor);
    await user.type(editor, "My reviewed draft");
    await user.click(screen.getByRole("button", { name: "Save preferences" }));

    expect(await screen.findByRole("heading", { name: "Review concurrent changes" })).toBeVisible();
    const comparison = screen.getByRole("heading", {
      name: "Review concurrent changes",
    }).closest("section");
    expect(comparison).not.toBeNull();
    expect(within(comparison as HTMLElement).getByText("My reviewed draft")).toBeVisible();
    expect(within(comparison as HTMLElement).getByText(/Latest elsewhere/)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Reapply my draft…" }));

    expect(savePreferenceDocument).toHaveBeenLastCalledWith(
      session,
      "My reviewed draft",
      "pref_latest",
    );
  });

  it("confirms and restores a private preference revision", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByLabelText("Preference Markdown");
    await user.click(screen.getByText("Private revision history (1)"));
    await user.click(screen.getByRole("button", { name: "Restore…" }));

    expect(window.confirm).toHaveBeenCalledWith(
      "Restore this private preference revision? The current text will be retained in revision history.",
    );
    expect(restorePreferenceRevision).toHaveBeenCalledWith(
      session,
      "pref_previous",
      "pref_current",
    );
  });

  it("applies runtime and workspace changes through revision-checked APIs", async () => {
    const user = userEvent.setup();
    vi.mocked(updateRuntimeSettings).mockResolvedValue({
      ...runtime,
      mutation: {
        changed: true,
        operation: "set_timeout",
        providerSessionInvalidated: true,
        runtimeAction: "restarted",
      },
      revision: "sws_timeout",
      timeoutSeconds: {
        ...runtime.timeoutSeconds,
        source: "principal override",
        value: 601,
      },
    });
    renderSettings();
    const runtimeTimeout = await screen.findByLabelText("Response timeout");
    await user.clear(runtimeTimeout);
    await user.type(runtimeTimeout, "601");
    const runtimeForm = runtimeTimeout.closest("form");
    expect(runtimeForm).not.toBeNull();
    await user.click(within(runtimeForm as HTMLFormElement).getByRole("button", { name: "Apply" }));

    expect(updateRuntimeSettings).toHaveBeenCalledWith(
      session,
      "sws_current",
      { field: "timeout", value: 601 },
    );
    expect(await screen.findByText(/provider-session continuity was cleared/)).toBeVisible();

    await user.selectOptions(screen.getByLabelText("Active workspace"), "1");
    expect(switchWorkspace).toHaveBeenCalledWith(session, "/srv/home", "sws_timeout");

    const prompt = screen.getByLabelText("Workspace system prompt");
    await user.type(prompt, "Use the project conventions.");
    const promptForm = prompt.closest("form");
    expect(promptForm).not.toBeNull();
    await user.click(within(promptForm as HTMLFormElement).getByRole("button", { name: "Apply" }));
    expect(updateWorkspaceConfig).toHaveBeenCalledWith(
      session,
      "sws_workspace",
      { field: "prompt", value: "Use the project conventions." },
    );
    expect(window.confirm).toHaveBeenCalled();
  });

  it("switches only the authenticated principal's backend after confirmation", async () => {
    const user = userEvent.setup();
    vi.mocked(updateRuntimeSettings).mockResolvedValue({
      ...runtime,
      backend: "codex",
      backendOptionId: "codex:openai",
      provider: "openai",
      backendOptions: runtime.backendOptions.map((option) => ({
        ...option,
        current: option.optionId === "codex:openai",
      })),
      mutation: {
        changed: true,
        operation: "set_runtime_backend",
        providerSessionInvalidated: true,
        runtimeAction: "restarted",
      },
      revision: "sws_backend",
    });
    renderSettings();
    const backend = await screen.findByLabelText("Backend");

    await user.selectOptions(backend, "codex:openai");
    await user.click(screen.getByRole("button", { name: "Switch backend" }));

    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining("other people and Kai itself will not restart"));
    expect(updateRuntimeSettings).toHaveBeenCalledWith(
      session,
      "sws_current",
      { field: "backend", value: "codex:openai" },
    );
    expect(await screen.findByText(/provider-session continuity was cleared/)).toBeVisible();
  });

  it("disables backend switching while this principal has an active run", async () => {
    renderSettings(vi.fn(), true);

    expect(await screen.findByLabelText("Backend")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Switch backend" })).toBeDisabled();
    expect(screen.getByText("Finish or stop the active run before switching.")).toBeVisible();
  });
});
