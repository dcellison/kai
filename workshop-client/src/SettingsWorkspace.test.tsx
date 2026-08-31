import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ChannelAccessError,
  deactivateOperatorModel,
  loadAppearancePreferences,
  loadGitHubSettings,
  loadNotificationPreferences,
  loadChannelNotificationPolicy,
  loadClientPreferences,
  loadModelCatalogue,
  loadPreferenceDocument,
  loadPreferenceHistory,
  loadSettingsWorkspace,
  loadWorkspaceConfig,
  PreferenceRevisionConflictError,
  refreshAllModelCatalogues,
  refreshModelCatalogue,
  restorePreferenceRevision,
  savePreferenceDocument,
  switchWorkspace,
  upsertOperatorModel,
  updateGitHubSettings,
  updateNotificationPreference,
  updateChannelNotificationPolicy,
  updateClientPreference,
  updateAppearancePreference,
  updateRuntimeSettings,
  updateWorkspaceConfig,
} from "./api";
import { AgentRuntimeControls, SettingsWorkspace } from "./SettingsWorkspace";
import { WORKSHOP_THEME_CATALOG } from "./theme";
import type {
  WorkshopPreferenceDocument,
  WorkshopGitHubSettings,
  WorkshopModelCatalogue,
  WorkshopNotificationPreferences,
  WorkshopChannelNotificationPolicy,
  WorkshopClientPreferences,
  WorkshopAppearancePreferences,
  WorkshopSettingsWorkspace,
  WorkshopWorkspaceConfig,
} from "./types";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    loadPreferenceDocument: vi.fn(),
    loadGitHubSettings: vi.fn(),
    loadNotificationPreferences: vi.fn(),
    loadChannelNotificationPolicy: vi.fn(),
    loadClientPreferences: vi.fn(),
    loadModelCatalogue: vi.fn(),
    loadAppearancePreferences: vi.fn(),
    loadPreferenceHistory: vi.fn(),
    loadSettingsWorkspace: vi.fn(),
    loadWorkspaceConfig: vi.fn(),
    restorePreferenceRevision: vi.fn(),
    refreshAllModelCatalogues: vi.fn(),
    refreshModelCatalogue: vi.fn(),
    savePreferenceDocument: vi.fn(),
    switchWorkspace: vi.fn(),
    upsertOperatorModel: vi.fn(),
    deactivateOperatorModel: vi.fn(),
    updateGitHubSettings: vi.fn(),
    updateNotificationPreference: vi.fn(),
    updateChannelNotificationPolicy: vi.fn(),
    updateClientPreference: vi.fn(),
    updateAppearancePreference: vi.fn(),
    updateRuntimeSettings: vi.fn(),
    updateWorkspaceConfig: vi.fn(),
  };
});

const session = {
  channelId: "chn_d3dfdfd7df9151ba8a1742b92403faa5",
  token: "session-secret",
};

const appearancePreferences: WorkshopAppearancePreferences = {
  mutation: null,
  revision: "apr_current",
  themeId: "atom-one-dark",
  themes: WORKSHOP_THEME_CATALOG.map((theme) => ({ ...theme })),
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
  modelCatalogue: {
    errorCode: null,
    errorDetail: null,
    lastAttemptAt: "2026-08-28T10:00:00Z",
    lastKnownGood: false,
    lastSuccessfulRefreshAt: "2026-08-28T10:00:00Z",
    stale: false,
    status: "succeeded",
  },
  modelOptions: [
    {
      displayName: "Claude Sonnet 4.6",
      modelId: "claude-sonnet-4-6",
      retained: true,
      selectable: true,
      sources: ["curated"],
      status: "available",
    },
    {
      displayName: "Claude Opus 4.1",
      modelId: "claude-opus-4-1",
      retained: false,
      selectable: true,
      sources: ["curated"],
      status: "available",
    },
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

const modelCatalogue: WorkshopModelCatalogue = {
  lastKnownGood: false,
  models: runtime.modelOptions ?? [],
  optionId: "claude:anthropic",
  refresh: {
    errorCode: null,
    errorDetail: null,
    expiresAt: "2026-08-29T10:00:00Z",
    generation: 1,
    lastAttemptAt: "2026-08-28T10:00:00Z",
    lastSuccessfulRefreshAt: "2026-08-28T10:00:00Z",
    status: "succeeded",
  },
  runtimeProfileId: runtime.runtimeProfileId,
  stale: false,
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

const githubSettings: WorkshopGitHubSettings = {
  githubLogin: "dcellison",
  issueTriage: { enabled: false, resettable: true, source: "user" },
  mutation: null,
  prReview: { enabled: true, resettable: false, source: "operator" },
  repositories: [
    {
      automationAuthorized: true,
      repository: "dcellison/kai",
      source: "operator",
    },
    {
      automationAuthorized: false,
      repository: "dcellison/notes",
      source: "user",
    },
  ],
  repositoriesResettable: true,
  revision: "ghs_current",
  tokenStored: true,
};

const notificationPreferences: WorkshopNotificationPreferences = {
  destinations: [
    {
      choiceId: "ndst_home",
      displayName: "Home",
      kind: "direct",
      supportedClasses: ["github", "generic"],
    },
    {
      choiceId: "ndst_notifications",
      displayName: "Notifications",
      kind: "notification",
      supportedClasses: ["github", "generic"],
    },
  ],
  mutation: null,
  preferences: [
    {
      destinationChoiceId: "ndst_notifications",
      destinationKind: "notification",
      destinationName: "Notifications",
      displayName: "GitHub",
      editable: true,
      integrationClass: "github",
      resettable: false,
      source: "protected policy",
    },
  ],
  revision: "nps_current",
};

const channelNotificationPolicy: WorkshopChannelNotificationPolicy = {
  channels: [
    {
      channelId: "chn_00000000000000000000000000000001",
      channelName: "General",
      level: "mentions_replies",
      source: "default",
    },
  ],
  doNotDisturb: {
    enabled: false,
    timezone: "America/Toronto",
    start: "22:00",
    end: "07:00",
  },
  levels: ["all", "mentions_replies", "muted"],
  mutedMentionsNotify: true,
  mutation: null,
  revision: "cnp_current",
};

const clientPreferences: WorkshopClientPreferences = {
  mutation: null,
  revision: "cvp_current",
  voiceOutput: {
    available: true,
    unavailableReason: null,
    modes: ["off", "text_and_voice", "voice_only"],
    voices: [
      { value: "alan", displayName: "Alan" },
      { value: "jenny", displayName: "Jenny" },
    ],
    bindings: [
      {
        choiceId: "cbd_telegram",
        clientName: "Telegram",
        mode: "off",
        voice: "alan",
        voiceName: "Alan",
        editable: true,
      },
    ],
  },
};

function renderSettings(
  onDirtyChange = vi.fn(),
  runActive = false,
  onChannelAccessFailure = vi.fn(),
  isAdministrator = true,
): void {
  render(
    <SettingsWorkspace
      onAuthenticationFailure={vi.fn()}
      onChannelAccessFailure={onChannelAccessFailure}
      onClose={vi.fn()}
      onDirtyChange={onDirtyChange}
      isAdministrator={isAdministrator}
      principalName="Daniel"
      roleLabel="Workshop administrator"
      runtimeLabel="Kai"
      runActive={runActive}
      session={session}
    />,
  );
}

function renderAgentRuntime(
  runActive = false,
  onChannelAccessFailure = vi.fn(),
  isAdministrator = true,
): void {
  render(
    <AgentRuntimeControls
      onAuthenticationFailure={vi.fn()}
      onChannelAccessFailure={onChannelAccessFailure}
      isAdministrator={isAdministrator}
      principalName="Daniel"
      roleLabel="Workshop administrator"
      runtimeLabel="Kai"
      runActive={runActive}
      session={session}
    />,
  );
}

async function acceptConfirmation(
  user: ReturnType<typeof userEvent.setup>,
  message: string | RegExp,
): Promise<void> {
  const dialog = await screen.findByRole("dialog", { name: "Continue?" });
  expect(within(dialog).getByText(message)).toBeVisible();
  await user.click(within(dialog).getByRole("button", { name: "Continue" }));
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
    vi.mocked(loadModelCatalogue).mockResolvedValue(modelCatalogue);
    vi.mocked(loadGitHubSettings).mockResolvedValue(githubSettings);
    vi.mocked(loadNotificationPreferences).mockResolvedValue(notificationPreferences);
    vi.mocked(loadChannelNotificationPolicy).mockResolvedValue(channelNotificationPolicy);
    vi.mocked(loadClientPreferences).mockResolvedValue(clientPreferences);
    vi.mocked(loadAppearancePreferences).mockResolvedValue(appearancePreferences);
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
    vi.mocked(refreshModelCatalogue).mockResolvedValue(modelCatalogue);
    vi.mocked(refreshAllModelCatalogues).mockResolvedValue({
      contexts: 7,
      statuses: { unsupported: 3, succeeded: 3, timed_out: 1 },
    });
    vi.mocked(upsertOperatorModel).mockResolvedValue(modelCatalogue);
    vi.mocked(deactivateOperatorModel).mockResolvedValue(modelCatalogue);
    vi.mocked(updateRuntimeSettings).mockResolvedValue(runtime);
    vi.mocked(updateGitHubSettings).mockResolvedValue(githubSettings);
    vi.mocked(updateNotificationPreference).mockResolvedValue(notificationPreferences);
    vi.mocked(updateChannelNotificationPolicy).mockResolvedValue(channelNotificationPolicy);
    vi.mocked(updateClientPreference).mockResolvedValue(clientPreferences);
    vi.mocked(updateAppearancePreference).mockResolvedValue(appearancePreferences);
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

  it("navigates principal-wide settings without agent-specific sections", async () => {
    const user = userEvent.setup();
    renderSettings();
    const editor = await screen.findByLabelText("Preference Markdown");
    const navigation = screen.getByRole("navigation", { name: "Settings sections" });
    const sectionLabels = [
      "Personal preferences",
      "GitHub",
      "Notification delivery",
      "Client preferences",
    ];

    for (const label of sectionLabels) {
      expect(within(navigation).getByRole("button", { name: label })).toBeVisible();
    }
    expect(within(navigation).queryByRole("button", { name: "Routing eligibility" }))
      .toBeNull();
    expect(within(navigation).queryByRole("button", { name: "Runtime settings" }))
      .toBeNull();
    expect(within(navigation).queryByRole("button", { name: "Workspace settings" }))
      .toBeNull();
    expect(screen.queryByLabelText("Task class")).toBeNull();
    expect(
      within(navigation).getByRole("button", { name: "Personal preferences" }),
    ).toHaveAttribute("aria-current", "location");

    await user.type(editor, "Unsaved navigation draft");
    const githubSection = document.getElementById("settings-section-github");
    expect(githubSection).not.toBeNull();
    const scrollIntoView = vi.fn();
    Object.defineProperty(githubSection as HTMLElement, "scrollIntoView", {
      configurable: true,
      value: scrollIntoView,
    });
    await user.click(within(navigation).getByRole("button", { name: "GitHub" }));

    expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth", block: "start" });
    expect(within(navigation).getByRole("button", { name: "GitHub" })).toHaveAttribute(
      "aria-current",
      "location",
    );
    expect(editor).toHaveValue(`${preference.content}Unsaved navigation draft`);
  });

  it("keeps agent model refresh available in a compact catalogue for members", async () => {
    const user = userEvent.setup();
    renderAgentRuntime(false, vi.fn(), false);

    const summary = await screen.findByLabelText("Model catalogue details");
    const catalogue = summary.closest("details");
    expect(catalogue).not.toBeNull();
    if (!catalogue) {
      throw new Error("Model catalogue disclosure is missing");
    }
    expect(catalogue).not.toHaveAttribute("open");
    expect(await within(catalogue).findByText("2")).toBeVisible();
    expect(within(catalogue).getByText("claude:anthropic")).toBeVisible();
    expect(within(catalogue).getByText("succeeded")).toBeVisible();
    expect(within(catalogue).getByText(/Aug 28, 2026/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Refresh models" })).not.toBeVisible();

    await user.click(summary);

    expect(screen.getByRole("button", { name: "Refresh models" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Refresh all contexts" })).toBeNull();
    expect(screen.queryByLabelText("Operator-managed model ID")).toBeNull();
  });

  it("groups compact agent runtime controls into explicit paired rows", async () => {
    renderAgentRuntime();

    expect(await screen.findByRole("heading", { name: "Runtime settings" })).toBeVisible();
    const controls = document.getElementById("agent-runtime-settings");
    expect(controls).toHaveClass("agent-runtime-controls");
    expect(controls).not.toHaveClass("settings-workspace");
    expect(controls?.querySelector(".settings-scroll")).toBeNull();
    expect(screen.getByText("01")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Workspace settings" })).toBeVisible();
    expect(screen.getByText("02")).toBeVisible();

    const backendPair = screen.getByLabelText("Backend").closest(".settings-card-pair");
    expect(backendPair).not.toBeNull();
    expect(within(backendPair as HTMLElement).getByText("Policy-controlled")).toBeVisible();

    const runtimePair = screen.getByLabelText("Runtime model").closest(".settings-card-pair");
    expect(runtimePair).not.toBeNull();
    expect(within(runtimePair as HTMLElement).getByLabelText("Response timeout")).toBeVisible();

    const workspacePair = screen
      .getByLabelText("Workspace model override")
      .closest(".settings-card-pair");
    expect(workspacePair).not.toBeNull();
    expect(
      within(workspacePair as HTMLElement).getByLabelText("Workspace timeout override"),
    ).toBeVisible();

    const prompt = screen.getByLabelText("Workspace system prompt");
    expect(prompt.closest(".settings-card-stack")).not.toBeNull();
    expect(prompt.closest(".settings-card-pair")).toBeNull();
  });

  it("refreshes one catalogue or every context without changing runtime settings", async () => {
    const user = userEvent.setup();
    renderAgentRuntime();

    await user.click(await screen.findByText("Model catalogue"));
    await user.click(await screen.findByRole("button", { name: "Refresh models" }));
    await waitFor(() => {
      expect(refreshModelCatalogue).toHaveBeenCalledWith(session, "claude:anthropic");
    });
    expect(updateRuntimeSettings).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Refresh all contexts" }));
    await waitFor(() => {
      expect(refreshAllModelCatalogues).toHaveBeenCalledWith(session);
    });
    expect(
      screen.getByText(
        "Model catalogue refresh completed across 7 authorized contexts: 3 succeeded, 3 unsupported, 1 timed out.",
      ),
    ).toBeVisible();
    expect(updateRuntimeSettings).not.toHaveBeenCalled();
  });

  it("updates the current settings destination during manual scrolling", async () => {
    renderSettings();
    await screen.findByLabelText("Preference Markdown");
    await screen.findByRole("heading", { name: "Client preferences" });
    const navigation = screen.getByRole("navigation", { name: "Settings sections" });
    const scroll = document.querySelector<HTMLElement>(".settings-scroll");
    expect(scroll).not.toBeNull();
    Object.defineProperties(scroll as HTMLElement, {
      clientHeight: { configurable: true, value: 500 },
      scrollHeight: { configurable: true, value: 1600 },
      scrollTop: { configurable: true, value: 420 },
    });
    vi.spyOn(scroll as HTMLElement, "getBoundingClientRect").mockReturnValue({
      top: 100,
    } as DOMRect);
    const positions: Record<string, number> = {
      "settings-section-personal-preferences": -500,
      "settings-section-github": 110,
      "settings-section-notifications": 400,
      "settings-section-clients": 700,
    };
    for (const [id, top] of Object.entries(positions)) {
      const section = document.getElementById(id);
      expect(section).not.toBeNull();
      vi.spyOn(section as HTMLElement, "getBoundingClientRect").mockReturnValue({
        top,
      } as DOMRect);
    }

    fireEvent.scroll(scroll as HTMLElement);

    expect(within(navigation).getByRole("button", { name: "GitHub" })).toHaveAttribute(
      "aria-current",
      "location",
    );
  });

  it("shows private preferences without loading agent runtime settings", async () => {
    renderSettings();

    expect(await screen.findByLabelText("Preference Markdown")).toHaveValue(
      preference.content,
    );
    expect(screen.queryByRole("heading", { name: "Runtime settings" })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Workspace settings" })).toBeNull();
    expect(screen.queryByLabelText("Backend")).toBeNull();
    expect(loadSettingsWorkspace).not.toHaveBeenCalled();
    expect(loadWorkspaceConfig).not.toHaveBeenCalled();
  });

  it("shows policy-bounded settings only in the selected agent", async () => {
    renderAgentRuntime();

    expect(await screen.findByLabelText("Backend")).toHaveValue("claude:anthropic");
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
    await acceptConfirmation(
      user,
      "Replace the latest saved preferences with your reviewed draft?",
    );

    await waitFor(() => expect(savePreferenceDocument).toHaveBeenLastCalledWith(
      session,
      "My reviewed draft",
      "pref_latest",
    ));
  });

  it("confirms and restores a private preference revision", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByLabelText("Preference Markdown");
    await user.click(screen.getByText("Private revision history (1)"));
    await user.click(screen.getByRole("button", { name: "Restore…" }));
    await acceptConfirmation(
      user,
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
    renderAgentRuntime();
    const runtimeTimeout = await screen.findByLabelText("Response timeout");
    await user.clear(runtimeTimeout);
    await user.type(runtimeTimeout, "601");
    const runtimeForm = runtimeTimeout.closest("form");
    expect(runtimeForm).not.toBeNull();
    await user.click(within(runtimeForm as HTMLFormElement).getByRole("button", { name: "Apply" }));
    await acceptConfirmation(user, /Change the response timeout/);

    expect(updateRuntimeSettings).toHaveBeenCalledWith(
      session,
      "sws_current",
      { field: "timeout", value: 601 },
    );
    expect(await screen.findByText(/provider-session continuity was cleared/)).toBeVisible();

    await user.selectOptions(screen.getByLabelText("Active workspace"), "1");
    await acceptConfirmation(user, /Switch the active workspace/);
    expect(switchWorkspace).toHaveBeenCalledWith(session, "/srv/home", "sws_timeout");

    const prompt = screen.getByLabelText("Workspace system prompt");
    await user.type(prompt, "Use the project conventions.");
    const promptForm = prompt.closest("form");
    expect(promptForm).not.toBeNull();
    await user.click(within(promptForm as HTMLFormElement).getByRole("button", { name: "Apply" }));
    await acceptConfirmation(user, /Apply this system prompt/);
    expect(updateWorkspaceConfig).toHaveBeenCalledWith(
      session,
      "sws_workspace",
      { field: "prompt", value: "Use the project conventions." },
    );
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
    renderAgentRuntime();
    const backend = await screen.findByLabelText("Backend");

    await user.selectOptions(backend, "codex:openai");
    await user.click(screen.getByRole("button", { name: "Switch backend" }));
    await acceptConfirmation(user, /other people and Kai itself will not restart/);
    expect(window.confirm).not.toHaveBeenCalled();
    expect(updateRuntimeSettings).toHaveBeenCalledWith(
      session,
      "sws_current",
      { field: "backend", value: "codex:openai" },
    );
    expect(await screen.findByText(/provider-session continuity was cleared/)).toBeVisible();
  });

  it("disables backend switching while this principal has an active run", async () => {
    renderAgentRuntime(true);

    expect(await screen.findByLabelText("Backend")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Switch backend" })).toBeDisabled();
    expect(screen.getByText("Finish or stop the active run before switching.")).toBeVisible();
  });

  it("keeps valid runtime settings when workspace overrides are unavailable", async () => {
    const onChannelAccessFailure = vi.fn();
    vi.mocked(loadWorkspaceConfig).mockRejectedValue(
      new ChannelAccessError("This session cannot access that Workshop channel."),
    );

    renderAgentRuntime(false, onChannelAccessFailure);

    expect(await screen.findByLabelText("Backend")).toHaveValue("claude:anthropic");
    expect(screen.getByRole("heading", { name: "Workspace settings" })).toBeVisible();
    expect(screen.getByText("This session cannot access that Workshop channel.")).toBeVisible();
    expect(onChannelAccessFailure).not.toHaveBeenCalled();
    expect(loadSettingsWorkspace).toHaveBeenCalledTimes(1);
    expect(loadWorkspaceConfig).toHaveBeenCalledTimes(1);
  });

  it("manages redacted principal-owned GitHub settings", async () => {
    const user = userEvent.setup();
    vi.mocked(updateGitHubSettings)
      .mockResolvedValueOnce({
        ...githubSettings,
        issueTriage: { enabled: true, resettable: true, source: "user" },
        mutation: { changed: true, operation: "set_github_issue_triage" },
        revision: "ghs_toggle",
      })
      .mockResolvedValueOnce({
        ...githubSettings,
        mutation: { changed: true, operation: "subscribe_github_repository" },
        repositories: [
          ...githubSettings.repositories,
          {
            automationAuthorized: false,
            repository: "dcellison/new-repo",
            source: "user",
          },
        ],
        revision: "ghs_repo",
      })
      .mockResolvedValueOnce({
        ...githubSettings,
        mutation: { changed: true, operation: "replace_github_token" },
        revision: "ghs_token",
      });
    renderSettings();

    expect(await screen.findByRole("heading", { name: "GitHub" })).toBeVisible();
    expect(screen.getByText("dcellison")).toBeVisible();
    expect(screen.getByText("dcellison/kai")).toBeVisible();
    expect(screen.getByText("operator · automation authorized")).toBeVisible();
    expect(screen.getByText("user · notifications only")).toBeVisible();
    expect(screen.queryByDisplayValue(/token/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Turn on" }));
    expect(updateGitHubSettings).toHaveBeenNthCalledWith(
      1,
      session,
      "ghs_current",
      { field: "toggle", name: "issue_triage", enabled: true },
    );

    await user.type(screen.getByLabelText("Add notification subscription"), "dcellison/new-repo");
    await user.click(screen.getByRole("button", { name: "Add" }));
    expect(updateGitHubSettings).toHaveBeenNthCalledWith(
      2,
      session,
      "ghs_toggle",
      { field: "repository", name: "dcellison/new-repo", subscribed: true },
    );

    await user.type(screen.getByLabelText("GitHub access token"), "replacement-secret");
    await user.click(screen.getByRole("button", { name: "Replace token" }));
    await acceptConfirmation(user, "Replace the stored GitHub token?");
    expect(updateGitHubSettings).toHaveBeenNthCalledWith(
      3,
      session,
      "ghs_repo",
      { field: "token", token: "replacement-secret" },
    );
    expect(screen.getByLabelText("GitHub access token")).toHaveValue("");
  });

  it("selects an authorized personal notification destination", async () => {
    const user = userEvent.setup();
    vi.mocked(updateNotificationPreference).mockResolvedValue({
      ...notificationPreferences,
      mutation: {
        changed: true,
        operation: "select_github_notification_destination",
      },
      preferences: [
        {
          ...notificationPreferences.preferences[0],
          destinationChoiceId: "ndst_home",
          destinationKind: "direct",
          destinationName: "Home",
          resettable: true,
          source: "personal override",
        },
      ],
      revision: "nps_selected",
    });
    renderSettings();

    const selector = await screen.findByLabelText("GitHub");
    const card = selector.closest("article");
    expect(card).not.toBeNull();
    expect(selector).toHaveValue("ndst_notifications");
    expect(within(card as HTMLElement).getByText(/Effective:/)).toHaveTextContent(
      "Effective: Notifications · protected policy",
    );

    await user.selectOptions(selector, "ndst_home");

    expect(updateNotificationPreference).toHaveBeenCalledWith(
      session,
      "nps_current",
      {
        field: "destination",
        integrationClass: "github",
        choiceId: "ndst_home",
      },
    );
    await waitFor(() => expect(
      within(card as HTMLElement).getByText(/Effective:/),
    ).toHaveTextContent(
      "Effective: Home · personal override",
    ));
  });

  it("edits channel notification level and do-not-disturb policy", async () => {
    const user = userEvent.setup();
    vi.mocked(updateChannelNotificationPolicy).mockImplementation(async (_session, _revision, change) => ({
      ...channelNotificationPolicy,
      channels: change.field === "channel"
        ? [{ ...channelNotificationPolicy.channels[0], level: change.level }]
        : channelNotificationPolicy.channels,
      doNotDisturb: change.field === "do_not_disturb"
        ? {
            enabled: change.enabled,
            timezone: change.timezone,
            start: change.start,
            end: change.end,
          }
        : channelNotificationPolicy.doNotDisturb,
      mutation: { changed: true, operation: `set_${change.field}` },
      revision: "cnp_changed",
    }));
    renderSettings();

    const channel = await screen.findByLabelText("General");
    await user.selectOptions(channel, "muted");
    expect(updateChannelNotificationPolicy).toHaveBeenCalledWith(
      session,
      "cnp_current",
      {
        field: "channel",
        channelId: "chn_00000000000000000000000000000001",
        level: "muted",
      },
    );
    expect(channel).toHaveValue("muted");
    expect(screen.getByLabelText("Direct mentions override muted channels")).toBeChecked();
    expect(screen.getByLabelText("Timezone")).toHaveValue("America/Toronto");
  });

  it("keeps notification controls together and saves validated DND drafts", async () => {
    const user = userEvent.setup();
    vi.mocked(updateChannelNotificationPolicy).mockImplementation(async (_session, _revision, change) => ({
      ...channelNotificationPolicy,
      doNotDisturb: change.field === "do_not_disturb"
        ? {
            enabled: change.enabled,
            timezone: change.timezone,
            start: change.start,
            end: change.end,
          }
        : channelNotificationPolicy.doNotDisturb,
      mutation: { changed: true, operation: `set_${change.field}` },
      revision: "cnp_changed",
    }));
    renderSettings();

    const heading = await screen.findByRole("heading", { name: "Notification delivery" });
    const section = heading.closest("section");
    const content = section?.querySelector(".settings-section-content");
    expect(section).not.toBeNull();
    expect(content).not.toBeNull();
    expect(section?.children).toHaveLength(2);
    expect(content).toContainElement(screen.getByLabelText("GitHub").closest("article"));

    const timezone = screen.getByLabelText("Timezone");
    expect(timezone.tagName).toBe("SELECT");
    expect(within(timezone).getByRole("option", { name: "America/Toronto" })).toBeVisible();

    const start = screen.getByLabelText("Start");
    const end = screen.getByLabelText("End");
    expect(start).toHaveAttribute("type", "text");
    expect(end).toHaveAttribute("type", "text");
    await user.clear(start);
    expect(screen.getByRole("button", { name: "Save schedule" })).toBeDisabled();
    await user.type(start, "09:30");
    await user.clear(end);
    await user.type(end, "17:45");
    await user.click(screen.getByLabelText("Pause external delivery on this schedule"));
    await user.click(screen.getByRole("button", { name: "Save schedule" }));

    expect(updateChannelNotificationPolicy).toHaveBeenCalledWith(
      session,
      "cnp_current",
      {
        field: "do_not_disturb",
        enabled: true,
        timezone: "America/Toronto",
        start: "09:30",
        end: "17:45",
      },
    );
    const mutedMention = screen.getByLabelText("Direct mentions override muted channels");
    expect(mutedMention.nextElementSibling).toHaveTextContent(
      "Direct mentions override muted channels",
    );
  });

  it("edits voice output for the authenticated client binding", async () => {
    const user = userEvent.setup();
    vi.mocked(updateClientPreference).mockResolvedValue({
      ...clientPreferences,
      mutation: { changed: true, operation: "set_client_voice_mode" },
      revision: "cvp_changed",
      voiceOutput: {
        ...clientPreferences.voiceOutput,
        bindings: [
          { ...clientPreferences.voiceOutput.bindings[0], mode: "text_and_voice" },
        ],
      },
    });
    renderSettings();

    const mode = await screen.findByLabelText("Response format");
    expect(screen.getByRole("heading", { name: "Telegram voice output" })).toBeVisible();
    expect(mode).toHaveValue("off");
    await user.selectOptions(mode, "text_and_voice");

    expect(updateClientPreference).toHaveBeenCalledWith(
      session,
      "cvp_current",
      {
        field: "mode",
        bindingChoiceId: "cbd_telegram",
        value: "text_and_voice",
      },
    );
    await waitFor(() => expect(mode).toHaveValue("text_and_voice"));
  });

  it("loads the principal-scoped Workshop appearance", async () => {
    renderSettings();

    const selector = await screen.findByLabelText("Theme");
    expect(selector).toHaveValue("atom-one-dark");
    expect(screen.getByRole("option", { name: "Atom One Dark" })).toBeVisible();
    expect(screen.getByRole("option", { name: "GitHub Light Default" })).toBeVisible();
    expect(screen.getByRole("option", { name: "GitHub Dark Dimmed" })).toBeVisible();
    expect((selector as HTMLSelectElement).options).toHaveLength(11);
    expect(document.documentElement.dataset.workshopTheme).toBe("atom-one-dark");
  });

  it("shows unavailable voice capability without editable controls", async () => {
    vi.mocked(loadClientPreferences).mockResolvedValue({
      ...clientPreferences,
      voiceOutput: {
        ...clientPreferences.voiceOutput,
        available: false,
        unavailableReason: "Voice output is not enabled for an eligible client.",
        bindings: clientPreferences.voiceOutput.bindings.map((binding) => ({
          ...binding,
          editable: false,
        })),
      },
    });
    renderSettings();

    expect(await screen.findByText(
      "Voice output is not enabled for an eligible client.",
    )).toBeVisible();
    expect(screen.getByLabelText("Response format")).toBeDisabled();
    expect(screen.getByLabelText("Voice")).toBeDisabled();
  });
});
