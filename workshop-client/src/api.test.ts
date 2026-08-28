import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  EventStreamDecoder,
  MemoryRevisionConflictError,
  PreferenceRevisionConflictError,
  SettingsRevisionConflictError,
  cancelRun,
  createMemoryFact,
  deleteMemories,
  deleteMemory,
  deactivateOperatorModel,
  editMemory,
  loadArtifactBlob,
  loadAppearancePreferences,
  loadEarlierTimeline,
  loadMemoryDetail,
  loadMemoryRecords,
  loadMemorySource,
  loadMemoryStats,
  loadModelCatalogue,
  loadNavigation,
  loadGitHubSettings,
  loadNotificationPreferences,
  loadClientPreferences,
  loadPreferenceDocument,
  loadPreferenceHistory,
  loadRun,
  loadTimeline,
  loadWorkspaceConfig,
  moveMemoriesScope,
  moveMemoryScope,
  redeemEnrollment,
  refreshAllModelCatalogues,
  refreshModelCatalogue,
  searchMemories,
  restorePreferenceRevision,
  savePreferenceDocument,
  submitCommand,
  streamTimeline,
  updateRuntimeSettings,
  updateGitHubSettings,
  updateNotificationPreference,
  updateClientPreference,
  updateAppearancePreference,
  updateWorkspaceConfig,
  upsertOperatorModel,
} from "./api";
import type { WorkshopSession } from "./types";
import { WORKSHOP_THEME_CATALOG } from "./theme";

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

function memoryRecord(): Record<string, unknown> {
  return {
    memory_id: "memory-1",
    kind: "fact",
    source: "extracted",
    memory_type: "fact",
    preview: "Daniel prefers concise output.",
    revision: "mr1_test-revision",
    tags: ["preference"],
    speaker: "user",
    confidence: 1,
    created_at: "2026-08-24T10:00:00Z",
    updated_at: "2026-08-24T10:00:00Z",
    scope: {
      scope: "global",
      project_id: null,
      scope_confidence: 1,
      scope_source: "operator",
      legacy_defaulted: false,
      invalid_defaulted: false,
      retrievable: true,
      exclusion_reason: null,
    },
  };
}

function settingsPayload(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    version: 1,
    channel_id: channelId,
    principal_id: "prn_00000000000000000000000000000001",
    runtime_profile_id: "rtp_00000000000000000000000000000001",
    backend: "claude",
    backend_option_id: "claude:anthropic",
    provider: "anthropic",
    backend_options: [
      { option_id: "claude:anthropic", backend: "claude", provider: "anthropic", current: true },
      { option_id: "codex:openai", backend: "codex", provider: "openai", current: false },
    ],
    model: {
      value: "claude-sonnet-4-6",
      source: "runtime policy",
      default_value: "claude-sonnet-4-6",
    },
    model_options: [
      {
        model_id: "claude-sonnet-4-6",
        display_name: "Claude Sonnet 4.6",
        status: "available",
        selectable: true,
        retained: true,
      },
    ],
    model_catalogue: {
      status: "succeeded",
      stale: false,
      last_known_good: false,
      last_attempt_at: "2026-08-28T10:00:00Z",
      last_successful_refresh_at: "2026-08-28T10:00:00Z",
      error_code: null,
      error_detail: null,
    },
    timeout_seconds: {
      value: 1800,
      source: "runtime policy",
      default_value: 1800,
    },
    workspace: "/srv/kai",
    workspaces: [
      { path: "/srv/kai", name: "Kai", current: true, home: false },
    ],
    revision: "sws_current",
    capabilities: [
      {
        field: "timeout",
        scope: "runtime",
        value_type: "integer_seconds",
        resettable: true,
        choices: null,
        minimum: 1,
        maximum: 1800,
      },
    ],
    mutation: null,
    ...overrides,
  };
}

function modelCataloguePayload(): Record<string, unknown> {
  return {
    version: 1,
    principal_id: "prn_00000000000000000000000000000001",
    runtime_profile_id: "rtp_00000000000000000000000000000001",
    option_id: "claude:anthropic",
    stale: false,
    last_known_good: false,
    refresh: {
      status: "succeeded",
      generation: 2,
      last_attempt_at: "2026-08-28T10:00:00Z",
      last_successful_refresh_at: "2026-08-28T10:00:00Z",
      expires_at: "2026-08-29T10:00:00Z",
      error_code: null,
      error_detail: null,
    },
    models: [
      {
        model_id: "claude-sonnet-4-6",
        display_name: "Claude Sonnet 4.6",
        status: "available",
        selectable: true,
        retained: true,
        sources: ["discovered:claude"],
      },
    ],
  };
}

function workspaceConfigPayload(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    version: 1,
    workspace: "/srv/kai",
    model: {
      value: "claude-sonnet-4-6",
      source: "runtime policy",
      default_value: "claude-sonnet-4-6",
    },
    timeout_seconds: {
      value: 1800,
      source: "runtime policy",
      default_value: 1800,
    },
    environment_keys: ["PROTECTED_KEY"],
    prompt: null,
    has_prompt: false,
    prompt_source: null,
    override_fields: [],
    revision: "sws_workspace",
    capabilities: [
      {
        field: "prompt",
        scope: "workspace",
        value_type: "text",
        resettable: true,
        choices: null,
        minimum: null,
        maximum: 4000,
      },
    ],
    mutation: null,
    ...overrides,
  };
}

function githubSettingsPayload(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    version: 1,
    github_login: "dcellison",
    repositories_resettable: true,
    repositories: [
      {
        repository: "owner/repo",
        source: "operator",
        automation_authorized: true,
      },
    ],
    pr_review: { enabled: true, source: "operator", resettable: false },
    issue_triage: { enabled: false, source: "user", resettable: true },
    token_stored: true,
    revision: "ghs_current",
    mutation: null,
    ...overrides,
  };
}

function notificationPreferencesPayload(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    version: 1,
    destinations: [
      {
        choice_id: "ndst_home",
        display_name: "Home",
        kind: "direct",
        supported_classes: ["github", "generic"],
      },
      {
        choice_id: "ndst_notifications",
        display_name: "Notifications",
        kind: "notification",
        supported_classes: ["github", "generic"],
      },
    ],
    preferences: [
      {
        integration_class: "github",
        display_name: "GitHub",
        destination_choice_id: "ndst_notifications",
        destination_name: "Notifications",
        destination_kind: "notification",
        source: "protected policy",
        editable: true,
        resettable: false,
      },
    ],
    revision: "nps_current",
    mutation: null,
    ...overrides,
  };
}

function clientPreferencesPayload(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    version: 1,
    voice_output: {
      available: true,
      unavailable_reason: null,
      modes: ["off", "text_and_voice", "voice_only"],
      voices: [
        { value: "alan", display_name: "Alan" },
        { value: "jenny", display_name: "Jenny" },
      ],
      bindings: [
        {
          choice_id: "cbd_telegram",
          client_name: "Telegram",
          mode: "off",
          voice: "alan",
          voice_name: "Alan",
          editable: true,
        },
      ],
    },
    revision: "cvp_current",
    mutation: null,
    ...overrides,
  };
}

function preferencePayload(
  content = "# Preferences\n\nBe concise.\n",
  revision = "pref_current",
): Record<string, unknown> {
  return {
    version: 1,
    document: {
      content,
      revision,
      updated_at: "2026-08-26T10:00:00Z",
      size_bytes: new TextEncoder().encode(content).length,
      max_bytes: 65536,
      editable: true,
    },
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

  it("loads only the newest window with a single tail request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        version: 1,
        channel_id: channelId,
        messages: [message(10), message(20)],
        next_cursor: null,
        previous_cursor: "earlier-page",
        through_position: 20,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const snapshot = await loadTimeline(session, new AbortController().signal);

    expect(snapshot.throughPosition).toBe(20);
    expect(snapshot.previousCursor).toBe("earlier-page");
    expect(snapshot.messages.map((item) => item.eventPosition)).toEqual([10, 20]);
    expect(fetchMock).toHaveBeenCalledOnce();
    const [path, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toContain("tail=1");
    expect(path).toContain("limit=100");
    expect(new Headers(options.headers).get("Authorization")).toBe(
      "Bearer session-secret",
    );
  });

  it("loads an earlier page from the same snapshot via its cursor", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        version: 1,
        channel_id: channelId,
        messages: [message(5)],
        next_cursor: null,
        previous_cursor: null,
        through_position: 20,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const page = await loadEarlierTimeline(
      session,
      "earlier-page",
      20,
      new AbortController().signal,
    );

    expect(page.messages.map((item) => item.eventPosition)).toEqual([5]);
    expect(page.previousCursor).toBeNull();
    const [path] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toContain("cursor=earlier-page");
    expect(path).not.toContain("tail=");
  });

  it("parses the stable Workshop memory read contracts", async () => {
    const responses = [
      Response.json({
        version: 1,
        stats: {
          total: 1,
          facts: 1,
          episodes: 0,
          by_source: { extracted: 1 },
          by_type: { fact: 1 },
          by_scope: { global: 1 },
          allowed_projects: [{ project_id: "kai", display_name: "Kai" }],
        },
      }),
      Response.json({ version: 1, records: [memoryRecord()], next_cursor: null }),
      Response.json({
        version: 1,
        active_project_id: "kai",
        reason: "ok",
        hits: [
          {
            record: memoryRecord(),
            raw_score: 0.9,
            adjusted_score: 0.8,
            compact_recall: "{\"record_type\":\"memory\"}",
          },
        ],
      }),
      Response.json({
        version: 1,
        record: {
          ...memoryRecord(),
          content: "Daniel prefers concise output.",
          compact_recall: "{\"record_type\":\"memory\"}",
          confirmation_quote: null,
          prompt_version: "v1",
          episode: null,
        },
      }),
      Response.json({
        version: 1,
        source_context: {
          status: "unavailable",
          reason: "legacy_source",
          run_id: null,
          source: null,
          result: null,
        },
      }),
    ];
    const fetchMock = vi.fn().mockImplementation(() => responses.shift());
    vi.stubGlobal("fetch", fetchMock);

    await expect(loadMemoryStats("session-secret")).resolves.toMatchObject({
      total: 1,
      facts: 1,
      bySource: { extracted: 1 },
    });
    await expect(
      loadMemoryRecords("session-secret", {
        kind: "fact",
        projectId: "kai",
        limit: 25,
        order: "oldest",
      }),
    ).resolves.toMatchObject({
      nextCursor: null,
      records: [{ memoryId: "memory-1", source: "extracted" }],
    });
    await expect(
      searchMemories("session-secret", "concise output", {
        scope: "project",
        tag: "preference",
      }),
    ).resolves.toMatchObject({
      activeProjectId: "kai",
      hits: [{ record: { memoryId: "memory-1" }, adjustedScore: 0.8 }],
    });
    await expect(loadMemoryDetail("session-secret", "memory-1")).resolves.toMatchObject({
      memoryId: "memory-1",
      content: "Daniel prefers concise output.",
      promptVersion: "v1",
    });
    await expect(loadMemorySource("session-secret", "memory-1")).resolves.toEqual({
      status: "unavailable",
      reason: "legacy_source",
      runId: null,
      source: null,
      result: null,
    });
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      "/v1/memory/records?kind=fact&project_id=kai&limit=25&order=oldest",
    );
    expect(fetchMock.mock.calls[2]?.[0]).toBe(
      "/v1/memory/search?tag=preference&scope=project&q=concise+output",
    );
  });

  it("loads and mutates principal preferences with revision protection", async () => {
    const responses = [
      Response.json(preferencePayload()),
      Response.json({
        version: 1,
        limit: 20,
        revisions: [
          {
            revision: "pref_previous",
            updated_at: "2026-08-25T10:00:00Z",
            size_bytes: 24,
          },
        ],
      }),
      Response.json(preferencePayload("# Preferences\n\nUse examples.\n", "pref_next")),
      Response.json(preferencePayload("# Preferences\n\nBe concise.\n", "pref_restored")),
    ];
    const fetchMock = vi.fn().mockImplementation(() => responses.shift());
    vi.stubGlobal("fetch", fetchMock);

    await expect(loadPreferenceDocument(session)).resolves.toMatchObject({
      content: "# Preferences\n\nBe concise.\n",
      editable: true,
      revision: "pref_current",
    });
    await expect(loadPreferenceHistory(session)).resolves.toEqual({
      limit: 20,
      revisions: [
        {
          revision: "pref_previous",
          updatedAt: "2026-08-25T10:00:00Z",
          sizeBytes: 24,
        },
      ],
    });
    await expect(
      savePreferenceDocument(
        session,
        "# Preferences\n\nUse examples.\n",
        "pref_current",
      ),
    ).resolves.toMatchObject({ revision: "pref_next" });
    await expect(
      restorePreferenceRevision(session, "pref_previous", "pref_next"),
    ).resolves.toMatchObject({ revision: "pref_restored" });

    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      "/v1/preferences",
      "/v1/preferences/revisions",
      "/v1/preferences",
      "/v1/preferences/revisions/pref_previous/restore",
    ]);
    expect(JSON.parse((fetchMock.mock.calls[2]?.[1] as RequestInit).body as string)).toEqual({
      content: "# Preferences\n\nUse examples.\n",
      revision: "pref_current",
    });
    expect(JSON.parse((fetchMock.mock.calls[3]?.[1] as RequestInit).body as string)).toEqual({
      revision: "pref_next",
    });
  });

  it("surfaces preference conflicts with the authoritative revision", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json(
          {
            error: {
              code: "revision_conflict",
              message: "Preferences changed",
              current_revision: "pref_latest",
            },
          },
          { status: 409 },
        ),
      ),
    );

    const failure = await savePreferenceDocument(
      session,
      "Draft",
      "pref_stale",
    ).catch((caught: unknown) => caught);

    expect(failure).toBeInstanceOf(PreferenceRevisionConflictError);
    expect(failure).toMatchObject({
      currentRevision: "pref_latest",
      message: "Preferences changed",
    });
  });

  it("loads and mutates policy-bounded runtime and workspace settings", async () => {
    const responses = [
      Response.json(settingsPayload()),
      Response.json(workspaceConfigPayload()),
    ];
    const fetchMock = vi.fn().mockImplementation(() => responses.shift());
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      updateRuntimeSettings(session, "sws_current", {
        field: "timeout",
        value: 601,
      }),
    ).resolves.toMatchObject({
      backend: "claude",
      provider: "anthropic",
      timeoutSeconds: { value: 1800 },
    });
    await expect(loadWorkspaceConfig(session)).resolves.toMatchObject({
      environmentKeys: ["PROTECTED_KEY"],
      hasPrompt: false,
      revision: "sws_workspace",
    });

    expect(fetchMock.mock.calls[0]?.[0]).toBe(`/v1/channels/${channelId}/settings`);
    expect(JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string)).toEqual({
      revision: "sws_current",
      timeout_seconds: 601,
    });
  });

  it("uses the canonical catalogue for principal refresh and operator actions", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json(modelCataloguePayload()))
      .mockResolvedValueOnce(Response.json(modelCataloguePayload()))
      .mockResolvedValueOnce(
        Response.json({
          version: 1,
          contexts: 2,
          statuses: { succeeded: 2 },
          selection_changed: false,
        }),
      )
      .mockResolvedValueOnce(Response.json(modelCataloguePayload()))
      .mockResolvedValueOnce(Response.json(modelCataloguePayload()));
    vi.stubGlobal("fetch", fetchMock);

    await expect(loadModelCatalogue(session, "claude:anthropic")).resolves.toMatchObject({
      optionId: "claude:anthropic",
      stale: false,
      models: [{ modelId: "claude-sonnet-4-6", selectable: true }],
    });
    await expect(refreshModelCatalogue(session, "claude:anthropic")).resolves.toMatchObject({
      refresh: { status: "succeeded" },
    });
    await expect(refreshAllModelCatalogues(session)).resolves.toEqual({
      contexts: 2,
      statuses: { succeeded: 2 },
    });
    await expect(
      upsertOperatorModel(session, "claude:anthropic", "new-model", "New Model"),
    ).resolves.toMatchObject({ optionId: "claude:anthropic" });
    await expect(
      deactivateOperatorModel(session, "claude:anthropic", "new-model"),
    ).resolves.toMatchObject({ optionId: "claude:anthropic" });

    expect(fetchMock.mock.calls.map((call) => [call[0], (call[1] as RequestInit).method ?? "GET"])).toEqual([
      [`/v1/channels/${channelId}/models?option_id=claude%3Aanthropic`, "GET"],
      [`/v1/channels/${channelId}/models`, "POST"],
      ["/v1/settings/model-catalogue/refresh-all", "POST"],
      [`/v1/channels/${channelId}/models`, "PUT"],
      [`/v1/channels/${channelId}/models`, "DELETE"],
    ]);
  });

  it("uses typed workspace mutations and rejects stale settings", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json(workspaceConfigPayload()))
      .mockResolvedValueOnce(
        Response.json(
          { error: { code: "settings_conflict", message: "Stale settings" } },
          { status: 409 },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      updateWorkspaceConfig(session, "sws_workspace", {
        field: "prompt",
        value: "Use the project conventions.",
      }),
    ).resolves.toMatchObject({ revision: "sws_workspace" });
    expect(JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string)).toEqual({
      revision: "sws_workspace",
      field: "prompt",
      value: "Use the project conventions.",
    });

    await expect(
      updateRuntimeSettings(session, "sws_stale", {
        field: "reset",
        value: "timeout",
      }),
    ).rejects.toBeInstanceOf(SettingsRevisionConflictError);
  });

  it("loads and mutates redacted personal GitHub settings", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json(githubSettingsPayload()))
      .mockResolvedValueOnce(Response.json(githubSettingsPayload({
        mutation: { operation: "replace_github_token", changed: true },
      })));
    vi.stubGlobal("fetch", fetchMock);

    const loaded = await loadGitHubSettings(session);
    expect(loaded).toMatchObject({
      repositories: [{ repository: "owner/repo", automationAuthorized: true }],
      tokenStored: true,
    });
    const updated = await updateGitHubSettings(session, "ghs_current", {
      field: "token",
      token: "new-secret",
    });
    expect(updated).toMatchObject({
      mutation: { operation: "replace_github_token", changed: true },
      tokenStored: true,
    });

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/v1/settings/github");
    expect(JSON.parse((fetchMock.mock.calls[1]?.[1] as RequestInit).body as string)).toEqual({
      revision: "ghs_current",
      token: "new-secret",
    });
    expect(JSON.stringify(loaded)).not.toContain("new-secret");
    expect(JSON.stringify(updated)).not.toContain("new-secret");
  });

  it("rejects stale GitHub settings mutations", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json(
          { error: { code: "settings_conflict", message: "Stale GitHub settings" } },
          { status: 409 },
        ),
      ),
    );

    await expect(
      updateGitHubSettings(session, "ghs_stale", {
        field: "repository",
        name: "owner/repo",
        subscribed: false,
      }),
    ).rejects.toBeInstanceOf(SettingsRevisionConflictError);
  });

  it("loads and mutates opaque notification destinations", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json(notificationPreferencesPayload()))
      .mockResolvedValueOnce(Response.json(notificationPreferencesPayload({
        mutation: {
          operation: "select_github_notification_destination",
          changed: true,
        },
      })));
    vi.stubGlobal("fetch", fetchMock);

    const loaded = await loadNotificationPreferences(session);
    expect(loaded).toMatchObject({
      destinations: [
        { choiceId: "ndst_home", displayName: "Home" },
        { choiceId: "ndst_notifications", displayName: "Notifications" },
      ],
      preferences: [{ integrationClass: "github", destinationName: "Notifications" }],
    });
    await expect(
      updateNotificationPreference(session, "nps_current", {
        field: "destination",
        integrationClass: "github",
        choiceId: "ndst_home",
      }),
    ).resolves.toMatchObject({
      mutation: {
        operation: "select_github_notification_destination",
        changed: true,
      },
    });

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/v1/settings/notifications");
    expect(JSON.parse((fetchMock.mock.calls[1]?.[1] as RequestInit).body as string)).toEqual({
      revision: "nps_current",
      destination_choice_id: "ndst_home",
      integration_class: "github",
    });
    expect(JSON.stringify(loaded)).not.toContain("telegram");
  });

  it("loads and mutates opaque client-binding voice preferences", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json(clientPreferencesPayload()))
      .mockResolvedValueOnce(Response.json(clientPreferencesPayload({
        mutation: { operation: "set_client_voice_mode", changed: true },
      })));
    vi.stubGlobal("fetch", fetchMock);

    const loaded = await loadClientPreferences(session);
    expect(loaded.voiceOutput.bindings).toEqual([
      expect.objectContaining({
        choiceId: "cbd_telegram",
        clientName: "Telegram",
        mode: "off",
      }),
    ]);
    await expect(updateClientPreference(session, "cvp_current", {
      field: "mode",
      bindingChoiceId: "cbd_telegram",
      value: "text_and_voice",
    })).resolves.toMatchObject({
      mutation: { operation: "set_client_voice_mode", changed: true },
    });

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/v1/settings/clients");
    expect(JSON.parse((fetchMock.mock.calls[1]?.[1] as RequestInit).body as string)).toEqual({
      revision: "cvp_current",
      binding_choice_id: "cbd_telegram",
      mode: "text_and_voice",
    });
    expect(JSON.stringify(loaded)).not.toContain("12345");
  });

  it("loads and mutates allowlisted appearance preferences", async () => {
    const payload = {
      version: 1,
      theme_id: "atom-one-dark",
      themes: WORKSHOP_THEME_CATALOG.map((theme) => ({
        theme_id: theme.themeId,
        display_name: theme.displayName,
        color_scheme: theme.colorScheme,
      })),
      revision: "apr_current",
      mutation: null,
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json(payload))
      .mockResolvedValueOnce(Response.json({
        ...payload,
        mutation: { operation: "set_theme", changed: false },
      }));
    vi.stubGlobal("fetch", fetchMock);

    const loaded = await loadAppearancePreferences(session);
    expect(loaded).toMatchObject({
      themeId: "atom-one-dark",
    });
    expect(loaded.themes).toHaveLength(11);
    await expect(
      updateAppearancePreference(session, "apr_current", "atom-one-dark"),
    ).resolves.toMatchObject({
      mutation: { operation: "set_theme", changed: false },
    });

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/v1/settings/appearance");
    expect(JSON.parse((fetchMock.mock.calls[1]?.[1] as RequestInit).body as string)).toEqual({
      revision: "apr_current",
      theme_id: "atom-one-dark",
    });
  });

  it("rejects malformed memory records instead of rendering partial data", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json({
          version: 1,
          records: [{ ...memoryRecord(), tags: "not-an-array" }],
          next_cursor: null,
        }),
      ),
    );

    await expect(loadMemoryRecords("session-secret")).rejects.toThrow(
      "Kai returned an unsupported memory record.",
    );
  });

  it("sends typed memory management requests and parses per-target outcomes", async () => {
    const response = (operation: "move_scope" | "delete", ids: string[]) => Response.json({
      version: 1,
      operation,
      results: ids.map((memoryId) => ({
        memory_id: memoryId,
        outcome: "succeeded",
        prior_scope: null,
        new_scope: null,
      })),
    });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response("move_scope", ["memory-1"]))
      .mockResolvedValueOnce(response("move_scope", ["memory-1", "memory-2"]))
      .mockResolvedValueOnce(response("delete", ["memory-1"]))
      .mockResolvedValueOnce(response("delete", ["memory-1", "memory-2"]));
    vi.stubGlobal("fetch", fetchMock);

    await moveMemoryScope("session-secret", "memory-1", {
      scope: "project",
      projectId: "kai",
    });
    await moveMemoriesScope("session-secret", ["memory-1", "memory-2"], {
      scope: "global",
    });
    await deleteMemory("session-secret", "memory-1");
    const deleted = await deleteMemories("session-secret", ["memory-1", "memory-2"]);

    expect(deleted.results.map((result) => result.memoryId)).toEqual(["memory-1", "memory-2"]);
    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      "/v1/memory/records/memory-1/scope",
      "/v1/memory/actions/scope",
      "/v1/memory/records/memory-1",
      "/v1/memory/actions/delete",
    ]);
    expect(JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string)).toEqual({
      scope: "project",
      project_id: "kai",
    });
    expect((fetchMock.mock.calls[2]?.[1] as RequestInit).method).toBe("DELETE");
  });

  it("creates and edits memories through typed revision-bound requests", async () => {
    const detail = {
      ...memoryRecord(),
      content: "Daniel prefers concise output.",
      compact_recall: '{"record_type":"memory"}',
      confirmation_quote: null,
      prompt_version: "v1",
      episode: null,
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json({ version: 1, created: true, record: detail }, { status: 201 }))
      .mockResolvedValueOnce(Response.json({
        version: 1,
        record: { ...detail, revision: "mr1_updated" },
        changed_fields: ["content", "tags"],
        idempotent_replay: false,
      }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(createMemoryFact("session-secret", {
      content: "Explicit fact",
      tags: ["explicit"],
      target: { scope: "project", projectId: "kai" },
      requestId: "create-request",
    })).resolves.toMatchObject({ created: true, record: { memoryId: "memory-1" } });
    await expect(editMemory("session-secret", {
      kind: "fact",
      memoryId: "memory-1",
      revision: "mr1_test-revision",
      requestId: "edit-request",
      content: "Corrected fact",
      tags: ["corrected"],
    })).resolves.toMatchObject({ changedFields: ["content", "tags"] });

    expect(JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string)).toEqual({
      kind: "fact",
      content: "Explicit fact",
      tags: ["explicit"],
      scope: "project",
      project_id: "kai",
      request_id: "create-request",
    });
    expect(JSON.parse((fetchMock.mock.calls[1]?.[1] as RequestInit).body as string)).toEqual({
      kind: "fact",
      revision: "mr1_test-revision",
      request_id: "edit-request",
      content: "Corrected fact",
      tags: ["corrected"],
    });
  });

  it("surfaces the current revision when a memory edit conflicts", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({
      error: {
        code: "memory_revision_conflict",
        message: "Memory changed since it was opened",
        current_revision: "mr1_current",
      },
    }, { status: 409 })));

    const failure = editMemory("session-secret", {
      kind: "fact",
      memoryId: "memory-1",
      revision: "mr1_stale",
      requestId: "edit-request",
      content: "Correction",
      tags: [],
    });
    await expect(failure).rejects.toBeInstanceOf(MemoryRevisionConflictError);
    await expect(failure).rejects.toMatchObject({ currentRevision: "mr1_current" });
  });

  it("rejects an earlier page whose snapshot bound does not match", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json({
          version: 1,
          channel_id: channelId,
          messages: [message(5)],
          next_cursor: null,
          previous_cursor: null,
          through_position: 21,
        }),
      ),
    );

    await expect(
      loadEarlierTimeline(session, "earlier-page", 20, new AbortController().signal),
    ).rejects.toThrow("The timeline snapshot changed while it was loading.");
  });

  it("loads authority-backed Workshop and channel navigation", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        version: 1,
        principal: {
          principal_id: "prn_00000000000000000000000000000001",
          display_name: "Daniel",
        },
        workshops: [
          {
            workshop_id: "wsp_00000000000000000000000000000001",
            name: "Kai Workshop",
            role: "admin",
            channels: [
              {
                channel_id: channelId,
                name: "Conversation",
                kind: "direct",
                role: "owner",
                can_submit_commands: true,
                agents: [
                  {
                    agent_id: "agt_00000000000000000000000000000001",
                    name: "Kai",
                  },
                ],
                participants: [
                  {
                    principal_id: "prn_00000000000000000000000000000002",
                    kind: "agent",
                    display_name: "Kai",
                  },
                ],
              },
            ],
          },
        ],
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(loadNavigation("session-secret")).resolves.toEqual({
      principal: {
        displayName: "Daniel",
        principalId: "prn_00000000000000000000000000000001",
      },
      workshops: [
        {
          channels: [
            {
              agents: [
                {
                  agentId: "agt_00000000000000000000000000000001",
                  name: "Kai",
                },
              ],
              canSubmitCommands: true,
              channelId,
              kind: "direct",
              name: "Conversation",
              participants: [
                {
                  displayName: "Kai",
                  kind: "agent",
                  principalId: "prn_00000000000000000000000000000002",
                },
              ],
              role: "owner",
            },
          ],
          name: "Kai Workshop",
          role: "admin",
          workshopId: "wsp_00000000000000000000000000000001",
        },
      ],
    });
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/v1/client/navigation");
    expect(
      new Headers((fetchMock.mock.calls[0]?.[1] as RequestInit).headers).get(
        "Authorization",
      ),
    ).toBe("Bearer session-secret");
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

  it("submits one attachment as ordered multipart data under bearer authority", async () => {
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
    const artifact = new File(["attachment body"], "notes.txt", {
      type: "text/plain",
    });

    await submitCommand(
      session,
      "browser-artifact-1",
      "Inspect this",
      artifact,
    );

    const [path, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe(`/v1/channels/${channelId}/commands`);
    expect(new Headers(options.headers).get("Authorization")).toBe(
      "Bearer session-secret",
    );
    expect(new Headers(options.headers).has("Content-Type")).toBe(false);
    expect(options.body).toBeInstanceOf(FormData);
    const fields = Array.from((options.body as FormData).entries());
    expect(fields.map(([name]) => name)).toEqual([
      "client_message_id",
      "body",
      "file",
    ]);
    expect(fields[0]?.[1]).toBe("browser-artifact-1");
    expect(fields[1]?.[1]).toBe("Inspect this");
    const submittedFile = fields[2]?.[1];
    expect(submittedFile).toBeInstanceOf(File);
    expect((submittedFile as File).name).toBe("notes.txt");
    await expect((submittedFile as File).text()).resolves.toBe("attachment body");
  });

  it("loads an opaque channel-scoped artifact with bearer authority", async () => {
    const blob = new Blob(["artifact"], { type: "text/plain" });
    const fetchMock = vi.fn().mockResolvedValue({
      blob: vi.fn().mockResolvedValue(blob),
      ok: true,
      status: 200,
    });
    vi.stubGlobal("fetch", fetchMock);
    const artifactId = "art_00000000000000000000000000000001";

    await expect(loadArtifactBlob(session, artifactId)).resolves.toEqual(blob);

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      `/v1/channels/${channelId}/artifacts/${artifactId}/content`,
    );
    expect(
      new Headers((fetchMock.mock.calls[0]?.[1] as RequestInit).headers).get(
        "Authorization",
      ),
    ).toBe("Bearer session-secret");
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
      { onConnected, onMessage, onRunActivity, onRunPreview: vi.fn(), onRunTrace: vi.fn() },
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

  it("parses run preview frames without an id and rejects malformed or foreign ones", async () => {
    const preview = (payload: Record<string, unknown>): string =>
      ["event: run.preview.updated", `data: ${JSON.stringify(payload)}`, "", ""].join("\n");
    const rawMessage = message(33, "Canonical answer");
    const frames = [
      // Valid: no id line, current channel.
      preview({
        version: 1,
        channel_id: channelId,
        run_id: "run_00000000000000000000000000000030",
        sequence: 2,
        text: "First sentence.",
      }),
      // Malformed: text missing.
      preview({
        version: 1,
        channel_id: channelId,
        run_id: "run_00000000000000000000000000000030",
        sequence: 3,
      }),
      // Foreign channel: must never reach the handler.
      preview({
        version: 1,
        channel_id: "chn_99999999999999999999999999999999",
        run_id: "run_00000000000000000000000000000031",
        sequence: 4,
        text: "Private to another channel.",
      }),
      // A durable event after the previews proves the resume path is intact.
      [
        "id: 33",
        "event: timeline.message.created",
        `data: ${JSON.stringify({ version: 1, channel_id: channelId, message: rawMessage })}`,
        "",
        "",
      ].join("\n"),
    ].join("");
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(frames));
        controller.close();
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(stream, { status: 200 })),
    );
    const onRunPreview = vi.fn();
    const onMessage = vi.fn();

    await streamTimeline(
      session,
      "32",
      { onConnected: vi.fn(), onMessage, onRunActivity: vi.fn(), onRunPreview, onRunTrace: vi.fn() },
      new AbortController().signal,
    );

    expect(onRunPreview).toHaveBeenCalledOnce();
    expect(onRunPreview).toHaveBeenCalledWith({
      runId: "run_00000000000000000000000000000030",
      sequence: 2,
      text: "First sentence.",
    });
    expect(onMessage).toHaveBeenCalledWith(
      expect.objectContaining({ body: "Canonical answer", eventPosition: 33 }),
      "33",
    );
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
      { onConnected: vi.fn(), onMessage: vi.fn(), onRunActivity, onRunPreview: vi.fn(), onRunTrace: vi.fn() },
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
