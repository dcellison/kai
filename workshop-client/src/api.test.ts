import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  EventStreamDecoder,
  MemoryRevisionConflictError,
  PreferenceRevisionConflictError,
  SettingsRevisionConflictError,
  attachChannelAgent,
  cancelRun,
  changeChannelMember,
  activateAgentRevision,
  addAgentRevision,
  archiveAgentDefinition,
  createChannel,
  createAgentDefinition,
  createMemoryFact,
  deleteMemories,
  deleteMemory,
  deactivateOperatorModel,
  detachChannelAgent,
  dismissChannelAgent,
  disableAgentDefinition,
  editMemory,
  enableAgentDefinition,
  loadArtifactBlob,
  loadChannelMembers,
  loadAgentDefinitions,
  loadAgentEnablements,
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
  loadRoutingEligibility,
  loadRoutingPolicy,
  loadRun,
  loadTimeline,
  loadThreadTimeline,
  loadWorkspaceConfig,
  moveMemoriesScope,
  moveMemoryScope,
  redeemEnrollment,
  refreshAllModelCatalogues,
  refreshModelCatalogue,
  searchMemories,
  restorePreferenceRevision,
  savePreferenceDocument,
  setMessageReaction,
  submitCommand,
  streamTimeline,
  streamAgentChanges,
  updateRuntimeSettings,
  updateRoutingPolicy,
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
const agentDefinitionId = "adf_00000000000000000000000000000001";
const agentRevisionId = "adr_00000000000000000000000000000001";
const agentEnablementId = "aen_00000000000000000000000000000001";
const agentId = "agt_00000000000000000000000000000001";
const runtimeProfileId = "rtp_00000000000000000000000000000001";

function agentDefinitionPayload(): Record<string, unknown> {
  return {
    active_revision_id: agentRevisionId,
    agent_id: agentId,
    created_at: "2026-08-29T10:00:00Z",
    created_by_principal_id: "prn_00000000000000000000000000000001",
    definition_id: agentDefinitionId,
    description: "A focused coding agent.",
    display_name: "Builder",
    handle: "builder",
    lifecycle_state: "active",
    presentation: { avatar: "B" },
    revisions: [
      {
        capabilities: ["text_generation", "workspace_execution"],
        created_at: "2026-08-29T10:00:00Z",
        created_by_principal_id: "prn_00000000000000000000000000000001",
        event_position: 91,
        instructions: "Work carefully.",
        purpose: "Implement bounded coding tasks.",
        revision_id: agentRevisionId,
        revision_number: 1,
      },
    ],
    state_version: 3,
  };
}

function agentEnablementPayload(): Record<string, unknown> {
  return {
    agent_id: agentId,
    definition_id: agentDefinitionId,
    direct_channel_id: channelId,
    display_name: "Builder",
    eligible_runtimes: [
      {
        backend_options: ["claude:anthropic", "opencode:deepseek"],
        display_name: "Daniel's runtime",
        runtime_profile_id: runtimeProfileId,
      },
    ],
    enablement_id: agentEnablementId,
    handle: "builder",
    lifecycle_state: "enabled",
    runtime_profile_id: runtimeProfileId,
    state_version: 2,
  };
}

function message(position: number, body = `Message ${position}`): Record<string, unknown> {
  return {
    author_display_name: position % 2 ? "Daniel" : "Kai",
    author_kind: position % 2 ? "human" : "agent",
    author_principal_id: position % 2
      ? "prn_00000000000000000000000000000001"
      : "prn_00000000000000000000000000000002",
    body,
    channel_id: channelId,
    created_at: "2026-08-13T09:00:00Z",
    event_position: position,
    latest_reply_at: null,
    message_id: `msg_${position.toString().padStart(32, "0")}`,
    mentions: [],
    reply_count: 0,
    reply_to_message_id: null,
    thread_root_id: null,
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

function routingEligibilityPayload(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    version: 1,
    task_class: "coding",
    required_capabilities: ["text_generation", "tool_activity", "workspace_execution"],
    principal_id: "prn_00000000000000000000000000000001",
    channel_id: channelId,
    agent_id: "agt_00000000000000000000000000000001",
    runtime_profile_id: "rtp_00000000000000000000000000000001",
    workspace: "/srv/kai",
    candidates: [
      {
        option_id: "claude:anthropic",
        backend: "claude",
        provider: "anthropic",
        allowed_services: ["perplexity"],
        model_id: "claude-sonnet-4-6",
        model_source: "current_selection",
        selected: true,
        eligible: true,
        capabilities: [
          {
            capability: "text_generation",
            support: "supported",
            evidence: "agent_backend_contract_v1",
          },
        ],
        reasons: [{ code: "eligible", detail: "All required capability checks passed." }],
      },
    ],
    ...overrides,
  };
}

function routingPolicyPayload(): Record<string, unknown> {
  return {
    version: 1,
    principal_id: "prn_00000000000000000000000000000001",
    channel_id: channelId,
    agent_id: "agt_00000000000000000000000000000001",
    runtime_profile_id: "rtp_00000000000000000000000000000001",
    entries: [
      {
        task_class: "coding",
        backend_option_id: "claude:anthropic",
        fallback: "selected",
        revision: 1,
        authorized_option_ids: ["claude:anthropic", "codex:openai"],
        eligible_option_ids: ["claude:anthropic"],
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

  it("loads a root-bound thread page and submits a canonical thread reply", async () => {
    const rootId = "msg_00000000000000000000000000000010";
    const reply = {
      ...message(11, "Thread reply"),
      reply_to_message_id: rootId,
      thread_root_id: rootId,
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        Response.json({
          version: 1,
          channel_id: channelId,
          thread_root_id: rootId,
          root: { ...message(10, "Root"), reply_count: 1 },
          messages: [reply],
          next_cursor: null,
          through_position: 11,
        }),
      )
      .mockResolvedValueOnce(
        Response.json({
          version: 3,
          acceptance: "newly_accepted",
          message_id: "msg_00000000000000000000000000000012",
          runs: [],
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const page = await loadThreadTimeline(session, rootId);
    expect(page.root.replyCount).toBe(1);
    expect(page.messages[0].threadRootId).toBe(rootId);
    await submitCommand(session, "thread-reply", "Continue", null, rootId);
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      `/v1/channels/${channelId}/commands`,
    );
    expect(JSON.parse((fetchMock.mock.calls[1]?.[1] as RequestInit).body as string)).toEqual({
      body: "Continue",
      client_message_id: "thread-reply",
      thread_root_id: rootId,
    });
  });

  it("accepts canonical mention spans without reparsing display names", async () => {
    const rawMessage = message(10, "Ask @kAi");
    rawMessage.mentions = [
      {
        principal_id: "prn_00000000000000000000000000000002",
        kind: "agent",
        start: 4,
        length: 4,
      },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json({
          version: 1,
          channel_id: channelId,
          messages: [rawMessage],
          next_cursor: null,
          previous_cursor: null,
          through_position: 10,
        }),
      ),
    );

    const snapshot = await loadTimeline(session, new AbortController().signal);

    expect(snapshot.messages[0]?.mentions).toEqual([
      {
        principalId: "prn_00000000000000000000000000000002",
        kind: "agent",
        start: 4,
        length: 4,
      },
    ]);
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

  it("loads strict read-only routing eligibility for an explicit task class", async () => {
    const fetchMock = vi.fn().mockResolvedValue(Response.json(routingEligibilityPayload()));
    vi.stubGlobal("fetch", fetchMock);

    await expect(loadRoutingEligibility(session, "coding")).resolves.toMatchObject({
      candidates: [
        {
          allowedServices: ["perplexity"],
          backend: "claude",
          eligible: true,
          selected: true,
        },
      ],
      requiredCapabilities: ["text_generation", "tool_activity", "workspace_execution"],
      taskClass: "coding",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `/v1/channels/${channelId}/routing-eligibility?task_class=coding`,
      expect.objectContaining({ headers: expect.any(Headers) }),
    );

    fetchMock.mockResolvedValueOnce(Response.json(
      routingEligibilityPayload({ candidates: [{ backend: "claude" }] }),
    ));
    await expect(loadRoutingEligibility(session, "coding")).rejects.toThrow(
      "Kai returned unsupported routing candidate.",
    );
  });

  it("loads and updates a principal-scoped explicit routing policy", async () => {
    const fetchMock = vi.fn().mockImplementation(
      () => Promise.resolve(Response.json(routingPolicyPayload())),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(loadRoutingPolicy(session)).resolves.toMatchObject({
      entries: [{ backendOptionId: "claude:anthropic", taskClass: "coding" }],
    });
    await expect(
      updateRoutingPolicy(session, "coding", "codex:openai", "fail_closed", 1),
    ).resolves.toMatchObject({ version: 1 });

    expect(fetchMock.mock.calls[1]).toEqual([
      `/v1/channels/${channelId}/routing-policy`,
      expect.objectContaining({
        body: JSON.stringify({
          task_class: "coding",
          backend_option_id: "codex:openai",
          fallback: "fail_closed",
          expected_revision: 1,
        }),
        method: "PATCH",
      }),
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
          handle: "daniel",
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
                    principal_id: "prn_00000000000000000000000000000002",
                    available: true,
                    engaged: false,
                    engaged_until: null,
                    memory_scope: "private",
                    handle: "kai",
                    name: "Kai",
                    runtime_profile_id:
                      "rtp_00000000000000000000000000000001",
                    sponsor_display_name: "Daniel",
                    sponsor_principal_id:
                      "prn_00000000000000000000000000000001",
                  },
                ],
                participants: [
                  {
                    principal_id: "prn_00000000000000000000000000000002",
                    kind: "agent",
                    display_name: "Kai",
                    handle: "kai",
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
        handle: "daniel",
        principalId: "prn_00000000000000000000000000000001",
      },
      workshops: [
        {
          channels: [
            {
              agents: [
                {
                  agentId: "agt_00000000000000000000000000000001",
                  available: true,
                  engaged: false,
                  engagedUntil: null,
                  handle: "kai",
                  memoryScope: "private",
                  name: "Kai",
                  principalId: "prn_00000000000000000000000000000002",
                  runtimeProfileId:
                    "rtp_00000000000000000000000000000001",
                  sponsorDisplayName: "Daniel",
                  sponsorPrincipalId:
                    "prn_00000000000000000000000000000001",
                },
              ],
              canSubmitCommands: true,
              channelId,
              kind: "direct",
              name: "Conversation",
              participants: [
                {
                  displayName: "Kai",
                  handle: "kai",
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

  it("creates a channel with canonical agents and an optional origin", async () => {
    const createdChannelId = "chn_22222222222222222222222222222222";
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        version: 1,
        channel: { channel_id: createdChannelId },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      createChannel("session-secret", {
        agentIds: ["agt_00000000000000000000000000000001"],
        name: "Release planning",
        originChannelId: channelId,
      }),
    ).resolves.toBe(createdChannelId);

    const [path, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/v1/channels");
    expect(new Headers(options.headers).get("Authorization")).toBe(
      "Bearer session-secret",
    );
    expect(JSON.parse(options.body as string)).toEqual({
      agent_ids: ["agt_00000000000000000000000000000001"],
      name: "Release planning",
      origin_channel_id: channelId,
    });
  });

  it("dismisses an engaged channel agent under session authority", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({ version: 1, dismissed: true, replayed: false }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      dismissChannelAgent(
        session,
        "agt_00000000000000000000000000000001",
        "browser-dismissal-1",
      ),
    ).resolves.toBeUndefined();

    const [path, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe(
      `/v1/channels/${channelId}/agents/` +
        "agt_00000000000000000000000000000001/dismiss",
    );
    expect(JSON.parse(options.body as string)).toEqual({
      client_dismissal_id: "browser-dismissal-1",
    });
  });

  it("attaches and detaches a channel agent under session authority", async () => {
    const agentId = "agt_00000000000000000000000000000001";
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        Response.json({ version: 1, operation: "attach", changed: true }),
      )
      .mockResolvedValueOnce(
        Response.json({ version: 1, operation: "detach", changed: true }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await attachChannelAgent(session, agentId, "attach-operation-1");
    await detachChannelAgent(session, agentId, "detach-operation-1");

    const [attachPath, attachOptions] = fetchMock.mock.calls[0] as [
      string,
      RequestInit,
    ];
    const [detachPath, detachOptions] = fetchMock.mock.calls[1] as [
      string,
      RequestInit,
    ];
    expect(attachPath).toBe(
      `/v1/channels/${channelId}/agents/${agentId}/attach`,
    );
    expect(detachPath).toBe(
      `/v1/channels/${channelId}/agents/${agentId}/detach`,
    );
    expect(JSON.parse(attachOptions.body as string)).toEqual({
      client_operation_id: "attach-operation-1",
    });
    expect(JSON.parse(detachOptions.body as string)).toEqual({
      client_operation_id: "detach-operation-1",
    });
  });

  it("sets a message reaction under session authority", async () => {
    const messageId = "msg_00000000000000000000000000000001";
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        version: 1,
        message_id: messageId,
        reaction: "eyes",
        active: true,
        changed: true,
        event_position: 42,
        reactions: [
          { reaction: "eyes", count: 2, reacted_by_viewer: true },
        ],
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      setMessageReaction(session, messageId, "eyes", true),
    ).resolves.toEqual([
      { reaction: "eyes", count: 2, reactedByViewer: true },
    ]);

    const [path, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe(
      `/v1/channels/${channelId}/messages/${messageId}/reactions`,
    );
    expect(JSON.parse(options.body as string)).toEqual({
      reaction: "eyes",
      active: true,
    });
  });

  it("scopes an agent dismissal to a thread when supplied", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({ version: 1, dismissed: true, replayed: false }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const threadRootId = "msg_00000000000000000000000000000001";

    await expect(
      dismissChannelAgent(
        session,
        "agt_00000000000000000000000000000001",
        "browser-dismissal-2",
        threadRootId,
      ),
    ).resolves.toBeUndefined();

    const [, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(options.body as string)).toEqual({
      client_dismissal_id: "browser-dismissal-2",
      thread_root_id: threadRootId,
    });
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

  it("accepts message-only and multi-agent group command responses", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        Response.json({
          version: 3,
          acceptance: "message_only",
          message_id: "msg_00000000000000000000000000000001",
          runs: [],
        }),
      )
      .mockResolvedValueOnce(
        Response.json({
          version: 3,
          acceptance: "newly_accepted",
          message_id: "msg_00000000000000000000000000000002",
          runs: [
            run(),
            {
              ...run(),
              run_id: "run_00000000000000000000000000000002",
            },
          ],
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      submitCommand(session, "browser-command-2", "Hello group"),
    ).resolves.toEqual({
      acceptance: "message_only",
      messageId: "msg_00000000000000000000000000000001",
      run: null,
    });
    await expect(
      submitCommand(session, "browser-command-3", "@Kai and @Nova hello"),
    ).resolves.toMatchObject({
      acceptance: "newly_accepted",
      messageId: "msg_00000000000000000000000000000002",
      run: { runId: "run_00000000000000000000000000000001" },
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
    expect(headers.get("X-Kai-Stream-ID")).toMatch(/^[0-9a-f]{32}$/);
  });

  it("applies live canonical reaction changes", async () => {
    const messageId = "msg_00000000000000000000000000000031";
    const event = [
      "id: 32",
      "event: timeline.message.reactions_changed",
      `data: ${JSON.stringify({
        version: 1,
        channel_id: channelId,
        message_id: messageId,
        reactions: [
          { reaction: "celebrate", count: 3, reacted_by_viewer: false },
        ],
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
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(stream, { status: 200 })));
    const onReactions = vi.fn();

    await streamTimeline(
      session,
      "31",
      {
        onConnected: vi.fn(),
        onMessage: vi.fn(),
        onReactions,
        onRunActivity: vi.fn(),
        onRunPreview: vi.fn(),
        onRunTrace: vi.fn(),
      },
      new AbortController().signal,
    );

    expect(onReactions).toHaveBeenCalledWith(
      messageId,
      [{ reaction: "celebrate", count: 3, reactedByViewer: false }],
      "32",
    );
  });

  it("creates the stream identity without secure-context randomUUID", async () => {
    sessionStorage.removeItem("kai.workshop.event-stream-id.v1");
    const availableCrypto = globalThis.crypto;
    vi.stubGlobal("crypto", {
      getRandomValues: availableCrypto.getRandomValues.bind(availableCrypto),
    });
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.close();
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(new Response(stream, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await streamTimeline(
      session,
      "30",
      {
        onConnected: vi.fn(),
        onMessage: vi.fn(),
        onRunActivity: vi.fn(),
        onRunPreview: vi.fn(),
        onRunTrace: vi.fn(),
      },
      new AbortController().signal,
    );

    const request = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(new Headers(request[1].headers).get("X-Kai-Stream-ID")).toMatch(
      /^[0-9a-f]{32}$/,
    );
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
        routing_decision: {
          backend: "opencode",
          decided_at: "2026-08-13T09:00:00Z",
          disposition: "routed",
          evidence_version: 1,
          model: "deepseek-chat",
          policy_revision: 1,
          provider: "deepseek",
          reason_code: "configured_route_eligible",
          requested_backend_option_id: "opencode:deepseek",
          requested_task_class: "coding",
          selected_backend_option_id: "opencode:deepseek",
        },
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
        run: expect.objectContaining({
          status: "started",
          routingDecision: expect.objectContaining({
            disposition: "routed",
            selectedBackendOptionId: "opencode:deepseek",
          }),
        }),
      },
      "32",
    );
  });

  it("loads strict agent definitions and principal-scoped enablements", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        agents: [agentDefinitionPayload()],
        version: 1,
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        agents: [agentEnablementPayload()],
        version: 1,
      }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const definitions = await loadAgentDefinitions("session-secret");
    const enablements = await loadAgentEnablements("session-secret");

    expect(definitions).toEqual([
      expect.objectContaining({
        definitionId: agentDefinitionId,
        handle: "builder",
        lifecycleState: "active",
      }),
    ]);
    expect(enablements).toEqual([
      expect.objectContaining({
        definitionId: agentDefinitionId,
        directChannelId: channelId,
        lifecycleState: "enabled",
        runtimeProfileId,
      }),
    ]);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/v1/client/agents",
      expect.objectContaining({ cache: "no-store" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/v1/client/agent-enablement",
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("uses versioned agent lifecycle and enablement mutation contracts", async () => {
    const definitionResponse = () => new Response(JSON.stringify({
      agent: agentDefinitionPayload(),
      version: 1,
    }), { status: 200 });
    const enablementResponse = () => new Response(JSON.stringify({
      agent: agentEnablementPayload(),
      version: 1,
    }), { status: 200 });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(definitionResponse())
      .mockResolvedValueOnce(definitionResponse())
      .mockResolvedValueOnce(definitionResponse())
      .mockResolvedValueOnce(definitionResponse())
      .mockResolvedValueOnce(enablementResponse())
      .mockResolvedValueOnce(enablementResponse());
    vi.stubGlobal("fetch", fetchMock);

    await createAgentDefinition("session-secret", {
      avatar: "B",
      capabilities: ["text_generation"],
      description: "Builds things.",
      displayName: "Builder",
      handle: "builder",
      idempotencyKey: "create-key",
      instructions: "Work carefully.",
      purpose: "Build bounded changes.",
    });
    await addAgentRevision("session-secret", agentDefinitionId, {
      capabilities: ["text_generation", "workspace_execution"],
      expectedVersion: 3,
      idempotencyKey: "revision-key",
      instructions: "Use tests.",
      purpose: "Build tested changes.",
    });
    await activateAgentRevision("session-secret", agentDefinitionId, {
      expectedVersion: 4,
      idempotencyKey: "activate-key",
      revisionId: agentRevisionId,
    });
    await archiveAgentDefinition("session-secret", agentDefinitionId, {
      expectedVersion: 5,
      idempotencyKey: "archive-key",
    });
    await enableAgentDefinition("session-secret", agentDefinitionId, {
      expectedVersion: null,
      idempotencyKey: "enable-key",
      runtimeProfileId,
    });
    await disableAgentDefinition("session-secret", agentDefinitionId, {
      expectedVersion: 2,
      idempotencyKey: "disable-key",
    });

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/v1/client/agents",
      `/v1/client/agents/${agentDefinitionId}/revisions`,
      `/v1/client/agents/${agentDefinitionId}/activate`,
      `/v1/client/agents/${agentDefinitionId}/archive`,
      `/v1/client/agents/${agentDefinitionId}/enable`,
      `/v1/client/agents/${agentDefinitionId}/disable`,
    ]);
    expect(JSON.parse(String(fetchMock.mock.calls[4][1]?.body))).toEqual({
      idempotency_key: "enable-key",
      runtime_profile_id: runtimeProfileId,
    });
  });

  it("streams typed agent changes over a distinct live connection", async () => {
    const frame = [
      "id: 96",
      "event: agent.enablement.changed",
      `data: ${JSON.stringify({
        definition_id: agentDefinitionId,
        event_position: 96,
        event_type: "principal_agent.enabled",
        occurred_at: "2026-08-29T10:05:00Z",
        revision_id: null,
        version: 1,
      })}`,
      "",
      "",
    ].join("\n");
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(frame));
        controller.close();
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(stream, { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const onChanged = vi.fn();

    await streamAgentChanges(
      "session-secret",
      "95",
      { onChanged, onConnected: vi.fn() },
      new AbortController().signal,
    );

    expect(onChanged).toHaveBeenCalledWith(
      expect.objectContaining({
        definitionId: agentDefinitionId,
        eventPosition: 96,
        kind: "enablement",
      }),
      "96",
    );
    const headers = fetchMock.mock.calls[0][1]?.headers as Headers;
    expect(headers.get("Last-Event-ID")).toBe("95");
    expect(headers.get("X-Kai-Stream-ID")).toMatch(/:agents$/);
  });

  it("loads and changes canonical human channel membership", async () => {
    const targetPrincipalId = "prn_00000000000000000000000000000003";
    const membershipResponse = new Response(JSON.stringify({
      archived: false,
      can_manage: true,
      channel_id: channelId,
      eligible_humans: [{
        display_name: "Scott",
        handle: "scott",
        principal_id: targetPrincipalId,
        role: null,
      }],
      members: [{
        display_name: "Daniel",
        handle: "daniel",
        principal_id: "prn_00000000000000000000000000000001",
        role: "owner",
      }],
      state_version: 88,
      version: 1,
      workshop_id: "wsp_00000000000000000000000000000001",
    }), { status: 200 });
    const mutationResponse = new Response(JSON.stringify({
      changed: true,
      operation: "add",
      state_version: 89,
      version: 1,
    }), { status: 200 });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(membershipResponse)
      .mockResolvedValueOnce(mutationResponse);
    vi.stubGlobal("fetch", fetchMock);

    const membership = await loadChannelMembers(session);
    const stateVersion = await changeChannelMember(
      session,
      targetPrincipalId,
      "add",
      membership.stateVersion,
      "membership-operation",
    );

    expect(membership.members[0]).toEqual(expect.objectContaining({
      handle: "daniel",
      role: "owner",
    }));
    expect(membership.eligibleHumans[0]).toEqual(expect.objectContaining({
      handle: "scott",
      role: null,
    }));
    expect(stateVersion).toBe(89);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(`/v1/channels/${channelId}/members`);
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      `/v1/channels/${channelId}/members/${targetPrincipalId}/add`,
    );
    expect(JSON.parse(String(fetchMock.mock.calls[1]?.[1]?.body))).toEqual({
      client_operation_id: "membership-operation",
      expected_state_version: 88,
    });
  });

  it("streams privacy-safe navigation changes for channel membership", async () => {
    const frame = [
      "id: 97",
      "event: workshop.navigation.changed",
      `data: ${JSON.stringify({
        definition_id: null,
        event_position: 97,
        event_type: "channel.member_added",
        occurred_at: "2026-08-31T10:05:00Z",
        revision_id: null,
        version: 1,
      })}`,
      "",
      "",
    ].join("\n");
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(frame));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(stream, { status: 200 }),
    ));
    const onChanged = vi.fn();

    await streamAgentChanges(
      "session-secret",
      "96",
      { onChanged, onConnected: vi.fn() },
      new AbortController().signal,
    );

    expect(onChanged).toHaveBeenCalledWith(
      expect.objectContaining({
        definitionId: null,
        eventPosition: 97,
        kind: "navigation",
      }),
      "97",
    );
  });
});
