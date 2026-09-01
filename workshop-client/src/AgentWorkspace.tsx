import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import {
  AuthenticationError,
  ChannelAccessError,
  ResynchronizationRequired,
  activateAgentRevision,
  addAgentRevision,
  archiveAgentDefinition,
  createAgentDefinition,
  enableAgentDefinition,
  loadAgentDefinitions,
  loadAgentEnablements,
  streamAgentChanges,
} from "./api";
import type {
  WorkshopAgentCapability,
  WorkshopAgentDefinition,
  WorkshopAgentEnablement,
} from "./types";
import { useConfirmation } from "./ConfirmationDialog";
import { AgentRuntimeControls } from "./SettingsWorkspace";

const CAPABILITIES: {
  description: string;
  label: string;
  value: WorkshopAgentCapability;
}[] = [
  {
    description: "Create ordinary text responses.",
    label: "Text generation",
    value: "text_generation",
  },
  {
    description: "Expose bounded tool activity in the run inspector.",
    label: "Tool activity",
    value: "tool_activity",
  },
  {
    description: "Work within an already-authorized workspace.",
    label: "Workspace execution",
    value: "workspace_execution",
  },
  {
    description: "Accept images when the selected runtime supports them.",
    label: "Image input",
    value: "image_input",
  },
  {
    description:
      "Delegate bounded tasks to other active agents in a shared channel.",
    label: "Agent delegation",
    value: "agent_delegation",
  },
];

interface DefinitionFormState {
  avatar: string;
  capabilities: WorkshopAgentCapability[];
  description: string;
  displayName: string;
  handle: string;
  instructions: string;
  purpose: string;
}

const EMPTY_DEFINITION: DefinitionFormState = {
  avatar: "",
  capabilities: ["text_generation"],
  description: "",
  displayName: "",
  handle: "",
  instructions: "",
  purpose: "",
};

function operationKey(kind: string): string {
  if (typeof globalThis.crypto?.getRandomValues !== "function") {
    throw new Error("This browser cannot create secure operation identities.");
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  const suffix = Array.from(bytes, (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
  return `workshop-agent-${kind}-${suffix}`;
}

function activeRevision(agent: WorkshopAgentDefinition) {
  return agent.revisions.find(
    (revision) => revision.revisionId === agent.activeRevisionId,
  ) ?? null;
}

function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(parsed);
}

function CapabilityChoices({
  disabled,
  selected,
  onChange,
}: {
  disabled: boolean;
  selected: WorkshopAgentCapability[];
  onChange: (capabilities: WorkshopAgentCapability[]) => void;
}): React.JSX.Element {
  return (
    <fieldset className="agent-capability-choices" disabled={disabled}>
      <legend>Declared capabilities</legend>
      <p>
        Capabilities describe requirements. They never grant tools, credentials,
        workspaces, services, or data access.
      </p>
      {CAPABILITIES.map((capability) => (
        <label key={capability.value}>
          <input
            type="checkbox"
            checked={selected.includes(capability.value)}
            onChange={(event) => {
              onChange(
                event.target.checked
                  ? [...selected, capability.value]
                  : selected.filter((item) => item !== capability.value),
              );
            }}
          />
          <span>
            <strong>{capability.label}</strong>
            <small>{capability.description}</small>
          </span>
        </label>
      ))}
    </fieldset>
  );
}

function AgentCreationForm({
  busy,
  onCancel,
  onCreate,
}: {
  busy: boolean;
  onCancel: () => void;
  onCreate: (form: DefinitionFormState) => Promise<void>;
}): React.JSX.Element {
  const [form, setForm] = useState(EMPTY_DEFINITION);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (form.capabilities.length === 0) {
      setError("Select at least one declared capability.");
      return;
    }
    setError(null);
    try {
      await onCreate(form);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not create this agent.");
    }
  };

  return (
    <form className="agent-editor" onSubmit={(event) => void submit(event)}>
      <div className="agent-editor-heading">
        <div>
          <p className="overline">New software participant</p>
          <h2>Create agent</h2>
        </div>
        <button className="quiet-button" type="button" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
      <div className="agent-editor-grid">
        <label>
          Stable handle
          <span className="agent-field-hint">Lowercase letters, numbers, and underscores</span>
          <div className="agent-handle-input">
            <span aria-hidden="true">@</span>
            <input
              autoFocus
              maxLength={32}
              pattern="[a-z][a-z0-9_]{0,31}"
              required
              value={form.handle}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  handle: event.target.value.toLowerCase(),
                }))
              }
            />
          </div>
        </label>
        <label>
          Display name
          <input
            maxLength={80}
            required
            value={form.displayName}
            onChange={(event) =>
              setForm((current) => ({ ...current, displayName: event.target.value }))
            }
          />
        </label>
        <label>
          Avatar text
          <span className="agent-field-hint">Optional, up to 16 characters</span>
          <input
            maxLength={16}
            value={form.avatar}
            onChange={(event) =>
              setForm((current) => ({ ...current, avatar: event.target.value }))
            }
          />
        </label>
      </div>
      <label>
        Description
        <textarea
          maxLength={1000}
          rows={3}
          value={form.description}
          onChange={(event) =>
            setForm((current) => ({ ...current, description: event.target.value }))
          }
        />
      </label>
      <label>
        Purpose
        <textarea
          maxLength={2000}
          required
          rows={3}
          value={form.purpose}
          onChange={(event) =>
            setForm((current) => ({ ...current, purpose: event.target.value }))
          }
        />
      </label>
      <label>
        Instructions
        <span className="agent-field-hint">
          Behavioral guidance only. Instructions cannot grant authority.
        </span>
        <textarea
          maxLength={20000}
          required
          rows={10}
          value={form.instructions}
          onChange={(event) =>
            setForm((current) => ({ ...current, instructions: event.target.value }))
          }
        />
      </label>
      <CapabilityChoices
        disabled={busy}
        selected={form.capabilities}
        onChange={(capabilities) =>
          setForm((current) => ({ ...current, capabilities }))
        }
      />
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={busy}>
          {busy ? "Creating…" : "Create draft"}
        </button>
        <button className="quiet-button" type="button" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
    </form>
  );
}

function RevisionEditor({
  agent,
  busy,
  onCancel,
  onSave,
}: {
  agent: WorkshopAgentDefinition;
  busy: boolean;
  onCancel: () => void;
  onSave: (input: {
    capabilities: WorkshopAgentCapability[];
    instructions: string;
    purpose: string;
  }) => Promise<void>;
}): React.JSX.Element {
  const latest = agent.revisions[agent.revisions.length - 1];
  const [purpose, setPurpose] = useState(latest?.purpose ?? "");
  const [instructions, setInstructions] = useState(latest?.instructions ?? "");
  const [capabilities, setCapabilities] = useState<WorkshopAgentCapability[]>(
    latest?.capabilities ?? ["text_generation"],
  );
  const [error, setError] = useState<string | null>(null);
  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (capabilities.length === 0) {
      setError("Select at least one declared capability.");
      return;
    }
    setError(null);
    try {
      await onSave({ capabilities, instructions, purpose });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not save this revision.");
    }
  };
  return (
    <form className="agent-revision-editor" onSubmit={(event) => void submit(event)}>
      <h3>New definition revision</h3>
      <p>
        Saving creates an immutable revision. Activate it separately after review.
      </p>
      <label>
        Purpose
        <textarea
          maxLength={2000}
          required
          rows={3}
          value={purpose}
          onChange={(event) => setPurpose(event.target.value)}
        />
      </label>
      <label>
        Instructions
        <textarea
          maxLength={20000}
          required
          rows={10}
          value={instructions}
          onChange={(event) => setInstructions(event.target.value)}
        />
      </label>
      <CapabilityChoices
        disabled={busy}
        selected={capabilities}
        onChange={setCapabilities}
      />
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={busy}>
          {busy ? "Saving…" : "Save revision"}
        </button>
        <button className="quiet-button" type="button" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
    </form>
  );
}

export function AgentWorkspace({
  activeChannelId,
  initialCreating,
  initialDefinitionId,
  initialSection,
  isAdministrator,
  onAuthenticationFailure,
  onChannelAccessFailure,
  onClose,
  onCreateAgent,
  onNavigationChanged,
  onOpenChannel,
  onSelectAgent,
  principalName,
  runActive,
  token,
}: {
  activeChannelId: string;
  initialCreating: boolean;
  initialDefinitionId: string | null;
  initialSection: "runtime" | null;
  isAdministrator: boolean;
  onAuthenticationFailure: (message: string) => void;
  onChannelAccessFailure: (message: string) => void;
  onClose: () => void;
  onCreateAgent: () => void;
  onNavigationChanged: () => Promise<void>;
  onOpenChannel: (channelId: string) => Promise<void>;
  onSelectAgent: (
    definitionId: string | null,
    section?: "runtime" | null,
  ) => void;
  principalName: string;
  runActive: boolean;
  token: string;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const [definitions, setDefinitions] = useState<WorkshopAgentDefinition[]>([]);
  const [enablements, setEnablements] = useState<WorkshopAgentEnablement[]>([]);
  const [selectedDefinitionId, setSelectedDefinitionId] = useState<string | null>(
    initialDefinitionId,
  );
  const [creating, setCreating] = useState(initialCreating);
  const [editingRevision, setEditingRevision] = useState(false);
  const [runtimeProfileId, setRuntimeProfileId] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveState, setLiveState] = useState<"connected" | "connecting">(
    "connecting",
  );

  const handleError = useCallback((caught: unknown, fallback: string): void => {
    if (caught instanceof AuthenticationError) {
      onAuthenticationFailure(caught.message);
    } else if (caught instanceof ChannelAccessError) {
      onChannelAccessFailure(caught.message);
    }
    setError(caught instanceof Error ? caught.message : fallback);
  }, [onAuthenticationFailure, onChannelAccessFailure]);

  const refresh = useCallback(async (): Promise<void> => {
    const [nextDefinitions, nextEnablements] = await Promise.all([
      loadAgentDefinitions(token),
      loadAgentEnablements(token),
    ]);
    setDefinitions(nextDefinitions);
    setEnablements(nextEnablements);
    setSelectedDefinitionId((current) => {
      const candidate = current ?? initialDefinitionId;
      if (candidate && nextDefinitions.some((item) => item.definitionId === candidate)) {
        return candidate;
      }
      return null;
    });
  }, [initialDefinitionId, token]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void refresh().then(
      () => {
        if (!cancelled) {
          setLoading(false);
          setError(null);
        }
      },
      (caught: unknown) => {
        if (!cancelled) {
          setLoading(false);
          handleError(caught, "Could not load agents.");
        }
      },
    );
    return () => { cancelled = true; };
  }, [handleError, refresh]);

  useEffect(() => {
    const controller = new AbortController();
    let lastEventId: string | null = null;
    const synchronize = async (): Promise<void> => {
      while (!controller.signal.aborted) {
        try {
          setLiveState("connecting");
          await streamAgentChanges(
            token,
            lastEventId,
            {
              onChanged: (_signal, eventId) => {
                lastEventId = eventId;
                void Promise.all([refresh(), onNavigationChanged()]).catch(
                  (caught: unknown) => handleError(caught, "Could not refresh agents."),
                );
              },
              onConnected: () => setLiveState("connected"),
            },
            controller.signal,
          );
        } catch (caught) {
          if (controller.signal.aborted) {
            return;
          }
          if (caught instanceof ResynchronizationRequired) {
            lastEventId = null;
            try {
              await Promise.all([refresh(), onNavigationChanged()]);
            } catch (refreshError) {
              handleError(refreshError, "Could not resynchronize agents.");
            }
          } else if (
            caught instanceof AuthenticationError ||
            caught instanceof ChannelAccessError
          ) {
            handleError(caught, "Live agent updates are unavailable.");
            return;
          }
          await new Promise<void>((resolve) => {
            const timeout = window.setTimeout(resolve, 2000);
            controller.signal.addEventListener(
              "abort",
              () => {
                window.clearTimeout(timeout);
                resolve();
              },
              { once: true },
            );
          });
        }
      }
    };
    void synchronize();
    return () => controller.abort();
  }, [handleError, onNavigationChanged, refresh, token]);

  const selected = definitions.find(
    (definition) => definition.definitionId === selectedDefinitionId,
  ) ?? null;
  const enablement = enablements.find(
    (item) => item.definitionId === selectedDefinitionId,
  ) ?? null;
  const selectedRevision = selected ? activeRevision(selected) : null;
  const runtimeSession = useMemo(() => (
    enablement?.lifecycleState === "enabled" && enablement.directChannelId
      ? { channelId: enablement.directChannelId, token }
      : null
  ), [enablement?.directChannelId, enablement?.lifecycleState, token]);

  useEffect(() => {
    setCreating(initialCreating);
    if (initialDefinitionId) {
      setSelectedDefinitionId(initialDefinitionId);
    }
  }, [initialCreating, initialDefinitionId]);

  useEffect(() => {
    setEditingRevision(false);
  }, [selectedDefinitionId]);

  useEffect(() => {
    setRuntimeProfileId(
      enablement?.runtimeProfileId ??
      enablement?.eligibleRuntimes[0]?.runtimeProfileId ??
      "",
    );
  }, [enablement?.eligibleRuntimes, enablement?.runtimeProfileId]);

  useEffect(() => {
    if (initialSection !== "runtime" || !runtimeSession) {
      return;
    }
    const scroll = window.setTimeout(() => {
      document.getElementById("agent-runtime-settings")?.scrollIntoView?.({
        block: "start",
      });
    }, 0);
    return () => window.clearTimeout(scroll);
  }, [initialSection, runtimeSession]);

  const runMutation = async (
    operation: () => Promise<void>,
    fallback: string,
  ): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      await operation();
    } catch (caught) {
      handleError(caught, fallback);
    } finally {
      setBusy(false);
    }
  };

  const create = (form: DefinitionFormState): Promise<void> =>
    runMutation(async () => {
      const created = await createAgentDefinition(token, {
        ...form,
        idempotencyKey: operationKey("create"),
      });
      await refresh();
      setCreating(false);
      setSelectedDefinitionId(created.definitionId);
      onSelectAgent(created.definitionId);
    }, "Could not create this agent.");

  const addRevision = (input: {
    capabilities: WorkshopAgentCapability[];
    instructions: string;
    purpose: string;
  }): Promise<void> => {
    if (!selected) {
      return Promise.resolve();
    }
    return runMutation(async () => {
      await addAgentRevision(token, selected.definitionId, {
        ...input,
        expectedVersion: selected.stateVersion,
        idempotencyKey: operationKey("revision"),
      });
      await refresh();
      setEditingRevision(false);
    }, "Could not save this revision.");
  };

  const activate = (revisionId: string): Promise<void> => {
    if (!selected) {
      return Promise.resolve();
    }
    return runMutation(async () => {
      await activateAgentRevision(token, selected.definitionId, {
        expectedVersion: selected.stateVersion,
        idempotencyKey: operationKey("activate"),
        revisionId,
      });
      await refresh();
      await onNavigationChanged();
    }, "Could not activate this agent revision.");
  };

  const archive = async (): Promise<void> => {
    if (!selected || !await confirm(
      `Archive @${selected.handle}? Existing history remains available, but the agent will no longer be runnable.`,
    )) {
      return;
    }
    await runMutation(async () => {
      await archiveAgentDefinition(token, selected.definitionId, {
        expectedVersion: selected.stateVersion,
        idempotencyKey: operationKey("archive"),
      });
      await refresh();
      await onNavigationChanged();
    }, "Could not archive this agent.");
  };

  const enable = (): Promise<void> => {
    if (!selected || !enablement || !runtimeProfileId) {
      return Promise.resolve();
    }
    return runMutation(async () => {
      await enableAgentDefinition(token, selected.definitionId, {
        expectedVersion: enablement.stateVersion,
        idempotencyKey: operationKey("enable"),
        runtimeProfileId,
      });
      await refresh();
      await onNavigationChanged();
    }, "Could not enable this agent.");
  };

  const counts = useMemo(() => ({
    active: definitions.filter((item) => item.lifecycleState === "active").length,
    enabled: enablements.filter((item) => item.lifecycleState === "enabled").length,
  }), [definitions, enablements]);

  return (
    <main className="agent-workspace" aria-label="Agents workspace">
      <header className="agent-workspace-header">
        <div>
          <p className="overline">Software participants</p>
          <h1>Agents</h1>
          <p>
            {counts.enabled} enabled · {counts.active} active · {principalName}
          </p>
        </div>
        <div className="agent-header-actions">
          <span className={`agent-live-state ${liveState}`} role="status">
            {liveState === "connected" ? "Live" : "Connecting"}
          </span>
          <div className="agent-header-controls">
            <button
                className="panel-icon-button"
                type="button"
                aria-label="Create agent"
                title="Create agent"
                disabled={busy}
                onClick={() => {
                  setCreating(true);
                  setEditingRevision(false);
                  onCreateAgent();
                }}
              >
                <span aria-hidden="true">+</span>
            </button>
            <button
              className="panel-icon-button"
              type="button"
              aria-label="Close agents"
              title="Close agents"
              onClick={onClose}
            >
              <span aria-hidden="true">×</span>
            </button>
          </div>
        </div>
      </header>

      <div className="agent-workspace-body">
        <section className="agent-detail" aria-live="polite">
          {creating ? (
            <AgentCreationForm
              busy={busy}
              onCancel={() => {
                setCreating(false);
                if (selectedDefinitionId) {
                  onSelectAgent(selectedDefinitionId);
                }
              }}
              onCreate={create}
            />
          ) : selected ? (
            <>
              <div className="agent-detail-identity">
                <span className="agent-detail-avatar" aria-hidden="true">
                  {selected.presentation.avatar ||
                    selected.displayName.slice(0, 1).toUpperCase()}
                </span>
                <div>
                  <div className="agent-detail-title-row">
                    <h2>{selected.displayName}</h2>
                    <span className={`agent-status ${selected.lifecycleState}`}>
                      {selected.lifecycleState}
                    </span>
                  </div>
                  <p className="agent-handle">@{selected.handle}</p>
                  <p>{selected.description || "No description has been provided."}</p>
                </div>
              </div>

              <section className="agent-authority-note">
                <strong>
                  {enablement?.canManage
                    ? "You own and manage this agent."
                    : `Owned and managed by ${selected.ownerDisplayName ?? "another Workshop member"}.`}
                </strong>
                <p>
                  Everyone talks to the same @{selected.handle} definition and owner-sponsored
                  runtime. Conversations, transcripts, and memory remain private to each person.
                </p>
              </section>

              {selected.lifecycleState === "active" && enablement && (
                <section className="agent-enablement-card">
                  <div>
                    <p className="overline">
                      {enablement.canManage ? "Owner runtime" : "Conversation access"}
                    </p>
                    <h3>
                      {enablement.lifecycleState === "enabled"
                        ? enablement.canManage ? "Runtime active" : "Available to you"
                        : enablement.canManage
                          ? "Available to enable"
                          : "Conversation available"}
                    </h3>
                  </div>
                  {enablement.canManage && enablement.eligibleRuntimes.length > 0 ? (
                    <label>
                      Authorized runtime
                      <select
                        value={runtimeProfileId}
                        disabled={busy}
                        onChange={(event) => setRuntimeProfileId(event.target.value)}
                      >
                        {enablement.eligibleRuntimes.map((runtime) => (
                          <option
                            value={runtime.runtimeProfileId}
                            key={runtime.runtimeProfileId}
                          >
                            {runtime.displayName}
                            {runtime.backendOptions.length > 0
                              ? ` · ${runtime.backendOptions.join(", ")}`
                              : ""}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : enablement.canManage ? (
                    <p className="agent-state-copy">
                      No authorized runtime can satisfy this agent yet.
                    </p>
                  ) : null}
                  <div className="form-actions">
                    {enablement.lifecycleState === "enabled" &&
                      enablement.directChannelId && (
                        <button
                          className="primary-button"
                          type="button"
                          disabled={busy}
                          onClick={() => {
                            const directChannelId = enablement.directChannelId;
                            if (!directChannelId) {
                              return;
                            }
                            void onOpenChannel(directChannelId).catch(
                              (caught: unknown) => handleError(
                                caught,
                                "Could not open this agent conversation.",
                              ),
                            );
                          }}
                        >
                          Start conversation
                        </button>
                      )}
                    {!enablement.canManage &&
                      enablement.lifecycleState !== "enabled" &&
                      enablement.eligibleRuntimes.length > 0 && (
                        <button
                          className="primary-button"
                          type="button"
                          disabled={busy}
                          onClick={() => void enable()}
                        >
                          Start conversation
                        </button>
                      )}
                    {enablement.canManage && enablement.eligibleRuntimes.length > 0 &&
                      (enablement.lifecycleState !== "enabled" ||
                        runtimeProfileId !== enablement.runtimeProfileId) && (
                        <button
                          className={
                            enablement.lifecycleState === "enabled"
                              ? "quiet-button"
                              : "primary-button"
                          }
                          type="button"
                          disabled={busy || !runtimeProfileId}
                          onClick={() => void enable()}
                        >
                          {busy
                            ? "Updating…"
                            : enablement.lifecycleState === "enabled"
                              ? "Change runtime"
                              : "Enable agent"}
                        </button>
                      )}
                  </div>
                </section>
              )}

              {runtimeSession && selected && enablement?.canManage && (
                <AgentRuntimeControls
                  isAdministrator={isAdministrator}
                  onAuthenticationFailure={onAuthenticationFailure}
                  onChannelAccessFailure={onChannelAccessFailure}
                  principalName={principalName}
                  roleLabel={
                    isAdministrator ? "Workshop administrator" : "Workshop member"
                  }
                  runtimeLabel={selected.displayName}
                  runActive={
                    activeChannelId === runtimeSession.channelId && runActive
                  }
                  session={runtimeSession}
                />
              )}

              {selectedRevision ? (
                <section className="agent-active-revision">
                  <p className="overline">Active definition</p>
                  <h3>Revision {selectedRevision.revisionNumber}</h3>
                  <p>{selectedRevision.purpose}</p>
                  <div className="agent-capability-tags">
                    {selectedRevision.capabilities.map((capability) => (
                      <span key={capability}>{capability.replaceAll("_", " ")}</span>
                    ))}
                  </div>
                </section>
              ) : (
                <p className="agent-state-copy">
                  This draft is not active and cannot be enabled or run.
                </p>
              )}

              {enablement?.canManage && selected.lifecycleState !== "archived" && (
                <section className="agent-admin-controls">
                  <div className="agent-section-heading">
                    <div>
                      <p className="overline">Owner controls</p>
                      <h3>Definition revisions</h3>
                    </div>
                    {!editingRevision && (
                      <button
                        className="quiet-button"
                        type="button"
                        disabled={busy}
                        onClick={() => setEditingRevision(true)}
                      >
                        New revision
                      </button>
                    )}
                  </div>
                  {editingRevision ? (
                    <RevisionEditor
                      agent={selected}
                      busy={busy}
                      onCancel={() => setEditingRevision(false)}
                      onSave={addRevision}
                    />
                  ) : (
                    <ol className="agent-revision-list">
                      {[...selected.revisions].reverse().map((revision) => (
                        <li key={revision.revisionId}>
                          <div>
                            <strong>Revision {revision.revisionNumber}</strong>
                            <small>{formatTimestamp(revision.createdAt)}</small>
                          </div>
                          <p>{revision.purpose}</p>
                          {revision.revisionId === selected.activeRevisionId ? (
                            <span className="agent-status active">active</span>
                          ) : (
                            <button
                              className="quiet-button"
                              type="button"
                              disabled={busy}
                              onClick={() => void activate(revision.revisionId)}
                            >
                              Activate
                            </button>
                          )}
                        </li>
                      ))}
                    </ol>
                  )}
                  <div className="agent-archive-row">
                    <div>
                      <strong>Archive definition</strong>
                      <p>Preserves messages and provenance while preventing future runs.</p>
                    </div>
                    <button
                      className="danger-button"
                      type="button"
                      disabled={busy}
                      onClick={() => void archive()}
                    >
                      Archive
                    </button>
                  </div>
                </section>
              )}

              {selected.lifecycleState === "archived" && (
                <p className="agent-archived-note">
                  This definition is archived. Historical conversations and revision
                  provenance remain available, but it cannot be enabled or run.
                </p>
              )}
            </>
          ) : loading ? (
            <p className="agent-state-copy">Loading agent details…</p>
          ) : (
            <div className="agent-empty-detail">
              <span aria-hidden="true">@</span>
              <h2>No agent selected</h2>
              <p>Choose an active agent from the sidebar or open the agent archive.</p>
            </div>
          )}
          {error && <p className="agent-workspace-error" role="alert">{error}</p>}
        </section>
      </div>
    </main>
  );
}
