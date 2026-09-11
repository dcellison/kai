import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import {
  type AgentProvisioningInput,
  type AgentProvisioningSetup,
  AuthenticationError,
  ChannelAccessError,
  activateAgentRevision,
  addAgentRevision,
  archiveAgentDefinition,
  createAgentDefinition,
  enableAgentDefinition,
  loadAgentCreationOptions,
  loadAgentCollaborationPolicy,
  loadAgentDefinitions,
  loadAgentEnablements,
  loadAgentProvisioningSetups,
  provisionAgent,
  startAgentConversation,
  updateAgentCollaborationPolicy,
  revokeAgentCollaborationGrants,
} from "./api";
import {
  AgentCreationDialog,
  type AgentDefinitionFormState,
} from "./AgentCreationDialog";
import { AGENT_CAPABILITIES, COLLABORATION_TOOLS } from "./agentChoices";
import type {
  WorkshopAgentCapability,
  WorkshopAgentDefinition,
  WorkshopAgentEnablement,
  WorkshopAgentCreationOptions,
  WorkshopAgentProvisioning,
  WorkshopCollaborationOperation,
  WorkshopCollaborationPolicy,
} from "./types";
import { useConfirmation } from "./ConfirmationDialog";
import { AgentRuntimeControls } from "./SettingsWorkspace";
import type { WorkshopPrincipalEvents } from "./usePrincipalEvents";

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

function agentStateLabel(state: WorkshopAgentDefinition["lifecycleState"]): string {
  return state === "active" ? "Ready" : state === "draft" ? "Draft" : "Archived";
}

function ConversationIcon(): React.JSX.Element {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      focusable="false"
      viewBox="0 0 24 24"
    >
      <path
        d="M20 15a3 3 0 0 1-3 3H9l-5 3V7a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3z"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
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
      {AGENT_CAPABILITIES.map((capability) => (
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

function CollaborationToolChoices({
  disabled,
  selected,
  onChange,
}: {
  disabled: boolean;
  selected: WorkshopCollaborationOperation[];
  onChange: (operations: WorkshopCollaborationOperation[]) => void;
}): React.JSX.Element {
  return (
    <fieldset className="agent-capability-choices" disabled={disabled}>
      <legend>Requested collaboration tools</legend>
      <p>
        This immutable revision requests operations. Owner and host policy still
        decide what a run actually receives.
      </p>
      {COLLABORATION_TOOLS.map((operation) => (
        <label key={operation.value}>
          <input
            type="checkbox"
            checked={selected.includes(operation.value)}
            onChange={(event) => onChange(
              event.target.checked
                ? [...selected, operation.value]
                : selected.filter((item) => item !== operation.value),
            )}
          />
          <span><strong>{operation.label}</strong><small>{operation.description}</small></span>
        </label>
      ))}
    </fieldset>
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
    collaborationOperations: WorkshopCollaborationOperation[];
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
  const [collaborationOperations, setCollaborationOperations] = useState<WorkshopCollaborationOperation[]>(
    latest?.collaborationOperations ?? [],
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
      await onSave({ capabilities, collaborationOperations, instructions, purpose });
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
      <CollaborationToolChoices
        disabled={busy}
        selected={collaborationOperations}
        onChange={setCollaborationOperations}
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

function CollaborationPolicyControls({
  busy,
  policy,
  onRevoke,
  onSave,
}: {
  busy: boolean;
  policy: WorkshopCollaborationPolicy;
  onRevoke: () => Promise<void>;
  onSave: (allowed: WorkshopCollaborationOperation[]) => Promise<void>;
}): React.JSX.Element {
  const [allowed, setAllowed] = useState<WorkshopCollaborationOperation[]>(
    policy.operations.filter((item) => item.ownerAllowed).map((item) => item.operation),
  );
  useEffect(() => {
    setAllowed(policy.operations.filter((item) => item.ownerAllowed).map((item) => item.operation));
  }, [policy]);
  const changed = policy.canManage && policy.operations.some(
    (item) => item.ownerAllowed !== allowed.includes(item.operation),
  );
  return (
    <section className="agent-collaboration-policy">
      <div className="agent-section-heading">
        <div>
          <p className="overline">{policy.canManage ? "Owner policy" : "Collaboration access"}</p>
          <h3>Collaboration tools</h3>
          {policy.canManage ? (
            <p>
              Policy changes apply to newly accepted attempts. Active attempts retain
              their immutable snapshot unless you revoke them below.
            </p>
          ) : (
            <p>
              This read-only view shows revision requests and resulting availability.
              Owner policy and active-run details remain private.
            </p>
          )}
        </div>
      </div>
      <div className="collaboration-policy-grid" role="list">
        {policy.operations.map((item) => {
          const choice = COLLABORATION_TOOLS.find((candidate) => candidate.value === item.operation);
          return (
            <article key={item.operation} role="listitem">
              <label>
                {policy.canManage && (
                  <input
                    type="checkbox"
                    checked={allowed.includes(item.operation)}
                    disabled={busy || !item.hostAllowed}
                    onChange={(event) => setAllowed((current) =>
                      event.target.checked
                        ? [...current, item.operation]
                        : current.filter((operation) => operation !== item.operation),
                    )}
                  />
                )}
                <strong>{choice?.label ?? item.operation.replaceAll("_", " ")}</strong>
              </label>
              <dl>
                <div><dt>Revision</dt><dd>{item.requested ? "requested" : "not requested"}</dd></div>
                {policy.canManage && (
                  <div><dt>Owner</dt><dd>{item.ownerAllowed ? "allowed" : "blocked"}</dd></div>
                )}
                <div><dt>Host</dt><dd>{item.hostAllowed ? "available" : "unavailable"}</dd></div>
                <div><dt>Next attempt</dt><dd>{item.effectiveForNewAttempt ? "effective" : "not effective"}</dd></div>
              </dl>
              {item.unavailableReason && <small>{item.unavailableReason}</small>}
              {item.quota !== null && <small>Per-attempt quota: {item.quota}</small>}
            </article>
          );
        })}
      </div>
      {policy.canManage && (
        <div className="collaboration-policy-actions">
          <button
            className="primary-button"
            type="button"
            disabled={busy || !changed}
            onClick={() => void onSave(allowed)}
          >
            Save policy
          </button>
          <button
            className="danger-button"
            type="button"
            disabled={busy || (policy.activeGrants ?? 0) === 0}
            onClick={() => void onRevoke()}
          >
            Revoke active access ({policy.activeGrants ?? 0})
          </button>
        </div>
      )}
    </section>
  );
}

export function AgentWorkspace({
  activeChannelId,
  initialCreating,
  initialDefinitionId,
  initialSetupId,
  initialSection,
  isAdministrator,
  onAuthenticationFailure,
  onChannelAccessFailure,
  onClose,
  onCreateAgent,
  onNavigationChanged,
  onOpenChannel,
  onSelectAgent,
  principalId,
  principalName,
  principalEvents,
  runActive,
  token,
}: {
  activeChannelId: string;
  initialCreating: boolean;
  initialDefinitionId: string | null;
  initialSetupId: string | null;
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
  principalId: string;
  principalName: string;
  principalEvents: WorkshopPrincipalEvents;
  runActive: boolean;
  token: string;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const [definitions, setDefinitions] = useState<WorkshopAgentDefinition[]>([]);
  const [enablements, setEnablements] = useState<WorkshopAgentEnablement[]>([]);
  const [provisioningSetups, setProvisioningSetups] = useState<AgentProvisioningSetup[]>([]);
  const [collaborationPolicy, setCollaborationPolicy] = useState<WorkshopCollaborationPolicy | null>(null);
  const [selectedDefinitionId, setSelectedDefinitionId] = useState<string | null>(
    initialDefinitionId,
  );
  const [creating, setCreating] = useState(initialCreating);
  const [creationOptions, setCreationOptions] = useState<WorkshopAgentCreationOptions | null>(null);
  const [creationOptionsError, setCreationOptionsError] = useState<string | null>(null);
  const [creationOptionsLoading, setCreationOptionsLoading] = useState(false);
  const [creationReadyDefinitionId, setCreationReadyDefinitionId] = useState<string | null>(null);
  const [editingRevision, setEditingRevision] = useState(false);
  const [runtimeProfileId, setRuntimeProfileId] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const subscribePrincipalEvents = principalEvents.subscribe;

  const handleError = useCallback((caught: unknown, fallback: string): void => {
    if (caught instanceof AuthenticationError) {
      onAuthenticationFailure(caught.message);
    } else if (caught instanceof ChannelAccessError) {
      onChannelAccessFailure(caught.message);
    }
    setError(caught instanceof Error ? caught.message : fallback);
  }, [onAuthenticationFailure, onChannelAccessFailure]);

  const refresh = useCallback(async (): Promise<void> => {
    const [nextDefinitions, nextEnablements, nextSetups] = await Promise.all([
      loadAgentDefinitions(token),
      loadAgentEnablements(token),
      loadAgentProvisioningSetups(token),
    ]);
    setDefinitions(nextDefinitions);
    setEnablements(nextEnablements);
    setProvisioningSetups(nextSetups);
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
    return subscribePrincipalEvents((event) => {
      if (
        event.kind === "synchronize" ||
        event.batch.changes.some((change) => change.agentChanges.length > 0)
      ) {
        void Promise.all([refresh(), onNavigationChanged()]).catch(
          (caught: unknown) => handleError(caught, "Could not refresh agents."),
        );
      }
    });
  }, [handleError, onNavigationChanged, refresh, subscribePrincipalEvents]);

  const selected = definitions.find(
    (definition) => definition.definitionId === selectedDefinitionId,
  ) ?? null;
  const enablement = enablements.find(
    (item) => item.definitionId === selectedDefinitionId,
  ) ?? null;
  const selectedRevision = selected ? activeRevision(selected) : null;
  const resumeSetup = provisioningSetups.find(
    (setup) => setup.provisioning.operationId === initialSetupId,
  ) ?? null;
  const resumeDraft = creating && !resumeSetup && selected?.lifecycleState === "draft"
    ? (() => {
        const revision = selected.revisions[0];
        if (!revision) return null;
        return {
          avatar: selected.presentation.avatar ?? "",
          capabilities: revision.capabilities,
          collaborationOperations: revision.collaborationOperations,
          definitionId: selected.definitionId,
          description: selected.description,
          displayName: selected.displayName,
          handle: selected.handle,
          instructions: revision.instructions,
          purpose: revision.purpose,
        };
      })()
    : null;
  const freshCreation = !initialDefinitionId && !initialSetupId;
  const canManage = selected?.ownerPrincipalId === principalId;
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
    if (
      loading ||
      !creating ||
      freshCreation ||
      resumeSetup ||
      resumeDraft
    ) return;
    // A saved operation can become ready between navigation and reload. In
    // that case show its canonical definition instead of an empty new-agent
    // form with none of the submitted choices.
    setCreating(false);
    onSelectAgent(selectedDefinitionId);
  }, [
    creating,
    freshCreation,
    loading,
    onSelectAgent,
    resumeDraft,
    resumeSetup,
    selectedDefinitionId,
  ]);

  useEffect(() => {
    let cancelled = false;
    if (!creating) {
      setCreationOptions(null);
      setCreationOptionsError(null);
      setCreationOptionsLoading(false);
      return () => { cancelled = true; };
    }
    setCreationOptions(null);
    setCreationOptionsError(null);
    setCreationOptionsLoading(true);
    void loadAgentCreationOptions(token).then(
      (nextOptions) => {
        if (!cancelled) {
          setCreationOptions(nextOptions);
          setCreationOptionsLoading(false);
        }
      },
      (caught: unknown) => {
        if (cancelled) return;
        if (caught instanceof AuthenticationError) {
          onAuthenticationFailure(caught.message);
        } else if (caught instanceof ChannelAccessError) {
          onChannelAccessFailure(caught.message);
        }
        setCreationOptionsError(
          caught instanceof Error ? caught.message : "Could not load agent creation options.",
        );
        setCreationOptionsLoading(false);
      },
    );
    return () => { cancelled = true; };
  }, [creating, onAuthenticationFailure, onChannelAccessFailure, token]);

  useEffect(() => {
    setEditingRevision(false);
  }, [selectedDefinitionId]);

  useEffect(() => {
    let cancelled = false;
    setCollaborationPolicy(null);
    if (!selectedDefinitionId) return () => { cancelled = true; };
    void loadAgentCollaborationPolicy(token, selectedDefinitionId).then(
      (policy) => { if (!cancelled) setCollaborationPolicy(policy); },
      (caught: unknown) => { if (!cancelled) handleError(caught, "Could not load collaboration policy."); },
    );
    return () => { cancelled = true; };
  }, [handleError, selectedDefinitionId, token]);

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

  const finishCreation = async (
    definitionId: string,
    ready: boolean,
  ): Promise<void> => {
    await refresh();
    await onNavigationChanged();
    setCreating(false);
    setSelectedDefinitionId(definitionId);
    setCreationReadyDefinitionId(ready ? definitionId : null);
    onSelectAgent(definitionId);
  };

  const createDraft = async (form: AgentDefinitionFormState): Promise<void> => {
    setBusy(true);
    try {
      const created = await createAgentDefinition(token, {
        ...form,
        idempotencyKey: operationKey("create"),
      });
      await finishCreation(created.definitionId, false);
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        onAuthenticationFailure(caught.message);
      } else if (caught instanceof ChannelAccessError) {
        onChannelAccessFailure(caught.message);
      }
      throw caught;
    } finally {
      setBusy(false);
    }
  };

  const createReady = async (
    input: AgentProvisioningInput,
  ): Promise<WorkshopAgentProvisioning> => {
    setBusy(true);
    try {
      const result = await provisionAgent(token, input);
      if (result.status === "ready" && result.definitionId) {
        await finishCreation(result.definitionId, true);
      } else {
        await refresh();
        await onNavigationChanged();
      }
      return result;
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        onAuthenticationFailure(caught.message);
      } else if (caught instanceof ChannelAccessError) {
        onChannelAccessFailure(caught.message);
      }
      throw caught;
    } finally {
      setBusy(false);
    }
  };

  const addRevision = (input: {
    capabilities: WorkshopAgentCapability[];
    collaborationOperations: WorkshopCollaborationOperation[];
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

  const saveCollaborationPolicy = (
    allowedOperations: WorkshopCollaborationOperation[],
  ): Promise<void> => {
    if (!selected || !collaborationPolicy) return Promise.resolve();
    return runMutation(async () => {
      const updated = await updateAgentCollaborationPolicy(token, selected.definitionId, {
        allowedOperations,
        clientOperationId: operationKey("collaboration-policy"),
        expectedPolicyVersion: collaborationPolicy.policyVersion,
      });
      setCollaborationPolicy(updated);
    }, "Could not update collaboration policy.");
  };

  const revokeCollaboration = async (): Promise<void> => {
    if (!selected || !collaborationPolicy || !await confirm(
      `Immediately revoke active collaboration access for @${selected.handle}? Current attempts will be fenced.`,
    )) return;
    await runMutation(async () => {
      const updated = await revokeAgentCollaborationGrants(
        token,
        selected.definitionId,
        operationKey("collaboration-revoke"),
      );
      setCollaborationPolicy(updated);
    }, "Could not revoke active collaboration access.");
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
      setCollaborationPolicy(
        await loadAgentCollaborationPolicy(token, selected.definitionId),
      );
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

  const startConversation = (): Promise<void> => {
    if (!selected || !enablement) {
      return Promise.resolve();
    }
    const definitionId = selected.definitionId;
    return runMutation(async () => {
      let conversation = enablement;
      if (conversation.lifecycleState !== "enabled") {
        const eligibleRuntime = conversation.eligibleRuntimes[0];
        if (conversation.canManage || !eligibleRuntime) {
          throw new Error("This agent conversation is not available.");
        }
        conversation = await enableAgentDefinition(token, definitionId, {
          expectedVersion: conversation.stateVersion,
          idempotencyKey: operationKey("conversation-access"),
          runtimeProfileId: eligibleRuntime.runtimeProfileId,
        });
      }
      if (!conversation.directChannelId || conversation.stateVersion === null) {
        throw new Error("This agent conversation is not available.");
      }
      if (!conversation.conversationStarted) {
        conversation = await startAgentConversation(token, definitionId, {
          expectedVersion: conversation.stateVersion,
          idempotencyKey: operationKey("conversation"),
        });
      }
      if (!conversation.directChannelId) {
        throw new Error("This agent conversation is not available.");
      }
      await refresh();
      await onNavigationChanged();
      await onOpenChannel(conversation.directChannelId);
    }, "Could not open this agent conversation.");
  };

  const counts = useMemo(() => {
    const incomplete = new Set(
      provisioningSetups.map((setup) => setup.provisioning.definitionId),
    );
    return {
      archived: definitions.filter((item) => item.lifecycleState === "archived").length,
      drafts: definitions.filter(
        (item) => item.lifecycleState === "draft" && !incomplete.has(item.definitionId),
      ).length,
      needsAttention: provisioningSetups.length,
      ready: definitions.filter(
        (item) => item.lifecycleState === "active" && !incomplete.has(item.definitionId),
      ).length,
    };
  }, [definitions, provisioningSetups]);
  const executionProfileControl =
    selected &&
    enablement?.canManage &&
    enablement.eligibleRuntimes.length > 1 ? (
      <details className="settings-card agent-execution-profile-card">
        <summary>Advanced execution profile</summary>
        <p>
          Choose which authorized execution context sponsors @{selected.handle}.
          Backend and model choices remain in Runtime settings.
        </p>
        <label>
          Execution profile
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
              </option>
            ))}
          </select>
        </label>
        {enablement.lifecycleState === "enabled" &&
          runtimeProfileId !== enablement.runtimeProfileId && (
            <div className="settings-actions">
              <button
                className="quiet-button"
                type="button"
                disabled={busy || !runtimeProfileId}
                onClick={() => void enable()}
              >
                {busy ? "Updating…" : "Change execution profile"}
              </button>
            </div>
          )}
      </details>
    ) : null;

  return (
    <main className="agent-workspace" aria-label="Agents workspace">
      <header className="agent-workspace-header">
        <div>
          <p className="overline">Software participants</p>
          <h1>Agents</h1>
          <p>
            {counts.ready} ready · {counts.drafts} drafts · {counts.needsAttention} need attention · {principalName}
          </p>
        </div>
        <div className="agent-header-actions">
          <span className={`agent-live-state ${principalEvents.connection.tone}`} role="status">
            {principalEvents.connection.label}
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
          {selected ? (
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
                      {agentStateLabel(selected.lifecycleState)}
                    </span>
                    {selected.lifecycleState === "active" &&
                      enablement &&
                      ((enablement.lifecycleState === "enabled" &&
                        enablement.directChannelId) ||
                        (!enablement.canManage &&
                          enablement.lifecycleState !== "enabled" &&
                          enablement.ownerRuntimeProfileId &&
                          enablement.eligibleRuntimes.length > 0)) && (
                        <button
                          className="panel-icon-button agent-conversation-button"
                          type="button"
                          aria-label={`${
                            enablement.conversationStarted ? "Open" : "Start"
                          } conversation with ${selected.displayName}`}
                          title={`${
                            enablement.conversationStarted ? "Open" : "Start"
                          } conversation with ${selected.displayName}`}
                          disabled={busy}
                          onClick={() => void startConversation()}
                        >
                          <ConversationIcon />
                        </button>
                      )}
                  </div>
                  <p className="agent-handle">@{selected.handle}</p>
                  <p>{selected.description || "No description has been provided."}</p>
                </div>
              </div>

              {creationReadyDefinitionId === selected.definitionId && (
                <section className="agent-ready-summary" role="status">
                  <strong>Agent ready</strong>
                  <p>
                    Revision 1 and the selected runtime settings are active. Start a
                    conversation with the speech-bubble action when you are ready;
                    no conversation was created automatically.
                  </p>
                </section>
              )}

              <section className="agent-authority-note">
                <strong>
                  {canManage
                    ? "You own and manage this agent."
                    : `Owned and managed by ${selected.ownerDisplayName ?? "another Workshop member"}.`}
                </strong>
                <p>
                  Everyone talks to the same @{selected.handle} definition and owner-sponsored
                  runtime. Conversations, transcripts, and memory remain private to each person.
                </p>
              </section>

              {selected.lifecycleState === "active" &&
                enablement &&
                !enablement.canManage &&
                enablement.lifecycleState !== "enabled" &&
                (!enablement.ownerRuntimeProfileId ||
                  enablement.eligibleRuntimes.length === 0) && (
                  <section className="agent-enablement-card">
                    <div>
                      <p className="overline">Conversation unavailable</p>
                      <h3>
                        {!enablement.ownerRuntimeProfileId
                          ? "Waiting for the agent owner’s runtime"
                          : "Unavailable to this Workshop account"}
                      </h3>
                    </div>
                  </section>
                )}

              {selected.lifecycleState === "active" &&
                enablement?.canManage &&
                enablement.lifecycleState !== "enabled" && (
                  <section
                    className="agent-runtime-controls agent-runtime-setup"
                    aria-label={`${selected.displayName} runtime settings`}
                    id="agent-runtime-settings"
                  >
                    <section className="settings-section">
                      <div>
                        <h2>Runtime settings</h2>
                        <p>
                          Enable this agent on one of your authorized execution
                          profiles before choosing its backend, model, and workspace.
                        </p>
                      </div>
                      <div className="settings-card-stack">
                        <article className="settings-card agent-runtime-setup-card">
                          <p className="settings-card-label">Enable agent</p>
                          {enablement.eligibleRuntimes.length === 0 ? (
                            <p>
                              No authorized execution profile can satisfy this agent yet.
                            </p>
                          ) : (
                            <>
                              <p>
                                {enablement.eligibleRuntimes.length === 1
                                  ? "Your authorized execution profile will be used automatically."
                                  : "An eligible execution profile is selected. Use Advanced execution profile to choose another."}
                              </p>
                              <div className="settings-actions">
                                <button
                                  className="primary-button"
                                  type="button"
                                  disabled={busy || !runtimeProfileId}
                                  onClick={() => void enable()}
                                >
                                  {busy ? "Enabling…" : "Enable agent"}
                                </button>
                              </div>
                            </>
                          )}
                        </article>
                        {executionProfileControl}
                      </div>
                    </section>
                  </section>
                )}

              {runtimeSession && selected && canManage && (
                <AgentRuntimeControls
                  executionProfileControl={executionProfileControl}
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
                  {selectedRevision.collaborationOperations.length > 0 && (
                    <div className="agent-capability-tags agent-collaboration-tags">
                      {selectedRevision.collaborationOperations.map((operation) => (
                        <span key={operation}>{operation.replaceAll("_", " ")}</span>
                      ))}
                    </div>
                  )}
                </section>
              ) : (
                <p className="agent-state-copy">
                  This draft is not active and cannot be enabled or run.
                </p>
              )}

              {collaborationPolicy && (
                <CollaborationPolicyControls
                  busy={busy}
                  policy={collaborationPolicy}
                  onRevoke={revokeCollaboration}
                  onSave={saveCollaborationPolicy}
                />
              )}

              {canManage && selected.lifecycleState !== "archived" && (
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
      {creating && (freshCreation || resumeSetup || resumeDraft) && (
        <AgentCreationDialog
          key={resumeSetup?.provisioning.operationId ?? resumeDraft?.definitionId ?? "new"}
          busy={busy}
          existingHandles={definitions.map((definition) => definition.handle)}
          options={creationOptions}
          optionsError={creationOptionsError}
          optionsLoading={creationOptionsLoading}
          resumeDraft={resumeDraft}
          resumeInput={resumeSetup?.input ?? null}
          resumeProvisioning={resumeSetup?.provisioning ?? null}
          onCancel={() => {
            setCreating(false);
            onSelectAgent(selectedDefinitionId);
          }}
          onCreateReady={createReady}
          onSaveDraft={createDraft}
        />
      )}
    </main>
  );
}
