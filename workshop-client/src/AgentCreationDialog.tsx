import { FormEvent, useEffect, useMemo, useState } from "react";

import type { AgentProvisioningInput } from "./api";
import { AGENT_CAPABILITIES, COLLABORATION_TOOLS } from "./agentChoices";
import { useConfirmation } from "./ConfirmationDialog";
import type {
  WorkshopAgentCapability,
  WorkshopAgentCreationBackendOption,
  WorkshopAgentCreationOptions,
  WorkshopAgentCreationRuntimeOption,
  WorkshopAgentProvisioning,
  WorkshopCollaborationOperation,
} from "./types";

export interface AgentDefinitionFormState {
  avatar: string;
  capabilities: WorkshopAgentCapability[];
  collaborationOperations: WorkshopCollaborationOperation[];
  description: string;
  displayName: string;
  handle: string;
  instructions: string;
  purpose: string;
}

export const EMPTY_AGENT_DEFINITION: AgentDefinitionFormState = {
  avatar: "",
  capabilities: ["text_generation"],
  collaborationOperations: [],
  description: "",
  displayName: "",
  handle: "",
  instructions: "",
  purpose: "",
};

const CREATION_STAGES = ["Identity", "Role", "Runtime", "Review"] as const;
type CreationStage = (typeof CREATION_STAGES)[number];

function creationOperationKey(): string {
  if (typeof globalThis.crypto?.getRandomValues !== "function") {
    throw new Error("This browser cannot create secure operation identities.");
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  const suffix = Array.from(bytes, (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
  return `workshop-agent-provision-${suffix}`;
}

function deriveHandle(displayName: string): string {
  const normalized = displayName
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!normalized) return "";
  const prefixed = /^[a-z]/.test(normalized) ? normalized : `agent_${normalized}`;
  return prefixed.slice(0, 32).replace(/_+$/g, "");
}

function backendLabel(backend: WorkshopAgentCreationBackendOption): string {
  return `${backend.backend} · ${backend.provider}`;
}

function backendSelectable(backend: WorkshopAgentCreationBackendOption): boolean {
  return (
    backend.blockers.length === 0 &&
    backend.readiness !== "unavailable" &&
    backend.readiness !== "misconfigured"
  );
}

function preferredBackend(
  runtime: WorkshopAgentCreationRuntimeOption,
): WorkshopAgentCreationBackendOption | null {
  const current = runtime.backends.find(
    (backend) => backend.optionId === runtime.currentBackendOptionId,
  );
  if (current && backendSelectable(current)) return current;
  return runtime.backends.find(backendSelectable) ?? current ?? runtime.backends[0] ?? null;
}

function preferredModel(backend: WorkshopAgentCreationBackendOption): string {
  const configured = backend.models.find(
    (model) => model.modelId === backend.defaultModel && model.selectable,
  );
  return configured?.modelId ?? backend.models.find((model) => model.selectable)?.modelId ?? "";
}

function preferredWorkspace(runtime: WorkshopAgentCreationRuntimeOption): string {
  const configured = runtime.workspaces.find(
    (workspace) => workspace.path === runtime.defaultWorkspace && workspace.available,
  );
  return configured?.path ?? runtime.workspaces.find((workspace) => workspace.available)?.path ?? "";
}

function choiceList({
  disabled,
  kind,
  selected,
  onChange,
}: {
  disabled: boolean;
  kind: "capabilities" | "collaboration";
  selected: (WorkshopAgentCapability | WorkshopCollaborationOperation)[];
  onChange: (
    value: (WorkshopAgentCapability | WorkshopCollaborationOperation)[],
  ) => void;
}): React.JSX.Element {
  const choices = kind === "capabilities" ? AGENT_CAPABILITIES : COLLABORATION_TOOLS;
  const heading = kind === "capabilities" ? "Declared capabilities" : "Collaboration tools";
  const description = kind === "capabilities"
    ? "Capabilities describe requirements; they do not grant credentials or data access."
    : "Requested tools remain bounded by owner and host policy on every attempt.";
  return (
    <fieldset className="agent-capability-choices" disabled={disabled}>
      <legend>{heading}</legend>
      <p>{description}</p>
      {choices.map((choice) => (
        <label key={choice.value}>
          <input
            type="checkbox"
            checked={selected.includes(choice.value)}
            onChange={(event) => {
              onChange(
                event.target.checked
                  ? [...selected, choice.value]
                  : selected.filter((item) => item !== choice.value),
              );
            }}
          />
          <span>
            <strong>{choice.label}</strong>
            <small>{choice.description}</small>
          </span>
        </label>
      ))}
    </fieldset>
  );
}

export function AgentCreationDialog({
  busy,
  existingHandles,
  options,
  optionsError,
  optionsLoading,
  onCancel,
  onCreateReady,
  onSaveDraft,
}: {
  busy: boolean;
  existingHandles: string[];
  options: WorkshopAgentCreationOptions | null;
  optionsError: string | null;
  optionsLoading: boolean;
  onCancel: () => void;
  onCreateReady: (input: AgentProvisioningInput) => Promise<WorkshopAgentProvisioning>;
  onSaveDraft: (form: AgentDefinitionFormState) => Promise<void>;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const [form, setForm] = useState(EMPTY_AGENT_DEFINITION);
  const [stage, setStage] = useState<CreationStage>("Identity");
  const [dirty, setDirty] = useState(false);
  const [handleAutomatic, setHandleAutomatic] = useState(true);
  const [runtimeProfileId, setRuntimeProfileId] = useState("");
  const [backendOptionId, setBackendOptionId] = useState("");
  const [model, setModel] = useState("");
  const [modelSelections, setModelSelections] = useState<Record<string, string>>({});
  const [workspace, setWorkspace] = useState("");
  const [runtimeNotice, setRuntimeNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submittedInput, setSubmittedInput] = useState<AgentProvisioningInput | null>(null);
  const [provisioning, setProvisioning] = useState<WorkshopAgentProvisioning | null>(null);
  const [clientOperationId] = useState(creationOperationKey);

  const selectedRuntime = options?.runtimes.find(
    (runtime) => runtime.runtimeProfileId === runtimeProfileId,
  ) ?? null;
  const selectedBackend = selectedRuntime?.backends.find(
    (backend) => backend.optionId === backendOptionId,
  ) ?? null;
  const selectedModel = selectedBackend?.models.find(
    (candidate) => candidate.modelId === model,
  ) ?? null;
  const selectedWorkspace = selectedRuntime?.workspaces.find(
    (candidate) => candidate.path === workspace,
  ) ?? null;
  const locked = submittedInput !== null;

  useEffect(() => {
    if (!options || runtimeProfileId || options.runtimes.length === 0) return;
    const runtime = options.runtimes.find((candidate) => candidate.ready) ?? options.runtimes[0];
    if (!runtime) return;
    const backend = preferredBackend(runtime);
    const nextModel = backend ? preferredModel(backend) : "";
    setRuntimeProfileId(runtime.runtimeProfileId);
    setBackendOptionId(backend?.optionId ?? "");
    setModel(nextModel);
    setModelSelections(backend ? { [backend.optionId]: nextModel } : {});
    setWorkspace(preferredWorkspace(runtime));
  }, [options, runtimeProfileId]);

  const readinessBlockers = useMemo(() => {
    const blockers: string[] = [];
    if (optionsLoading) blockers.push("Runtime choices are still loading.");
    if (optionsError) blockers.push(optionsError);
    if (!optionsLoading && !options && !optionsError) {
      blockers.push("Runtime choices are unavailable.");
    }
    if (options && !options.ready) blockers.push(...options.blockers.map((item) => item.detail));
    if (selectedRuntime && !selectedRuntime.ready) {
      blockers.push(...selectedRuntime.blockers.map((item) => item.detail));
      if (selectedRuntime.blockers.length === 0) blockers.push("The selected runtime is not ready.");
    }
    if (selectedRuntime && !selectedBackend) blockers.push("Choose an authorized backend.");
    if (selectedBackend) {
      blockers.push(...selectedBackend.blockers.map((item) => item.detail));
      if (!backendSelectable(selectedBackend) && selectedBackend.blockers.length === 0) {
        blockers.push(`${backendLabel(selectedBackend)} is not ready for agent creation.`);
      }
    }
    if (selectedBackend && (!selectedModel || !selectedModel.selectable)) {
      blockers.push("Choose an available model for the selected backend.");
    }
    if (selectedRuntime && (!selectedWorkspace || !selectedWorkspace.available)) {
      blockers.push("Choose an available workspace.");
    }
    return [...new Set(blockers)];
  }, [
    options,
    optionsError,
    optionsLoading,
    selectedBackend,
    selectedModel,
    selectedRuntime,
    selectedWorkspace,
  ]);

  const updateForm = (change: Partial<AgentDefinitionFormState>): void => {
    setForm((current) => ({ ...current, ...change }));
    setDirty(true);
    setError(null);
  };

  const close = async (): Promise<void> => {
    if (busy) return;
    if (dirty && !await confirm("Discard your unsaved agent changes?")) return;
    onCancel();
  };

  const validateIdentity = (): string | null => {
    if (!form.displayName.trim()) return "Enter a display name.";
    if (!/^[a-z][a-z0-9_]{0,31}$/.test(form.handle)) {
      return "Use a handle beginning with a lowercase letter and containing only lowercase letters, numbers, and underscores.";
    }
    if (existingHandles.includes(form.handle)) {
      return `@${form.handle} is already in use. Choose another handle.`;
    }
    return null;
  };

  const validateRole = (): string | null => {
    if (!form.purpose.trim()) return "Describe what this agent should do.";
    if (!form.instructions.trim()) return "Describe how this agent should behave.";
    if (form.capabilities.length === 0) return "Select at least one declared capability.";
    return null;
  };

  const continueForward = (): void => {
    const validation = stage === "Identity"
      ? validateIdentity()
      : stage === "Role"
        ? validateRole()
        : null;
    if (validation) {
      setError(validation);
      return;
    }
    const index = CREATION_STAGES.indexOf(stage);
    const next = CREATION_STAGES[index + 1];
    if (next) {
      setError(null);
      setStage(next);
    }
  };

  const goBack = (): void => {
    const index = CREATION_STAGES.indexOf(stage);
    const previous = CREATION_STAGES[index - 1];
    if (previous && !locked) {
      setError(null);
      setStage(previous);
    }
  };

  const changeRuntime = (nextRuntimeProfileId: string): void => {
    const runtime = options?.runtimes.find(
      (candidate) => candidate.runtimeProfileId === nextRuntimeProfileId,
    );
    if (!runtime) return;
    const backend = preferredBackend(runtime);
    const nextModel = backend
      ? modelSelections[backend.optionId] || preferredModel(backend)
      : "";
    setRuntimeProfileId(runtime.runtimeProfileId);
    setBackendOptionId(backend?.optionId ?? "");
    setModel(nextModel);
    setWorkspace(preferredWorkspace(runtime));
    setRuntimeNotice(null);
    setDirty(true);
    setError(null);
  };

  const changeBackend = (nextBackendOptionId: string): void => {
    const backend = selectedRuntime?.backends.find(
      (candidate) => candidate.optionId === nextBackendOptionId,
    );
    if (!backend) return;
    const remembered = modelSelections[backend.optionId];
    const compatibleCurrent = backend.models.find(
      (candidate) => candidate.modelId === model && candidate.selectable,
    )?.modelId;
    const nextModel = remembered || compatibleCurrent || preferredModel(backend);
    setBackendOptionId(backend.optionId);
    setModel(nextModel);
    setModelSelections((current) => ({ ...current, [backend.optionId]: nextModel }));
    setRuntimeNotice(
      compatibleCurrent || remembered
        ? null
        : nextModel
          ? `Model set to ${nextModel}, the protected default for ${backendLabel(backend)}.`
          : `No available model was found for ${backendLabel(backend)}.`,
    );
    setDirty(true);
    setError(null);
  };

  const saveDraft = async (): Promise<void> => {
    const validation = validateIdentity() ?? validateRole();
    if (validation) {
      setError(validation);
      return;
    }
    setError(null);
    try {
      await onSaveDraft(form);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not save this draft.");
    }
  };

  const createReady = async (): Promise<void> => {
    const validation = validateIdentity() ?? validateRole();
    if (validation) {
      setError(validation);
      return;
    }
    if (readinessBlockers.length > 0 || !selectedRuntime || !selectedBackend) {
      setError(readinessBlockers[0] ?? "Runtime choices are incomplete.");
      return;
    }
    const input = submittedInput ?? {
      allowedCollaborationOperations: form.collaborationOperations,
      avatar: form.avatar,
      backendOptionId: selectedBackend.optionId,
      capabilities: form.capabilities,
      clientOperationId,
      collaborationOperations: form.collaborationOperations,
      description: form.description,
      displayName: form.displayName,
      handle: form.handle,
      instructions: form.instructions,
      model,
      purpose: form.purpose,
      runtimeProfileId: selectedRuntime.runtimeProfileId,
      timeoutSeconds: selectedRuntime.defaultTimeoutSeconds,
      workspace,
    };
    setSubmittedInput(input);
    setError(null);
    try {
      const result = await onCreateReady(input);
      if (result.status !== "ready") {
        setProvisioning(result);
        setError(
          result.blockers[0]?.detail ??
          "Agent setup needs attention. Your progress was saved and can be retried safely.",
        );
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not create this agent.");
    }
  };

  const submit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (stage === "Review") {
      void createReady();
    } else {
      continueForward();
    }
  };

  const stageIndex = CREATION_STAGES.indexOf(stage);
  return (
    <div className="modal-backdrop" role="presentation">
      <section
        className="channel-creation-dialog agent-creation-dialog agent-creation-wizard"
        role="dialog"
        aria-modal="true"
        aria-labelledby="create-agent-title"
      >
        <form className="agent-editor" onSubmit={submit}>
          <div className="agent-creation-header">
            <div>
              <p className="overline">New software participant</p>
              <h2 id="create-agent-title">Create agent</h2>
            </div>
            <button
              className="panel-icon-button"
              type="button"
              aria-label="Close agent creation"
              title="Close agent creation"
              onClick={() => void close()}
              disabled={busy}
            >
              <span aria-hidden="true">×</span>
            </button>
          </div>

          <ol className="agent-creation-progress" aria-label="Agent creation progress">
            {CREATION_STAGES.map((item, index) => (
              <li
                className={index <= stageIndex ? "active" : undefined}
                aria-current={item === stage ? "step" : undefined}
                key={item}
              >
                <span aria-hidden="true">{index + 1}</span>
                {item}
              </li>
            ))}
          </ol>

          {stage === "Identity" && (
            <section className="agent-creation-stage" aria-labelledby="agent-identity-heading">
              <div className="agent-creation-stage-heading">
                <h3 id="agent-identity-heading">Identity</h3>
                <p>How this agent will appear to people in Workshop.</p>
              </div>
              <div className="agent-editor-grid">
                <label>
                  Display name
                  <input
                    autoFocus
                    maxLength={80}
                    required
                    disabled={locked}
                    value={form.displayName}
                    onChange={(event) => {
                      const displayName = event.target.value;
                      updateForm({
                        displayName,
                        ...(handleAutomatic ? { handle: deriveHandle(displayName) } : {}),
                      });
                    }}
                  />
                </label>
                <label>
                  Handle
                  <span className="agent-field-hint">Unique, stable, and editable</span>
                  <div className="agent-handle-input">
                    <span aria-hidden="true">@</span>
                    <input
                      maxLength={32}
                      pattern="[a-z][a-z0-9_]{0,31}"
                      required
                      disabled={locked}
                      value={form.handle}
                      onChange={(event) => {
                        setHandleAutomatic(false);
                        updateForm({ handle: event.target.value.toLowerCase() });
                      }}
                    />
                  </div>
                </label>
                <label>
                  Avatar text
                  <span className="agent-field-hint">Optional, up to 16 characters</span>
                  <input
                    maxLength={16}
                    disabled={locked}
                    value={form.avatar}
                    onChange={(event) => updateForm({ avatar: event.target.value })}
                  />
                </label>
              </div>
              <label>
                Description
                <span className="agent-field-hint">A concise summary shown to other Workshop members</span>
                <textarea
                  maxLength={1000}
                  rows={3}
                  disabled={locked}
                  value={form.description}
                  onChange={(event) => updateForm({ description: event.target.value })}
                />
              </label>
            </section>
          )}

          {stage === "Role" && (
            <section className="agent-creation-stage" aria-labelledby="agent-role-heading">
              <div className="agent-creation-stage-heading">
                <h3 id="agent-role-heading">Role</h3>
                <p>Define the agent's purpose and behavioral boundaries.</p>
              </div>
              <label>
                What should this agent do?
                <textarea
                  autoFocus
                  maxLength={2000}
                  required
                  rows={4}
                  disabled={locked}
                  value={form.purpose}
                  onChange={(event) => updateForm({ purpose: event.target.value })}
                />
              </label>
              <label>
                How should it behave?
                <span className="agent-field-hint">Instructions cannot grant authority.</span>
                <textarea
                  maxLength={20000}
                  required
                  rows={8}
                  disabled={locked}
                  value={form.instructions}
                  onChange={(event) => updateForm({ instructions: event.target.value })}
                />
              </label>
              <details className="agent-creation-advanced">
                <summary>Advanced capabilities</summary>
                {choiceList({
                  disabled: busy || locked,
                  kind: "capabilities",
                  selected: form.capabilities,
                  onChange: (capabilities) => updateForm({
                    capabilities: capabilities as WorkshopAgentCapability[],
                  }),
                })}
                {choiceList({
                  disabled: busy || locked,
                  kind: "collaboration",
                  selected: form.collaborationOperations,
                  onChange: (collaborationOperations) => updateForm({
                    collaborationOperations:
                      collaborationOperations as WorkshopCollaborationOperation[],
                  }),
                })}
              </details>
            </section>
          )}

          {stage === "Runtime" && (
            <section className="agent-creation-stage" aria-labelledby="agent-runtime-heading">
              <div className="agent-creation-stage-heading">
                <h3 id="agent-runtime-heading">Runtime</h3>
                <p>Choose from runtime access already authorized for you.</p>
              </div>
              {optionsLoading && <p className="agent-creation-loading" role="status">Loading runtime choices…</p>}
              {!optionsLoading && selectedRuntime && (
                <div className="agent-runtime-choice-grid">
                  {options && options.runtimes.length > 1 && (
                    <details className="agent-creation-advanced agent-runtime-profile-choice">
                      <summary>Advanced runtime</summary>
                      <label>
                        Execution profile
                        <select
                          value={runtimeProfileId}
                          disabled={busy || locked}
                          onChange={(event) => changeRuntime(event.target.value)}
                        >
                          {options.runtimes.map((runtime) => (
                            <option value={runtime.runtimeProfileId} key={runtime.runtimeProfileId}>
                              {runtime.displayName}
                            </option>
                          ))}
                        </select>
                      </label>
                    </details>
                  )}
                  <label>
                    Backend
                    <select
                      autoFocus
                      value={backendOptionId}
                      disabled={busy || locked}
                      onChange={(event) => changeBackend(event.target.value)}
                    >
                      {selectedRuntime.backends.map((backend) => (
                        <option
                          value={backend.optionId}
                          disabled={!backendSelectable(backend)}
                          key={backend.optionId}
                        >
                          {backendLabel(backend)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Model
                    <select
                      value={model}
                      disabled={busy || locked || !selectedBackend}
                      onChange={(event) => {
                        const nextModel = event.target.value;
                        setModel(nextModel);
                        setModelSelections((current) => ({
                          ...current,
                          [backendOptionId]: nextModel,
                        }));
                        setRuntimeNotice(null);
                        setDirty(true);
                        setError(null);
                      }}
                    >
                      {selectedBackend?.models.map((candidate) => (
                        <option
                          value={candidate.modelId}
                          disabled={!candidate.selectable}
                          key={candidate.modelId}
                        >
                          {candidate.displayName}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Workspace
                    <select
                      value={workspace}
                      disabled={busy || locked}
                      onChange={(event) => {
                        setWorkspace(event.target.value);
                        setDirty(true);
                        setError(null);
                      }}
                    >
                      {selectedRuntime.workspaces.map((candidate) => (
                        <option
                          value={candidate.path}
                          disabled={!candidate.available}
                          key={candidate.path}
                        >
                          {candidate.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  {runtimeNotice && <p className="agent-runtime-notice" role="status">{runtimeNotice}</p>}
                </div>
              )}
              {!optionsLoading && readinessBlockers.length > 0 && (
                <section className="agent-creation-blockers" aria-label="Creation readiness">
                  <strong>Ready creation is not currently available</strong>
                  <ul>{readinessBlockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul>
                  <p>You can continue reviewing this agent and save it as a draft.</p>
                </section>
              )}
            </section>
          )}

          {stage === "Review" && (
            <section className="agent-creation-stage" aria-labelledby="agent-review-heading">
              <div className="agent-creation-stage-heading">
                <h3 id="agent-review-heading">Review</h3>
                <p>Confirm the definition and runtime choices before creating the agent.</p>
              </div>
              <dl className="agent-creation-review">
                <div><dt>Agent</dt><dd>{form.displayName} · @{form.handle}</dd></div>
                <div><dt>Purpose</dt><dd>{form.purpose}</dd></div>
                <div><dt>Backend</dt><dd>{selectedBackend ? backendLabel(selectedBackend) : "Not available"}</dd></div>
                <div><dt>Model</dt><dd>{(selectedModel?.displayName ?? model) || "Not available"}</dd></div>
                <div><dt>Workspace</dt><dd>{selectedWorkspace?.name ?? "Not available"}</dd></div>
                <div>
                  <dt>Advanced requests</dt>
                  <dd>
                    {form.collaborationOperations.length > 0
                      ? form.collaborationOperations.map((item) => item.replaceAll("_", " ")).join(", ")
                      : "None"}
                  </dd>
                </div>
              </dl>
              <p className="agent-conversation-notice">
                Creating this agent will not start a direct conversation or add it to a channel.
              </p>
              {readinessBlockers.length > 0 && (
                <section className="agent-creation-blockers" aria-label="Creation readiness">
                  <strong>Save as draft to keep this definition</strong>
                  <ul>{readinessBlockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul>
                </section>
              )}
              {busy && <p className="agent-creation-loading" role="status">Creating the agent and applying its runtime settings…</p>}
              {provisioning && provisioning.status !== "ready" && (
                <p className="agent-creation-recovery">
                  Setup progress is stored durably. Retry to continue the same operation without creating a duplicate.
                </p>
              )}
            </section>
          )}

          {error && <p className="form-error agent-creation-error" role="alert">{error}</p>}
          <div className="form-actions agent-creation-actions">
            <span>
              {stageIndex > 0 && (
                <button
                  className="panel-icon-button"
                  type="button"
                  aria-label={`Back to ${CREATION_STAGES[stageIndex - 1]}`}
                  title={`Back to ${CREATION_STAGES[stageIndex - 1]}`}
                  disabled={busy || locked}
                  onClick={goBack}
                >
                  <span aria-hidden="true">←</span>
                </button>
              )}
            </span>
            <span className="agent-creation-primary-actions">
              {stage === "Review" ? (
                <>
                  <button
                    className="quiet-button"
                    type="button"
                    disabled={busy || locked}
                    onClick={() => void saveDraft()}
                  >
                    Save as draft
                  </button>
                  <button
                    className="primary-button"
                    type="submit"
                    disabled={busy || readinessBlockers.length > 0}
                  >
                    {busy ? "Creating agent…" : locked ? "Retry setup" : "Create agent"}
                  </button>
                </>
              ) : (
                <button className="primary-button" type="submit" disabled={busy || locked}>
                  Continue
                </button>
              )}
            </span>
          </div>
        </form>
      </section>
    </div>
  );
}
