import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import {
  AuthenticationError,
  ChannelAccessError,
  loadPreferenceDocument,
  loadPreferenceHistory,
  loadSettingsWorkspace,
  loadWorkspaceConfig,
  PreferenceRevisionConflictError,
  restorePreferenceRevision,
  savePreferenceDocument,
  SettingsRevisionConflictError,
  switchWorkspace,
  updateRuntimeSettings,
  updateWorkspaceConfig,
} from "./api";
import type {
  WorkshopEditableCapability,
  WorkshopPreferenceDocument,
  WorkshopPreferenceHistory,
  WorkshopRuntimeSettingsChange,
  WorkshopSession,
  WorkshopSettingsMutation,
  WorkshopSettingsWorkspace,
  WorkshopWorkspaceConfig,
  WorkshopWorkspaceSettingChange,
} from "./types";

function formatDate(value: string | null): string {
  if (!value) {
    return "Not saved yet";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(parsed);
}

function capability(
  capabilities: WorkshopEditableCapability[],
  field: string,
): WorkshopEditableCapability | null {
  return capabilities.find((item) => item.field === field) ?? null;
}

function mutationMessage(mutation: WorkshopSettingsMutation | null): string {
  if (!mutation || !mutation.changed || mutation.runtimeAction === "unchanged") {
    return "No runtime change was needed.";
  }
  if (mutation.runtimeAction === "restarted") {
    return mutation.providerSessionInvalidated
      ? "Saved. The active runtime restarted and provider-session continuity was cleared."
      : "Saved. The active runtime restarted."
  }
  return "Saved. The change will apply when this runtime starts next.";
}

function errorText(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback;
}

export function SettingsWorkspace({
  onAuthenticationFailure,
  onChannelAccessFailure,
  onClose,
  onDirtyChange,
  principalName,
  roleLabel,
  runtimeLabel,
  runActive,
  session,
}: {
  onAuthenticationFailure: (message: string) => void;
  onChannelAccessFailure: (message: string) => void;
  onClose: () => void;
  onDirtyChange: (dirty: boolean) => void;
  principalName: string;
  roleLabel: string;
  runtimeLabel: string;
  runActive: boolean;
  session: WorkshopSession;
}): React.JSX.Element {
  const [preference, setPreference] = useState<WorkshopPreferenceDocument | null>(null);
  const [preferenceDraft, setPreferenceDraft] = useState("");
  const [preferenceHistory, setPreferenceHistory] =
    useState<WorkshopPreferenceHistory | null>(null);
  const [conflictDocument, setConflictDocument] =
    useState<WorkshopPreferenceDocument | null>(null);
  const [preferenceLoading, setPreferenceLoading] = useState(true);
  const [preferenceBusy, setPreferenceBusy] = useState(false);
  const [preferenceError, setPreferenceError] = useState<string | null>(null);
  const [preferenceNotice, setPreferenceNotice] = useState<string | null>(null);

  const [runtime, setRuntime] = useState<WorkshopSettingsWorkspace | null>(null);
  const [workspaceConfig, setWorkspaceConfig] = useState<WorkshopWorkspaceConfig | null>(null);
  const [runtimeLoading, setRuntimeLoading] = useState(true);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [runtimeNotice, setRuntimeNotice] = useState<string | null>(null);
  const [runtimeBackend, setRuntimeBackend] = useState("");
  const [runtimeModel, setRuntimeModel] = useState("");
  const [runtimeTimeout, setRuntimeTimeout] = useState("");
  const [workspaceModel, setWorkspaceModel] = useState("");
  const [workspaceTimeout, setWorkspaceTimeout] = useState("");
  const [workspacePrompt, setWorkspacePrompt] = useState("");

  const preferenceDirty = preference !== null && preferenceDraft !== preference.content;
  const preferenceBytes = useMemo(
    () => new TextEncoder().encode(preferenceDraft).length,
    [preferenceDraft],
  );

  const handleAccessFailure = useCallback((caught: unknown): boolean => {
    if (caught instanceof AuthenticationError) {
      onAuthenticationFailure(caught.message);
      return true;
    }
    if (caught instanceof ChannelAccessError) {
      onChannelAccessFailure(caught.message);
      return true;
    }
    return false;
  }, [onAuthenticationFailure, onChannelAccessFailure]);

  const adoptPreference = useCallback((document: WorkshopPreferenceDocument): void => {
    setPreference(document);
    setPreferenceDraft(document.content);
    setConflictDocument(null);
  }, []);

  const refreshPreferenceHistory = useCallback(async (): Promise<void> => {
    setPreferenceHistory(await loadPreferenceHistory(session));
  }, [session]);

  const refreshPreferences = useCallback(async (): Promise<void> => {
    setPreferenceLoading(true);
    setPreferenceError(null);
    try {
      const [document, history] = await Promise.all([
        loadPreferenceDocument(session),
        loadPreferenceHistory(session),
      ]);
      adoptPreference(document);
      setPreferenceHistory(history);
    } catch (caught) {
      if (!handleAccessFailure(caught)) {
        setPreferenceError(errorText(caught, "Could not load personal preferences."));
      }
    } finally {
      setPreferenceLoading(false);
    }
  }, [adoptPreference, handleAccessFailure, session]);

  const adoptRuntime = useCallback((snapshot: WorkshopSettingsWorkspace): void => {
    setRuntime(snapshot);
    setRuntimeBackend(snapshot.backend);
    setRuntimeModel(snapshot.model.value);
    setRuntimeTimeout(String(snapshot.timeoutSeconds.value));
  }, []);

  const adoptWorkspace = useCallback((snapshot: WorkshopWorkspaceConfig): void => {
    setWorkspaceConfig(snapshot);
    setWorkspaceModel(snapshot.model.value);
    setWorkspaceTimeout(String(snapshot.timeoutSeconds.value));
    setWorkspacePrompt(snapshot.prompt ?? "");
  }, []);

  const refreshRuntime = useCallback(async (): Promise<void> => {
    setRuntimeLoading(true);
    setRuntimeError(null);
    try {
      const [settings, workspace] = await Promise.all([
        loadSettingsWorkspace(session),
        loadWorkspaceConfig(session),
      ]);
      adoptRuntime(settings);
      adoptWorkspace(workspace);
    } catch (caught) {
      if (!handleAccessFailure(caught)) {
        setRuntimeError(errorText(caught, "Could not load runtime settings."));
      }
    } finally {
      setRuntimeLoading(false);
    }
  }, [adoptRuntime, adoptWorkspace, handleAccessFailure, session]);

  useEffect(() => {
    void refreshPreferences();
    void refreshRuntime();
  }, [refreshPreferences, refreshRuntime]);

  useEffect(() => {
    onDirtyChange(preferenceDirty);
    return () => onDirtyChange(false);
  }, [onDirtyChange, preferenceDirty]);

  useEffect(() => {
    if (!preferenceDirty) {
      return;
    }
    const warn = (event: BeforeUnloadEvent): void => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [preferenceDirty]);

  const savePreferences = async (
    revision = preference?.revision ?? "",
  ): Promise<void> => {
    if (!preference || preferenceBytes > preference.maxBytes) {
      return;
    }
    setPreferenceBusy(true);
    setPreferenceError(null);
    setPreferenceNotice(null);
    try {
      const saved = await savePreferenceDocument(
        session,
        preferenceDraft,
        revision,
      );
      adoptPreference(saved);
      await refreshPreferenceHistory();
      setPreferenceNotice("Preferences saved. They will apply to the next eligible agent turn.");
    } catch (caught) {
      if (caught instanceof PreferenceRevisionConflictError) {
        try {
          const latest = await loadPreferenceDocument(session);
          setConflictDocument(latest);
          setPreferenceError(
            "Preferences changed in another session. Compare both versions before choosing what to keep.",
          );
        } catch (refreshError) {
          if (!handleAccessFailure(refreshError)) {
            setPreferenceError(errorText(refreshError, caught.message));
          }
        }
      } else if (!handleAccessFailure(caught)) {
        setPreferenceError(errorText(caught, "Could not save preferences."));
      }
    } finally {
      setPreferenceBusy(false);
    }
  };

  const restoreRevision = async (targetRevision: string): Promise<void> => {
    if (!preference || !window.confirm(
      "Restore this private preference revision? The current text will be retained in revision history.",
    )) {
      return;
    }
    setPreferenceBusy(true);
    setPreferenceError(null);
    setPreferenceNotice(null);
    try {
      const restored = await restorePreferenceRevision(
        session,
        targetRevision,
        preference.revision,
      );
      adoptPreference(restored);
      await refreshPreferenceHistory();
      setPreferenceNotice("Preference revision restored.");
    } catch (caught) {
      if (caught instanceof PreferenceRevisionConflictError) {
        setPreferenceError("Preferences changed before the restore. Reload and review the latest revision.");
      } else if (!handleAccessFailure(caught)) {
        setPreferenceError(errorText(caught, "Could not restore preferences."));
      }
    } finally {
      setPreferenceBusy(false);
    }
  };

  const mutateRuntime = async (
    change: WorkshopRuntimeSettingsChange,
    confirmation: string,
  ): Promise<void> => {
    if (!runtime || !window.confirm(confirmation)) {
      return;
    }
    setRuntimeBusy(true);
    setRuntimeError(null);
    setRuntimeNotice(null);
    try {
      const changed = await updateRuntimeSettings(
        session,
        runtime.revision,
        change,
      );
      adoptRuntime(changed);
      adoptWorkspace(await loadWorkspaceConfig(session));
      setRuntimeNotice(mutationMessage(changed.mutation));
    } catch (caught) {
      if (change.field === "backend") {
        setRuntimeBackend(runtime.backend);
      }
      if (caught instanceof SettingsRevisionConflictError) {
        await refreshRuntime();
        setRuntimeError("Runtime settings changed elsewhere. The latest values have been reloaded.");
      } else if (!handleAccessFailure(caught)) {
        setRuntimeError(errorText(caught, "Could not update runtime settings."));
      }
    } finally {
      setRuntimeBusy(false);
    }
  };

  const mutateWorkspace = async (
    change: WorkshopWorkspaceSettingChange,
    confirmation: string,
  ): Promise<void> => {
    if (!workspaceConfig || !window.confirm(confirmation)) {
      return;
    }
    setRuntimeBusy(true);
    setRuntimeError(null);
    setRuntimeNotice(null);
    try {
      const changed = await updateWorkspaceConfig(
        session,
        workspaceConfig.revision,
        change,
      );
      adoptWorkspace(changed);
      adoptRuntime(await loadSettingsWorkspace(session));
      setRuntimeNotice(mutationMessage(changed.mutation));
    } catch (caught) {
      if (caught instanceof SettingsRevisionConflictError) {
        await refreshRuntime();
        setRuntimeError("Workspace settings changed elsewhere. The latest values have been reloaded.");
      } else if (!handleAccessFailure(caught)) {
        setRuntimeError(errorText(caught, "Could not update workspace settings."));
      }
    } finally {
      setRuntimeBusy(false);
    }
  };

  const selectWorkspace = async (path: string): Promise<void> => {
    if (!runtime || path === runtime.workspace || !window.confirm(
      "Switch the active workspace? An active runtime may restart and provider-session continuity may be cleared.",
    )) {
      return;
    }
    setRuntimeBusy(true);
    setRuntimeError(null);
    setRuntimeNotice(null);
    try {
      const changed = await switchWorkspace(session, path, runtime.revision);
      adoptRuntime(changed);
      adoptWorkspace(await loadWorkspaceConfig(session));
      setRuntimeNotice(mutationMessage(changed.mutation));
    } catch (caught) {
      if (caught instanceof SettingsRevisionConflictError) {
        await refreshRuntime();
        setRuntimeError("Workspace selection changed elsewhere. The latest state has been reloaded.");
      } else if (!handleAccessFailure(caught)) {
        setRuntimeError(errorText(caught, "Could not switch workspace."));
      }
    } finally {
      setRuntimeBusy(false);
    }
  };

  const runtimeBackendCapability = runtime
    ? capability(runtime.capabilities, "backend")
    : null;
  const runtimeModelCapability = runtime
    ? capability(runtime.capabilities, "model")
    : null;
  const runtimeTimeoutCapability = runtime
    ? capability(runtime.capabilities, "timeout")
    : null;
  const workspaceModelCapability = workspaceConfig
    ? capability(workspaceConfig.capabilities, "model")
    : null;
  const workspaceTimeoutCapability = workspaceConfig
    ? capability(workspaceConfig.capabilities, "timeout")
    : null;
  const workspacePromptCapability = workspaceConfig
    ? capability(workspaceConfig.capabilities, "prompt")
    : null;
  const activeWorkspaceIndex = runtime
    ? runtime.workspaces.findIndex((item) => item.path === runtime.workspace)
    : -1;

  const modelControl = (
    id: string,
    value: string,
    choices: string[] | null,
    onChange: (value: string) => void,
  ): React.JSX.Element => choices ? (
    <select id={id} value={value} onChange={(event) => onChange(event.target.value)}>
      {choices.map((choice) => {
        const display = runtime?.modelOptions?.find((item) => item.modelId === choice)?.displayName;
        return <option key={choice} value={choice}>{display ?? choice}</option>;
      })}
    </select>
  ) : (
    <input id={id} type="text" value={value} onChange={(event) => onChange(event.target.value)} />
  );

  return (
    <section className="settings-workspace" aria-labelledby="settings-title">
      <header className="settings-header">
        <div>
          <p className="overline">Personal workspace</p>
          <h1 id="settings-title">Settings</h1>
          <p>{principalName} · {roleLabel}</p>
        </div>
        <button className="quiet-button settings-mobile-back" type="button" onClick={onClose}>
          Back to conversation
        </button>
      </header>

      <div className="settings-scroll">
        <section className="settings-intro">
          <div>
            <p className="section-number">01</p>
            <h2>Personal preferences</h2>
            <p>
              Principal-wide guidance supplied to your agents. This is private to
              your Workshop identity and is not semantic memory.
            </p>
          </div>
          {preferenceLoading ? (
            <p role="status">Loading preferences…</p>
          ) : preference ? (
            <div className="preference-editor-card">
              <label htmlFor="personal-preferences">Preference Markdown</label>
              <textarea
                id="personal-preferences"
                value={preferenceDraft}
                readOnly={!preference.editable}
                disabled={preferenceBusy}
                onChange={(event) => {
                  setPreferenceDraft(event.target.value);
                  setPreferenceNotice(null);
                }}
                rows={14}
                spellCheck
              />
              <div className="preference-editor-meta">
                <span>{preferenceBytes.toLocaleString()} / {preference.maxBytes.toLocaleString()} bytes</span>
                <span>Updated {formatDate(preference.updatedAt)}</span>
                <span>{preferenceDirty ? "Unsaved changes" : "Saved"}</span>
              </div>
              <div className="settings-actions">
                <button
                  className="primary-button"
                  type="button"
                  disabled={
                    preferenceBusy ||
                    !preference.editable ||
                    !preferenceDirty ||
                    preferenceBytes > preference.maxBytes
                  }
                  onClick={() => void savePreferences()}
                >
                  {preferenceBusy ? "Saving…" : "Save preferences"}
                </button>
                <button
                  className="quiet-button"
                  type="button"
                  disabled={preferenceBusy || !preferenceDirty}
                  onClick={() => setPreferenceDraft(preference.content)}
                >
                  Discard changes
                </button>
              </div>
              {preferenceBytes > preference.maxBytes && (
                <p className="settings-error" role="alert">Preference text exceeds the protected size limit.</p>
              )}
              {preferenceNotice && <p className="settings-notice" role="status">{preferenceNotice}</p>}
              {preferenceError && <p className="settings-error" role="alert">{preferenceError}</p>}
            </div>
          ) : (
            <div className="settings-failure">
              <p role="alert">{preferenceError ?? "Preferences are unavailable."}</p>
              <button className="quiet-button" type="button" onClick={() => void refreshPreferences()}>Retry</button>
            </div>
          )}

          {conflictDocument && preference && (
            <section className="preference-conflict" aria-labelledby="preference-conflict-title">
              <h3 id="preference-conflict-title">Review concurrent changes</h3>
              <p>Your draft remains untouched. Compare it with the latest saved text before replacing either version.</p>
              <div className="preference-compare">
                <div><strong>Your draft</strong><pre>{preferenceDraft}</pre></div>
                <div><strong>Latest saved</strong><pre>{conflictDocument.content}</pre></div>
              </div>
              <div className="settings-actions">
                <button
                  className="quiet-button"
                  type="button"
                  onClick={() => adoptPreference(conflictDocument)}
                >
                  Use latest saved
                </button>
                <button
                  className="primary-button"
                  type="button"
                  disabled={preferenceBusy}
                  onClick={() => {
                    if (window.confirm("Replace the latest saved preferences with your reviewed draft?")) {
                      void savePreferences(conflictDocument.revision);
                    }
                  }}
                >
                  Reapply my draft…
                </button>
              </div>
            </section>
          )}

          {preferenceHistory && preferenceHistory.revisions.length > 0 && (
            <details className="preference-history">
              <summary>Private revision history ({preferenceHistory.revisions.length})</summary>
              <ol>
                {preferenceHistory.revisions.map((item) => (
                  <li key={item.revision}>
                    <span><strong>{formatDate(item.updatedAt)}</strong><small>{item.sizeBytes.toLocaleString()} bytes</small></span>
                    <button
                      className="quiet-button"
                      type="button"
                      disabled={preferenceBusy || item.revision === preference?.revision}
                      onClick={() => void restoreRevision(item.revision)}
                    >
                      Restore…
                    </button>
                  </li>
                ))}
              </ol>
            </details>
          )}
        </section>

        <section className="settings-section">
          <div>
            <p className="section-number">02</p>
            <h2>Runtime settings</h2>
            <p>
              Policy-bounded controls for {runtimeLabel}. Changes can restart an
              active runtime and clear provider-session continuity.
            </p>
          </div>
          {runtimeLoading ? (
            <p role="status">Loading runtime policy…</p>
          ) : runtime ? (
            <div className="settings-card-grid">
              {runtimeBackendCapability && (
                <form
                  className="settings-card"
                  onSubmit={(event: FormEvent) => {
                    event.preventDefault();
                    void mutateRuntime(
                      { field: "backend", value: runtimeBackend },
                      "Switch backend? Your current provider session will end. Your next message will start on the selected backend; other people and Kai itself will not restart.",
                    );
                  }}
                >
                  <label htmlFor="runtime-backend">Backend</label>
                  <select
                    id="runtime-backend"
                    value={runtimeBackend}
                    disabled={runtimeBusy || runActive}
                    onChange={(event) => setRuntimeBackend(event.target.value)}
                  >
                    {runtime.backendOptions.map((option) => (
                      <option key={option.backend} value={option.backend}>
                        {option.backend} · {option.provider}
                      </option>
                    ))}
                  </select>
                  <p>Active: <strong>{runtime.backend}</strong> · {runtime.provider}</p>
                  <p>
                    {runActive
                      ? "Finish or stop the active run before switching."
                      : "Only your runtime lane and provider session are replaced. If sign-in is required, your next message will explain how; you can switch back here immediately."}
                  </p>
                  <div className="settings-actions">
                    <button
                      className="primary-button"
                      type="submit"
                      disabled={runtimeBusy || runActive || runtimeBackend === runtime.backend}
                    >
                      Switch backend
                    </button>
                  </div>
                </form>
              )}

              <article className="settings-card policy-card">
                <p className="settings-card-label">Policy-controlled</p>
                <dl>
                  <div><dt>Authorized choices</dt><dd>{runtime.backendOptions.length}</dd></div>
                </dl>
                <p>Credentials, identity mappings, executable paths, assignments, and workspace grants remain operator managed.</p>
              </article>

              {runtimeModelCapability && (
                <form
                  className="settings-card"
                  onSubmit={(event: FormEvent) => {
                    event.preventDefault();
                    void mutateRuntime(
                      { field: "model", value: runtimeModel },
                      "Change the runtime model? The active runtime may restart and provider-session continuity will be cleared.",
                    );
                  }}
                >
                  <label htmlFor="runtime-model">Runtime model</label>
                  {modelControl("runtime-model", runtimeModel, runtimeModelCapability.choices, setRuntimeModel)}
                  <p>Effective: <strong>{runtime.model.value}</strong> · {runtime.model.source}</p>
                  <p>Policy default: {runtime.model.defaultValue}</p>
                  <div className="settings-actions">
                    <button className="primary-button" type="submit" disabled={runtimeBusy || runtimeModel === runtime.model.value}>Apply</button>
                    {runtimeModelCapability.resettable && (
                      <button className="quiet-button" type="button" disabled={runtimeBusy || runtime.model.source === "runtime policy"} onClick={() => void mutateRuntime({ field: "reset", value: "model" }, "Reset the runtime model to protected policy? The active runtime may restart.")}>Reset</button>
                    )}
                  </div>
                </form>
              )}

              {runtimeTimeoutCapability && (
                <form
                  className="settings-card"
                  onSubmit={(event: FormEvent) => {
                    event.preventDefault();
                    void mutateRuntime(
                      { field: "timeout", value: Number(runtimeTimeout) },
                      "Change the response timeout? The active runtime may restart and provider-session continuity will be cleared.",
                    );
                  }}
                >
                  <label htmlFor="runtime-timeout">Response timeout</label>
                  <div className="settings-number-input"><input id="runtime-timeout" type="number" min={runtimeTimeoutCapability.minimum ?? undefined} max={runtimeTimeoutCapability.maximum ?? undefined} value={runtimeTimeout} onChange={(event) => setRuntimeTimeout(event.target.value)} /><span>seconds</span></div>
                  <p>Effective: <strong>{runtime.timeoutSeconds.value}s</strong> · {runtime.timeoutSeconds.source}</p>
                  <p>Allowed: {runtimeTimeoutCapability.minimum}–{runtimeTimeoutCapability.maximum}s</p>
                  <div className="settings-actions">
                    <button className="primary-button" type="submit" disabled={runtimeBusy || !runtimeTimeout || Number(runtimeTimeout) === runtime.timeoutSeconds.value}>Apply</button>
                    {runtimeTimeoutCapability.resettable && (
                      <button className="quiet-button" type="button" disabled={runtimeBusy || runtime.timeoutSeconds.source === "runtime policy"} onClick={() => void mutateRuntime({ field: "reset", value: "timeout" }, "Reset the timeout to protected policy? The active runtime may restart.")}>Reset</button>
                    )}
                  </div>
                </form>
              )}
            </div>
          ) : (
            <div className="settings-failure"><p role="alert">{runtimeError ?? "Runtime settings are unavailable."}</p><button className="quiet-button" type="button" onClick={() => void refreshRuntime()}>Retry</button></div>
          )}
          {runtimeNotice && <p className="settings-notice" role="status">{runtimeNotice}</p>}
          {runtimeError && runtime && <p className="settings-error" role="alert">{runtimeError}</p>}
        </section>

        {runtime && workspaceConfig && (
          <section className="settings-section">
            <div>
              <p className="section-number">03</p>
              <h2>Workspace settings</h2>
              <p>Choose an existing authorized workspace and manage overrides that apply only within it.</p>
            </div>
            <div className="workspace-settings-heading">
              <label htmlFor="settings-workspace">Active workspace</label>
              <select
                id="settings-workspace"
                value={String(activeWorkspaceIndex)}
                disabled={runtimeBusy}
                onChange={(event) => {
                  const selected = runtime.workspaces[Number(event.target.value)];
                  if (selected) {
                    void selectWorkspace(selected.path);
                  }
                }}
              >
                {runtime.workspaces.map((item, index) => (
                  <option key={item.path} value={String(index)}>{item.name}</option>
                ))}
              </select>
            </div>
            <div className="settings-card-grid workspace-overrides">
              {workspaceModelCapability && (
                <form className="settings-card" onSubmit={(event) => { event.preventDefault(); void mutateWorkspace({ field: "model", value: workspaceModel }, "Apply this model only to the active workspace? The active runtime may restart."); }}>
                  <label htmlFor="workspace-model">Workspace model override</label>
                  {modelControl("workspace-model", workspaceModel, workspaceModelCapability.choices, setWorkspaceModel)}
                  <p>Effective: <strong>{workspaceConfig.model.value}</strong> · {workspaceConfig.model.source}</p>
                  <div className="settings-actions"><button className="primary-button" type="submit" disabled={runtimeBusy || workspaceModel === workspaceConfig.model.value}>Apply</button><button className="quiet-button" type="button" disabled={runtimeBusy || !workspaceConfig.overrideFields.includes("model")} onClick={() => void mutateWorkspace({ field: "reset", value: "model" }, "Remove this workspace model override?")}>Reset</button></div>
                </form>
              )}
              {workspaceTimeoutCapability && (
                <form className="settings-card" onSubmit={(event) => { event.preventDefault(); void mutateWorkspace({ field: "timeout", value: workspaceTimeout }, "Apply this timeout only to the active workspace? The active runtime may restart."); }}>
                  <label htmlFor="workspace-timeout">Workspace timeout override</label>
                  <div className="settings-number-input"><input id="workspace-timeout" type="number" min={workspaceTimeoutCapability.minimum ?? undefined} max={workspaceTimeoutCapability.maximum ?? undefined} value={workspaceTimeout} onChange={(event) => setWorkspaceTimeout(event.target.value)} /><span>seconds</span></div>
                  <p>Effective: <strong>{workspaceConfig.timeoutSeconds.value}s</strong> · {workspaceConfig.timeoutSeconds.source}</p>
                  <div className="settings-actions"><button className="primary-button" type="submit" disabled={runtimeBusy || !workspaceTimeout || Number(workspaceTimeout) === workspaceConfig.timeoutSeconds.value}>Apply</button><button className="quiet-button" type="button" disabled={runtimeBusy || !workspaceConfig.overrideFields.includes("timeout")} onClick={() => void mutateWorkspace({ field: "reset", value: "timeout" }, "Remove this workspace timeout override?")}>Reset</button></div>
                </form>
              )}
              {workspacePromptCapability && (
                <form className="settings-card prompt-card" onSubmit={(event) => { event.preventDefault(); void mutateWorkspace({ field: "prompt", value: workspacePrompt }, "Apply this system prompt only to the active workspace? The active runtime may restart."); }}>
                  <label htmlFor="workspace-prompt">Workspace system prompt</label>
                  <textarea id="workspace-prompt" rows={7} maxLength={workspacePromptCapability.maximum ?? undefined} value={workspacePrompt} onChange={(event) => setWorkspacePrompt(event.target.value)} />
                  <p>{workspaceConfig.hasPrompt ? `Effective source: ${workspaceConfig.promptSource ?? "workspace"}` : "No workspace prompt is active."}</p>
                  <div className="settings-actions"><button className="primary-button" type="submit" disabled={runtimeBusy || workspacePrompt === (workspaceConfig.prompt ?? "")}>Apply</button><button className="quiet-button" type="button" disabled={runtimeBusy || !workspaceConfig.overrideFields.includes("prompt")} onClick={() => void mutateWorkspace({ field: "reset", value: "prompt" }, "Remove this workspace prompt override?")}>Reset</button></div>
                </form>
              )}
            </div>
          </section>
        )}
      </div>
    </section>
  );
}
