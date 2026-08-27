import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  AuthenticationError,
  ChannelAccessError,
  loadAppearancePreferences,
  loadPreferenceDocument,
  loadPreferenceHistory,
  loadGitHubSettings,
  loadNotificationPreferences,
  loadClientPreferences,
  loadSettingsWorkspace,
  loadWorkspaceConfig,
  PreferenceRevisionConflictError,
  restorePreferenceRevision,
  savePreferenceDocument,
  SettingsRevisionConflictError,
  switchWorkspace,
  updateRuntimeSettings,
  updateGitHubSettings,
  updateNotificationPreference,
  updateClientPreference,
  updateAppearancePreference,
  updateWorkspaceConfig,
} from "./api";
import type {
  WorkshopEditableCapability,
  WorkshopPreferenceDocument,
  WorkshopPreferenceHistory,
  WorkshopGitHubSettings,
  WorkshopGitHubSettingsChange,
  WorkshopNotificationPreferences,
  WorkshopNotificationPreferenceChange,
  WorkshopClientPreferences,
  WorkshopClientPreferenceChange,
  WorkshopAppearancePreferences,
  WorkshopRuntimeSettingsChange,
  WorkshopSession,
  WorkshopSettingsMutation,
  WorkshopSettingsWorkspace,
  WorkshopWorkspaceConfig,
  WorkshopWorkspaceSettingChange,
} from "./types";
import { applyWorkshopTheme } from "./theme";

const VOICE_MODE_LABELS = {
  off: "Text only",
  text_and_voice: "Text and voice",
  voice_only: "Voice only",
} as const;

const SETTINGS_SECTIONS = [
  { id: "settings-section-personal-preferences", label: "Personal preferences" },
  { id: "settings-section-runtime", label: "Runtime settings" },
  { id: "settings-section-workspace", label: "Workspace settings" },
  { id: "settings-section-github", label: "GitHub" },
  { id: "settings-section-notifications", label: "Notification delivery" },
  { id: "settings-section-clients", label: "Client preferences" },
] as const;

type SettingsSectionId = (typeof SETTINGS_SECTIONS)[number]["id"];

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
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [runtimeNotice, setRuntimeNotice] = useState<string | null>(null);
  const [runtimeBackend, setRuntimeBackend] = useState("");
  const [runtimeModel, setRuntimeModel] = useState("");
  const [runtimeTimeout, setRuntimeTimeout] = useState("");
  const [workspaceModel, setWorkspaceModel] = useState("");
  const [workspaceTimeout, setWorkspaceTimeout] = useState("");
  const [workspacePrompt, setWorkspacePrompt] = useState("");

  const [github, setGitHub] = useState<WorkshopGitHubSettings | null>(null);
  const [githubLoading, setGitHubLoading] = useState(true);
  const [githubBusy, setGitHubBusy] = useState(false);
  const [githubError, setGitHubError] = useState<string | null>(null);
  const [githubNotice, setGitHubNotice] = useState<string | null>(null);
  const [githubRepository, setGitHubRepository] = useState("");
  const [githubToken, setGitHubToken] = useState("");

  const [notifications, setNotifications] =
    useState<WorkshopNotificationPreferences | null>(null);
  const [notificationLoading, setNotificationLoading] = useState(true);
  const [notificationBusy, setNotificationBusy] = useState(false);
  const [notificationError, setNotificationError] = useState<string | null>(null);
  const [notificationNotice, setNotificationNotice] = useState<string | null>(null);

  const [clients, setClients] = useState<WorkshopClientPreferences | null>(null);
  const [clientLoading, setClientLoading] = useState(true);
  const [clientBusy, setClientBusy] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);
  const [clientNotice, setClientNotice] = useState<string | null>(null);

  const [appearance, setAppearance] = useState<WorkshopAppearancePreferences | null>(null);
  const [appearanceDraft, setAppearanceDraft] = useState("");
  const [appearanceLoading, setAppearanceLoading] = useState(true);
  const [appearanceBusy, setAppearanceBusy] = useState(false);
  const [appearanceError, setAppearanceError] = useState<string | null>(null);
  const [appearanceNotice, setAppearanceNotice] = useState<string | null>(null);

  const settingsScrollRef = useRef<HTMLDivElement>(null);
  const [activeSection, setActiveSection] = useState<SettingsSectionId>(
    SETTINGS_SECTIONS[0].id,
  );

  const preferenceDirty = preference !== null && preferenceDraft !== preference.content;
  const preferenceBytes = useMemo(
    () => new TextEncoder().encode(preferenceDraft).length,
    [preferenceDraft],
  );

  const navigateToSection = (sectionId: SettingsSectionId): void => {
    const section = document.getElementById(sectionId);
    if (!section) {
      return;
    }
    const reduceMotion = typeof window.matchMedia === "function"
      && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setActiveSection(sectionId);
    section.scrollIntoView({
      behavior: reduceMotion ? "auto" : "smooth",
      block: "start",
    });
  };

  const updateActiveSection = (): void => {
    const scroll = settingsScrollRef.current;
    if (!scroll) {
      return;
    }
    const available = SETTINGS_SECTIONS.flatMap((item) => {
      const element = document.getElementById(item.id);
      return element ? [{ item, element }] : [];
    });
    if (available.length === 0) {
      return;
    }
    if (scroll.scrollTop + scroll.clientHeight >= scroll.scrollHeight - 2) {
      setActiveSection(available[available.length - 1].item.id);
      return;
    }
    const threshold = scroll.getBoundingClientRect().top + 32;
    let current = available[0].item.id;
    for (const candidate of available) {
      if (candidate.element.getBoundingClientRect().top > threshold) {
        break;
      }
      current = candidate.item.id;
    }
    setActiveSection(current);
  };

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
    setRuntimeBackend(snapshot.backendOptionId);
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
    setWorkspaceError(null);
    try {
      const settings = await loadSettingsWorkspace(session);
      adoptRuntime(settings);
    } catch (caught) {
      if (!handleAccessFailure(caught)) {
        setRuntimeError(errorText(caught, "Could not load runtime settings."));
      }
      setRuntime(null);
      setWorkspaceConfig(null);
      setRuntimeLoading(false);
      return;
    }
    try {
      adoptWorkspace(await loadWorkspaceConfig(session));
    } catch (caught) {
      if (caught instanceof AuthenticationError) {
        onAuthenticationFailure(caught.message);
      } else {
        // A valid runtime snapshot proves this session owns the channel. Keep
        // those settings visible if the narrower workspace-override surface
        // is unavailable instead of treating its 403 as lost channel access
        // and starting a navigation-refresh loop.
        setWorkspaceConfig(null);
        setWorkspaceError(errorText(caught, "Could not load workspace settings."));
      }
    } finally {
      setRuntimeLoading(false);
    }
  }, [adoptRuntime, adoptWorkspace, handleAccessFailure, onAuthenticationFailure, session]);

  const refreshGitHub = useCallback(async (): Promise<void> => {
    setGitHubLoading(true);
    setGitHubError(null);
    try {
      setGitHub(await loadGitHubSettings(session));
    } catch (caught) {
      if (!handleAccessFailure(caught)) {
        setGitHubError(errorText(caught, "Could not load GitHub settings."));
      }
    } finally {
      setGitHubLoading(false);
    }
  }, [handleAccessFailure, session]);

  const refreshNotifications = useCallback(async (): Promise<void> => {
    setNotificationLoading(true);
    setNotificationError(null);
    try {
      setNotifications(await loadNotificationPreferences(session));
    } catch (caught) {
      if (!handleAccessFailure(caught)) {
        setNotificationError(errorText(caught, "Could not load notification preferences."));
      }
    } finally {
      setNotificationLoading(false);
    }
  }, [handleAccessFailure, session]);

  const refreshClients = useCallback(async (): Promise<void> => {
    setClientLoading(true);
    setClientError(null);
    try {
      setClients(await loadClientPreferences(session));
    } catch (caught) {
      if (!handleAccessFailure(caught)) {
        setClientError(errorText(caught, "Could not load client preferences."));
      }
    } finally {
      setClientLoading(false);
    }
  }, [handleAccessFailure, session]);

  const refreshAppearance = useCallback(async (): Promise<void> => {
    setAppearanceLoading(true);
    setAppearanceError(null);
    try {
      const snapshot = await loadAppearancePreferences(session);
      setAppearance(snapshot);
      setAppearanceDraft(snapshot.themeId);
      applyWorkshopTheme(snapshot.themeId);
    } catch (caught) {
      if (!handleAccessFailure(caught)) {
        setAppearanceError(errorText(caught, "Could not load appearance preferences."));
      }
    } finally {
      setAppearanceLoading(false);
    }
  }, [handleAccessFailure, session]);

  useEffect(() => {
    void refreshPreferences();
    void refreshRuntime();
    void refreshGitHub();
    void refreshNotifications();
    void refreshClients();
    void refreshAppearance();
  }, [
    refreshAppearance,
    refreshClients,
    refreshGitHub,
    refreshNotifications,
    refreshPreferences,
    refreshRuntime,
  ]);

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
        setRuntimeBackend(runtime.backendOptionId);
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

  const mutateGitHub = async (
    change: WorkshopGitHubSettingsChange,
    confirmation: string | null = null,
  ): Promise<boolean> => {
    if (!github || (confirmation !== null && !window.confirm(confirmation))) {
      return false;
    }
    setGitHubBusy(true);
    setGitHubError(null);
    setGitHubNotice(null);
    try {
      const changed = await updateGitHubSettings(session, github.revision, change);
      setGitHub(changed);
      setGitHubToken("");
      setGitHubNotice(changed.mutation?.changed ? "GitHub settings saved." : "No change was needed.");
      return true;
    } catch (caught) {
      setGitHubToken("");
      if (caught instanceof SettingsRevisionConflictError) {
        await refreshGitHub();
        setGitHubError("GitHub settings changed elsewhere. The latest values have been reloaded.");
      } else if (!handleAccessFailure(caught)) {
        setGitHubError(errorText(caught, "Could not update GitHub settings."));
      }
      return false;
    } finally {
      setGitHubBusy(false);
    }
  };

  const mutateNotificationPreference = async (
    change: WorkshopNotificationPreferenceChange,
    confirmation: string | null = null,
  ): Promise<void> => {
    if (!notifications || (confirmation !== null && !window.confirm(confirmation))) {
      return;
    }
    setNotificationBusy(true);
    setNotificationError(null);
    setNotificationNotice(null);
    try {
      const changed = await updateNotificationPreference(
        session,
        notifications.revision,
        change,
      );
      setNotifications(changed);
      setNotificationNotice(
        changed.mutation?.changed
          ? "Notification destination saved."
          : "No change was needed.",
      );
    } catch (caught) {
      if (caught instanceof SettingsRevisionConflictError) {
        await refreshNotifications();
        setNotificationError(
          "Notification preferences changed elsewhere. The latest values have been reloaded.",
        );
      } else if (!handleAccessFailure(caught)) {
        setNotificationError(
          errorText(caught, "Could not update notification preferences."),
        );
      }
    } finally {
      setNotificationBusy(false);
    }
  };

  const mutateClientPreference = async (
    change: WorkshopClientPreferenceChange,
  ): Promise<void> => {
    if (!clients) {
      return;
    }
    setClientBusy(true);
    setClientError(null);
    setClientNotice(null);
    try {
      const changed = await updateClientPreference(
        session,
        clients.revision,
        change,
      );
      setClients(changed);
      setClientNotice(
        changed.mutation?.changed
          ? "Client preference saved."
          : "No change was needed.",
      );
    } catch (caught) {
      if (caught instanceof SettingsRevisionConflictError) {
        await refreshClients();
        setClientError(
          "Client preferences changed elsewhere. The latest values have been reloaded.",
        );
      } else if (!handleAccessFailure(caught)) {
        setClientError(errorText(caught, "Could not update client preferences."));
      }
    } finally {
      setClientBusy(false);
    }
  };

  const mutateAppearance = async (themeId: string): Promise<void> => {
    if (!appearance) {
      return;
    }
    const previousTheme = appearance.themeId;
    setAppearanceDraft(themeId);
    applyWorkshopTheme(themeId);
    setAppearanceBusy(true);
    setAppearanceError(null);
    setAppearanceNotice(null);
    try {
      const changed = await updateAppearancePreference(
        session,
        appearance.revision,
        themeId,
      );
      setAppearance(changed);
      setAppearanceDraft(changed.themeId);
      applyWorkshopTheme(changed.themeId);
      setAppearanceNotice(
        changed.mutation?.changed
          ? "Workshop appearance saved."
          : "No change was needed.",
      );
    } catch (caught) {
      if (caught instanceof SettingsRevisionConflictError) {
        await refreshAppearance();
        setAppearanceError(
          "Appearance preferences changed elsewhere. The latest value has been reloaded.",
        );
      } else {
        setAppearanceDraft(previousTheme);
        applyWorkshopTheme(previousTheme);
        if (!handleAccessFailure(caught)) {
          setAppearanceError(errorText(caught, "Could not update appearance preferences."));
        }
      }
    } finally {
      setAppearanceBusy(false);
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

      <nav className="settings-section-navigation" aria-label="Settings sections">
        <div>
          {SETTINGS_SECTIONS.map((section) => (
            <button
              key={section.id}
              type="button"
              aria-current={activeSection === section.id ? "location" : undefined}
              onClick={() => navigateToSection(section.id)}
            >
              {section.label}
            </button>
          ))}
        </div>
      </nav>

      <div className="settings-scroll" ref={settingsScrollRef} onScroll={updateActiveSection}>
        <section className="settings-intro" id="settings-section-personal-preferences">
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

        <section className="settings-section" id="settings-section-runtime">
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
                      <option key={option.optionId} value={option.optionId}>
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
                      disabled={runtimeBusy || runActive || runtimeBackend === runtime.backendOptionId}
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
          <section className="settings-section" id="settings-section-workspace">
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
        {runtime && !workspaceConfig && !runtimeLoading && (
          <section className="settings-section" id="settings-section-workspace">
            <div>
              <p className="section-number">03</p>
              <h2>Workspace settings</h2>
              <p>Runtime settings remain available, but workspace-specific overrides could not be loaded.</p>
            </div>
            <div className="settings-failure">
              <p role="alert">{workspaceError ?? "Workspace settings are unavailable."}</p>
              <button className="quiet-button" type="button" onClick={() => void refreshRuntime()}>Retry</button>
            </div>
          </section>
        )}

        <section className="settings-section" id="settings-section-github">
          <div>
            <p className="section-number">04</p>
            <h2>GitHub</h2>
            <p>
              Personal notification subscriptions, automation choices, and a
              write-only access token. Repository execution authority remains
              operator controlled.
            </p>
          </div>
          {githubLoading ? (
            <p role="status">Loading GitHub settings…</p>
          ) : github ? (
            <div className="settings-card-grid github-settings-grid">
              <article className="settings-card policy-card">
                <p className="settings-card-label">Protected identity</p>
                <dl>
                  <div>
                    <dt>GitHub login</dt>
                    <dd>{github.githubLogin ?? "Not configured"}</dd>
                  </div>
                </dl>
                <p>This identity is operator managed and is used for GitHub actor routing.</p>
              </article>

              <article className="settings-card">
                <p className="settings-card-label">Automation</p>
                <div className="github-toggle-row">
                  <div>
                    <strong>PR reviews</strong>
                    <p>{github.prReview.enabled ? "On" : "Off"} · {github.prReview.source}</p>
                  </div>
                  <div className="settings-actions">
                    <button
                      className="primary-button"
                      type="button"
                      disabled={githubBusy}
                      onClick={() => void mutateGitHub({
                        field: "toggle",
                        name: "pr_review",
                        enabled: !github.prReview.enabled,
                      })}
                    >
                      Turn {github.prReview.enabled ? "off" : "on"}
                    </button>
                    {github.prReview.resettable && (
                      <button
                        className="quiet-button"
                        type="button"
                        disabled={githubBusy}
                        onClick={() => void mutateGitHub(
                          { field: "toggle", name: "pr_review", enabled: null },
                          "Reset PR reviews to protected policy?",
                        )}
                      >
                        Reset
                      </button>
                    )}
                  </div>
                </div>
                <div className="github-toggle-row">
                  <div>
                    <strong>Issue triage</strong>
                    <p>{github.issueTriage.enabled ? "On" : "Off"} · {github.issueTriage.source}</p>
                  </div>
                  <div className="settings-actions">
                    <button
                      className="primary-button"
                      type="button"
                      disabled={githubBusy}
                      onClick={() => void mutateGitHub({
                        field: "toggle",
                        name: "issue_triage",
                        enabled: !github.issueTriage.enabled,
                      })}
                    >
                      Turn {github.issueTriage.enabled ? "off" : "on"}
                    </button>
                    {github.issueTriage.resettable && (
                      <button
                        className="quiet-button"
                        type="button"
                        disabled={githubBusy}
                        onClick={() => void mutateGitHub(
                          { field: "toggle", name: "issue_triage", enabled: null },
                          "Reset issue triage to protected policy?",
                        )}
                      >
                        Reset
                      </button>
                    )}
                  </div>
                </div>
              </article>

              <article className="settings-card github-repositories-card">
                <p className="settings-card-label">Subscribed repositories</p>
                {github.repositories.length > 0 ? (
                  <ul className="github-repository-list">
                    {github.repositories.map((item) => (
                      <li key={item.repository}>
                        <span>
                          <strong>{item.repository}</strong>
                          <small>
                            {item.source} · {item.automationAuthorized
                              ? "automation authorized"
                              : "notifications only"}
                          </small>
                        </span>
                        <button
                          className="quiet-button"
                          type="button"
                          disabled={githubBusy}
                          onClick={() => void mutateGitHub(
                            { field: "repository", name: item.repository, subscribed: false },
                            `Unsubscribe from ${item.repository}?`,
                          )}
                        >
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p>No repository subscriptions.</p>
                )}
                <form
                  className="github-add-repository"
                  onSubmit={(event: FormEvent) => {
                    event.preventDefault();
                    const repository = githubRepository.trim();
                    if (!repository) {
                      return;
                    }
                    void mutateGitHub({ field: "repository", name: repository, subscribed: true })
                      .then((changed) => {
                        if (changed) {
                          setGitHubRepository("");
                        }
                      });
                  }}
                >
                  <label htmlFor="github-repository">Add notification subscription</label>
                  <div className="settings-inline-input">
                    <input
                      id="github-repository"
                      type="text"
                      value={githubRepository}
                      disabled={githubBusy}
                      placeholder="owner/repository"
                      autoCapitalize="none"
                      autoCorrect="off"
                      onChange={(event) => setGitHubRepository(event.target.value)}
                    />
                    <button
                      className="primary-button"
                      type="submit"
                      disabled={githubBusy || !githubRepository.trim()}
                    >
                      Add
                    </button>
                  </div>
                  <p>Subscriptions outside protected repository policy receive notifications only.</p>
                </form>
                {github.repositoriesResettable && (
                  <div className="settings-actions">
                    <button
                      className="quiet-button"
                      type="button"
                      disabled={githubBusy}
                      onClick={() => void mutateGitHub(
                        { field: "repository_reset" },
                        "Reset all repository subscriptions to protected policy?",
                      )}
                    >
                      Reset subscriptions
                    </button>
                  </div>
                )}
              </article>

              <form
                className="settings-card github-token-card"
                onSubmit={(event: FormEvent) => {
                  event.preventDefault();
                  if (githubToken.trim()) {
                    void mutateGitHub(
                      { field: "token", token: githubToken },
                      github.tokenStored ? "Replace the stored GitHub token?" : null,
                    );
                  }
                }}
              >
                <label htmlFor="github-token">GitHub access token</label>
                <p>Status: <strong>{github.tokenStored ? "Stored" : "Not set"}</strong></p>
                <input
                  id="github-token"
                  type="password"
                  value={githubToken}
                  disabled={githubBusy}
                  autoComplete="new-password"
                  placeholder={github.tokenStored ? "Enter a replacement token" : "Enter a token"}
                  onChange={(event) => setGitHubToken(event.target.value)}
                />
                <p>The existing token is never returned to this browser.</p>
                <div className="settings-actions">
                  <button
                    className="primary-button"
                    type="submit"
                    disabled={githubBusy || !githubToken.trim()}
                  >
                    {github.tokenStored ? "Replace token" : "Store token"}
                  </button>
                  {github.tokenStored && (
                    <button
                      className="quiet-button"
                      type="button"
                      disabled={githubBusy}
                      onClick={() => void mutateGitHub(
                        { field: "token", token: null },
                        "Remove the stored GitHub token? Automated GitHub actions may stop working.",
                      )}
                    >
                      Remove token
                    </button>
                  )}
                </div>
              </form>
            </div>
          ) : (
            <div className="settings-failure">
              <p role="alert">{githubError ?? "GitHub settings are unavailable."}</p>
              <button className="quiet-button" type="button" onClick={() => void refreshGitHub()}>Retry</button>
            </div>
          )}
          {githubNotice && <p className="settings-notice" role="status">{githubNotice}</p>}
          {githubError && github && <p className="settings-error" role="alert">{githubError}</p>}
        </section>

        <section className="settings-section" id="settings-section-notifications">
          <div>
            <p className="section-number">05</p>
            <h2>Notification delivery</h2>
            <p>
              Choose where personal integration notifications appear. Only
              canonical destinations authorized for your Workshop identity are listed.
            </p>
          </div>
          {notificationLoading ? (
            <p role="status">Loading notification destinations…</p>
          ) : notifications ? (
            <div className="settings-card-grid notification-preference-grid">
              {notifications.preferences.map((preference) => {
                const choices = notifications.destinations.filter((destination) =>
                  destination.supportedClasses.includes(preference.integrationClass));
                return (
                  <article className="settings-card" key={preference.integrationClass}>
                    <label htmlFor={`notification-${preference.integrationClass}`}>
                      {preference.displayName}
                    </label>
                    <select
                      id={`notification-${preference.integrationClass}`}
                      value={preference.destinationChoiceId}
                      disabled={notificationBusy || !preference.editable}
                      onChange={(event) => void mutateNotificationPreference({
                        field: "destination",
                        integrationClass: preference.integrationClass,
                        choiceId: event.target.value,
                      })}
                    >
                      {choices.map((destination) => (
                        <option key={destination.choiceId} value={destination.choiceId}>
                          {destination.displayName} · {destination.kind}
                        </option>
                      ))}
                    </select>
                    <p>
                      Effective: <strong>{preference.destinationName}</strong>
                      {` · ${preference.source}`}
                    </p>
                    {preference.resettable && (
                      <div className="settings-actions">
                        <button
                          className="quiet-button"
                          type="button"
                          disabled={notificationBusy}
                          onClick={() => void mutateNotificationPreference(
                            {
                              field: "reset",
                              integrationClass: preference.integrationClass,
                            },
                            `Reset ${preference.displayName} delivery to protected policy?`,
                          )}
                        >
                          Reset
                        </button>
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          ) : (
            <div className="settings-failure">
              <p role="alert">
                {notificationError ?? "Notification preferences are unavailable."}
              </p>
              <button
                className="quiet-button"
                type="button"
                onClick={() => void refreshNotifications()}
              >
                Retry
              </button>
            </div>
          )}
          {notificationNotice && (
            <p className="settings-notice" role="status">{notificationNotice}</p>
          )}
          {notificationError && notifications && (
            <p className="settings-error" role="alert">{notificationError}</p>
          )}
        </section>

        <section className="settings-section" id="settings-section-clients">
          <div>
            <p className="section-number">06</p>
            <h2>Client preferences</h2>
            <p>
              Control presentation on each connected client without changing
              how your other Kai clients behave.
            </p>
          </div>
          <div className="settings-client-preferences">
          {appearanceLoading ? (
            <p role="status">Loading Workshop appearance…</p>
          ) : appearance ? (
            <article className="settings-card settings-appearance-card">
              <h3>Workshop appearance</h3>
              <p>
                This theme follows your Workshop account across enrolled browsers.
              </p>
              <label htmlFor="workshop-theme">Theme</label>
              <select
                id="workshop-theme"
                value={appearanceDraft}
                disabled={appearanceBusy}
                onChange={(event) => void mutateAppearance(event.target.value)}
              >
                {appearance.themes.map((theme) => (
                  <option key={theme.themeId} value={theme.themeId}>
                    {theme.displayName}
                  </option>
                ))}
              </select>
            </article>
          ) : (
            <div className="settings-failure">
              <p role="alert">
                {appearanceError ?? "Appearance preferences are unavailable."}
              </p>
              <button className="quiet-button" type="button" onClick={() => void refreshAppearance()}>
                Retry
              </button>
            </div>
          )}
          {appearanceNotice && <p className="settings-notice" role="status">{appearanceNotice}</p>}
          {appearanceError && appearance && <p className="settings-error" role="alert">{appearanceError}</p>}
          {clientLoading ? (
            <p role="status">Loading client preferences…</p>
          ) : clients ? (
            clients.voiceOutput.bindings.length > 0 ? (
              <div className="settings-card-grid">
                {clients.voiceOutput.bindings.map((binding) => (
                  <article className="settings-card" key={binding.choiceId}>
                    <h3>{binding.clientName} voice output</h3>
                    {!clients.voiceOutput.available && (
                      <p>{clients.voiceOutput.unavailableReason}</p>
                    )}
                    <label htmlFor={`client-mode-${binding.choiceId}`}>Response format</label>
                    <select
                      id={`client-mode-${binding.choiceId}`}
                      value={binding.mode}
                      disabled={clientBusy || !binding.editable}
                      onChange={(event) => void mutateClientPreference({
                        field: "mode",
                        bindingChoiceId: binding.choiceId,
                        value: event.target.value as "off" | "text_and_voice" | "voice_only",
                      })}
                    >
                      {clients.voiceOutput.modes.map((mode) => (
                        <option key={mode} value={mode}>{VOICE_MODE_LABELS[mode]}</option>
                      ))}
                    </select>
                    <label htmlFor={`client-voice-${binding.choiceId}`}>Voice</label>
                    <select
                      id={`client-voice-${binding.choiceId}`}
                      value={binding.voice}
                      disabled={clientBusy || !binding.editable}
                      onChange={(event) => void mutateClientPreference({
                        field: "voice",
                        bindingChoiceId: binding.choiceId,
                        value: event.target.value,
                      })}
                    >
                      {clients.voiceOutput.voices.map((voice) => (
                        <option key={voice.value} value={voice.value}>{voice.displayName}</option>
                      ))}
                    </select>
                  </article>
                ))}
              </div>
            ) : (
              <p>No voice-capable client binding is available for this account.</p>
            )
          ) : (
            <div className="settings-failure">
              <p role="alert">{clientError ?? "Client preferences are unavailable."}</p>
              <button className="quiet-button" type="button" onClick={() => void refreshClients()}>
                Retry
              </button>
            </div>
          )}
          {clientNotice && <p className="settings-notice" role="status">{clientNotice}</p>}
          {clientError && clients && <p className="settings-error" role="alert">{clientError}</p>}
          </div>
        </section>
      </div>
    </section>
  );
}
