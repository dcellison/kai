import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import {
  AuthenticationError,
  ChannelAccessError,
  SettingsRevisionConflictError,
  addExistingWorkspace,
  createWorkspace,
  deleteWorkspace,
  loadMemoryProjects,
  loadSettingsWorkspace,
  loadWorkspaceGrants,
  registerMemoryProject,
  removeExistingWorkspace,
  unregisterMemoryProject,
} from "./api";
import type {
  WorkshopMemoryProject,
  WorkshopMemoryProjectRegistry,
  WorkshopSession,
  WorkshopSettingsWorkspace,
  WorkshopWorkspaceGrant,
  WorkshopWorkspaceOption,
  WorkshopWorkspaceGrants,
} from "./types";
import { useConfirmation } from "./ConfirmationDialog";

function WorkspaceAddIcon(): React.JSX.Element {
  return (
    <svg aria-hidden="true" fill="none" focusable="false" viewBox="0 0 24 24">
      <path d="M3.5 6.5h6l2 2h9v10h-17z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />
      <path d="M12 11v5m-2.5-2.5h5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}

function ExistingWorkspaceIcon(): React.JSX.Element {
  return (
    <svg aria-hidden="true" fill="none" focusable="false" viewBox="0 0 24 24">
      <path d="M3.5 6.5h6l2 2h9v10h-17z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />
      <path d="M12 11v5m-2.5-2.5h5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      <path d="M7 3.5h10" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}

function DeleteIcon(): React.JSX.Element {
  return (
    <svg aria-hidden="true" fill="none" focusable="false" viewBox="0 0 24 24">
      <path d="M5 7h14M9 7V4h6v3m2 0-1 13H8L7 7m3 4v5m4-5v5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
    </svg>
  );
}

function errorText(caught: unknown, fallback: string): string {
  return caught instanceof Error && caught.message ? caught.message : fallback;
}

function grantProvenance(provenance: string | null, home: boolean): string {
  if (home) return "Private home";
  switch (provenance) {
    case "principal_created": return "Created by you";
    case "principal_added": return "Added by you";
    case "legacy_migrated": return "Migrated access";
    case "profile": return "Runtime profile";
    default: return provenance ? provenance.replaceAll("_", " ") : "Authorized workspace";
  }
}

function projectProvenance(project: WorkshopMemoryProject): string {
  switch (project.provenance) {
    case "operator_pinned": return "Operator pinned";
    case "principal_created": return "Created with this workspace";
    case "principal_registered": return "Registered by you";
    case "legacy_migrated": return "Migrated registration";
  }
}

type CatalogueEntry = {
  grant: WorkshopWorkspaceGrant | null;
  option: WorkshopWorkspaceOption;
  project: WorkshopMemoryProject | null;
};

export function WorkspacesWorkspace({
  onAuthenticationFailure,
  onChannelAccessFailure,
  onClose,
  session,
}: {
  onAuthenticationFailure: (message: string) => void;
  onChannelAccessFailure: (message: string) => void;
  onClose: () => void;
  session: WorkshopSession;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const [runtime, setRuntime] = useState<WorkshopSettingsWorkspace | null>(null);
  const [grants, setGrants] = useState<WorkshopWorkspaceGrants | null>(null);
  const [projects, setProjects] = useState<WorkshopMemoryProjectRegistry | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState("");
  const [addOpen, setAddOpen] = useState(false);
  const [addPath, setAddPath] = useState("");
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [projectName, setProjectName] = useState("");

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

  const refresh = useCallback(async (preferredPath: string | null = null): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const [nextRuntime, nextGrants, nextProjects] = await Promise.all([
        loadSettingsWorkspace(session),
        loadWorkspaceGrants(session),
        loadMemoryProjects(session),
      ]);
      setRuntime(nextRuntime);
      setGrants(nextGrants);
      setProjects(nextProjects);
      setSelectedPath((current) => {
        const requested = preferredPath ?? current ?? nextRuntime.workspace;
        return nextRuntime.workspaces.some((item) => item.path === requested)
          ? requested
          : nextRuntime.workspace;
      });
    } catch (caught) {
      if (!handleAccessFailure(caught)) {
        setError(errorText(caught, "Could not load workspaces."));
      }
    } finally {
      setLoading(false);
    }
  }, [handleAccessFailure, session]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const entries = useMemo<CatalogueEntry[]>(() => {
    if (!runtime) return [];
    const grantByPath = new Map((grants?.grants ?? []).map((item) => [item.path, item]));
    const projectByPath = new Map<string, WorkshopMemoryProject>();
    for (const project of projects?.projects ?? []) {
      for (const root of project.workspaceRoots) projectByPath.set(root, project);
    }
    return runtime.workspaces.map((option) => ({
      grant: grantByPath.get(option.path) ?? null,
      option,
      project: projectByPath.get(option.path) ?? null,
    }));
  }, [grants, projects, runtime]);
  const selected = entries.find((entry) => entry.option.path === selectedPath) ?? entries[0] ?? null;

  const createPrivateWorkspace = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (!runtime || busy || !createName.trim()) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const created = await createWorkspace(session, createName.trim(), runtime.revision);
      const messages = [created.directoryCreated
        ? "Workspace created and selected."
        : "That workspace already existed and is now selected."];
      if (!created.gitReady) messages.push("Git initialization failed; the workspace is still usable.");
      if (!created.memoryProjectRegistered) messages.push(created.memoryProjectNote);
      setCreateOpen(false);
      setCreateName("");
      setNotice(messages.join(" "));
      await refresh(created.path);
    } catch (caught) {
      if (caught instanceof SettingsRevisionConflictError) {
        await refresh();
        setError("Workspace state changed elsewhere. The latest catalogue has been loaded.");
      } else if (!handleAccessFailure(caught)) {
        setError(errorText(caught, "Could not create workspace."));
      }
    } finally {
      setBusy(false);
    }
  };

  const addExisting = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const path = addPath.trim();
    if (!path || busy) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const changed = await addExistingWorkspace(session, path);
      setAddOpen(false);
      setAddPath("");
      setNotice(changed.mutation?.changed
        ? "Existing workspace access added."
        : "That workspace was already available.");
      await refresh(path);
    } catch (caught) {
      if (!handleAccessFailure(caught)) setError(errorText(caught, "Could not add workspace access."));
    } finally {
      setBusy(false);
    }
  };

  const removeAccess = async (): Promise<void> => {
    if (!selected?.grant?.removable || busy || !await confirm(
      `Remove access to ${selected.option.name}? The directory and its files will not be deleted.`,
    )) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const changed = await removeExistingWorkspace(session, selected.option.path);
      setNotice(changed.mutation?.changed
        ? `Access to ${selected.option.name} was removed. The directory was not deleted.`
        : `Access to ${selected.option.name} was already absent.`);
      await refresh();
    } catch (caught) {
      if (!handleAccessFailure(caught)) setError(errorText(caught, "Could not remove workspace access."));
    } finally {
      setBusy(false);
    }
  };

  const deletePermanently = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (!runtime || !selected?.option.deletable || deleteConfirmation !== selected.option.name || busy) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const deleted = await deleteWorkspace(
        session,
        selected.option.name,
        deleteConfirmation,
        runtime.revision,
      );
      setDeleteOpen(false);
      setDeleteConfirmation("");
      setNotice(deleted.directoryDeleted
        ? `Workspace ${selected.option.name} was permanently deleted.`
        : `Workspace ${selected.option.name} was already absent; its Kai records were cleaned.`);
      await refresh();
    } catch (caught) {
      if (caught instanceof SettingsRevisionConflictError) {
        await refresh();
        setError("Workspace state changed elsewhere. The latest catalogue has been loaded.");
      } else if (!handleAccessFailure(caught)) {
        setError(errorText(caught, "Could not delete workspace."));
      }
    } finally {
      setBusy(false);
    }
  };

  const registerProject = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (!projects || !selected || !projectName.trim() || busy) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const changed = await registerMemoryProject(
        session,
        projects.revision,
        projectName.trim(),
        selected.option.path,
      );
      setProjects(changed);
      setProjectName("");
      setNotice(changed.mutation?.note ?? "Memory project registered.");
    } catch (caught) {
      if (caught instanceof SettingsRevisionConflictError) {
        await refresh(selected.option.path);
        setError("Memory projects changed elsewhere. The latest registry has been loaded.");
      } else if (!handleAccessFailure(caught)) {
        setError(errorText(caught, "Could not register memory project."));
      }
    } finally {
      setBusy(false);
    }
  };

  const unregisterProject = async (): Promise<void> => {
    if (!projects || !selected?.project?.removable || busy || !await confirm(
      `Unregister memory project ${selected.project.projectId}? Existing memories will not be deleted.`,
    )) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const changed = await unregisterMemoryProject(
        session,
        projects.revision,
        selected.project.projectId,
      );
      setProjects(changed);
      setNotice(changed.mutation?.note ?? "Memory project unregistered.");
    } catch (caught) {
      if (caught instanceof SettingsRevisionConflictError) {
        await refresh(selected.option.path);
        setError("Memory projects changed elsewhere. The latest registry has been loaded.");
      } else if (!handleAccessFailure(caught)) {
        setError(errorText(caught, "Could not unregister memory project."));
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="workspaces-workspace" aria-label="Workspaces">
      <header className="workspaces-header">
        <div>
          <p className="overline">Workspace</p>
          <h1>Workspaces</h1>
          <p>Directories you have authorized Kai to use.</p>
        </div>
        <div className="workspaces-header-actions">
          <button className="panel-icon-button" type="button" aria-label="Create workspace" title="Create private workspace" disabled={busy} onClick={() => setCreateOpen(true)}><WorkspaceAddIcon /></button>
          <button className="panel-icon-button" type="button" aria-label="Add existing workspace" title="Add existing workspace" disabled={busy} onClick={() => setAddOpen(true)}><ExistingWorkspaceIcon /></button>
          <button className="panel-icon-button" type="button" aria-label="Back to conversation" title="Back to conversation" onClick={onClose}><span aria-hidden="true">←</span></button>
        </div>
      </header>

      {(notice || error) && <div className="workspaces-notices" aria-live="polite">
        {notice && <p className="settings-notice" role="status">{notice}</p>}
        {error && <p className="settings-error" role="alert">{error}</p>}
      </div>}

      <div className="workspaces-body">
        <section className="workspace-catalogue" aria-label="Authorized workspaces">
          {loading ? <p className="workspaces-empty">Loading workspaces…</p> : entries.length === 0 ? <p className="workspaces-empty">No authorized workspaces.</p> : (
            <ul>
              {entries.map((entry) => (
                <li key={entry.option.path}>
                  <button className={entry.option.path === selected?.option.path ? "selected" : ""} type="button" onClick={() => setSelectedPath(entry.option.path)}>
                    <span><strong>{entry.option.name}</strong>{entry.option.current && <span className="workspace-current">Current</span>}</span>
                    <small>{entry.option.path}</small>
                    <small>{grantProvenance(entry.grant?.provenance ?? null, entry.option.home)}{entry.grant && !entry.grant.available ? " · unavailable" : ""}</small>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <aside className="workspace-detail" aria-live="polite">
          {selected ? (
            <>
              <div className="workspace-detail-heading">
                <div><p className="overline">Selected workspace</p><h2>{selected.option.name}</h2></div>
                {selected.option.current && <span className="workspace-status">Current</span>}
              </div>
              <dl>
                <div><dt>Path</dt><dd><code>{selected.option.path}</code></dd></div>
                <div><dt>Access</dt><dd>{grantProvenance(selected.grant?.provenance ?? null, selected.option.home)}</dd></div>
                <div><dt>Availability</dt><dd>{selected.grant?.available === false ? "Unavailable" : "Available"}</dd></div>
              </dl>

              <section className="workspace-project-detail">
                <h3>Memory project</h3>
                {selected.project ? (
                  <>
                    <p><strong>{selected.project.displayName}</strong></p>
                    <p>{projectProvenance(selected.project)}{!selected.project.available ? " · unavailable" : ""}</p>
                    {selected.project.removable && <button className="quiet-button" type="button" disabled={busy} onClick={() => void unregisterProject()}>Unregister project</button>}
                  </>
                ) : (
                  <form className="memory-project-registration" onSubmit={(event) => void registerProject(event)}>
                    <p>Not registered as a memory project.</p>
                    <label htmlFor="workspace-project-name">Project name</label>
                    <div><input id="workspace-project-name" type="text" maxLength={64} pattern="[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}" placeholder={selected.option.name.toLowerCase().replaceAll(/[^a-z0-9_-]/g, "-")} value={projectName} disabled={busy} onChange={(event) => setProjectName(event.target.value)} /><button className="panel-icon-button" type="submit" aria-label="Register memory project" title="Register memory project" disabled={busy || !projectName.trim()}><WorkspaceAddIcon /></button></div>
                  </form>
                )}
              </section>

              {(selected.grant?.removable || selected.option.deletable) && <section className="workspace-detail-actions">
                <h3>Workspace actions</h3>
                {selected.grant?.removable && <button className="quiet-button" type="button" disabled={busy || selected.option.current} title={selected.option.current ? "Switch away from this workspace before removing access" : undefined} onClick={() => void removeAccess()}>Remove access</button>}
                {selected.option.deletable && <button className="danger-button" type="button" disabled={busy} onClick={() => { setDeleteConfirmation(""); setDeleteOpen(true); }}>Delete permanently</button>}
              </section>}
            </>
          ) : <p className="workspaces-empty">Select a workspace.</p>}
        </aside>
      </div>

      {createOpen && runtime && <div className="modal-backdrop" role="presentation"><section className="channel-creation-dialog workspace-creation-dialog" role="dialog" aria-modal="true" aria-labelledby="create-workspace-title"><header className="workspace-creation-header"><div><p className="overline">Private workspace</p><h2 id="create-workspace-title">Create workspace</h2></div><button className="panel-icon-button" type="button" aria-label="Close workspace creation" title="Close workspace creation" disabled={busy} onClick={() => { setCreateOpen(false); setCreateName(""); }}><span aria-hidden="true">×</span></button></header><p>This creates a private directory inside your Kai home.</p><form onSubmit={(event) => void createPrivateWorkspace(event)}><label htmlFor="workspace-name">Workspace name</label><input id="workspace-name" type="text" autoFocus maxLength={64} value={createName} disabled={busy} onChange={(event) => setCreateName(event.target.value)} /><div className="form-actions"><button className="primary-button" type="submit" disabled={busy || !createName.trim()}>{busy ? "Creating…" : "Create workspace"}</button></div></form></section></div>}

      {addOpen && <div className="modal-backdrop" role="presentation"><section className="channel-creation-dialog workspace-creation-dialog" role="dialog" aria-modal="true" aria-labelledby="add-existing-workspace-title"><header className="workspace-creation-header"><div><p className="overline">Existing directory</p><h2 id="add-existing-workspace-title">Add existing workspace</h2></div><button className="panel-icon-button" type="button" aria-label="Close existing workspace" title="Close existing workspace" disabled={busy} onClick={() => { setAddOpen(false); setAddPath(""); }}><span aria-hidden="true">×</span></button></header><p>This grants Kai access to a directory that already exists. It does not create, move, or delete files.</p><form onSubmit={(event) => void addExisting(event)}><label htmlFor="workspace-existing-path">Absolute directory path</label><input id="workspace-existing-path" type="text" autoFocus value={addPath} disabled={busy} placeholder="/Users/you/projects/example" autoCapitalize="none" autoCorrect="off" onChange={(event) => setAddPath(event.target.value)} /><div className="form-actions"><button className="primary-button" type="submit" disabled={busy || !addPath.trim()}>{busy ? "Adding…" : "Add workspace access"}</button></div></form></section></div>}

      {deleteOpen && selected && <div className="modal-backdrop" role="presentation"><section className="channel-creation-dialog workspace-creation-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-workspace-title"><header className="workspace-creation-header"><div><p className="overline">Permanent deletion</p><h2 id="delete-workspace-title">Delete workspace</h2></div><button className="panel-icon-button" type="button" aria-label="Close workspace deletion" title="Close workspace deletion" disabled={busy} onClick={() => { setDeleteOpen(false); setDeleteConfirmation(""); }}><span aria-hidden="true">×</span></button></header><p>This permanently deletes <strong>{selected.option.name}</strong> and removes it from Kai.</p><form onSubmit={(event) => void deletePermanently(event)}><label htmlFor="workspace-delete-confirmation">Type <strong>{selected.option.name}</strong> to confirm</label><input id="workspace-delete-confirmation" type="text" autoFocus value={deleteConfirmation} disabled={busy} onChange={(event) => setDeleteConfirmation(event.target.value)} /><div className="form-actions"><button className="danger-button" type="submit" disabled={busy || deleteConfirmation !== selected.option.name}>{busy ? "Deleting…" : "Delete permanently"}</button></div></form></section></div>}
    </main>
  );
}
