import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
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
import { ConfirmationProvider } from "./ConfirmationDialog";
import type {
  WorkshopMemoryProjectRegistry,
  WorkshopSettingsWorkspace,
  WorkshopWorkspaceGrants,
} from "./types";
import { WorkspacesWorkspace } from "./WorkspacesWorkspace";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    addExistingWorkspace: vi.fn(),
    createWorkspace: vi.fn(),
    deleteWorkspace: vi.fn(),
    loadMemoryProjects: vi.fn(),
    loadSettingsWorkspace: vi.fn(),
    loadWorkspaceGrants: vi.fn(),
    registerMemoryProject: vi.fn(),
    removeExistingWorkspace: vi.fn(),
    unregisterMemoryProject: vi.fn(),
  };
});

const session = {
  channelId: "chn_d3dfdfd7df9151ba8a1742b92403faa5",
  token: "session-secret",
};

const runtime: WorkshopSettingsWorkspace = {
  backend: "codex",
  backendOptionId: "codex:openai",
  backendOptions: [{ backend: "codex", current: true, optionId: "codex:openai", provider: "openai" }],
  capabilities: [],
  channelId: session.channelId,
  model: { defaultValue: "gpt-5.6-sol", source: "runtime policy", value: "gpt-5.6-sol" },
  modelCatalogue: null,
  modelOptions: null,
  mutation: null,
  principalId: "prn_00000000000000000000000000000001",
  provider: "openai",
  revision: "sws_current",
  runtimeProfileId: "rtp_00000000000000000000000000000001",
  timeoutSeconds: { defaultValue: 1800, source: "runtime policy", value: 1800 },
  workspace: "/srv/kai",
  workspaces: [
    { current: true, home: false, name: "Kai", path: "/srv/kai" },
    { current: false, home: true, name: "Home", path: "/srv/home" },
    {
      current: false,
      deletable: true,
      home: false,
      name: "qualification-1520",
      path: "/srv/home/workspaces/qualification-1520",
    },
  ],
};

const grants: WorkshopWorkspaceGrants = {
  grants: [
    { available: true, current: true, name: "Kai", path: "/srv/kai", provenance: "profile", removable: false },
    { available: true, current: false, name: "Home", path: "/srv/home", provenance: "profile", removable: false },
    {
      available: true,
      current: false,
      name: "qualification-1520",
      path: "/srv/home/workspaces/qualification-1520",
      provenance: "principal_created",
      removable: true,
    },
  ],
  mutation: null,
  principalId: runtime.principalId,
  runtimeProfileId: runtime.runtimeProfileId,
  workspaceBase: "/srv/home/workspaces",
};

const projects: WorkshopMemoryProjectRegistry = {
  activeProjectId: "kai",
  currentWorkspace: "/srv/kai",
  mutation: null,
  principalId: runtime.principalId,
  projects: [
    {
      available: true,
      current: true,
      displayName: "Kai",
      projectId: "kai",
      provenance: "operator_pinned",
      removable: false,
      stateVersion: null,
      workspaceRoots: ["/srv/kai"],
    },
  ],
  revision: "mpr_current",
  runtimeProfileId: runtime.runtimeProfileId,
};

function renderWorkspace(): void {
  render(
    <ConfirmationProvider>
      <WorkspacesWorkspace
        onAuthenticationFailure={vi.fn()}
        onChannelAccessFailure={vi.fn()}
        onClose={vi.fn()}
        session={session}
      />
    </ConfirmationProvider>,
  );
}

describe("Workspaces workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadSettingsWorkspace).mockResolvedValue(runtime);
    vi.mocked(loadWorkspaceGrants).mockResolvedValue(grants);
    vi.mocked(loadMemoryProjects).mockResolvedValue(projects);
    vi.mocked(registerMemoryProject).mockResolvedValue({
      ...projects,
      activeProjectId: null,
      mutation: { changed: true, note: "Registered memory project 'qualification'.", projectId: "qualification" },
      projects: [
        ...projects.projects,
        {
          available: true,
          current: false,
          displayName: "qualification",
          projectId: "qualification",
          provenance: "principal_registered",
          removable: true,
          stateVersion: 0,
          workspaceRoots: ["/srv/home/workspaces/qualification-1520"],
        },
      ],
    });
    vi.mocked(unregisterMemoryProject).mockResolvedValue(projects);
    vi.mocked(addExistingWorkspace).mockResolvedValue(grants);
    vi.mocked(removeExistingWorkspace).mockResolvedValue({
      ...grants,
      grants: grants.grants.filter((item) => item.path !== "/srv/home/workspaces/qualification-1520"),
      mutation: { changed: true, path: "/srv/home/workspaces/qualification-1520" },
    });
    vi.mocked(createWorkspace).mockResolvedValue({
      directoryCreated: true,
      gitReady: true,
      memoryProjectNote: "Registered.",
      memoryProjectRegistered: true,
      path: "/srv/home/workspaces/research",
      settings: runtime,
    });
    vi.mocked(deleteWorkspace).mockResolvedValue({
      directoryDeleted: true,
      memoryProjectUnregistered: null,
      path: "/srv/home/workspaces/qualification-1520",
      settings: runtime,
    });
  });

  it("presents the principal catalogue and selected workspace detail", async () => {
    const user = userEvent.setup();
    renderWorkspace();

    expect(await screen.findByRole("heading", { name: "Workspaces", level: 1 })).toBeVisible();
    expect(screen.getByText("Kai Workshop / Workspaces")).toBeVisible();
    expect(screen.queryByText("Workspace", { selector: ".overline" })).not.toBeInTheDocument();
    expect(screen.getAllByText("/srv/kai")[0].closest("button")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /qualification-1520/ }));
    expect(screen.getByRole("heading", { name: "qualification-1520", level: 2 })).toBeVisible();
    expect(screen.getAllByText("Created by you")).toHaveLength(2);
    expect(screen.getByText("Not registered as a memory project.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Remove access" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Delete permanently" })).toBeVisible();
  });

  it("registers the selected workspace rather than the active workspace", async () => {
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(await screen.findByRole("button", { name: /qualification-1520/ }));
    await user.type(screen.getByLabelText("Project name"), "qualification");
    await user.click(screen.getByRole("button", { name: "Register memory project" }));

    await waitFor(() => expect(registerMemoryProject).toHaveBeenCalledWith(
      session,
      "mpr_current",
      "qualification",
      "/srv/home/workspaces/qualification-1520",
    ));
  });

  it("creates a private workspace and adds access to an existing directory", async () => {
    const user = userEvent.setup();
    renderWorkspace();

    await user.click(await screen.findByRole("button", { name: "Create workspace" }));
    const createDialog = screen.getByRole("dialog", { name: "Create workspace" });
    await user.type(within(createDialog).getByLabelText("Workspace name"), "research");
    await user.click(within(createDialog).getByRole("button", { name: "Create workspace" }));
    await waitFor(() => expect(createWorkspace).toHaveBeenCalledWith(
      session,
      "research",
      "sws_current",
    ));

    await user.click(screen.getByRole("button", { name: "Add existing workspace" }));
    const addDialog = screen.getByRole("dialog", { name: "Add existing workspace" });
    expect(addDialog).toHaveTextContent("It does not create, move, or delete files.");
    await user.type(
      within(addDialog).getByLabelText("Absolute directory path"),
      "/srv/existing",
    );
    await user.click(within(addDialog).getByRole("button", { name: "Add workspace access" }));
    await waitFor(() => expect(addExistingWorkspace).toHaveBeenCalledWith(
      session,
      "/srv/existing",
    ));
  });

  it("requires the exact workspace name before permanent deletion", async () => {
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(await screen.findByRole("button", { name: /qualification-1520/ }));
    await user.click(screen.getByRole("button", { name: "Delete permanently" }));

    const dialog = screen.getByRole("dialog", { name: "Delete workspace" });
    const submit = within(dialog).getByRole("button", { name: "Delete permanently" });
    expect(submit).toBeDisabled();
    await user.type(
      within(dialog).getByLabelText(/Type qualification-1520 to confirm/),
      "qualification-1520",
    );
    await user.click(submit);
    await waitFor(() => expect(deleteWorkspace).toHaveBeenCalledWith(
      session,
      "qualification-1520",
      "qualification-1520",
      "sws_current",
    ));
  });

  it("unregisters a removable memory project without deleting memories", async () => {
    const user = userEvent.setup();
    vi.mocked(loadMemoryProjects).mockResolvedValue({
      ...projects,
      projects: [
        ...projects.projects,
        {
          available: true,
          current: false,
          displayName: "qualification",
          projectId: "qualification",
          provenance: "principal_registered",
          removable: true,
          stateVersion: 0,
          workspaceRoots: ["/srv/home/workspaces/qualification-1520"],
        },
      ],
    });
    renderWorkspace();
    await user.click(await screen.findByRole("button", { name: /qualification-1520/ }));
    await user.click(screen.getByRole("button", { name: "Unregister project" }));
    const confirmation = screen.getByRole("dialog", { name: "Continue?" });
    expect(confirmation).toHaveTextContent("Existing memories will not be deleted.");
    await user.click(within(confirmation).getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(unregisterMemoryProject).toHaveBeenCalledWith(
      session,
      "mpr_current",
      "qualification",
    ));
  });

  it("keeps access removal distinct from permanent deletion", async () => {
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(await screen.findByRole("button", { name: /qualification-1520/ }));

    await user.click(screen.getByRole("button", { name: "Remove access" }));
    const confirmation = screen.getByRole("dialog", { name: "Continue?" });
    expect(confirmation).toHaveTextContent("The directory and its files will not be deleted.");
    await user.click(within(confirmation).getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(removeExistingWorkspace).toHaveBeenCalledWith(
      session,
      "/srv/home/workspaces/qualification-1520",
    ));
  });
});
