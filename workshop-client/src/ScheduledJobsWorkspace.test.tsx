import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { cancelScheduledJob, loadScheduledJobs } from "./api";
import { ConfirmationProvider } from "./ConfirmationDialog";
import { ScheduledJobsWorkspace } from "./ScheduledJobsWorkspace";
import type { WorkshopScheduledJob } from "./types";
import type { WorkshopPrincipalEvents } from "./usePrincipalEvents";

vi.mock("./api", async (importOriginal) => ({
  ...await importOriginal<typeof import("./api")>(),
  cancelScheduledJob: vi.fn(),
  loadScheduledJobs: vi.fn(),
}));

const session = {
  channelId: "chn_d3dfdfd7df9151ba8a1742b92403faa5",
  token: "session-secret",
};
const job: WorkshopScheduledJob = {
  active: true,
  agentName: "Kai",
  autoRemove: false,
  channelName: "Kai",
  createdAt: "2026-09-18T12:00:00Z",
  deliveryTransports: ["telegram", "workshop_client"],
  id: 7,
  jobType: "agent",
  name: "Daily status",
  nextRunAt: "2026-09-19T13:00:00Z",
  notifyOnCheck: false,
  prompt: "Report project status.",
  scheduleData: '{"times":["13:00"]}',
  scheduleType: "daily",
};

function renderJobs(): void {
  const principalEvents: WorkshopPrincipalEvents = {
    connection: { label: "Live", tone: "connected" },
    subscribe: vi.fn(() => () => undefined),
  };
  render(<ConfirmationProvider><ScheduledJobsWorkspace
    onAuthenticationFailure={vi.fn()}
    onClose={vi.fn()}
    principalEvents={principalEvents}
    session={session}
  /></ConfirmationProvider>);
}

describe("Scheduled jobs workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadScheduledJobs).mockResolvedValue([job]);
    vi.mocked(cancelScheduledJob).mockResolvedValue({
      cancelled: true,
      changed: true,
      eventPosition: 90,
      jobId: 7,
      replayed: false,
    });
  });

  it("shows the active collection and principal-safe job detail", async () => {
    renderJobs();
    expect(await screen.findByRole("heading", { name: "Scheduled jobs", level: 1 })).toBeVisible();
    expect(screen.getByRole("button", { name: /Daily status/ })).toBeVisible();
    expect(screen.getByText("Report project status.")).toBeVisible();
    expect(screen.getByText(/Kai · Telegram, Workshop/)).toBeVisible();
    expect(screen.queryByRole("button", { name: /create/i })).not.toBeInTheDocument();
  });

  it("confirms cancellation and keeps history wording explicit", async () => {
    const user = userEvent.setup();
    vi.mocked(loadScheduledJobs)
      .mockResolvedValueOnce([job])
      .mockResolvedValue([]);
    renderJobs();
    await user.click(await screen.findByRole("button", { name: "Cancel scheduled job" }));
    const confirmation = screen.getByRole("dialog", { name: "Continue?" });
    expect(confirmation).toHaveTextContent("existing conversation and delivery history will remain");
    await user.click(within(confirmation).getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(cancelScheduledJob).toHaveBeenCalledWith(
      session,
      7,
      expect.stringMatching(/^workshop-job-cancel-7-/),
    ));
    expect(await screen.findByText("No active scheduled jobs")).toBeVisible();
  });
});
