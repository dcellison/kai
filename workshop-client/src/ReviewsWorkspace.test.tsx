import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { loadReviewJobs } from "./api";
import { ConfirmationProvider } from "./ConfirmationDialog";
import { ReviewsWorkspace } from "./ReviewsWorkspace";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return { ...original, loadReviewJobs: vi.fn() };
});

const session = {
  channelId: "chn_d3dfdfd7df9151ba8a1742b92403faa5",
  token: "session-secret",
};

describe("ReviewsWorkspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadReviewJobs).mockResolvedValue({
      jobs: [
        {
          artifact: {
            artifactId: "rva_00000000000000000000000000000001",
            filename: "review.md",
            warningCount: 1,
            warnings: [{ message: "One optional source was unavailable.", source: "issue_context" }],
          },
          attemptCount: 1,
          createdAt: "2026-09-20T12:00:00Z",
          lastErrorCode: null,
          pullRequestNumber: 1695,
          repository: "dcellison/kai",
          replayed: false,
          reviewJobId: "rvj_00000000000000000000000000000001",
          status: "succeeded",
          updatedAt: "2026-09-20T12:05:00Z",
        },
      ],
      submission: {
        activeWorkspace: "/srv/kai",
        activeWorkspaceRepository: "dcellison/kai",
        inferredRepository: "dcellison/kai",
        repositories: ["dcellison/kai"],
      },
    });
  });

  it("presents durable reviews as Activity and exposes the launch action", async () => {
    const user = userEvent.setup();
    const onReviewPullRequest = vi.fn();
    render(
      <ConfirmationProvider>
        <ReviewsWorkspace
          connection={{ label: "Live", tone: "connected" }}
          onAuthenticationFailure={vi.fn()}
          onChannelAccessFailure={vi.fn()}
          onReviewPullRequest={onReviewPullRequest}
          revision={0}
          session={session}
          workshopName="Kai Workshop"
        />
      </ConfirmationProvider>,
    );

    expect(await screen.findByRole("heading", { name: "Reviews" })).toBeVisible();
    expect(screen.getByText("Kai Workshop / Activity")).toBeVisible();
    expect(screen.getByText("dcellison/kai#1695")).toBeVisible();
    expect(screen.getByText(/^Completed ·/)).toBeVisible();
    expect(screen.getByText("1 collection warning")).toBeVisible();
    expect(screen.getByRole("button", { name: "Open" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Download" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Review pull request" }));
    expect(onReviewPullRequest).toHaveBeenCalledTimes(1);
  });
});
