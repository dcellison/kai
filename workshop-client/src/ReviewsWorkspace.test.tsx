import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { loadReviewArtifact, loadReviewJobs } from "./api";
import { ConfirmationProvider } from "./ConfirmationDialog";
import { ReviewsWorkspace } from "./ReviewsWorkspace";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return { ...original, loadReviewArtifact: vi.fn(), loadReviewJobs: vi.fn() };
});

const session = {
  channelId: "chn_d3dfdfd7df9151ba8a1742b92403faa5",
  token: "session-secret",
};

describe("ReviewsWorkspace", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

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

  it("opens the artifact window synchronously before loading its contents", async () => {
    const user = userEvent.setup();
    let resolveArtifact!: (artifact: Blob) => void;
    vi.mocked(loadReviewArtifact).mockReturnValue(new Promise((resolve) => {
      resolveArtifact = resolve;
    }));
    const popup = {
      close: vi.fn(),
      location: { href: "about:blank" },
      opener: window,
    } as unknown as Window;
    const open = vi.spyOn(window, "open").mockReturnValue(popup);
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:review-artifact"),
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });
    render(
      <ConfirmationProvider>
        <ReviewsWorkspace
          connection={{ label: "Live", tone: "connected" }}
          onAuthenticationFailure={vi.fn()}
          onChannelAccessFailure={vi.fn()}
          onReviewPullRequest={vi.fn()}
          revision={0}
          session={session}
          workshopName="Kai Workshop"
        />
      </ConfirmationProvider>,
    );

    await user.click(await screen.findByRole("button", { name: "Open" }));
    expect(open).toHaveBeenCalledWith("about:blank", "_blank");
    expect(loadReviewArtifact).toHaveBeenCalledTimes(1);
    expect(popup.location.href).toBe("about:blank");

    resolveArtifact(new Blob(["review"]));
    await waitFor(() => expect(popup.location.href).toBe("blob:review-artifact"));
    expect(popup.opener).toBeNull();
  });
});
