import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { loadReviewJobs, submitReviewJob } from "./api";
import { ReviewDialog } from "./ReviewDialog";

vi.mock("./api", async (importOriginal) => ({
  ...await importOriginal<typeof import("./api")>(),
  loadReviewJobs: vi.fn(),
  submitReviewJob: vi.fn(),
}));

const session = { channelId: "chn_" + "1".repeat(32), token: "secret" };

describe("ReviewDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadReviewJobs).mockResolvedValue({
      jobs: [],
      submission: {
        activeWorkspace: "/Users/daniel/work/kai",
        activeWorkspaceRepository: "dcellison/kai",
        inferredRepository: null,
        repositories: ["dcellison/kai", "dcellison/anvil"],
      },
    });
    vi.mocked(submitReviewJob).mockResolvedValue({
      artifact: null,
      attemptCount: 0,
      createdAt: "2026-09-20T12:00:00Z",
      lastErrorCode: null,
      pullRequestNumber: 42,
      repository: "dcellison/kai",
      replayed: false,
      reviewJobId: "rvj_" + "2".repeat(32),
      status: "pending",
      updatedAt: "2026-09-20T12:00:00Z",
    });
  });

  it("requires an explicit repository when inference is ambiguous and submits canonically", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onSubmitted = vi.fn();
    render(
      <ReviewDialog
        onAuthenticationFailure={vi.fn()}
        onChannelAccessFailure={vi.fn()}
        onClose={onClose}
        onSubmitted={onSubmitted}
        session={session}
      />,
    );

    expect(await screen.findByText("/Users/daniel/work/kai")).toBeVisible();
    await user.type(screen.getByLabelText("Pull request"), "42");
    expect(screen.getByRole("button", { name: "Start review" })).toBeDisabled();
    await user.selectOptions(screen.getByLabelText("Authorized repository"), "dcellison/kai");
    await user.click(screen.getByRole("button", { name: "Start review" }));

    await waitFor(() => expect(submitReviewJob).toHaveBeenCalledWith(
      session,
      "dcellison/kai",
      42,
      expect.stringMatching(/^workshop-review:/),
    ));
    expect(onSubmitted).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("accepts a GitHub pull-request URL only when its repository is authorized", async () => {
    const user = userEvent.setup();
    render(
      <ReviewDialog
        onAuthenticationFailure={vi.fn()}
        onChannelAccessFailure={vi.fn()}
        onClose={vi.fn()}
        onSubmitted={vi.fn()}
        session={session}
      />,
    );
    await screen.findByText("/Users/daniel/work/kai");
    await user.type(screen.getByLabelText("Pull request"), "https://github.com/unknown/repo/pull/9");
    expect(screen.getByText("That repository is not authorized for review.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Start review" })).toBeDisabled();
  });
});
