import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { RunTraceCard } from "./RunTraceCard";
import type { WorkshopRunTraceEntry } from "./types";

function entry(overrides: Partial<WorkshopRunTraceEntry>): WorkshopRunTraceEntry {
  return {
    createdAt: "2026-08-17T18:00:00+00:00",
    detail: "",
    isDiff: false,
    isError: false,
    kind: "tool_call",
    seq: 1,
    summary: "",
    toolName: null,
    toolUseId: null,
    ...overrides,
  };
}

describe("RunTraceCard", () => {
  it("renders collapsed rows that expand on click, folding results into calls", async () => {
    const user = userEvent.setup();
    render(
      <RunTraceCard
        loaded
        runId="run_1"
        entries={[
          entry({
            seq: 1,
            kind: "tool_call",
            toolName: "Bash",
            toolUseId: "t1",
            summary: "Bash: ls",
            detail: '{"command": "ls"}',
          }),
          entry({
            seq: 2,
            kind: "tool_result",
            toolUseId: "t1",
            summary: "total 48",
            detail: "total 48\nREADME.md",
          }),
          entry({
            seq: 3,
            kind: "tool_result",
            toolUseId: "t9",
            summary: "boom",
            detail: "stack trace",
            isError: true,
          }),
        ]}
      />,
    );

    const rows = screen.getAllByRole("listitem");
    expect(rows).toHaveLength(2);
    expect(screen.getByText("Bash: ls")).toBeVisible();
    expect(screen.getByText("done")).toBeVisible();
    expect(screen.queryByText(/README\.md/)).toBeNull();
    expect(rows[1].className).toContain("trace-error");
    expect(screen.getByText("boom")).toBeVisible();

    await user.click(screen.getByRole("button", { name: /Bash: ls/ }));
    expect(screen.getByText(/README\.md/)).toBeVisible();
    await user.click(screen.getByRole("button", { name: /Bash: ls/ }));
    expect(screen.queryByText(/README\.md/)).toBeNull();
  });

  it("renders diff rows expanded by default with line-prefix classes", () => {
    const { container } = render(
      <RunTraceCard
        loaded
        runId="run_1"
        entries={[
          entry({
            seq: 1,
            kind: "tool_call",
            toolName: "edit",
            toolUseId: "t1",
            summary: "Edit config.py",
            isDiff: true,
            detail: "/w/config.py\n- a = 1\n+ a = 2",
          }),
        ]}
      />,
    );

    const added = container.querySelectorAll(".trace-diff-add");
    const removed = container.querySelectorAll(".trace-diff-del");
    expect(added).toHaveLength(1);
    expect(removed).toHaveLength(1);
    expect(added[0].textContent).toContain("+ a = 2");
    expect(removed[0].textContent).toContain("- a = 1");
  });

  it("shows honest empty states and the truncation marker as its own row", () => {
    const { rerender } = render(<RunTraceCard loaded runId="run_1" entries={[]} />);
    expect(screen.getByText("No steps recorded for this run.")).toBeVisible();

    rerender(<RunTraceCard loaded runId={null} entries={[]} />);
    expect(screen.getByText("No runs yet in this channel.")).toBeVisible();

    rerender(
      <RunTraceCard
        loaded
        runId="run_1"
        entries={[entry({ seq: 1, kind: "truncated", summary: "trace truncated at 500 steps" })]}
      />,
    );
    expect(screen.getByText("trace truncated at 500 steps")).toBeVisible();
    expect(screen.queryByRole("button")).toBeNull();
  });
});
