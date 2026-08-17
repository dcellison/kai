import { fireEvent, render, screen } from "@testing-library/react";
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
        failed={false}
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
        failed={false}
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
    // Every diff line, tinted or not, is a block-level trace-line so the
    // add/del backgrounds span the full row.
    expect(container.querySelectorAll(".trace-line")).toHaveLength(3);
  });

  it("pretty-prints JSON details and copies the rendered text", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <RunTraceCard
        failed={false}
        loaded
        runId="run_1"
        entries={[
          entry({
            seq: 1,
            kind: "tool_call",
            toolName: "Bash",
            toolUseId: "t1",
            summary: "Bash: ls",
            detail: '{"command":"ls","cwd":"/w"}',
          }),
        ]}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Bash: ls/ }));
    const pretty = '{\n  "command": "ls",\n  "cwd": "/w"\n}';
    expect(container.querySelector(".trace-detail")?.textContent).toBe(pretty);

    const copyButton = screen.getByRole("button", { name: "Copy detail" });
    expect(copyButton.textContent).not.toBe("✓");
    await user.click(copyButton);
    expect(copyButton.textContent).toBe("✓");
    await expect(window.navigator.clipboard.readText()).resolves.toBe(pretty);
  });

  it("renders a detail that fails to parse as JSON exactly as stored", async () => {
    const user = userEvent.setup();
    // Source-side truncation can cut a payload mid-JSON; it must render raw.
    const truncated = '{"command":"ls","cwd":"/w';
    const { container } = render(
      <RunTraceCard
        failed={false}
        loaded
        runId="run_1"
        entries={[
          entry({
            seq: 1,
            kind: "tool_call",
            toolName: "Bash",
            toolUseId: "t1",
            summary: "Bash: ls",
            detail: truncated,
          }),
        ]}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Bash: ls/ }));
    expect(container.querySelector(".trace-detail")?.textContent).toBe(truncated);
  });

  it("omits the copy button when the clipboard API is unavailable", () => {
    // userEvent.setup() installs a clipboard stub on the shared jsdom
    // navigator, so this test forces the property absent and clicks via
    // fireEvent; calling setup() here would put the stub right back.
    const original = Object.getOwnPropertyDescriptor(window.navigator, "clipboard");
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: undefined,
    });
    try {
      render(
        <RunTraceCard
          failed={false}
          loaded
          runId="run_1"
          entries={[
            entry({
              seq: 1,
              kind: "tool_call",
              toolName: "Bash",
              toolUseId: "t1",
              summary: "Bash: ls",
              detail: '{"command":"ls"}',
            }),
          ]}
        />,
      );
      fireEvent.click(screen.getByRole("button", { name: /Bash: ls/ }));
      // The detail block itself still renders; only the button is gone.
      expect(document.querySelector(".trace-detail")).not.toBeNull();
      expect(screen.queryByRole("button", { name: "Copy detail" })).toBeNull();
    } finally {
      if (original) {
        Object.defineProperty(window.navigator, "clipboard", original);
      } else {
        Reflect.deleteProperty(window.navigator, "clipboard");
      }
    }
  });

  it("shows honest empty states and the truncation marker as its own row", () => {
    const { rerender } = render(<RunTraceCard failed={false} loaded runId="run_1" entries={[]} />);
    expect(screen.getByText("No steps recorded for this run.")).toBeVisible();

    rerender(<RunTraceCard failed={false} loaded runId={null} entries={[]} />);
    expect(screen.getByText("No runs yet in this channel.")).toBeVisible();

    rerender(<RunTraceCard failed loaded={false} runId="run_1" entries={[]} />);
    expect(screen.getByText("Trace unavailable for this run.")).toBeVisible();

    rerender(
      <RunTraceCard
        failed={false}
        loaded
        runId="run_1"
        entries={[entry({ seq: 1, kind: "truncated", summary: "trace truncated at 500 steps" })]}
      />,
    );
    expect(screen.getByText("trace truncated at 500 steps")).toBeVisible();
    expect(screen.queryByRole("button")).toBeNull();
  });
});
