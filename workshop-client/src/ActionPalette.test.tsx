import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ActionPalette, type WorkshopActionPaletteEntry } from "./ActionPalette";

function action(
  operationId: string,
  label: string,
  targetLabel: string | null = null,
): WorkshopActionPaletteEntry {
  return {
    capability: {
      available: true,
      confirmation: "none",
      description: `Description for ${label}`,
      disposition: "native_surface",
      inputShape: "none",
      label,
      mutatesState: false,
      operationId,
      paletteEntry: true,
      scope: targetLabel ? "principal_agent" : "principal",
      surface: "action_palette",
      unavailableReason: null,
    },
    key: `${operationId}-${targetLabel ?? "global"}`,
    targetAgentId: targetLabel ? "agt_00000000000000000000000000000001" : null,
    targetLabel,
  };
}

describe("Action palette", () => {
  it("filters registry-backed entries and identifies the target agent", async () => {
    const user = userEvent.setup();
    const close = vi.fn();
    render(<ActionPalette
      actions={[
        action("runtime.status.read", "Show runtime status", "Kai"),
        action("memory.manage", "Open memory"),
      ]}
      error={null}
      loading={false}
      onClose={close}
      onInvoke={vi.fn().mockResolvedValue(null)}
    />);

    expect(screen.getByRole("dialog", { name: "Help and actions" })).toBeVisible();
    expect(screen.getByRole("option", { name: /Show runtime status/ })).toHaveTextContent("Kai");
    await user.type(screen.getByRole("combobox", { name: "Search actions" }), "memory");
    expect(screen.getByRole("option", { name: /Open memory/ })).toBeVisible();
    expect(screen.queryByRole("option", { name: /runtime status/ })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(close).toHaveBeenCalledOnce();
  });

  it("supports keyboard selection, results, and dismissal", async () => {
    const user = userEvent.setup();
    const close = vi.fn();
    const invoke = vi.fn()
      .mockResolvedValueOnce("Agent: Kai\nProcess: alive")
      .mockResolvedValueOnce(null);
    render(<ActionPalette
      actions={[
        action("memory.manage", "Open memory"),
        action("runtime.status.read", "Show runtime status", "Kai"),
      ]}
      error={null}
      loading={false}
      onClose={close}
      onInvoke={invoke}
    />);

    const search = screen.getByRole("combobox", { name: "Search actions" });
    await user.type(search, "{ArrowDown}{Enter}");
    expect(await screen.findByText(/Agent: Kai/)).toBeVisible();
    expect(invoke).toHaveBeenCalledWith(expect.objectContaining({
      targetLabel: "Kai",
    }));
    await user.clear(search);
    await user.type(search, "memory{Enter}");
    expect(close).toHaveBeenCalledOnce();
  });
});
