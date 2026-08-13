import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MarkdownMessage } from "./MarkdownMessage";

describe("Workshop Markdown messages", () => {
  it("renders GitHub-flavored structure and secure external links", () => {
    render(
      <MarkdownMessage
        body={[
          "## Result",
          "",
          "**Complete** with `inline code`.",
          "",
          "- [x] Verified",
          "- [ ] Follow-up",
          "",
          "| Item | State |",
          "| --- | --- |",
          "| Build | Green |",
          "",
          "[Open the PR](https://github.com/dcellison/kai/pull/910)",
        ].join("\n")}
      />,
    );

    expect(screen.getByRole("heading", { name: "Result" })).toBeVisible();
    expect(screen.getByText("Complete").tagName).toBe("STRONG");
    expect(screen.getByText("inline code").tagName).toBe("CODE");
    expect(screen.getAllByRole("checkbox")).toHaveLength(2);
    expect(screen.getAllByRole("checkbox")[0]).toBeDisabled();
    expect(screen.getByRole("table")).toHaveTextContent("BuildGreen");
    expect(screen.getByRole("link", { name: "Open the PR" })).toHaveAttribute(
      "target",
      "_blank",
    );
    expect(screen.getByRole("link", { name: "Open the PR" })).toHaveAttribute(
      "rel",
      "noopener noreferrer",
    );
  });

  it("keeps embedded HTML inert and never fetches Markdown images", () => {
    const { container } = render(
      <MarkdownMessage
        body={[
          '<img src=x onerror="alert(1)">',
          "",
          "![tracking pixel](https://example.invalid/pixel.png)",
          "",
          "[unsafe](javascript:alert(1))",
        ].join("\n")}
      />,
    );

    expect(screen.getByText('<img src=x onerror="alert(1)">')).toBeVisible();
    expect(screen.getByText("[Image: tracking pixel]")).toBeVisible();
    expect(container.querySelector("img")).toBeNull();
    expect(screen.queryByRole("link", { name: "unsafe" })).toBeNull();
    expect(container.querySelector("a")).toBeNull();
    expect(screen.getByText("unsafe")).toHaveClass("markdown-unsafe-link");
  });
});
