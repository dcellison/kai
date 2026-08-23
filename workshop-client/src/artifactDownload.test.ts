import { afterEach, describe, expect, it, vi } from "vitest";

import { startArtifactDownload } from "./artifactDownload";

const channelId = "chn_d3dfdfd7df9151ba8a1742b92403faa5";
const artifactId = "art_00000000000000000000000000000001";

describe("artifact download", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    document.querySelectorAll('iframe[name^="kai-artifact-download-"]').forEach(
      (frame) => frame.remove(),
    );
  });

  it("submits native download authority synchronously through a temporary frame", () => {
    vi.useFakeTimers();
    const submit = vi.spyOn(HTMLFormElement.prototype, "submit").mockImplementation(
      function (this: HTMLFormElement): void {
        expect(document.body.contains(this)).toBe(true);
        expect(this.method).toBe("post");
        expect(this.action.endsWith(
          `/v1/channels/${channelId}/artifacts/${artifactId}/download`,
        )).toBe(true);
        expect(this.target).toMatch(/^kai-artifact-download-[0-9]+$/);
        expect(document.querySelector(`iframe[name="${this.target}"]`)).toHaveAttribute(
          "sandbox",
          "allow-downloads",
        );
        expect(this.elements.namedItem("session_token")).toHaveValue("session-secret");
      },
    );

    startArtifactDownload({ channelId, token: "session-secret" }, artifactId);

    expect(submit).toHaveBeenCalledOnce();
    expect(document.querySelector("form")).toBeNull();
    expect(document.querySelector('iframe[name^="kai-artifact-download-"]')).not.toBeNull();
    vi.advanceTimersByTime(60_000);
    expect(document.querySelector('iframe[name^="kai-artifact-download-"]')).toBeNull();
  });

  it("rejects malformed authority before creating a form", () => {
    expect(() =>
      startArtifactDownload(
        { channelId: "not-a-channel", token: "session-secret" },
        artifactId,
      )
    ).toThrow("Invalid artifact download authority.");
    expect(document.querySelector("form")).toBeNull();
  });
});
