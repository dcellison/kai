import { afterEach, describe, expect, it, vi } from "vitest";

import { startArtifactDownload } from "./artifactDownload";

const channelId = "chn_d3dfdfd7df9151ba8a1742b92403faa5";
const artifactId = "art_00000000000000000000000000000001";

describe("artifact download", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("submits native download authority synchronously in the active page", () => {
    const submit = vi.spyOn(HTMLFormElement.prototype, "submit").mockImplementation(
      function (this: HTMLFormElement): void {
        expect(document.body.contains(this)).toBe(true);
        expect(this.method).toBe("post");
        expect(this.action.endsWith(
          `/v1/channels/${channelId}/artifacts/${artifactId}/download`,
        )).toBe(true);
        expect(this.target).toBe("_self");
        expect(this.elements.namedItem("session_token")).toHaveValue("session-secret");
      },
    );

    startArtifactDownload({ channelId, token: "session-secret" }, artifactId);

    expect(submit).toHaveBeenCalledOnce();
    expect(document.querySelector("form")).toBeNull();
    expect(document.querySelector("iframe")).toBeNull();
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
