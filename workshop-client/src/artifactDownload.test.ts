import { afterEach, describe, expect, it, vi } from "vitest";

import { downloadArtifactBlob } from "./artifactDownload";

describe("artifact download", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("keeps the object URL alive while the browser claims the download", () => {
    vi.useFakeTimers();
    const createObjectURL = vi.spyOn(URL, "createObjectURL").mockReturnValue(
      "blob:workshop-artifact",
    );
    const revokeObjectURL = vi.spyOn(URL, "revokeObjectURL");
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
      function (this: HTMLAnchorElement): void {
        expect(document.body.contains(this)).toBe(true);
        expect(this.download).toBe("qualification.aiff");
      },
    );
    const blob = new Blob(["audio bytes"], { type: "audio/aiff" });

    downloadArtifactBlob(blob, "qualification.aiff");

    expect(createObjectURL).toHaveBeenCalledWith(blob);
    expect(click).toHaveBeenCalledOnce();
    expect(document.querySelector('a[href="blob:workshop-artifact"]')).toBeNull();
    expect(revokeObjectURL).not.toHaveBeenCalled();

    vi.advanceTimersByTime(59_999);
    expect(revokeObjectURL).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(revokeObjectURL).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:workshop-artifact");
  });
});
