import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  HumanAvatar,
  HumanAvatarCacheProvider,
  HumanAvatarImageCache,
} from "./HumanAvatar";

const ACTIVE_AVATAR = {
  active: true,
  stateVersion: 1,
  url: "/v1/principals/prn_test/avatar/1",
} as const;

function pngResponse(): Response {
  return {
    blob: async () => new Blob(["png"], { type: "image/png" }),
    headers: new Headers({ "Content-Type": "image/png" }),
    ok: true,
  } as Response;
}

describe("human avatar image cache", () => {
  beforeEach(() => {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn()
        .mockReturnValueOnce("blob:avatar-1")
        .mockReturnValueOnce("blob:avatar-2")
        .mockReturnValueOnce("blob:avatar-3"),
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetches once with bearer authorization and reuses the object URL", async () => {
    const fetchMock = vi.fn().mockResolvedValue(pngResponse());
    vi.stubGlobal("fetch", fetchMock);
    const cache = new HumanAvatarImageCache("avatar-token");

    await expect(Promise.all([
      cache.load("prn_test", ACTIVE_AVATAR.url),
      cache.load("prn_test", ACTIVE_AVATAR.url),
    ])).resolves.toEqual(["blob:avatar-1", "blob:avatar-1"]);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith(ACTIVE_AVATAR.url, {
      cache: "no-store",
      headers: { Authorization: "Bearer avatar-token" },
    });

    cache.dispose();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:avatar-1");
  });

  it("revokes replaced and least-recently-used object URLs", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(pngResponse())));
    const cache = new HumanAvatarImageCache("avatar-token", 1);

    await cache.load("prn_one", "/v1/principals/prn_one/avatar/1");
    await cache.load("prn_one", "/v1/principals/prn_one/avatar/2");
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:avatar-1");

    await cache.load("prn_two", "/v1/principals/prn_two/avatar/1");
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:avatar-2");
    cache.dispose();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:avatar-3");
  });

  it("revokes an in-flight image that is replaced before it arrives", async () => {
    let resolveResponse!: (response: Response) => void;
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(new Promise<Response>((resolve) => {
      resolveResponse = resolve;
    })));
    const cache = new HumanAvatarImageCache("avatar-token");
    const pending = cache.load("prn_test", ACTIVE_AVATAR.url);

    cache.replace("prn_test", { active: false, stateVersion: 2, url: null });
    resolveResponse(pngResponse());

    await expect(pending).rejects.toThrow("no longer current");
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:avatar-1");
    cache.dispose();
  });

  it("keeps the accessible initial fallback when image retrieval fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      headers: new Headers(),
      ok: false,
    } as Response));
    render(
      <HumanAvatarCacheProvider token="avatar-token">
        <HumanAvatar
          avatar={ACTIVE_AVATAR}
          className="test-avatar"
          displayName="Daniel"
          label="Daniel's avatar"
          principalId="prn_test"
        />
      </HumanAvatarCacheProvider>,
    );

    const avatar = screen.getByRole("img", { name: "Daniel's avatar" });
    expect(avatar).toHaveTextContent("D");
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalledOnce());
    expect(avatar).toHaveTextContent("D");
  });
});
