import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import type { WorkshopHumanAvatarDescriptor } from "./types";

type CachedAvatar = {
  objectUrl: string;
  touched: number;
};

export class HumanAvatarImageCache {
  private readonly entries = new Map<string, CachedAvatar>();
  private readonly pending = new Map<string, Promise<string>>();
  private readonly principalUrls = new Map<string, string>();
  private clock = 0;
  private disposed = false;

  constructor(
    private readonly token: string,
    private readonly capacity = 64,
  ) {
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new Error("Avatar cache capacity must be positive.");
    }
  }

  async load(principalId: string, url: string): Promise<string> {
    if (this.disposed) {
      throw new Error("Avatar cache is unavailable.");
    }
    const previous = this.principalUrls.get(principalId);
    if (previous && previous !== url) {
      this.evict(previous);
    }
    this.principalUrls.set(principalId, url);
    const cached = this.entries.get(url);
    if (cached) {
      cached.touched = ++this.clock;
      return cached.objectUrl;
    }
    const inFlight = this.pending.get(url);
    if (inFlight) {
      return inFlight;
    }
    const request = this.fetch(url).finally(() => this.pending.delete(url));
    this.pending.set(url, request);
    return request;
  }

  replace(principalId: string, descriptor: WorkshopHumanAvatarDescriptor): void {
    if (this.disposed) {
      return;
    }
    const previous = this.principalUrls.get(principalId);
    if (previous && previous !== descriptor.url) {
      this.evict(previous);
    }
    if (descriptor.url) {
      this.principalUrls.set(principalId, descriptor.url);
    } else {
      this.principalUrls.delete(principalId);
    }
  }

  dispose(): void {
    this.disposed = true;
    for (const entry of this.entries.values()) {
      URL.revokeObjectURL(entry.objectUrl);
    }
    this.entries.clear();
    this.pending.clear();
    this.principalUrls.clear();
  }

  private async fetch(url: string): Promise<string> {
    const response = await globalThis.fetch(url, {
      cache: "no-store",
      headers: { Authorization: `Bearer ${this.token}` },
    });
    if (!response.ok || response.headers.get("Content-Type")?.split(";", 1)[0] !== "image/png") {
      throw new Error("Avatar image is unavailable.");
    }
    const objectUrl = URL.createObjectURL(await response.blob());
    if (this.disposed || ![...this.principalUrls.values()].includes(url)) {
      URL.revokeObjectURL(objectUrl);
      throw new Error("Avatar image is no longer current.");
    }
    this.entries.set(url, { objectUrl, touched: ++this.clock });
    this.trim();
    return objectUrl;
  }

  private trim(): void {
    while (this.entries.size > this.capacity) {
      const oldest = [...this.entries.entries()].reduce((left, right) =>
        left[1].touched <= right[1].touched ? left : right,
      );
      this.evict(oldest[0]);
    }
  }

  private evict(url: string): void {
    const cached = this.entries.get(url);
    if (cached) {
      URL.revokeObjectURL(cached.objectUrl);
      this.entries.delete(url);
    }
    for (const [principalId, principalUrl] of this.principalUrls) {
      if (principalUrl === url) {
        this.principalUrls.delete(principalId);
      }
    }
  }
}

type HumanAvatarContextValue = {
  cache: HumanAvatarImageCache;
  overrides: ReadonlyMap<string, WorkshopHumanAvatarDescriptor>;
  setOverride: (principalId: string, descriptor: WorkshopHumanAvatarDescriptor) => void;
};

const HumanAvatarCacheContext = createContext<HumanAvatarContextValue | null>(null);

export function HumanAvatarCacheProvider({
  children,
  token,
}: {
  children: ReactNode;
  token: string;
}): React.JSX.Element {
  const cache = useMemo(() => new HumanAvatarImageCache(token), [token]);
  const [overrides, setOverrides] = useState<ReadonlyMap<string, WorkshopHumanAvatarDescriptor>>(
    () => new Map(),
  );
  const value = useMemo<HumanAvatarContextValue>(() => ({
    cache,
    overrides,
    setOverride: (principalId, descriptor) => {
      cache.replace(principalId, descriptor);
      setOverrides((current) => new Map(current).set(principalId, descriptor));
    },
  }), [cache, overrides]);
  useEffect(() => () => cache.dispose(), [cache]);
  return (
    <HumanAvatarCacheContext.Provider value={value}>
      {children}
    </HumanAvatarCacheContext.Provider>
  );
}

export function useHumanAvatarCache(): HumanAvatarImageCache {
  const context = useContext(HumanAvatarCacheContext);
  if (!context) {
    throw new Error("Human avatars require an avatar cache provider.");
  }
  return context.cache;
}

export function useHumanAvatarOverride(): HumanAvatarContextValue["setOverride"] {
  const context = useContext(HumanAvatarCacheContext);
  if (!context) {
    throw new Error("Human avatars require an avatar cache provider.");
  }
  return context.setOverride;
}

export function HumanAvatar({
  avatar,
  className,
  displayName,
  label,
  principalId,
}: {
  avatar: WorkshopHumanAvatarDescriptor;
  className: string;
  displayName: string;
  label?: string;
  principalId: string;
}): React.JSX.Element {
  const context = useContext(HumanAvatarCacheContext);
  if (!context) {
    throw new Error("Human avatars require an avatar cache provider.");
  }
  const { cache } = context;
  const effectiveAvatar = context.overrides.get(principalId) ?? avatar;
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const initial = displayName.trim().slice(0, 1).toUpperCase() || "?";

  useEffect(() => {
    let active = true;
    setImageUrl(null);
    cache.replace(principalId, effectiveAvatar);
    if (!effectiveAvatar.active || !effectiveAvatar.url) {
      return () => { active = false; };
    }
    void cache.load(principalId, effectiveAvatar.url)
      .then((url) => {
        if (active) {
          setImageUrl(url);
        }
      })
      .catch(() => {
        if (active) {
          setImageUrl(null);
        }
      });
    return () => { active = false; };
  }, [cache, effectiveAvatar.active, effectiveAvatar.stateVersion, effectiveAvatar.url, principalId]);

  return (
    <span
      className={className}
      aria-hidden={label ? undefined : "true"}
      aria-label={label}
      role={label ? "img" : undefined}
    >
      {imageUrl ? (
        <img src={imageUrl} alt="" onError={() => setImageUrl(null)} />
      ) : initial}
    </span>
  );
}
