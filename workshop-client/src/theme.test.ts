import { beforeEach, describe, expect, it } from "vitest";

import {
  DEFAULT_WORKSHOP_THEME_ID,
  WORKSHOP_THEME_CATALOG,
  applyWorkshopTheme,
  clearWorkshopThemeHint,
  restoreWorkshopThemeHint,
} from "./theme";

describe("Workshop theme application", () => {
  beforeEach(() => {
    sessionStorage.clear();
    document.documentElement.removeAttribute("data-workshop-theme");
    document.documentElement.style.colorScheme = "";
  });

  it("applies and remembers an allowlisted theme", () => {
    expect(applyWorkshopTheme(DEFAULT_WORKSHOP_THEME_ID)).toBe(DEFAULT_WORKSHOP_THEME_ID);
    expect(document.documentElement.dataset.workshopTheme).toBe(DEFAULT_WORKSHOP_THEME_ID);
    expect(document.documentElement.style.colorScheme).toBe("dark");
    expect(sessionStorage.getItem("kai.workshop.theme-hint.v1")).toBe(DEFAULT_WORKSHOP_THEME_ID);
  });

  it("exposes the complete curated light and dark catalog", () => {
    expect(WORKSHOP_THEME_CATALOG.map((theme) => theme.themeId)).toEqual([
      "atom-one-dark",
      "atom-one-light",
      "dracula",
      "nord",
      "solarized-dark",
      "solarized-light",
      "catppuccin-mocha",
      "catppuccin-latte",
      "github-light-default",
      "github-dark-default",
      "github-dark-dimmed",
    ]);
    expect(WORKSHOP_THEME_CATALOG.filter((theme) => theme.colorScheme === "light")).toHaveLength(4);
    expect(WORKSHOP_THEME_CATALOG.filter((theme) => theme.colorScheme === "dark")).toHaveLength(7);
  });

  it("applies light themes before rendering and retains them across reload hints", () => {
    expect(applyWorkshopTheme("github-light-default")).toBe("github-light-default");
    expect(document.documentElement.dataset.workshopTheme).toBe("github-light-default");
    expect(document.documentElement.style.colorScheme).toBe("light");
    expect(restoreWorkshopThemeHint()).toBe("github-light-default");
  });

  it("fails closed when a render hint is unknown", () => {
    sessionStorage.setItem("kai.workshop.theme-hint.v1", "../../custom.css");
    expect(restoreWorkshopThemeHint()).toBe(DEFAULT_WORKSHOP_THEME_ID);
    expect(document.documentElement.dataset.workshopTheme).toBe(DEFAULT_WORKSHOP_THEME_ID);
  });

  it("clears a prior principal's hint", () => {
    applyWorkshopTheme(DEFAULT_WORKSHOP_THEME_ID);
    clearWorkshopThemeHint();
    expect(sessionStorage.getItem("kai.workshop.theme-hint.v1")).toBeNull();
    expect(document.documentElement.dataset.workshopTheme).toBe(DEFAULT_WORKSHOP_THEME_ID);
  });
});
