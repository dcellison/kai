import { beforeEach, describe, expect, it } from "vitest";

import {
  DEFAULT_WORKSHOP_THEME_ID,
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
